"""Shared scientific discussion, separate from agent execution and Graph claims."""
from __future__ import annotations

import json
import time
import uuid
from pathlib import Path
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.storage import connect_workspace_db


class DiscussionPostRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    body: str = Field(min_length=1)
    title: str = ""
    reply_to: str = ""
    node_id: str = ""
    target_task_id: str = ""
    references: list[str] = Field(default_factory=list)
    resolves_message_id: str = ""
    review_outcome: Literal["addressed", "deferred", "follow_up"] = "addressed"


def ensure_discussion_schema(connection):
    connection.executescript("""
        CREATE TABLE IF NOT EXISTS research_discussions (
            seq INTEGER PRIMARY KEY AUTOINCREMENT,
            message_id TEXT NOT NULL UNIQUE,
            graph_id TEXT NOT NULL REFERENCES research_graphs(graph_id) ON DELETE CASCADE,
            discussion_id TEXT NOT NULL,
            title TEXT NOT NULL,
            body TEXT NOT NULL,
            author_thread_id TEXT NOT NULL,
            author_kind TEXT NOT NULL CHECK(author_kind IN ('agent', 'user')),
            target_task_id TEXT NOT NULL DEFAULT '',
            node_id TEXT NOT NULL DEFAULT '',
            reply_to TEXT NOT NULL DEFAULT '',
            references_json TEXT NOT NULL DEFAULT '[]',
            created_at REAL NOT NULL
        );
        CREATE INDEX IF NOT EXISTS research_discussions_graph_seq ON research_discussions(graph_id, seq);
        CREATE INDEX IF NOT EXISTS research_discussions_topic ON research_discussions(graph_id, discussion_id, seq);
        CREATE INDEX IF NOT EXISTS research_discussions_target ON research_discussions(graph_id, target_task_id, seq);
        CREATE INDEX IF NOT EXISTS research_discussions_node ON research_discussions(graph_id, node_id, seq);
    """)
    # Additive workspace data, independent of native conversation checkpoints.
    connection.execute("BEGIN IMMEDIATE")
    columns = {row[1] for row in connection.execute("PRAGMA table_info(research_discussions)")}
    for name, declaration in {
        "review_status": "TEXT NOT NULL DEFAULT ''",
        "review_reason": "TEXT NOT NULL DEFAULT ''",
        "review_response_id": "TEXT NOT NULL DEFAULT ''",
        "review_notified_run_id": "TEXT NOT NULL DEFAULT ''",
    }.items():
        if name not in columns:
            connection.execute(f"ALTER TABLE research_discussions ADD COLUMN {name} {declaration}")


class ResearchDiscussions:
    def __init__(self, workspace: Path | str):
        from .knowledge_graph.store import ResearchGraphStore
        from catmaster.webui.thread_store import ThreadStore

        self.workspace = Path(workspace).resolve()
        self.graphs = ResearchGraphStore(self.workspace)
        self.threads = ThreadStore(workspace=self.workspace)

    @staticmethod
    def _row(row):
        result = dict(row)
        result["references"] = json.loads(result.pop("references_json"))
        result["review_notified"] = bool(result.pop("review_notified_run_id"))
        return result

    def _bound_thread(self, graph_id, thread_id):
        thread = self.threads.get_thread(thread_id)
        if thread.active_research_graph_id != graph_id:
            raise ValueError("The research task belongs to a different graph.")
        return thread

    def persistent_owner(self, graph_id, thread_id=""):
        """The declared persistent owner and its descendants opt into collaboration.

        An unrelated ordinary Research thread does not opt in by viewing the same
        graph. Branches retain their real Research entrypoint and native topology.
        """
        graph = self.graphs.get_graph(graph_id)
        owner_id = graph['orchestration_thread_id']
        if not owner_id:
            return ''
        try:
            owner = self._bound_thread(graph_id, owner_id)
            if owner.entrypoint != 'persistent_research':
                return ''
            if not thread_id:
                return owner_id
            thread = self._bound_thread(graph_id, thread_id)
            while thread.thread_id != owner_id:
                if not thread.parent_thread_id:
                    return ''
                thread = self.threads.get_thread(thread.parent_thread_id)
            return owner_id
        except (KeyError, ValueError):
            return ''

    def post(self, graph_id: str, payload: DiscussionPostRequest, *, author_thread_id="",
             author_kind="agent", message_id=""):
        if author_thread_id:
            self._bound_thread(graph_id, author_thread_id)
        elif author_kind != "user":
            raise ValueError("An agent discussion requires its bound research thread.")
        if not self.persistent_owner(graph_id, author_thread_id):
            raise ValueError("Shared research discussion belongs to a Persistent Research session and its branches.")
        if not payload.body.strip():
            raise ValueError("Discussion text is required.")
        message_id = message_id or "discussion_" + uuid.uuid4().hex
        with connect_workspace_db(self.workspace) as connection:
            connection.execute("BEGIN IMMEDIATE")
            graph = connection.execute("SELECT revision, orchestration_thread_id FROM research_graphs WHERE graph_id=?", (graph_id,)).fetchone()
            if graph is None:
                raise KeyError(graph_id)
            existing = connection.execute("SELECT * FROM research_discussions WHERE message_id=? AND graph_id=?",
                                          (message_id, graph_id)).fetchone()
            if existing:
                return self._row(existing)
            if payload.resolves_message_id:
                if author_kind != 'user' and author_thread_id != graph['orchestration_thread_id']:
                    raise ValueError("Only the main research thread or user can record the review disposition.")
                requested = connection.execute("SELECT * FROM research_discussions WHERE graph_id=? AND message_id=?",
                    (graph_id, payload.resolves_message_id)).fetchone()
                if requested is None:
                    raise ValueError("The discussion message is not in this graph.")
                if payload.reply_to != payload.resolves_message_id:
                    raise ValueError("Reply to the source message when recording the research decision.")
            parent = None
            if payload.reply_to:
                parent = connection.execute("SELECT * FROM research_discussions WHERE graph_id=? AND message_id=?",
                                            (graph_id, payload.reply_to)).fetchone()
                if parent is None:
                    raise ValueError("The replied-to message is not in this research graph.")
            title = parent["title"] if parent else payload.title.strip()
            if not title:
                raise ValueError("A new discussion needs a title; a reply inherits its topic.")
            node_id = parent["node_id"] if parent else payload.node_id
            if parent and payload.node_id and payload.node_id != node_id:
                raise ValueError("A reply keeps its discussion's node; add other sources in references.")
            if node_id and not connection.execute("SELECT 1 FROM research_nodes WHERE graph_id=? AND node_id=?",
                                                 (graph_id, node_id)).fetchone():
                raise ValueError("The associated scientific node is not in this graph.")
            target = payload.target_task_id
            if not target and parent and parent["author_kind"] == "agent":
                target = parent["author_thread_id"]
            if target:
                recipient = self._bound_thread(graph_id, target)
                if recipient.entrypoint not in {"research", "persistent_research"}:
                    raise ValueError("Only Research entrypoints receive targeted notices; leave target_task_id empty for shared discussion.")
                if not self.persistent_owner(graph_id, target):
                    raise ValueError("The recipient is not a branch of this Persistent Research session.")
            discussion_id = parent["discussion_id"] if parent else message_id
            connection.execute("""INSERT INTO research_discussions
                (message_id, graph_id, discussion_id, title, body, author_thread_id, author_kind,
                 target_task_id, node_id, reply_to, references_json, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)""",
                (message_id, graph_id, discussion_id, title, payload.body.strip(), author_thread_id,
                 author_kind, target, node_id, payload.reply_to,
                 json.dumps(payload.references, ensure_ascii=False), time.time()))
            if payload.resolves_message_id:
                connection.execute("UPDATE research_discussions SET review_status=?, review_response_id=? WHERE message_id=?",
                    (payload.review_outcome, message_id, payload.resolves_message_id))
            # UI outbox. Execution admission is handled by the existing thread service.
            self.graphs._write_event(connection, graph_id=graph_id, revision=graph["revision"],
                change="discussion.posted", thread_id=author_thread_id,
                node_ids=[node_id] if node_id else [], details={"message_id": message_id, "discussion_id": discussion_id})
            row = connection.execute("SELECT * FROM research_discussions WHERE message_id=?", (message_id,)).fetchone()
        return self._row(row)

    def messages(self, graph_id, *, discussion_id="", node_id="", target_task_id="", after_seq=0, before_seq=0, limit=50):
        self.graphs.get_graph(graph_id)
        where, args = ["graph_id=?", "seq> ?"], [graph_id, max(0, after_seq)]
        if before_seq:
            where.append("seq<?")
            args.append(before_seq)
        for key, value in (("discussion_id", discussion_id), ("node_id", node_id), ("target_task_id", target_task_id)):
            if value:
                where.append(f"{key}=?")
                args.append(value)
        limit = max(1, min(100, limit))
        with connect_workspace_db(self.workspace) as connection:
            rows = connection.execute("SELECT * FROM research_discussions WHERE " + " AND ".join(where)
                                      + " ORDER BY seq DESC LIMIT ?", (*args, limit + 1)).fetchall()
        owner = self.persistent_owner(graph_id)
        return {"messages": [self._row(row) for row in reversed(rows[:limit])],
                "collaboration_enabled": bool(owner), "main_thread_id": owner,
                "next_before_seq": rows[limit - 1]["seq"] if len(rows) > limit else None}

    def notice(self, graph_id, thread_id, after_seq, *, all_graph=False):
        # One aggregate and stable source range, not an ever-growing transcript broadcast.
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute("""SELECT COUNT(*) AS count, MAX(seq) AS last_seq
                FROM research_discussions WHERE graph_id=? AND seq>? AND author_thread_id != ?"""
                + ("" if all_graph else " AND target_task_id=?"),
                (graph_id, after_seq, thread_id, *([] if all_graph else [thread_id]))).fetchone()
        return dict(row)
