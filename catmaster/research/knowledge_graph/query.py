from __future__ import annotations

import re
import secrets
import sqlite3
from pathlib import Path
from typing import Any
from urllib.parse import quote

from catmaster.storage.workspace_db import workspace_database_path


_LOGICAL_TABLES = {
    "research_graphs",
    "research_nodes",
    "research_edges",
    "research_refs",
    "research_launches",
    "research_planning",
    "workspace_artifacts",
    "thread_messages",
}
_SCHEMA_QUALIFIER_RE = re.compile(
    r"(?ix)(?<![\w])(?:main|temp|\"main\"|\"temp\"|`main`|`temp`|\[main\]|\[temp\])\s*\."
)


def _sql_literal(value: str) -> str:
    return "'" + str(value).replace("'", "''") + "'"


def _without_sql_literals_and_comments(statement: str) -> str:
    """Blank string literals and comments before a narrow qualifier check."""

    text = str(statement)
    out: list[str] = []
    index = 0
    while index < len(text):
        char = text[index]
        next_char = text[index + 1] if index + 1 < len(text) else ""
        if char == "'":
            out.append(" ")
            index += 1
            while index < len(text):
                if text[index] == "'":
                    if index + 1 < len(text) and text[index + 1] == "'":
                        out.extend((" ", " "))
                        index += 2
                        continue
                    out.append(" ")
                    index += 1
                    break
                out.append("\n" if text[index] == "\n" else " ")
                index += 1
            continue
        if char == "-" and next_char == "-":
            out.extend((" ", " "))
            index += 2
            while index < len(text) and text[index] != "\n":
                out.append(" ")
                index += 1
            continue
        if char == "/" and next_char == "*":
            out.extend((" ", " "))
            index += 2
            while index < len(text):
                if text[index] == "*" and index + 1 < len(text) and text[index + 1] == "/":
                    out.extend((" ", " "))
                    index += 2
                    break
                out.append("\n" if text[index] == "\n" else " ")
                index += 1
            continue
        out.append(char)
        index += 1
    return "".join(out)


class ResearchGraphSQLQuery:
    """Execute standard SQL through graph-bound logical views."""

    def __init__(self, workspace: Path | str) -> None:
        self.workspace = Path(workspace).expanduser().resolve()

    @staticmethod
    def _install_views(
        connection: sqlite3.Connection,
        graph_id: str,
    ) -> dict[str, str]:
        graph = _sql_literal(graph_id)
        nonce = secrets.token_hex(16)
        internal = {
            table: f"_catmaster_bound_{nonce}_{table}"
            for table in _LOGICAL_TABLES
        }
        connection.executescript(
            f"""
            CREATE TEMP VIEW {internal['research_graphs']} AS
            SELECT graph_id, title, question, completion_criterion, completed,
                   orchestration_mode, archived, revision, created_at, updated_at
            FROM main.research_graphs
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_graphs AS
            SELECT * FROM {internal['research_graphs']};

            CREATE TEMP VIEW {internal['research_nodes']} AS
            SELECT graph_id, node_id, kind, title, state, body_json, revision,
                   created_at, updated_at
            FROM main.research_nodes
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_nodes AS
            SELECT * FROM {internal['research_nodes']};

            CREATE TEMP VIEW {internal['research_edges']} AS
            SELECT graph_id, source_node_id, target_node_id, relation
            FROM main.research_edges
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_edges AS
            SELECT * FROM {internal['research_edges']};

            CREATE TEMP VIEW {internal['research_refs']} AS
            SELECT graph_id, node_id, ref_kind, ref_id
            FROM main.research_refs
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_refs AS
            SELECT * FROM {internal['research_refs']};

            CREATE TEMP VIEW {internal['research_launches']} AS
            SELECT launch_id, graph_id, experiment_node_id, idempotency_key,
                   status, thread_id, run_id, lease_owner, lease_until,
                   created_at, updated_at
            FROM main.research_launches
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_launches AS
            SELECT * FROM {internal['research_launches']};

            CREATE TEMP VIEW {internal['research_planning']} AS
            SELECT planning_id, graph_id, start_revision, status, thread_id,
                   preview_json, lease_until, created_at, updated_at
            FROM main.research_planning
            WHERE graph_id = {graph};
            CREATE TEMP VIEW research_planning AS
            SELECT * FROM {internal['research_planning']};

            CREATE TEMP VIEW {internal['workspace_artifacts']} AS
            SELECT artifact_id, thread_id, payload_json, created_at, updated_at
            FROM main.workspace_artifacts AS artifact
            WHERE EXISTS (
                SELECT 1
                FROM main.research_refs AS ref
                WHERE ref.graph_id = {graph}
                  AND ref.ref_kind = 'artifact'
                  AND ref.ref_id = artifact.artifact_id
            );
            CREATE TEMP VIEW workspace_artifacts AS
            SELECT * FROM {internal['workspace_artifacts']};

            CREATE TEMP VIEW {internal['thread_messages']} AS
            SELECT row_id, thread_id, message_id, created_at, updated_at,
                   payload_json, message_role, message_run_id
            FROM main.thread_messages AS message
            WHERE EXISTS (
                SELECT 1
                FROM main.research_refs AS ref
                WHERE ref.graph_id = {graph}
                  AND (
                    (ref.ref_kind = 'thread' AND ref.ref_id = message.thread_id)
                    OR (
                      ref.ref_kind = 'message'
                      AND (
                        ref.ref_id = message.thread_id || ':' || message.message_id
                        OR (
                          instr(ref.ref_id, ':') = 0
                          AND ref.ref_id = message.message_id
                          AND 1 = (
                            SELECT COUNT(DISTINCT legacy_message.thread_id)
                            FROM main.thread_messages AS legacy_message
                            WHERE legacy_message.message_id = ref.ref_id
                          )
                        )
                      )
                    )
                  )
            );
            CREATE TEMP VIEW thread_messages AS
            SELECT * FROM {internal['thread_messages']};
            """
        )
        return internal

    @staticmethod
    def _authorizer(
        internal: dict[str, str],
    ) -> tuple[Any, dict[str, bool]]:
        state = {"select_seen": False}
        internal_to_public = {
            internal_view: public
            for public, internal_view in internal.items()
        }
        trusted_main_reads = {
            internal["research_graphs"]: {"research_graphs"},
            internal["research_nodes"]: {"research_nodes"},
            internal["research_edges"]: {"research_edges"},
            internal["research_refs"]: {"research_refs"},
            internal["research_launches"]: {"research_launches"},
            internal["research_planning"]: {"research_planning"},
            internal["workspace_artifacts"]: {
                "workspace_artifacts",
                "research_refs",
            },
            internal["thread_messages"]: {
                "thread_messages",
                "research_refs",
            },
        }
        denied_functions = {
            "edit",
            "eval",
            "fts3_tokenizer",
            "fsdir",
            "getenv",
            "load_extension",
            "readfile",
            "shell",
            "system",
            "writefile",
        }

        def authorize(
            action: int,
            arg1: str | None,
            arg2: str | None,
            database: str | None,
            source: str | None,
        ) -> int:
            if action == sqlite3.SQLITE_SELECT:
                state["select_seen"] = True
                return sqlite3.SQLITE_OK
            if action == getattr(sqlite3, "SQLITE_RECURSIVE", -1):
                return sqlite3.SQLITE_OK
            if action == sqlite3.SQLITE_READ:
                table = str(arg1 or "")
                db_name = str(database or "")
                view = str(source or "")
                if db_name == "temp" and table in _LOGICAL_TABLES:
                    return sqlite3.SQLITE_OK
                expected_public_view = internal_to_public.get(table, "")
                if (
                    db_name == "temp"
                    and expected_public_view
                    and view == expected_public_view
                ):
                    return sqlite3.SQLITE_OK
                if (
                    db_name == "main"
                    and table in trusted_main_reads.get(view, set())
                ):
                    return sqlite3.SQLITE_OK
                return sqlite3.SQLITE_DENY
            if action == sqlite3.SQLITE_FUNCTION:
                function_name = str(arg2 or arg1 or "").casefold()
                return (
                    sqlite3.SQLITE_DENY
                    if function_name in denied_functions
                    or function_name.startswith("pragma_")
                    else sqlite3.SQLITE_OK
                )
            # Only SELECT/read/function actions are needed for SELECT and WITH,
            # including recursive CTEs, aggregates, windows, and JSON1.
            return sqlite3.SQLITE_DENY

        return authorize, state

    def execute(self, *, graph_id: str, sql: str) -> dict[str, Any]:
        statement = str(sql or "").strip()
        if not statement:
            raise ValueError("sql must contain one SELECT or WITH statement.")
        structural_sql = _without_sql_literals_and_comments(statement)
        leading = structural_sql.lstrip()
        if not re.match(r"(?is)^(?:select|with)\b", leading):
            raise ValueError("sql must contain one SELECT or WITH statement.")
        if _SCHEMA_QUALIFIER_RE.search(structural_sql):
            raise ValueError(
                "The bound Research Graph query was rejected: schema-qualified "
                "names are not available. Use the documented logical table names."
            )
        database_path = workspace_database_path(self.workspace).resolve()
        uri = f"file:{quote(str(database_path), safe='/')}?mode=ro"
        connection = sqlite3.connect(uri, uri=True)
        connection.row_factory = sqlite3.Row
        try:
            internal = self._install_views(connection, graph_id)
            connection.execute("PRAGMA query_only=ON")
            connection.execute("BEGIN")
            authorize, state = self._authorizer(internal)
            connection.set_authorizer(authorize)
            try:
                cursor = connection.execute(statement)
                rows = cursor.fetchall()
                revision_row = connection.execute(
                    "SELECT revision FROM research_graphs"
                ).fetchone()
            except sqlite3.Error as exc:
                raise ValueError(
                    "The bound Research Graph query was rejected or invalid: "
                    f"{exc}"
                ) from exc
            finally:
                connection.set_authorizer(None)
            if not state["select_seen"] or cursor.description is None:
                raise ValueError("sql must contain one SELECT or WITH statement.")
            if revision_row is None:
                raise ValueError("The bound Research Graph no longer exists.")
            columns = [str(item[0]) for item in cursor.description]
            return {
                "revision": int(revision_row["revision"]),
                "columns": columns,
                "rows": [dict(row) for row in rows],
                "row_count": len(rows),
            }
        finally:
            connection.close()


__all__ = ["ResearchGraphSQLQuery"]
