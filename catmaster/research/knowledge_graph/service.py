from __future__ import annotations

import asyncio
import json
import logging
import re
import socket
import threading
import time
from pathlib import Path
from typing import Any, Callable
from urllib.parse import quote, urlparse

from catmaster.tools.base import system_root
from catmaster.webui.artifact_registry import ArtifactRegistry
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import (
    ArtifactPart,
    ArtifactRecord,
    is_persistent_research_root,
    MessagePart,
    ThreadMessage,
    ThreadRole,
    ThreadSubmitRequest,
    ThreadStatus,
)
from catmaster.webui.thread_store import ThreadStore

from .context import (
    ResearchGraphContextBuilder,
    external_handoff_ids,
    runnable_frontier_ids,
)
from .models import (
    BoundExperimentCreateRequest,
    EdgeRelation,
    ExperimentCreateRequest,
    ExperimentState,
    GraphCreateRequest,
    GraphPatchRequest,
    HypothesisCreateRequest,
    NodeKind,
    NodePatchRequest,
    RefKind,
    ResearchExperimentPairOutcomeDraft,
    ResearchExperimentProposal,
    ResearchGraphPlanningDraft,
    ResearchGraphPlanningProposal,
    ResearchHypothesisProposal,
    ResearchRefInput,
    ResultCreateRequest,
    ResultJudgmentSetRequest,
)
from .planning import build_planning_preview
from .store import ResearchGraphConflict, ResearchGraphStore

_DOI_RE = re.compile(r"^10\.\d{4,9}/\S+$", re.IGNORECASE)
_ACTIVE_LAUNCH_STATUSES = {"claimed", "submitting", "running", "unknown"}
_PLANNING_THREAD_KIND = "research_graph_planning"
_COMPARISON_THREAD_KIND = "research_graph_experiment_comparison"
_WAIT_CANDIDATE_ID = "__wait__"
_THREAD_ROLE_MIGRATION_LOCK = threading.Lock()
_THREAD_ROLE_MIGRATED_WORKSPACES: set[str] = set()
logger = logging.getLogger(__name__)


class ResearchGraphService:
    """Workspace graph domain service plus child-thread orchestration."""

    def __init__(
        self,
        *,
        workspace: Path | str,
        workspace_id: str = "",
        agent_loop_factory: Callable[[Path, str], Any] | None = None,
        event_broker: ThreadEventBroker | None = None,
        worker_id: str = "",
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name).strip() or self.workspace.name
        self.store = ResearchGraphStore(self.workspace)
        self.thread_store = ThreadStore(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
        )
        migration_key = str(self.workspace)
        with _THREAD_ROLE_MIGRATION_LOCK:
            if migration_key not in _THREAD_ROLE_MIGRATED_WORKSPACES:
                changed_threads = self.thread_store.migrate_research_thread_roles()
                self._reconcile_migrated_orchestration_roots(changed_threads)
                _THREAD_ROLE_MIGRATED_WORKSPACES.add(migration_key)
        self.artifact_registry = ArtifactRegistry(
            workspace=self.workspace,
            workspace_id=self.workspace_id,
        )
        self.context_builder = ResearchGraphContextBuilder(
            workspace=self.workspace,
            store=self.store,
        )
        self.agent_loop_factory = agent_loop_factory
        self.event_broker = event_broker
        self.worker_id = str(worker_id or f"{socket.gethostname()}:{id(self)}")

    def _reconcile_migrated_orchestration_roots(
        self,
        changed_threads: list[Any],
    ) -> None:
        """Repair graph ownership only when role migration changed bound threads."""

        graph_ids = {
            str(thread.active_research_graph_id or "").strip()
            for thread in changed_threads
            if str(thread.active_research_graph_id or "").strip()
        }
        for graph_id in graph_ids:
            self.reconcile_orchestration_thread(graph_id)

    def reconcile_orchestration_thread(self, graph_id: str) -> str:
        """Keep orchestration ownership on one explicit Persistent Research root."""

        try:
            graph = self.store.get_graph(graph_id)
        except KeyError:
            return ""
        candidates = [
            thread.thread_id
            for thread in self.thread_store.list_threads()
            if thread.active_research_graph_id == graph_id
            and is_persistent_research_root(thread)
        ]
        current = str(graph.get("orchestration_thread_id") or "").strip()
        if current in candidates:
            return current
        replacement = candidates[0] if len(candidates) == 1 else ""
        if replacement != current:
            self.store.set_orchestration_thread(graph_id, replacement)
        return replacement

    def _session_root_id(self, graph_id: str, thread_id: str = "") -> str:
        """Resolve an explicit Research Session root without graph-title inference."""

        requested = str(thread_id or "").strip()
        if requested:
            try:
                thread = self.thread_store.get_thread(requested)
            except KeyError:
                return ""
            if (
                thread.active_research_graph_id == graph_id
                and is_persistent_research_root(thread)
            ):
                return thread.thread_id
            if (
                ThreadRole(thread.thread_role)
                in {
                    ThreadRole.RESEARCH_EXECUTION,
                    ThreadRole.RESEARCH_PLANNING,
                    ThreadRole.RESEARCH_COMPARISON,
                }
                and thread.parent_thread_id
            ):
                try:
                    parent = self.thread_store.get_thread(thread.parent_thread_id)
                except KeyError:
                    return ""
                if (
                    parent.active_research_graph_id == graph_id
                    and is_persistent_research_root(parent)
                ):
                    return parent.thread_id
            return ""
        try:
            graph = self.store.get_graph(graph_id)
        except KeyError:
            return ""
        routed = str(graph.get("orchestration_thread_id") or "").strip()
        if routed:
            try:
                thread = self.thread_store.get_thread(routed)
            except KeyError:
                pass
            else:
                if (
                    thread.active_research_graph_id == graph_id
                    and is_persistent_research_root(thread)
                ):
                    return thread.thread_id
        candidates = [
            thread.thread_id
            for thread in self.thread_store.list_threads()
            if thread.active_research_graph_id == graph_id
            and is_persistent_research_root(thread)
        ]
        return candidates[0] if len(candidates) == 1 else ""

    @staticmethod
    def _session_message_id(kind: str, token: str) -> str:
        normalized_kind = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(kind or "update"))
        normalized_token = re.sub(r"[^A-Za-z0-9_.:-]+", "_", str(token or "state"))
        return f"msg_rs_{normalized_kind}_{normalized_token}"[:120]

    def _append_session_message(
        self,
        *,
        root_thread_id: str,
        message_id: str,
        text: str,
        meta: dict[str, Any],
        artifact: ArtifactRecord | None = None,
        replace: bool = False,
    ) -> None:
        """Persist one host-authored Research Session update and stream it live."""

        if not root_thread_id:
            return
        try:
            existing = self.thread_store.get_message(root_thread_id, message_id)
            if existing and not replace:
                return
            parts: list[Any] = [
                MessagePart(
                    id=f"part_text_{message_id.removeprefix('msg_')}",
                    type="text",
                    text=str(text or "").strip(),
                    status="completed",
                )
            ]
            if artifact is not None:
                parts.append(
                    ArtifactPart(
                        id=f"part_artifact_{message_id.removeprefix('msg_')}",
                        type="artifact",
                        status="completed",
                        artifact_id=str(artifact.artifact_id),
                        renderer=str(artifact.renderer or "text"),
                        title=str(artifact.title or Path(artifact.path).name),
                        summary="Detailed research report",
                        path=str(artifact.path),
                    )
                )
            message = ThreadMessage(
                id=message_id,
                thread_id=root_thread_id,
                role="assistant",
                status="completed",
                parts=parts,
                meta={"kind": "research_session_writeback", **dict(meta or {})},
            )
            if existing:
                # Amend the same visible Result, retaining its original place
                # and source identity instead of publishing duplicate evidence.
                message = self.thread_store.update_message(root_thread_id, message_id,
                    parts=[part.model_dump(mode="json") for part in parts],
                    meta={**existing.meta, **{k: v for k, v in message.meta.items() if v != ""}})
            else:
                self.thread_store.append_message(message)
            broker = self.event_broker or ThreadEventBroker(workspace=self.workspace)
            broker.emit(
                root_thread_id,
                "message.updated" if existing else "message.created",
                message_id=message.id,
                status="completed",
                data={"message": message.model_dump(mode="json")},
            )
        except Exception:
            # Scientific graph mutation is already durable at this point. A UI
            # projection failure must not make the agent repeat that mutation.
            logger.exception(
                "Failed to project Research Session writeback %s to %s",
                message_id,
                root_thread_id,
            )

    @staticmethod
    def _report_path_score(path: str) -> int:
        candidate = Path(str(path or ""))
        name = candidate.name.casefold()
        suffix = candidate.suffix.casefold()
        score = 0
        if name == "scientific_report.md":
            score += 100
        if "report" in name:
            score += 60
        if "result" in name or "findings" in name:
            score += 45
        if "summary" in name or "conclusion" in name:
            score += 35
        if suffix in {".md", ".markdown", ".pdf", ".docx", ".html"}:
            score += 10
        return score

    def _report_artifact_for_refs(
        self,
        refs: list[dict[str, str]],
        *,
        owner_thread_id: str = "",
        run_id: str = "",
        register_missing: bool = False,
    ) -> tuple[ArtifactRecord | None, str]:
        candidates: list[tuple[int, ArtifactRecord | None, str]] = []
        for ref in refs:
            kind = str(ref.get("ref_kind") or "")
            ref_id = str(ref.get("ref_id") or "").strip()
            if kind == RefKind.ARTIFACT.value:
                artifact = self.artifact_registry.get(ref_id)
                if artifact is not None:
                    score = self._report_path_score(artifact.path)
                    descriptor = " ".join(
                        [
                            str(artifact.title or ""),
                            str(artifact.summary or ""),
                        ]
                    ).casefold()
                    if "report" in descriptor:
                        score += 60
                    if "result" in descriptor or "findings" in descriptor:
                        score += 45
                    if score < 40:
                        continue
                    if (
                        register_missing
                        and owner_thread_id
                        and artifact.thread_id != owner_thread_id
                    ):
                        try:
                            artifact = self.artifact_registry.register_path(
                                artifact.path,
                                thread_id=owner_thread_id,
                                run_id=run_id,
                                summary="Research result report",
                                meta={"source": "research_result"},
                            )
                        except (OSError, ValueError):
                            pass
                    candidates.append(
                        (score + 20, artifact, artifact.path)
                    )
                continue
            if kind != RefKind.NOTE.value:
                continue
            resolved = self._safe_note_path(ref_id)
            if resolved is None:
                continue
            path, _candidate = resolved
            score = self._report_path_score(path)
            if score < 40:
                continue
            artifact = self.artifact_registry.find_by_path(
                path,
                thread_id=owner_thread_id,
            )
            if artifact is None and register_missing:
                try:
                    artifact = self.artifact_registry.register_path(
                        path,
                        thread_id=owner_thread_id,
                        run_id=run_id,
                        summary="Research result report",
                        meta={"source": "research_result"},
                    )
                except (OSError, ValueError):
                    artifact = None
            if artifact is None:
                artifact = self.artifact_registry.find_by_path(path)
            candidates.append((score, artifact, path))
        if not candidates:
            return None, ""
        _score, artifact, path = max(
            candidates,
            key=lambda item: (
                item[0],
                float(getattr(item[1], "updated_at", 0.0) or 0.0),
                item[2],
            ),
        )
        return artifact, path

    def latest_session_result(
        self,
        graph_id: str,
        *,
        session_root_thread_id: str = "",
    ) -> dict[str, Any]:
        """Project the newest Result owned by one Persistent Research session."""

        root_id = self._session_root_id(graph_id, session_root_thread_id)
        snapshot = self.store.get_snapshot(graph_id)
        result_nodes = {
            str(node["node_id"]): node
            for node in snapshot["nodes"]
            if str(node.get("kind") or "") == NodeKind.RESULT.value
        }
        if not result_nodes:
            return {}
        child_ids = {
            thread.thread_id
            for thread in self.thread_store.list_threads()
            if root_id
            and str(thread.parent_thread_id or "") == root_id
            and ThreadRole(thread.thread_role) is ThreadRole.RESEARCH_EXECUTION
        }
        refs_by_node: dict[str, list[dict[str, str]]] = {}
        for ref in snapshot["refs"]:
            refs_by_node.setdefault(str(ref["node_id"]), []).append(dict(ref))
        thread_owned = {
            node_id
            for node_id, refs in refs_by_node.items()
            if any(
                str(ref.get("ref_kind") or "") == RefKind.THREAD.value
                and str(ref.get("ref_id") or "") in child_ids
                for ref in refs
            )
        }
        session_experiment_ids = {
            str(launch["experiment_node_id"])
            for launch in snapshot["launches"]
            if root_id
            and str(launch.get("session_root_thread_id") or "") == root_id
        }
        produced = {
            str(edge["target_node_id"])
            for edge in snapshot["edges"]
            if str(edge.get("relation") or "") == EdgeRelation.PRODUCES.value
            and str(edge.get("source_node_id") or "") in session_experiment_ids
        }
        candidate_ids = (thread_owned | produced) & set(result_nodes)
        graph_root = str(snapshot["graph"].get("orchestration_thread_id") or "")
        if root_id and graph_root == root_id:
            # The graph owner integrates every branch, including independent
            # async researchers without a legacy Experiment launch binding.
            candidate_ids = set(result_nodes)
        if not candidate_ids:
            return {}
        latest = max(
            (result_nodes[node_id] for node_id in candidate_ids),
            key=lambda node: (
                float(node.get("updated_at") or 0.0),
                float(node.get("created_at") or 0.0),
                str(node.get("node_id") or ""),
            ),
        )
        refs = refs_by_node.get(str(latest["node_id"]), [])
        artifact, report_path = self._report_artifact_for_refs(
            refs,
            owner_thread_id=root_id,
        )
        return {
            "node_id": str(latest["node_id"]),
            "title": str(latest.get("title") or "Research result"),
            "summary": str(latest.get("body", {}).get("summary") or ""),
            "updated_at": float(latest.get("updated_at") or 0.0),
            "report_artifact_id": str(getattr(artifact, "artifact_id", "") or ""),
            "report_path": str(getattr(artifact, "path", "") or report_path),
            "report_renderer": str(getattr(artifact, "renderer", "") or ""),
        }

    def _write_result_milestone(
        self,
        *,
        graph_id: str,
        result: dict[str, Any],
        refs: list[dict[str, str]],
        launch_id: str = "",
        run_id: str = "",
    ) -> None:
        launch = self.store.get_launch(launch_id) if launch_id else None
        owner_thread_id = str((launch or {}).get("thread_id") or "")
        root_id = self._session_root_id(
            graph_id,
            str((launch or {}).get("session_root_thread_id") or owner_thread_id),
        )
        if not root_id:
            return
        artifact, _report_path = self._report_artifact_for_refs(
            refs,
            owner_thread_id=root_id,
            run_id=run_id,
            register_missing=True,
        )
        title = str(result.get("title") or "Research result").strip()
        body = result.get("body", {})
        sections = ["## Research milestone", f"**{title}**", str(body.get("summary") or "").strip()]
        for field, label in (("conclusion", "Conclusion"), ("methods", "Methods")):
            value = str(body.get(field) or "").strip()
            if value and value != "missing due to old record":
                sections.append(f"**{label}**\n\n{value}")
        text = "\n\n".join(section for section in sections if section)
        self._append_session_message(
            root_thread_id=root_id,
            message_id=self._session_message_id(
                "result", f"{graph_id}_{result['node_id']}"
            ),
            text=text,
            artifact=artifact,
            replace=True,
            meta={
                "research_session_event": "result",
                "research_graph_id": graph_id,
                "research_result_node_id": str(result["node_id"]),
                "source_thread_id": owner_thread_id,
                "run_id": str(run_id or ""),
            },
        )

    def _write_session_state(
        self,
        *,
        graph_id: str,
        state: str,
        reason: str,
        token: str,
        session_root_thread_id: str = "",
    ) -> None:
        root_id = self._session_root_id(graph_id, session_root_thread_id)
        if not root_id:
            return
        snapshot = self.store.get_snapshot(graph_id)
        graph = snapshot["graph"]
        normalized_state = str(state or "waiting").strip().lower()
        latest = self.latest_session_result(
            graph_id,
            session_root_thread_id=root_id,
        )
        artifact = (
            self.artifact_registry.get(str(latest.get("report_artifact_id") or ""))
            if latest.get("report_artifact_id")
            else None
        )
        ready_titles = [
            str(node["title"])
            for node in snapshot["nodes"]
            if str(node.get("node_id") or "") in set(self._frontier_ids(snapshot))
        ]
        heading = {
            "completed": "Research completed",
            "blocked": "Experiment route blocked — research not completed",
            "no_change": "Automatic exploration stopped — not completed",
            "waiting": "Automatic research is waiting — not completed",
        }.get(normalized_state, "Research Session update")
        body: list[str] = [f"## {heading}"]
        normalized_reason = str(reason or "").strip()
        if normalized_reason:
            body.extend(["", normalized_reason])
        if normalized_state != "completed":
            body.extend(
                [
                    "",
                    "The Research Graph remains open; its completion criterion has not "
                    "been marked satisfied.",
                ]
            )
            if ready_titles:
                body.extend(
                    [
                        "",
                        "Ready scientific branches still recorded:",
                        *[f"- {title}" for title in ready_titles[:8]],
                    ]
                )
        elif latest:
            body.extend(
                [
                    "",
                    f"Latest recorded Result: **{latest['title']}**",
                ]
            )
        self._append_session_message(
            root_thread_id=root_id,
            message_id=self._session_message_id(
                normalized_state, f"{graph_id}_{token}"
            ),
            text="\n".join(body),
            artifact=artifact,
            meta={
                "research_session_event": normalized_state,
                "research_graph_id": graph_id,
                "graph_revision": int(graph["revision"]),
            },
        )

    @staticmethod
    def _hypothesis_evidence_states(
        nodes: list[dict[str, Any]],
        edges: list[dict[str, Any]],
    ) -> dict[str, str]:
        relations: dict[str, set[str]] = {}
        for edge in edges:
            if edge["relation"] in {"supports", "opposes", "inconclusive"}:
                relations.setdefault(str(edge["target_node_id"]), set()).add(
                    str(edge["relation"])
                )
        result: dict[str, str] = {}
        for node in nodes:
            if node["kind"] != "hypothesis":
                continue
            values = relations.get(str(node["node_id"]), set())
            if "supports" in values and "opposes" in values:
                state = "conflicting_evidence"
            elif "supports" in values:
                state = "supporting_evidence"
            elif "opposes" in values:
                state = "opposing_evidence"
            elif "inconclusive" in values:
                state = "not_distinguished"
            else:
                state = "no_results"
            result[str(node["node_id"])] = state
        return result

    @staticmethod
    def _frontier_ids(snapshot: dict[str, Any]) -> list[str]:
        return runnable_frontier_ids(snapshot["nodes"], snapshot["edges"])

    @staticmethod
    def _external_handoff_ids(snapshot: dict[str, Any]) -> list[str]:
        return external_handoff_ids(snapshot["nodes"], snapshot["edges"])

    @staticmethod
    def _is_internal_research_thread(thread: Any) -> bool:
        raw_role = getattr(thread, "thread_role", "")
        try:
            role = (
                raw_role
                if isinstance(raw_role, ThreadRole)
                else ThreadRole(str(raw_role or ""))
            )
        except ValueError:
            role = ThreadRole.PRIMARY
        if role in {
            ThreadRole.RESEARCH_PLANNING,
            ThreadRole.RESEARCH_COMPARISON,
        }:
            return True
        meta = thread.meta if isinstance(getattr(thread, "meta", None), dict) else {}
        if str(meta.get("internal_kind") or "") in {
            _PLANNING_THREAD_KIND,
            _COMPARISON_THREAD_KIND,
        }:
            return True
        return (
            str(getattr(thread, "thread_id", "")).startswith("thread_rg_")
            and str(getattr(thread, "title", "")).startswith("Plan next step:")
        )

    def _safe_note_path(self, ref_id: str) -> tuple[str, Path] | None:
        raw = str(ref_id or "").strip().replace("\\", "/").lstrip("/")
        if not raw or any(part in {"", ".", ".."} for part in Path(raw).parts):
            return None
        files_root = (self.workspace / "files").resolve()
        candidate = self.workspace.joinpath(*Path(raw).parts).resolve()
        if not candidate.exists() and not raw.startswith("files/"):
            candidate = files_root.joinpath(*Path(raw).parts).resolve()
        try:
            candidate.relative_to(files_root)
        except ValueError:
            return None
        if not candidate.is_file():
            return None
        relative = str(candidate.relative_to(self.workspace)).replace("\\", "/")
        return relative, candidate

    def _message_ref(self, ref_id: str) -> tuple[Any, Any] | None:
        raw = str(ref_id or "").strip()
        thread_hint = ""
        message_id = raw
        threads = self.thread_store.list_threads()
        if ":" in raw:
            prefix, suffix = raw.split(":", 1)
            # DBOS-backed message IDs themselves contain colons. Treat the
            # prefix as an owner only when it names an actual workspace thread.
            if any(thread.thread_id == prefix for thread in threads):
                thread_hint, message_id = prefix, suffix
        if thread_hint:
            threads = [
                thread for thread in threads if thread.thread_id == thread_hint
            ]
        matches: list[tuple[Any, Any]] = []
        for thread in threads:
            message = self.thread_store.get_message(thread.thread_id, message_id)
            if message is not None:
                matches.append((thread, message))
        # Bare message IDs are accepted when ownership is unambiguous.
        # Persisted refs are canonicalized below so later SQL
        # projection cannot accidentally expose a same-ID message in another
        # thread.
        return matches[0] if len(matches) == 1 else None

    def _discussion_ref(self, message_id: str) -> dict[str, Any] | None:
        from catmaster.storage import connect_workspace_db
        with connect_workspace_db(self.workspace) as connection:
            row = connection.execute("SELECT message_id, graph_id, discussion_id, title FROM research_discussions WHERE message_id=?",
                                     (message_id,)).fetchone()
        return dict(row) if row else None

    def validate_ref(self, ref: ResearchRefInput | dict[str, Any]) -> dict[str, str]:
        model = (
            ref
            if isinstance(ref, ResearchRefInput)
            else ResearchRefInput.model_validate(ref)
        )
        kind = model.ref_kind
        ref_id = model.ref_id
        if kind is RefKind.THREAD:
            try:
                self.thread_store.get_thread(ref_id)
            except (KeyError, ValueError) as exc:
                if not self._native_thread_owner(ref_id):
                    # Native async specialists need not have a WebUI thread row.
                    # The current tool binding is supplied by the workspace-bound
                    # assistant, not by the model's reference argument.
                    from catmaster.runtime.tool_runtime import current_tool_context
                    if ref_id != str(current_tool_context().get('native_thread_id') or ''):
                        raise ValueError("Thread reference is not available in this workspace.") from exc
        elif kind is RefKind.MESSAGE:
            # Shared discussion messages are stable sources in their own table,
            # not copied into a conversation merely to make them citeable.
            if self._discussion_ref(ref_id) is None:
                found = self._message_ref(ref_id)
                if found is None:
                    raise ValueError(
                        "Message reference is not available in this workspace. "
                        "Use a discussion message_id or thread_id:message_id for a conversation message."
                    )
                thread, message = found
                ref_id = f"{thread.thread_id}:{message.id}"
        elif kind is RefKind.ARTIFACT:
            if self.artifact_registry.get(ref_id) is None:
                raise ValueError(
                    "Artifact reference is not available in this workspace."
                )
        elif kind is RefKind.RUN:
            run_path = system_root(self.workspace) / "runs" / ref_id
            if not run_path.is_dir():
                raise ValueError(
                    "Run reference is not available in this workspace."
                )
        elif kind is RefKind.NOTE:
            if self._safe_note_path(ref_id) is None:
                raise ValueError(
                    "Note reference must point to an existing file under this "
                    "workspace's files directory."
                )
        elif kind is RefKind.DOI:
            normalized = re.sub(
                r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
                "",
                ref_id,
                flags=re.IGNORECASE,
            ).strip()
            if not _DOI_RE.fullmatch(normalized):
                raise ValueError("DOI reference is invalid.")
            ref_id = normalized
        elif kind is RefKind.URL:
            parsed = urlparse(ref_id)
            if parsed.scheme not in {"http", "https"} or not parsed.netloc:
                raise ValueError("URL reference must be an http or https URL.")
        return {"ref_kind": kind.value, "ref_id": ref_id}

    def _native_thread_owner(self, native_id: str) -> Any:
        for thread in self.thread_store.list_threads():
            if str(thread.deepagent_thread_id or '') == native_id:
                return thread
            for message in self.thread_store.list_messages(thread.thread_id):
                if any(part.type == 'subagent' and native_id in {
                    str((part.meta or {}).get('thread_id') or ''),
                    str((part.meta or {}).get('task_id') or '')
                } for part in message.parts):
                    return thread
        return None

    def resolve_ref(self, ref: dict[str, str]) -> dict[str, Any]:
        kind = RefKind(str(ref["ref_kind"]))
        ref_id = str(ref["ref_id"])
        base = {
            "ref_kind": kind.value,
            "ref_id": ref_id,
            "label": "Source unavailable",
            "available": False,
            "href": "",
        }
        try:
            if kind is RefKind.THREAD:
                try:
                    thread = self.thread_store.get_thread(ref_id)
                except KeyError:
                    thread = self._native_thread_owner(ref_id)
                    if thread is None:
                        return base
                return {
                    **base,
                    "label": thread.title or "Untitled thread",
                    "available": True,
                    "thread_id": thread.thread_id,
                }
            if kind is RefKind.MESSAGE:
                discussion = self._discussion_ref(ref_id)
                if discussion:
                    return {**base, "available": True, "label": discussion["title"],
                            "discussion_id": discussion["discussion_id"],
                            "discussion_graph_id": discussion["graph_id"], "message_id": ref_id}
                found = self._message_ref(ref_id)
                if found is None:
                    return base
                thread, message = found
                return {
                    **base,
                    "label": f"{message.role.title()} message in {thread.title or 'thread'}",
                    "available": True,
                    "thread_id": thread.thread_id,
                    "message_id": message.id,
                }
            if kind is RefKind.ARTIFACT:
                artifact = self.artifact_registry.get(ref_id)
                if artifact is None:
                    return base
                return {
                    **base,
                    "label": artifact.title or Path(artifact.path).name,
                    "available": True,
                    "artifact_id": artifact.artifact_id,
                    "preview_url": artifact.preview_url,
                    "download_url": artifact.download_url,
                }
            if kind is RefKind.RUN:
                path = system_root(self.workspace) / "runs" / ref_id
                if not path.is_dir():
                    return base
                return {
                    **base,
                    "label": f"Run {ref_id}",
                    "available": True,
                    "run_id": ref_id,
                }
            if kind is RefKind.NOTE:
                found = self._safe_note_path(ref_id)
                if found is None:
                    return base
                relative, path = found
                return {
                    **base,
                    "label": path.name,
                    "available": True,
                    "path": relative,
                }
            if kind is RefKind.DOI:
                return {
                    **base,
                    "label": f"DOI {ref_id}",
                    "available": True,
                    "href": f"https://doi.org/{quote(ref_id, safe='/()')}",
                }
            if kind is RefKind.URL:
                return {
                    **base,
                    "label": urlparse(ref_id).netloc or ref_id,
                    "available": True,
                    "href": ref_id,
                }
        except (KeyError, ValueError, OSError):
            return base
        return base

    @staticmethod
    def _public_launch(launch: dict[str, Any]) -> dict[str, Any]:
        return {
            "launch_id": str(launch["launch_id"]),
            "experiment_node_id": str(launch["experiment_node_id"]),
            "status": str(launch["status"]),
            "thread_id": str(launch.get("thread_id") or ""),
            "run_id": str(launch.get("run_id") or ""),
        }

    def _present_active_launch(self, launch: dict[str, Any]) -> dict[str, Any]:
        """Add one UI-only activity label derived from existing thread state."""

        presented = self._public_launch(launch)
        thread_id = presented["thread_id"]
        activity = "running"
        if thread_id:
            try:
                thread_status = self.thread_store.get_thread(thread_id).status
            except KeyError:
                activity = "operationally_incomplete"
            else:
                if thread_status is ThreadStatus.IDLE:
                    activity = "waiting_continue"
                elif thread_status is ThreadStatus.INTERRUPTED:
                    activity = "waiting_review"
                elif thread_status in {ThreadStatus.ERROR, ThreadStatus.STOPPED}:
                    activity = "operationally_incomplete"
        presented["activity"] = activity
        return presented

    @staticmethod
    def _public_graph(graph: dict[str, Any]) -> dict[str, Any]:
        return {
            "graph_id": str(graph["graph_id"]),
            "title": str(graph["title"]),
            "question": str(graph["question"]),
            "completion_criterion": str(graph["completion_criterion"]),
            "decision_preferences": str(graph["decision_preferences"]),
            "completed": bool(graph["completed"]),
            "orchestration_mode": str(graph["orchestration_mode"]),
            "archived": bool(graph["archived"]),
            "revision": int(graph["revision"]),
        }

    @staticmethod
    def _public_node(node: dict[str, Any]) -> dict[str, Any]:
        return {
            "node_id": str(node["node_id"]),
            "kind": str(node["kind"]),
            "title": str(node["title"]),
            "state": str(node.get("state") or ""),
            "body": dict(node.get("body") or {}),
            "revision": int(node["revision"]),
        }

    @staticmethod
    def _public_edge(edge: dict[str, Any]) -> dict[str, str]:
        return {
            "source_node_id": str(edge["source_node_id"]),
            "target_node_id": str(edge["target_node_id"]),
            "relation": str(edge["relation"]),
            "scope": str(edge.get("scope") or ""),
            "rationale": str(edge.get("rationale") or ""),
            "action": str(edge.get("action") or ""),
        }

    @staticmethod
    def _initial_selection_state(
        *,
        revision: int,
        candidate_ids: list[str],
        initial_incumbent_id: str = "",
        empty_reason: str = "",
    ) -> dict[str, Any]:
        """Create score-free reducer state for one exact graph revision."""

        candidates = list(dict.fromkeys(str(item) for item in candidate_ids if item))
        incumbent = (
            str(initial_incumbent_id)
            if str(initial_incumbent_id) in candidates
            else (candidates[0] if candidates else "")
        )
        remaining = [item for item in candidates if item != incumbent]
        if not candidates:
            return {
                "revision": int(revision),
                "status": "wait",
                "phase": "complete",
                "candidate_experiment_ids": [],
                "incumbent_id": "",
                "remaining_challenger_ids": [],
                "last_decisive_pair": {},
                "first_wait_winner_id": "",
                "pair_records": [],
                "active_comparison": {},
                "next_serial": 1,
                "recommended_experiment_id": "",
                "reason": (
                    str(empty_reason or "").strip()
                    or "No CatMaster-executable Experiment is ready in this revision."
                ),
                "decisive_source_refs": [],
                "unresolved_tradeoff": "",
            }
        return {
            "revision": int(revision),
            "status": "pending",
            "phase": "tournament" if remaining else "wait_forward",
            "candidate_experiment_ids": candidates,
            "incumbent_id": incumbent,
            "remaining_challenger_ids": remaining,
            "last_decisive_pair": {},
            "first_wait_winner_id": "",
            "pair_records": [],
            "active_comparison": {},
            "next_serial": 1,
            "recommended_experiment_id": "",
            "reason": "",
            "decisive_source_refs": [],
            "unresolved_tradeoff": "",
        }

    @staticmethod
    def _pair_winner_id(active: dict[str, Any], outcome: str) -> str:
        if outcome == "a":
            return str(active.get("candidate_a_id") or "")
        if outcome == "b":
            return str(active.get("candidate_b_id") or "")
        return ""

    @staticmethod
    def _finish_selection_wait(
        selection: dict[str, Any],
        record: dict[str, Any],
        *,
        fallback_tradeoff: str = "",
    ) -> None:
        selection.update(
            {
                "status": "wait",
                "phase": "complete",
                "recommended_experiment_id": "",
                "reason": str(record.get("reason") or "").strip(),
                "decisive_source_refs": list(
                    record.get("decisive_source_refs") or []
                ),
                "unresolved_tradeoff": str(
                    record.get("unresolved_tradeoff") or fallback_tradeoff
                ).strip(),
                "active_comparison": {},
            }
        )

    @staticmethod
    def _finish_selection_recommended(
        selection: dict[str, Any],
        record: dict[str, Any],
        experiment_id: str,
    ) -> None:
        selection.update(
            {
                "status": "recommended",
                "phase": "complete",
                "recommended_experiment_id": str(experiment_id),
                "reason": str(record.get("reason") or "").strip(),
                "decisive_source_refs": list(
                    record.get("decisive_source_refs") or []
                ),
                "unresolved_tradeoff": str(
                    record.get("unresolved_tradeoff") or ""
                ).strip(),
                "active_comparison": {},
            }
        )

    @classmethod
    def _next_selection_pair(
        cls,
        selection: dict[str, Any],
    ) -> dict[str, str] | None:
        """Advance non-model reducer phases and return the next clean pair."""

        if selection.get("active_comparison"):
            return None
        while selection.get("status") in {"pending", "comparing"}:
            phase = str(selection.get("phase") or "")
            incumbent = str(selection.get("incumbent_id") or "")
            if phase == "tournament":
                remaining = list(selection.get("remaining_challenger_ids") or [])
                if remaining:
                    return {
                        "purpose": "tournament",
                        "candidate_a_id": incumbent,
                        "candidate_b_id": str(remaining[0]),
                    }
                last = dict(selection.get("last_decisive_pair") or {})
                if last:
                    selection["phase"] = "tournament_reverse"
                    continue
                selection["phase"] = "wait_forward"
                continue
            if phase == "tournament_reverse":
                last = dict(selection.get("last_decisive_pair") or {})
                return {
                    "purpose": "tournament_reverse",
                    "candidate_a_id": str(last.get("candidate_b_id") or ""),
                    "candidate_b_id": str(last.get("candidate_a_id") or ""),
                }
            if phase == "tournament_adjudication":
                last = dict(selection.get("last_decisive_pair") or {})
                return {
                    "purpose": "tournament_adjudication",
                    "candidate_a_id": str(last.get("candidate_a_id") or ""),
                    "candidate_b_id": str(last.get("candidate_b_id") or ""),
                }
            if phase == "wait_forward":
                return {
                    "purpose": "wait_forward",
                    "candidate_a_id": incumbent,
                    "candidate_b_id": _WAIT_CANDIDATE_ID,
                }
            if phase == "wait_reverse":
                return {
                    "purpose": "wait_reverse",
                    "candidate_a_id": _WAIT_CANDIDATE_ID,
                    "candidate_b_id": incumbent,
                }
            if phase == "wait_adjudication":
                return {
                    "purpose": "wait_adjudication",
                    "candidate_a_id": incumbent,
                    "candidate_b_id": _WAIT_CANDIDATE_ID,
                }
            return None
        return None

    @classmethod
    def _apply_pair_record(
        cls,
        selection: dict[str, Any],
        record: dict[str, Any],
    ) -> None:
        """Reduce one structured pair result without deriving scientific scores."""

        purpose = str(record.get("purpose") or "")
        winner = cls._pair_winner_id(record, str(record.get("outcome") or ""))
        if purpose == "tournament":
            remaining = list(selection.get("remaining_challenger_ids") or [])
            if remaining:
                remaining.pop(0)
            selection["remaining_challenger_ids"] = remaining
            if not winner:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="The current ready Experiments could not be cleanly distinguished.",
                )
                return
            selection["incumbent_id"] = winner
            selection["last_decisive_pair"] = {
                "candidate_a_id": str(record.get("candidate_a_id") or ""),
                "candidate_b_id": str(record.get("candidate_b_id") or ""),
                "winner_id": winner,
            }
            selection["phase"] = "tournament"
        elif purpose == "tournament_reverse":
            expected = str(
                dict(selection.get("last_decisive_pair") or {}).get("winner_id")
                or ""
            )
            if not winner:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="The order-reversed comparison did not distinguish the finalists.",
                )
                return
            if winner == expected:
                selection["incumbent_id"] = winner
                selection["phase"] = "wait_forward"
            else:
                selection["phase"] = "tournament_adjudication"
        elif purpose == "tournament_adjudication":
            if not winner:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="Fresh adjudication did not resolve the order-sensitive finalist comparison.",
                )
                return
            selection["incumbent_id"] = winner
            selection["phase"] = "wait_forward"
        elif purpose == "wait_forward":
            if not winner:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="The candidate did not clearly outperform waiting.",
                )
                return
            selection["first_wait_winner_id"] = winner
            selection["phase"] = "wait_reverse"
        elif purpose == "wait_reverse":
            first = str(selection.get("first_wait_winner_id") or "")
            if not winner:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="The order-reversed comparison did not show that execution beats waiting.",
                )
                return
            if winner != first:
                selection["phase"] = "wait_adjudication"
                return
            if winner == _WAIT_CANDIDATE_ID:
                cls._finish_selection_wait(selection, record)
            else:
                cls._finish_selection_recommended(selection, record, winner)
        elif purpose == "wait_adjudication":
            if not winner or winner == _WAIT_CANDIDATE_ID:
                cls._finish_selection_wait(
                    selection,
                    record,
                    fallback_tradeoff="Fresh adjudication did not show that execution beats waiting.",
                )
            else:
                cls._finish_selection_recommended(selection, record, winner)

        if selection.get("status") not in {"recommended", "wait"}:
            selection["status"] = "pending"
            selection["active_comparison"] = {}

    @staticmethod
    def _public_selection(selection: dict[str, Any]) -> dict[str, Any]:
        records = []
        for item in list(selection.get("pair_records") or []):
            records.append(
                {
                    key: item.get(key)
                    for key in (
                        "sequence",
                        "purpose",
                        "candidate_a_id",
                        "candidate_b_id",
                        "outcome",
                        "reason",
                        "decisive_source_refs",
                        "unresolved_tradeoff",
                    )
                }
            )
        return {
            "status": str(selection.get("status") or "pending"),
            "candidate_experiment_ids": list(
                selection.get("candidate_experiment_ids") or []
            ),
            "pair_records": records,
            "recommended_experiment_id": str(
                selection.get("recommended_experiment_id") or ""
            ),
            "reason": str(selection.get("reason") or ""),
            "decisive_source_refs": list(
                selection.get("decisive_source_refs") or []
            ),
            "unresolved_tradeoff": str(
                selection.get("unresolved_tradeoff") or ""
            ),
        }

    def presentation(
        self,
        graph_id: str,
        *,
        current_thread_id: str = "",
    ) -> dict[str, Any]:
        snapshot = self.store.get_snapshot(graph_id)
        evidence_states = self._hypothesis_evidence_states(
            snapshot["nodes"], snapshot["edges"]
        )
        refs_by_node: dict[str, list[dict[str, Any]]] = {}
        for ref in snapshot["refs"]:
            refs_by_node.setdefault(str(ref["node_id"]), []).append(
                self.resolve_ref(ref)
            )
        active_by_experiment = {
            str(launch["experiment_node_id"]): self._present_active_launch(launch)
            for launch in snapshot["launches"]
            if launch["status"] in _ACTIVE_LAUNCH_STATUSES
        }
        nodes: list[dict[str, Any]] = []
        for node in snapshot["nodes"]:
            node_id = str(node["node_id"])
            presented = {
                **self._public_node(node),
                "evidence_state": evidence_states.get(node_id, ""),
                "refs": refs_by_node.get(node_id, []),
            }
            if node_id in active_by_experiment:
                presented["active_launch"] = active_by_experiment[node_id]
            nodes.append(presented)
        frontier_ids = self._frontier_ids(snapshot)
        external_ids = self._external_handoff_ids(snapshot)
        node_by_id = {str(node["node_id"]): node for node in nodes}
        bound_threads = [
            thread
            for thread in self.thread_store.list_threads()
            if thread.active_research_graph_id == graph_id
            and not self._is_internal_research_thread(thread)
        ]
        graph = {
            **self._public_graph(snapshot["graph"]),
            "counts": {
                "hypotheses": sum(node["kind"] == "hypothesis" for node in nodes),
                "experiments": sum(node["kind"] == "experiment" for node in nodes),
                "results": sum(node["kind"] == "result" for node in nodes),
                "external_handoffs": len(external_ids),
            },
            "frontier": [
                {
                    "node_id": node_id,
                    "title": str(node_by_id[node_id]["title"]),
                }
                for node_id in frontier_ids
                if node_id in node_by_id
            ],
            "external_handoffs": [
                {
                    "node_id": node_id,
                    "title": str(node_by_id[node_id]["title"]),
                }
                for node_id in external_ids
                if node_id in node_by_id
            ],
            "bound_thread_count": len(bound_threads),
            "bound_to_current_thread": any(
                thread.thread_id == current_thread_id for thread in bound_threads
            ),
        }
        recent_history = self.store.list_mutation_history(
            graph_id=graph_id,
            limit=51,
        )
        result = {
            "graph": graph,
            "nodes": nodes,
            "edges": [self._public_edge(edge) for edge in snapshot["edges"]],
            "decisions": snapshot["decisions"],
            "mutation_history": recent_history[:50],
            "mutation_history_scope": "recent",
            "mutation_history_has_more": len(recent_history) > 50,
            "mutation_history_ref": (
                f"/api/workspaces/{self.workspace_id}/research-graphs/"
                f"{graph_id}/mutation-history"
            ),
        }
        planning = self.store.latest_planning_preview(
            graph_id,
            current_revision_only=False,
        )
        if planning is not None and not graph["completed"]:
            stored_preview = dict(planning.get("preview") or {})
            raw_proposal = stored_preview.get("proposal")
            selection = dict(stored_preview.get("selection") or {})
            if (
                selection
                and int(selection.get("revision") or 0) == int(graph["revision"])
            ):
                public_selection = self._public_selection(selection)
                admitted = dict(stored_preview.get("admission") or {})
                status = str(public_selection["status"] or "pending")
                summary = str(public_selection.get("reason") or "").strip()
                if not summary:
                    summary = (
                        "Comparing the current ready Experiments in isolated pairs."
                        if status in {"pending", "comparing"}
                        else "No Experiment is selected for this revision."
                    )
                result["planning_preview"] = {
                    "planning_id": str(planning["planning_id"]),
                    "revision": int(selection["revision"]),
                    "status": status,
                    "summary": summary,
                    "admitted_node_ids": dict(admitted.get("node_ids") or {}),
                    "nodes": [],
                    "edges": [],
                    **public_selection,
                }
            elif (
                isinstance(raw_proposal, dict)
                and int(planning["revision"]) == int(graph["revision"])
            ):
                proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
                public_preview = build_planning_preview(
                    snapshot,
                    proposal,
                    focus_node_id=str(stored_preview.get("focus_node_id") or ""),
                )
                for node in public_preview.get("nodes", []):
                    node["refs"] = [
                        self.resolve_ref(ref)
                        for ref in list(node.get("refs") or [])
                    ]
                result["planning_preview"] = {
                    "planning_id": str(planning["planning_id"]),
                    "revision": int(planning["revision"]),
                    "status": str(planning["status"]),
                    **public_preview,
                }
            elif (
                str(stored_preview.get("no_change_reason") or "").strip()
                and int(planning["revision"]) == int(graph["revision"])
            ):
                reason = str(stored_preview["no_change_reason"]).strip()
                result["planning_preview"] = {
                    "planning_id": str(planning["planning_id"]),
                    "revision": int(planning["revision"]),
                    "status": str(planning["status"]),
                    "summary": reason,
                    "no_change_reason": reason,
                    "nodes": [],
                    "edges": [],
                    "candidate_experiment_ids": [],
                    "pair_records": [],
                    "recommended_experiment_id": "",
                    "reason": reason,
                    "decisive_source_refs": [],
                    "unresolved_tradeoff": "",
                }
        return result

    def catalog(
        self,
        *,
        include_archived: bool = True,
        current_thread_id: str = "",
    ) -> list[dict[str, Any]]:
        entries: list[dict[str, Any]] = []
        threads = self.thread_store.list_threads()
        for graph in self.store.list_graphs(include_archived=include_archived):
            snapshot = self.store.get_snapshot(graph["graph_id"])
            frontier_ids = self._frontier_ids(snapshot)
            external_ids = self._external_handoff_ids(snapshot)
            by_id = {
                str(node["node_id"]): node for node in snapshot["nodes"]
            }
            bound = [
                thread
                for thread in threads
                if thread.active_research_graph_id == graph["graph_id"]
                and not self._is_internal_research_thread(thread)
            ]
            entries.append(
                {
                    **self._public_graph(graph),
                    # The catalog renders this as a human-readable update label.
                    # Creation time and the rest of the storage row stay internal.
                    "updated_at": float(graph["updated_at"]),
                    "counts": {
                        "hypotheses": sum(
                            node["kind"] == "hypothesis"
                            for node in snapshot["nodes"]
                        ),
                        "experiments": sum(
                            node["kind"] == "experiment"
                            for node in snapshot["nodes"]
                        ),
                        "results": sum(
                            node["kind"] == "result"
                            for node in snapshot["nodes"]
                        ),
                        "external_handoffs": len(external_ids),
                    },
                    "frontier": [
                        {
                            "node_id": node_id,
                            "title": str(by_id[node_id]["title"]),
                        }
                        for node_id in frontier_ids
                        if node_id in by_id
                    ],
                    "external_handoffs": [
                        {
                            "node_id": node_id,
                            "title": str(by_id[node_id]["title"]),
                        }
                        for node_id in external_ids
                        if node_id in by_id
                    ],
                    "bound_thread_count": len(bound),
                    "bound_to_current_thread": any(
                        thread.thread_id == current_thread_id for thread in bound
                    ),
                }
            )
        return entries

    def create_graph(
        self,
        request: GraphCreateRequest,
        *,
        session_root_thread_id: str = "",
    ) -> dict[str, Any]:
        seeds = []
        for seed in request.initial_hypotheses:
            payload = seed.model_dump(mode="json")
            payload["refs"] = [self.validate_ref(ref) for ref in seed.refs]
            seeds.append(payload)
        requested_root_id = str(session_root_thread_id or "").strip()
        root_id = ""
        if requested_root_id:
            try:
                root = self.thread_store.get_thread(requested_root_id)
            except KeyError:
                root = None
            if (
                root is not None
                and is_persistent_research_root(root)
            ):
                root_id = root.thread_id
        graph = self.store.create_graph(
            title=request.title,
            question=request.question,
            completion_criterion=request.completion_criterion,
            decision_preferences=request.decision_preferences,
            orchestration_mode=("auto" if root_id and "orchestration_mode" not in request.model_fields_set
                                else request.orchestration_mode),
            orchestration_thread_id=root_id,
            initial_hypotheses=seeds,
        )
        if root_id:
            self.bind_thread(root_id, graph_id=graph["graph_id"])
        return self.presentation(graph["graph_id"])

    def patch_graph(
        self,
        graph_id: str,
        request: GraphPatchRequest,
        *,
        session_root_thread_id: str = "",
    ) -> dict[str, Any]:
        changes = {
            key: getattr(request, key)
            for key in request.model_fields_set
            if key != "expected_revision"
        }
        if changes.get("completed"):
            snapshot = self.store.get_snapshot(graph_id)
            if not any(node["kind"] == "result" for node in snapshot["nodes"]):
                raise ValueError(
                    "A Research Graph cannot be completed before it records a Result."
                )
        if (
            {"question", "completion_criterion"} & set(changes)
            and "completed" not in changes
        ):
            changes["completed"] = False
        self.store.update_graph(
            graph_id,
            expected_revision=request.expected_revision,
            changes=changes,
        )
        root_id = self._session_root_id(graph_id, session_root_thread_id)
        if root_id:
            self.store.set_orchestration_thread(graph_id, root_id)
        if bool(changes.get("completed")):
            completed_graph = self.store.get_graph(graph_id)
            self._write_session_state(
                graph_id=graph_id,
                state="completed",
                reason="The Research Graph completion criterion has been marked satisfied.",
                token=f"revision_{completed_graph['revision']}",
                session_root_thread_id=root_id,
            )
        return self.presentation(graph_id)

    def add_hypothesis(
        self,
        graph_id: str,
        request: HypothesisCreateRequest,
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        title = request.title or request.claim[:120]
        node_id = ""
        # Allocate the node ID in the store, then form suggests edges against it
        # by using one explicit stable ID for this atomic bundle.
        from uuid import uuid4

        node_id = f"hyp_{uuid4().hex[:16]}"
        edges = [
            {
                "source_node_id": result_id,
                "target_node_id": node_id,
                "relation": EdgeRelation.SUGGESTS.value,
            }
            for result_id in request.suggested_by_result_ids
        ]
        node, _event_id = self.store.add_node_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            kind=NodeKind.HYPOTHESIS,
            title=title,
            body={
                "claim": request.claim,
                "rationale": request.rationale,
                "predictions": request.predictions,
                "importance": request.importance,
            },
            edges=edges,
            refs=refs,
            node_id=node_id,
            change="hypothesis.added",
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def add_experiment(
        self,
        graph_id: str,
        request: ExperimentCreateRequest,
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        from uuid import uuid4

        node_id = f"exp_{uuid4().hex[:16]}"
        edges = [
            {
                "source_node_id": hypothesis_id,
                "target_node_id": node_id,
                "relation": EdgeRelation.TESTS.value,
            }
            for hypothesis_id in request.tests_hypothesis_ids
        ]
        edges.extend(
            {
                "source_node_id": node_id,
                "target_node_id": dependency_id,
                "relation": EdgeRelation.DEPENDS_ON.value,
            }
            for dependency_id in request.depends_on_experiment_ids
        )
        node, _event_id = self.store.add_node_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            kind=NodeKind.EXPERIMENT,
            title=request.title or request.objective[:120],
            body={
                "objective": request.objective,
                "plan_summary": request.plan_summary,
                "decision_rule": request.decision_rule,
                "blocking_reason": request.blocking_reason,
                "execution_lane": request.execution_lane,
                "estimated_compute_cost": request.estimated_compute_cost,
            },
            state=request.state.value,
            edges=edges,
            refs=refs,
            node_id=node_id,
            change="experiment.added",
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def add_bound_experiment(
        self,
        *,
        thread_id: str,
        graph_id: str,
        expected_revision: int,
        request: BoundExperimentCreateRequest,
    ) -> dict[str, Any]:
        """Create one Experiment and bind the requesting thread as one transaction."""

        thread = self.thread_store.get_thread(thread_id)
        if thread.active_research_graph_id != graph_id:
            raise ValueError("The current thread is not bound to this Research Graph.")
        refs = [self.validate_ref(ref) for ref in request.refs]
        from uuid import uuid4

        node_id = f"exp_{uuid4().hex[:16]}"
        edges = [
            {
                "source_node_id": hypothesis_id,
                "target_node_id": node_id,
                "relation": EdgeRelation.TESTS.value,
            }
            for hypothesis_id in request.tests_hypothesis_ids
        ]
        edges.extend(
            {
                "source_node_id": node_id,
                "target_node_id": dependency_id,
                "relation": EdgeRelation.DEPENDS_ON.value,
            }
            for dependency_id in request.depends_on_experiment_ids
        )
        state = (
            ExperimentState.READY.value
            if request.plan_summary and request.decision_rule
            else ExperimentState.DRAFT.value
        )
        previous_graph_id = str(thread.active_research_graph_id or "")
        previous_focus = str(thread.research_focus_node_id or "")
        rebound = False

        def bind_before_commit(created_node_id: str) -> None:
            nonlocal rebound
            self.thread_store.update_thread(
                thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=created_node_id,
            )
            rebound = True

        try:
            node, _event_id = self.store.add_node_bundle(
                graph_id,
                expected_revision=expected_revision,
                kind=NodeKind.EXPERIMENT,
                title=request.title or request.objective[:120],
                body={
                    "objective": request.objective,
                    "plan_summary": request.plan_summary,
                    "decision_rule": request.decision_rule,
                    "blocking_reason": "",
                    "execution_lane": "experiment",
                    "estimated_compute_cost": "",
                },
                state=state,
                edges=edges,
                refs=refs,
                node_id=node_id,
                change="experiment.adopted",
                before_commit=bind_before_commit,
            )
        except Exception:
            if rebound:
                self.thread_store.update_thread(
                    thread_id,
                    active_research_graph_id=previous_graph_id,
                    research_focus_node_id=previous_focus,
                )
            raise
        return {
            "node": self._public_node(node),
            "thread": self.thread_store.get_thread(thread_id).model_dump(mode="json"),
            **self.presentation(graph_id, current_thread_id=thread_id),
        }

    def set_thread_focus(
        self,
        *,
        thread_id: str,
        graph_id: str,
        node_id: str,
    ) -> dict[str, Any]:
        thread = self.thread_store.get_thread(thread_id)
        if thread.active_research_graph_id != graph_id:
            raise ValueError("The current thread is not bound to this Research Graph.")
        focus_id = str(node_id or "").strip()
        if focus_id:
            self.store.get_node(graph_id, focus_id)
        updated = self.bind_thread(
            thread_id,
            graph_id=graph_id,
            focus_node_id=focus_id,
        )
        if not focus_id:
            return {
                "thread": updated.model_dump(mode="json"),
                "focus": None,
                "neighbors": [],
                "edges": [],
            }
        snapshot = self.store.get_snapshot(graph_id)
        direct_edges = [
            self._public_edge(edge)
            for edge in snapshot["edges"]
            if focus_id
            in {str(edge["source_node_id"]), str(edge["target_node_id"])}
        ]
        neighbor_ids = {
            str(edge["source_node_id"])
            if str(edge["target_node_id"]) == focus_id
            else str(edge["target_node_id"])
            for edge in direct_edges
        }
        nodes = {
            str(node["node_id"]): self._public_node(node)
            for node in snapshot["nodes"]
        }
        return {
            "thread": updated.model_dump(mode="json"),
            "focus": nodes[focus_id],
            "neighbors": [nodes[item] for item in sorted(neighbor_ids) if item in nodes],
            "edges": direct_edges,
        }

    def update_bound_result(
        self,
        *,
        graph_id: str,
        experiment_node_id: str,
        result_node_id: str,
        summary: str,
        title: str,
        refs: list[ResearchRefInput],
        methods: str | None = None,
        conclusion: str | None = None,
    ) -> dict[str, Any]:
        graph = self.store.get_graph(graph_id)
        result = self.store.get_node(graph_id, result_node_id)
        validated_refs = [self.validate_ref(ref) for ref in refs]
        node, event_id = self.store.update_result_with_refs(
            graph_id,
            result_node_id,
            experiment_node_id=experiment_node_id,
            expected_revision=int(graph["revision"]),
            expected_node_revision=int(result["revision"]),
            title=title,
            summary=summary,
            methods=methods,
            conclusion=conclusion,
            refs=validated_refs,
        )
        self._write_result_milestone(graph_id=graph_id, result=node, refs=validated_refs)
        return {
            "node": self._public_node(node),
            "refs": validated_refs,
            "event_id": event_id,
            **self.presentation(graph_id),
        }

    def resume_bound_experiment(
        self,
        *,
        graph_id: str,
        experiment_node_id: str,
        reason: str,
        refs: list[ResearchRefInput],
    ) -> dict[str, Any]:
        graph = self.store.get_graph(graph_id)
        experiment = self.store.get_node(graph_id, experiment_node_id)
        validated_refs = [self.validate_ref(ref) for ref in refs]
        node, event_id = self.store.resume_experiment(
            graph_id,
            experiment_node_id,
            expected_revision=int(graph["revision"]),
            expected_node_revision=int(experiment["revision"]),
            reason=reason,
            refs=validated_refs,
        )
        return {
            "node": self._public_node(node),
            "refs": validated_refs,
            "event_id": event_id,
            **self.presentation(graph_id),
        }

    def retract_result(
        self,
        *,
        graph_id: str,
        result_node_id: str,
        expected_revision: int,
        expected_node_revision: int,
        reason: str,
        focused_experiment_node_id: str = "",
        owner_run_id: str = "",
        owner_thread_id: str = "",
    ) -> dict[str, Any]:
        changed = self.store.retract_result(
            graph_id,
            result_node_id,
            expected_revision=expected_revision,
            expected_node_revision=expected_node_revision,
            reason=reason,
            focused_experiment_node_id=focused_experiment_node_id,
            owner_run_id=owner_run_id,
            owner_thread_id=owner_thread_id,
        )
        return {**changed, **self.presentation(graph_id)}

    def record_result(
        self,
        graph_id: str,
        request: ResultCreateRequest,
        *,
        launch_id: str | None = None,
        run_id: str = "",
    ) -> dict[str, Any]:
        refs = [self.validate_ref(ref) for ref in request.refs]
        node, _event_id = self.store.add_result_bundle(
            graph_id,
            expected_revision=request.expected_revision,
            title=request.title,
            body={"summary": request.summary, "methods": request.methods, "conclusion": request.conclusion},
            experiment_node_id=request.experiment_node_id,
            judgments=[
                judgment.model_dump(mode="json")
                for judgment in request.judgments
            ],
            refs=refs,
            launch_id=launch_id,
            run_id=run_id,
        )
        self._write_result_milestone(
            graph_id=graph_id,
            result=node,
            refs=refs,
            launch_id=str(launch_id or ""),
            run_id=run_id,
        )
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def set_result_judgment(
        self,
        graph_id: str,
        result_node_id: str,
        hypothesis_node_id: str,
        request: ResultJudgmentSetRequest,
    ) -> dict[str, Any]:
        self.store.set_result_judgment(
            graph_id,
            expected_revision=request.expected_revision,
            result_node_id=result_node_id,
            hypothesis_node_id=hypothesis_node_id,
            relation=request.relation,
            scope=request.scope,
            rationale=request.rationale,
        )
        node = self.store.get_node(graph_id, result_node_id)
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def update_node(
        self,
        graph_id: str,
        node_id: str,
        request: NodePatchRequest,
    ) -> dict[str, Any]:
        node, _event_id = self.store.update_node(
            graph_id,
            node_id,
            expected_revision=request.expected_revision,
            expected_node_revision=request.expected_node_revision,
            title=request.title,
            state=request.state,
            body=request.body,
        )
        if node["kind"] == NodeKind.RESULT.value:
            refs = [ref for ref in self.store.get_snapshot(graph_id)["refs"]
                    if ref["node_id"] == node_id]
            self._write_result_milestone(graph_id=graph_id, result=node, refs=refs)
        return {"node": self._public_node(node), **self.presentation(graph_id)}

    def add_edge(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        source_node_id: str,
        target_node_id: str,
        relation: EdgeRelation,
    ) -> dict[str, Any]:
        if relation is EdgeRelation.REVISES:
            raise ValueError('Use a scientific revision with explicit action, scope and rationale.')
        self.store.add_edge(
            graph_id,
            expected_revision=expected_revision,
            source_node_id=source_node_id,
            target_node_id=target_node_id,
            relation=relation,
        )
        return self.presentation(graph_id)

    def add_ref(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        node_id: str,
        ref: ResearchRefInput,
    ) -> dict[str, Any]:
        validated = self.validate_ref(ref)
        self.store.add_ref(
            graph_id,
            expected_revision=expected_revision,
            node_id=node_id,
            ref_kind=validated["ref_kind"],
            ref_id=validated["ref_id"],
        )
        return self.presentation(graph_id)

    def mark_experiment_blocked(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        reason: str,
        launch_id: str | None = None,
        run_id: str = "",
    ) -> dict[str, Any]:
        self.store.mark_experiment_blocked(
            graph_id,
            experiment_node_id,
            expected_revision=expected_revision,
            reason=reason,
            launch_id=launch_id,
            run_id=run_id,
        )
        launch = self.store.get_launch(launch_id) if launch_id else None
        self._write_session_state(
            graph_id=graph_id,
            state="blocked",
            reason=str(reason or "").strip(),
            token=str((launch or {}).get("launch_id") or experiment_node_id),
            session_root_thread_id=str(
                (launch or {}).get("session_root_thread_id")
                or (launch or {}).get("thread_id")
                or ""
            ),
        )
        return self.presentation(graph_id)

    def bind_thread(
        self,
        thread_id: str,
        *,
        graph_id: str,
        focus_node_id: str = "",
    ) -> Any:
        thread = self.thread_store.get_thread(thread_id)
        thread_role = ThreadRole(thread.thread_role)
        if thread.entrypoint == "persistent_research" and not thread.parent_thread_id:
            thread_role = ThreadRole.RESEARCH_ROOT
        elif thread_role is ThreadRole.RESEARCH_ROOT:
            thread_role = ThreadRole.PRIMARY
        if not graph_id:
            return self.thread_store.update_thread(
                thread.thread_id,
                active_research_graph_id="",
                research_focus_node_id="",
                thread_role=thread_role,
            )
        graph = self.store.get_graph(graph_id)
        if focus_node_id:
            self.store.get_node(graph_id, focus_node_id)
        updated = self.thread_store.update_thread(
            thread.thread_id,
            active_research_graph_id=graph_id,
            research_focus_node_id=focus_node_id,
            thread_role=thread_role,
        )
        if is_persistent_research_root(updated):
            self.store.set_orchestration_thread(graph_id, updated.thread_id)
        return updated

    def _experiment_launch_prompt(
        self,
        *,
        graph_id: str,
        experiment: dict[str, Any],
        replicate: bool,
    ) -> str:
        body = experiment["body"]
        attempt = "replicate this experiment" if replicate else "run this experiment"
        return (
            f"You are continuing workspace Research Graph {graph_id}. "
            f"Prepare and {attempt} for node {experiment['node_id']}.\n\n"
            f"Objective: {body['objective']}\n"
            f"Plan: {body['plan_summary']}\n"
            f"Decision rule: {body['decision_rule']}\n\n"
            "Keep detailed calculations, receipts, and reports in their normal "
            "workspace owners. When execution reaches a scientifically "
            "meaningful outcome, collect a concise Result and attach the source "
            "run/artifacts as refs. Before the atomic graph writeback, ask the "
            "shared evidence judge to assess only the hypotheses the evidence "
            "actually distinguishes; an empty judgment set is valid. If execution "
            "cannot proceed, first use a bounded feasibility/preparation attempt with "
            "a materially different authorized route when prior evidence shows a "
            "method-domain failure. Unspecified implementation choices belong to the "
            "ExperimentSpecialist and worker. Record a blocker only after those feasible "
            "in-scope routes actually fail or a real external or authorization boundary "
            "prevents them."
        )

    async def launch_experiment(
        self,
        graph_id: str,
        experiment_node_id: str,
        *,
        expected_revision: int,
        replicate: bool = False,
        session_root_thread_id: str = "",
    ) -> dict[str, Any]:
        root_id = self._session_root_id(graph_id, session_root_thread_id)
        launch, claimed = self.store.claim_launch(
            graph_id,
            experiment_node_id,
            expected_revision=expected_revision,
            replicate=replicate,
            lease_owner=self.worker_id,
            session_root_thread_id=root_id,
        )
        if not claimed:
            return {
                "accepted": True,
                "deduplicated": True,
                "launch": self._public_launch(launch),
                **self.presentation(graph_id),
            }
        launch, child = await self._materialize_launch(
            launch,
            replicate=replicate,
        )
        return {
            "accepted": True,
            "deduplicated": False,
            "launch": self._public_launch(launch),
            "thread": child,
            **self.presentation(graph_id),
        }

    async def _materialize_launch(
        self,
        launch: dict[str, Any],
        *,
        replicate: bool,
    ) -> tuple[dict[str, Any], Any]:
        if self.agent_loop_factory is None:
            self.store.update_launch(
                launch["launch_id"],
                status="unknown",
                lease_owner="",
                lease_until=0,
            )
            raise RuntimeError("Research launch service is unavailable.")
        graph_id = str(launch["graph_id"])
        experiment_node_id = str(launch["experiment_node_id"])
        root_thread_id = str(launch.get("session_root_thread_id") or "")
        child_role = (
            ThreadRole.RESEARCH_EXECUTION
            if root_thread_id
            else ThreadRole.PRIMARY
        )
        child_thread_id = f"thread_rg_{launch['launch_id'].removeprefix('launch_')}"
        experiment = self.store.get_node(graph_id, experiment_node_id)
        entrypoint = str(experiment["body"]["execution_lane"])
        if entrypoint not in {"research", "experiment", "literature_review"}:
            entrypoint = "experiment"
        try:
            self.store.update_launch(
                launch["launch_id"],
                status="submitting",
                lease_owner=self.worker_id,
                lease_until=launch["lease_until"],
            )
            child = self.thread_store.create_thread(
                thread_id=child_thread_id,
                title=(
                    f"Replicate: {experiment['title']}"
                    if replicate
                    else f"Run: {experiment['title']}"
                ),
                entrypoint=entrypoint,
                parent_thread_id=root_thread_id,
                thread_role=child_role,
            )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=experiment_node_id,
                parent_thread_id=root_thread_id,
                thread_role=child_role,
            )
            self.store.update_launch(
                launch["launch_id"],
                status="submitting",
                thread_id=child.thread_id,
                lease_owner=self.worker_id,
                lease_until=launch["lease_until"],
            )
            existing_messages = self.thread_store.list_messages(child.thread_id)
            if not existing_messages:
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(
                        text=self._experiment_launch_prompt(
                            graph_id=graph_id,
                            experiment=experiment,
                            replicate=replicate,
                        ),
                        entrypoint=entrypoint,
                        llm_config=str((self.thread_store.get_thread(root_thread_id).meta or {}).get("model_config") or "") if root_thread_id else "",
                    ),
                )
            launch = self.store.update_launch(
                launch["launch_id"],
                status="running",
                thread_id=child.thread_id,
                lease_owner="",
                lease_until=0,
            )
        except Exception:
            # A child or remote submission may already exist. Preserve that
            # identity for reconciliation and never blindly resubmit.
            self.store.update_launch(
                launch["launch_id"],
                status="unknown",
                thread_id=child_thread_id,
                lease_owner="",
                lease_until=0,
            )
            raise
        return launch, child

    def reconcile_finished_child(
        self,
        *,
        child_thread_id: str,
        terminal_status: str,
        run_id: str = "",
        launch_id: str | None = None,
    ) -> bool:
        """Reconcile one child and report whether the scheduler should wake now."""

        normalized_status = str(terminal_status or "unknown").strip().lower()
        if launch_id is None:
            launch = self.store.find_active_launch_by_thread(child_thread_id)
        elif launch_id:
            launch = self.store.get_launch(launch_id)
            if str(launch.get("thread_id") or "") != child_thread_id:
                raise ValueError(
                    "The reconciled launch does not belong to this child thread."
                )
        else:
            launch = None
        if launch is None:
            comparison_planning = self.store.find_planning_by_comparison_thread(
                child_thread_id
            )
            if comparison_planning is not None:
                comparison_preview = dict(
                    comparison_planning.get("preview") or {}
                )
                selection = dict(comparison_preview.get("selection") or {})
                recorded = any(
                    str(item.get("comparison_thread_id") or "")
                    == child_thread_id
                    for item in list(selection.get("pair_records") or [])
                    if isinstance(item, dict)
                )
                if recorded:
                    return True
                active = dict(selection.get("active_comparison") or {})
                if str(active.get("thread_id") or "") == child_thread_id:
                    selection["active_comparison"] = {}
                    selection["status"] = "pending"
                    comparison_preview["selection"] = selection
                    self.store.set_planning_preview(
                        str(comparison_planning["graph_id"]),
                        str(comparison_planning["planning_id"]),
                        start_revision=int(comparison_planning["revision"]),
                        preview=comparison_preview,
                    )
                    # A model turn that did not publish an outcome is an
                    # operationally incomplete comparison, not scientific wait.
                    # Retry on the normal recovery cadence rather than spinning.
                    return False
            planning = self.store.find_planning_by_thread(child_thread_id)
            if planning is None:
                return True
            graph = self.store.get_graph(planning["graph_id"])
            preview = dict(planning.get("preview") or {})
            has_staged_proposal = isinstance(preview.get("proposal"), dict)
            selection = dict(preview.get("selection") or {})
            has_current_selection = bool(selection) and int(
                selection.get("revision") or 0
            ) == int(planning["revision"])
            has_explicit_no_change = bool(
                str(preview.get("no_change_reason") or "").strip()
            ) and not selection
            graph_revision = int(graph["revision"])
            planning_revision = int(planning["revision"])
            same_revision = graph_revision == planning_revision
            successful_turn = normalized_status in {
                "completed",
                "done",
                "idle",
                "steered",
            }
            finished = same_revision and successful_turn and has_staged_proposal
            if finished:
                try:
                    self.admit_planning_branches(
                        str(planning["graph_id"]),
                        str(planning["planning_id"]),
                        expected_revision=planning_revision,
                    )
                except (KeyError, TypeError, ValueError):
                    finished = False
            if finished or (
                same_revision and successful_turn and has_current_selection
            ):
                final_status = "finished"
                wake_scheduler = True
            elif same_revision and successful_turn and has_explicit_no_change:
                final_status = "no_change"
                wake_scheduler = False
            else:
                final_status = "stale"
                wake_scheduler = not same_revision
            self.store.update_planning(
                planning["graph_id"],
                planning["planning_id"],
                start_revision=planning_revision,
                status=final_status,
                thread_id=child_thread_id,
            )
            # A changed graph already represents new work. Failed or incomplete
            # planning retries on the worker recovery cadence instead of forming
            # an immediate failure loop.
            return wake_scheduler
        if str(launch["status"]) in {"completed", "blocked"}:
            if run_id and not str(launch.get("run_id") or ""):
                self.store.update_launch(
                    launch["launch_id"],
                    status=launch["status"],
                    run_id=run_id,
                    thread_id=child_thread_id,
                    lease_owner="",
                    lease_until=0,
                )
            return True
        if normalized_status in {
            "completed",
            "done",
            "idle",
            "interrupted",
            "paused",
            "awaiting_human_feedback",
            "queued",
            "running",
            "streaming",
            "steered",
        }:
            # A formal execution can span several ordinary turns in the same
            # child thread.  Conversation completion without an explicit
            # Result/blocker therefore waits for continuation.
            return True
        # A stopped or failed child is operational state, not scientific
        # evidence. Release it for an explicit retry without inventing a Result.
        self.store.release_incomplete_launch(
            launch["launch_id"],
            run_id=run_id,
            thread_id=child_thread_id,
        )
        return True

    def _validated_planning_proposal(
        self,
        snapshot: dict[str, Any],
        proposal: ResearchGraphPlanningProposal,
    ) -> ResearchGraphPlanningProposal:
        payload = proposal.model_dump(mode="json")
        durable_by_id = {
            str(node["node_id"]): node for node in snapshot["nodes"]
        }
        proposed_hypothesis_ids = {
            str(item["proposal_id"]) for item in payload["hypotheses"]
        }
        proposed_experiment_ids = {
            str(item["proposal_id"]) for item in payload["experiments"]
        }
        valid_hypothesis_ids = {
            node_id
            for node_id, node in durable_by_id.items()
            if node["kind"] == "hypothesis"
        } | proposed_hypothesis_ids
        valid_experiment_ids = {
            node_id
            for node_id, node in durable_by_id.items()
            if node["kind"] == "experiment"
        } | proposed_experiment_ids
        for item in payload["hypotheses"]:
            item["refs"] = [self.validate_ref(ref) for ref in item["refs"]]
        dependencies: dict[str, list[str]] = {}
        for item in payload["experiments"]:
            item["refs"] = [self.validate_ref(ref) for ref in item["refs"]]
            unknown_hypotheses = sorted(
                set(item["tests_hypothesis_ids"]) - valid_hypothesis_ids
            )
            if unknown_hypotheses:
                raise ValueError(
                    "A planning experiment references unknown hypothesis IDs: "
                    + ", ".join(unknown_hypotheses)
                )
            unknown_dependencies = sorted(
                set(item["depends_on_experiment_ids"]) - valid_experiment_ids
            )
            if unknown_dependencies:
                raise ValueError(
                    "A planning experiment references unknown dependency IDs: "
                    + ", ".join(unknown_dependencies)
                )
            dependencies[item["proposal_id"]] = [
                dependency_id
                for dependency_id in item["depends_on_experiment_ids"]
                if dependency_id in proposed_experiment_ids
            ]

        visiting: set[str] = set()
        visited: set[str] = set()

        def visit(proposal_id: str) -> None:
            if proposal_id in visiting:
                raise ValueError(
                    "Planning experiment dependencies must remain acyclic."
                )
            if proposal_id in visited:
                return
            visiting.add(proposal_id)
            for dependency_id in dependencies.get(proposal_id, []):
                visit(dependency_id)
            visiting.remove(proposal_id)
            visited.add(proposal_id)

        for proposal_id in sorted(dependencies):
            visit(proposal_id)

        recommended_target_id = str(payload["recommended_target_id"] or "")
        ready_ids = set(self._frontier_ids(snapshot))
        if (
            recommended_target_id
            and recommended_target_id not in proposed_experiment_ids
            and recommended_target_id not in ready_ids
        ):
            raise ValueError(
                "The recommended route must be a proposed Experiment or an "
                "existing ready Experiment."
            )
        return ResearchGraphPlanningProposal.model_validate(payload)

    @staticmethod
    def _planning_semantic_key(value: str) -> str:
        return re.sub(r"\s+", " ", str(value or "").strip()).casefold()

    @classmethod
    def _add_planning_alias(
        cls,
        aliases: dict[str, set[str]],
        value: str,
        target_id: str,
    ) -> None:
        key = cls._planning_semantic_key(value)
        if key:
            aliases.setdefault(key, set()).add(target_id)

    @classmethod
    def _resolve_planning_alias(
        cls,
        aliases: dict[str, set[str]],
        value: str,
        *,
        role: str,
    ) -> str:
        key = cls._planning_semantic_key(value)
        matches = aliases.get(key, set())
        if not matches:
            raise ValueError(
                f"The planning {role} '{value}' does not match an exact scientific "
                "title, claim, or objective in the bound graph or this draft."
            )
        if len(matches) > 1:
            raise ValueError(
                f"The planning {role} '{value}' is scientifically ambiguous. "
                "Use a unique exact title, claim, or objective."
            )
        return next(iter(matches))

    def _planning_source_refs(self, values: list[str]) -> list[ResearchRefInput]:
        refs: list[ResearchRefInput] = []
        for raw in values:
            value = str(raw or "").strip()
            if not value:
                continue
            normalized_doi = re.sub(
                r"^(?:https?://(?:dx\.)?doi\.org/|doi:\s*)",
                "",
                value,
                flags=re.IGNORECASE,
            ).strip()
            if _DOI_RE.fullmatch(normalized_doi):
                kind = RefKind.DOI
                ref_id = normalized_doi
            else:
                parsed = urlparse(value)
                if parsed.scheme in {"http", "https"} and parsed.netloc:
                    kind = RefKind.URL
                    ref_id = value
                else:
                    kind = RefKind.NOTE
                    ref_id = value
            refs.append(ResearchRefInput(ref_kind=kind, ref_id=ref_id))
        return refs

    def compile_planning_draft(
        self,
        snapshot: dict[str, Any],
        draft: ResearchGraphPlanningDraft,
        *,
        planning_id: str,
    ) -> ResearchGraphPlanningProposal:
        """Resolve scientific labels and add internal IDs outside the model contract."""

        prefix = re.sub(
            r"[^A-Za-z0-9_.:-]+",
            "_",
            str(planning_id or "planning").strip(),
        )[:96]
        hypothesis_ids = [
            f"{prefix}_hypothesis_{index}"
            for index, _item in enumerate(draft.hypotheses, start=1)
        ]
        experiment_ids = [
            f"{prefix}_experiment_{index}"
            for index, _item in enumerate(draft.experiments, start=1)
        ]

        hypothesis_aliases: dict[str, set[str]] = {}
        experiment_aliases: dict[str, set[str]] = {}
        ready_route_aliases: dict[str, set[str]] = {}
        ready_ids = set(self._frontier_ids(snapshot))
        for node in list(snapshot.get("nodes") or []):
            node_id = str(node.get("node_id") or "")
            body = dict(node.get("body") or {})
            if node.get("kind") == "hypothesis":
                self._add_planning_alias(
                    hypothesis_aliases,
                    str(node.get("title") or ""),
                    node_id,
                )
                self._add_planning_alias(
                    hypothesis_aliases,
                    str(body.get("claim") or ""),
                    node_id,
                )
            elif node.get("kind") == "experiment":
                self._add_planning_alias(
                    experiment_aliases,
                    str(node.get("title") or ""),
                    node_id,
                )
                self._add_planning_alias(
                    experiment_aliases,
                    str(body.get("objective") or ""),
                    node_id,
                )
                if node_id in ready_ids:
                    self._add_planning_alias(
                        ready_route_aliases,
                        str(node.get("title") or ""),
                        node_id,
                    )
                    self._add_planning_alias(
                        ready_route_aliases,
                        str(body.get("objective") or ""),
                        node_id,
                    )

        for target_id, item in zip(hypothesis_ids, draft.hypotheses, strict=True):
            self._add_planning_alias(hypothesis_aliases, item.title, target_id)
            self._add_planning_alias(hypothesis_aliases, item.claim, target_id)
        for target_id, item in zip(experiment_ids, draft.experiments, strict=True):
            self._add_planning_alias(experiment_aliases, item.title, target_id)
            self._add_planning_alias(experiment_aliases, item.objective, target_id)
            self._add_planning_alias(ready_route_aliases, item.title, target_id)
            self._add_planning_alias(ready_route_aliases, item.objective, target_id)

        hypotheses = [
            ResearchHypothesisProposal(
                proposal_id=target_id,
                claim=item.claim,
                title=item.title,
                rationale=item.rationale,
                predictions=item.predictions,
                importance=item.importance,
                refs=self._planning_source_refs(item.sources),
            )
            for target_id, item in zip(
                hypothesis_ids,
                draft.hypotheses,
                strict=True,
            )
        ]
        experiments = [
            ResearchExperimentProposal(
                proposal_id=target_id,
                objective=item.objective,
                title=item.title,
                plan_summary=item.plan_summary,
                decision_rule=item.decision_rule,
                execution_lane=item.execution_lane,
                estimated_compute_cost=item.estimated_compute_cost,
                tests_hypothesis_ids=[
                    self._resolve_planning_alias(
                        hypothesis_aliases,
                        reference,
                        role="hypothesis reference",
                    )
                    for reference in item.tests_hypotheses
                ],
                depends_on_experiment_ids=[
                    self._resolve_planning_alias(
                        experiment_aliases,
                        reference,
                        role="experiment prerequisite",
                    )
                    for reference in item.depends_on_experiments
                ],
                refs=self._planning_source_refs(item.sources),
            )
            for target_id, item in zip(
                experiment_ids,
                draft.experiments,
                strict=True,
            )
        ]
        recommended_target_id = ""
        if draft.recommended_route:
            recommended_target_id = self._resolve_planning_alias(
                ready_route_aliases,
                draft.recommended_route,
                role="recommended route",
            )
        return ResearchGraphPlanningProposal(
            hypotheses=hypotheses,
            experiments=experiments,
            recommended_target_id=recommended_target_id,
            recommendation_reason=draft.recommendation_reason,
        )

    def stage_planning_draft(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        draft: ResearchGraphPlanningDraft,
    ) -> dict[str, Any]:
        """Compile science-first input and publish its temporary graph preview."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        proposal = self.compile_planning_draft(
            self.store.get_snapshot(graph_id),
            draft,
            planning_id=str(planning["planning_id"]),
        )
        return self.stage_planning_proposal(
            graph_id,
            expected_revision=expected_revision,
            planning_thread_id=planning_thread_id,
            proposal=proposal,
        )

    def mark_planning_no_change(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        reason: str,
    ) -> dict[str, Any]:
        """Finish an active planning turn with an explicit scientific no-change."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        graph = self.store.get_graph(graph_id)
        if (
            int(planning["revision"]) != int(expected_revision)
            or int(graph["revision"]) != int(expected_revision)
        ):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(graph["revision"]),
            )
        normalized_reason = str(reason or "").strip()
        if not normalized_reason:
            raise ValueError("A scientific no-change reason is required.")
        thread = self.thread_store.get_thread(planning_thread_id)
        snapshot = self.store.get_snapshot(graph_id)
        frontier = self._frontier_ids(snapshot)
        selection = (
            self._initial_selection_state(
                revision=int(expected_revision),
                candidate_ids=frontier,
            )
            if frontier
            else {}
        )
        preview = {
            "focus_node_id": str(thread.research_focus_node_id or ""),
            "no_change_reason": normalized_reason,
        }
        if selection:
            preview["selection"] = selection
        self.store.set_planning_preview(
            graph_id,
            str(planning["planning_id"]),
            start_revision=expected_revision,
            preview=preview,
        )
        self.store.update_planning(
            graph_id,
            str(planning["planning_id"]),
            start_revision=expected_revision,
            status="finished" if selection else "no_change",
            thread_id=planning_thread_id,
        )
        if not selection:
            self._write_session_state(
                graph_id=graph_id,
                state="no_change",
                reason=normalized_reason,
                token=str(planning["planning_id"]),
                session_root_thread_id=str(
                    planning.get("session_root_thread_id") or planning_thread_id
                ),
            )
        return {
            "accepted": True,
            "planning_id": str(planning["planning_id"]),
            "revision": int(expected_revision),
            "reason": normalized_reason,
            "candidate_experiment_ids": frontier,
            "selection_started": bool(selection),
        }

    def stage_planning_proposal(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        planning_thread_id: str,
        proposal: ResearchGraphPlanningProposal,
    ) -> dict[str, Any]:
        """Publish one temporary evidence-aware route plan from the bound child."""

        planning = self.store.find_planning_by_thread(planning_thread_id)
        if planning is None or str(planning["graph_id"]) != str(graph_id):
            raise ValueError(
                "This tool is available only inside the active bound Research "
                "Graph planning thread."
            )
        if int(planning["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(self.store.get_graph(graph_id)["revision"]),
            )
        snapshot = self.store.get_snapshot(graph_id)
        if int(snapshot["graph"]["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(snapshot["graph"]["revision"]),
            )
        proposal = self._validated_planning_proposal(snapshot, proposal)
        thread = self.thread_store.get_thread(planning_thread_id)
        focus_node_id = str(thread.research_focus_node_id or "")
        public_preview = build_planning_preview(
            snapshot,
            proposal,
            focus_node_id=focus_node_id,
        )
        stored_preview = {
            "focus_node_id": focus_node_id,
            "proposal": proposal.model_dump(mode="json"),
        }
        self.store.set_planning_preview(
            graph_id,
            planning["planning_id"],
            start_revision=expected_revision,
            preview=stored_preview,
        )
        return {
            "accepted": True,
            "planning_id": str(planning["planning_id"]),
            "revision": int(expected_revision),
            "summary": str(public_preview.get("summary") or ""),
            "candidate_experiment_ids": list(
                public_preview.get("candidate_experiment_ids") or []
            ),
            "staged": {
                "hypotheses": len(proposal.hypotheses),
                "experiments": len(proposal.experiments),
            },
        }

    def admit_planning_branches(
        self,
        graph_id: str,
        planning_id: str,
        *,
        expected_revision: int,
    ) -> dict[str, Any]:
        """Atomically persist every staged H/E branch before route selection."""

        graph = self.store.get_graph(graph_id)
        if int(graph["revision"]) != int(expected_revision):
            raise ResearchGraphConflict(
                expected_revision=int(expected_revision),
                current_revision=int(graph["revision"]),
            )
        planning = self.store.get_planning(graph_id, planning_id)
        preview = dict(planning.get("preview") or {})
        if int(planning["revision"]) != int(expected_revision):
            raise ValueError(
                "This temporary plan is stale. Refresh and plan against the "
                "current graph revision."
            )
        raw_proposal = dict(preview.get("proposal") or {})
        proposal = ResearchGraphPlanningProposal.model_validate(raw_proposal)
        projected_preview = build_planning_preview(
            self.store.get_snapshot(graph_id),
            proposal,
            focus_node_id=str(preview.get("focus_node_id") or ""),
        )
        nodes: list[dict[str, Any]] = []
        for item in proposal.hypotheses:
            nodes.append(
                {
                    "proposal_id": item.proposal_id,
                    "kind": "hypothesis",
                    "title": item.title or item.claim[:120],
                    "state": "",
                    "body": item.model_dump(
                        mode="json",
                        exclude={"proposal_id", "title", "refs"},
                    ),
                    "refs": [
                        self.validate_ref(ref) for ref in item.refs
                    ],
                }
            )
        for item in proposal.experiments:
            experiment_state = (
                ExperimentState.READY.value
                if item.plan_summary and item.decision_rule
                else ExperimentState.DRAFT.value
            )
            nodes.append(
                {
                    "proposal_id": item.proposal_id,
                    "kind": "experiment",
                    "title": item.title or item.objective[:120],
                    "state": experiment_state,
                    "body": item.model_dump(
                        mode="json",
                        exclude={
                            "proposal_id",
                            "title",
                            "tests_hypothesis_ids",
                            "depends_on_experiment_ids",
                            "refs",
                        },
                    ),
                    "refs": [
                        self.validate_ref(ref) for ref in item.refs
                    ],
                }
            )

        durable_ids = {
            str(node["node_id"])
            for node in self.store.get_snapshot(graph_id)["nodes"]
        }
        staged_ids = {
            *(item.proposal_id for item in proposal.hypotheses),
            *(item.proposal_id for item in proposal.experiments),
        }
        staged_hypothesis_ids = {
            item.proposal_id for item in proposal.hypotheses
        }
        edges = [
            dict(edge)
            for edge in list(projected_preview.get("edges") or [])
            if (
                str(edge.get("source_node_id") or "") in staged_ids
                or str(edge.get("source_node_id") or "")
                in durable_ids
            )
            and (
                str(edge.get("target_node_id") or "") in staged_ids
                or str(edge.get("target_node_id") or "")
                in durable_ids
            )
        ]
        focus_node_id = str(preview.get("focus_node_id") or "")
        focus_node = (
            self.store.get_node(graph_id, focus_node_id)
            if focus_node_id
            else None
        )
        if focus_node is not None and focus_node["kind"] == NodeKind.RESULT.value:
            edges.extend(
                {
                    "source_node_id": focus_node_id,
                    "target_node_id": hypothesis_id,
                    "relation": EdgeRelation.SUGGESTS.value,
                }
                for hypothesis_id in sorted(staged_hypothesis_ids)
            )
        mapping: dict[str, str] = {}
        if nodes:
            mapping, _event_id = self.store.admit_plan_bundle(
                graph_id,
                expected_revision=expected_revision,
                nodes=nodes,
                edges=edges,
            )
        current = self.store.get_snapshot(graph_id)
        frontier = self._frontier_ids(current)
        external_handoffs = self._external_handoff_ids(current)
        admitted_revision = int(current["graph"]["revision"])
        proposer_incumbent_id = str(
            mapping.get(
                proposal.recommended_target_id,
                proposal.recommended_target_id,
            )
            or ""
        )
        if proposer_incumbent_id not in frontier:
            proposer_incumbent_id = ""
        selection = self._initial_selection_state(
            revision=admitted_revision,
            candidate_ids=frontier,
            initial_incumbent_id=proposer_incumbent_id,
            empty_reason=(
                "External Experiments are ready for laboratory or collaborator "
                "execution. Automatic orchestration is waiting for their Results "
                "to be recorded."
                if external_handoffs
                else ""
            ),
        )
        admitted_preview = {
            "focus_node_id": str(preview.get("focus_node_id") or ""),
            "admission": {
                "start_revision": int(planning["revision"]),
                "revision": admitted_revision,
                "node_ids": mapping,
            },
            "selection": selection,
        }
        self.store.set_planning_preview(
            graph_id,
            planning_id,
            start_revision=int(planning["revision"]),
            preview=admitted_preview,
        )
        if str(selection.get("status") or "") == "wait":
            self._write_session_state(
                graph_id=graph_id,
                state="waiting",
                reason=str(selection.get("reason") or "").strip(),
                token=f"{planning_id}_{admitted_revision}",
                session_root_thread_id=str(
                    planning.get("session_root_thread_id") or ""
                ),
            )
        return {
            "node_ids": mapping,
            "admitted_revision": admitted_revision,
            "candidate_experiment_ids": frontier,
            **self.presentation(graph_id),
        }

    @staticmethod
    def _candidate_comparison_packet(
        snapshot: dict[str, Any],
        candidate_id: str,
    ) -> dict[str, Any]:
        if candidate_id == _WAIT_CANDIDATE_ID:
            return {
                "candidate_id": _WAIT_CANDIDATE_ID,
                "kind": "wait",
                "meaning": (
                    "Launch no new Experiment in this revision. In Persistent Research "
                    "this is allowed only when canonical evidence shows a concrete "
                    "in-scope failure, an unresolved external dependency, or an "
                    "authorization boundary that prevents every bounded ready Experiment "
                    "from materially advancing the completion criterion. Method choice, "
                    "representation choice, implementation uncertainty, compute cost by "
                    "itself, or a risk of an inconclusive result is not enough because those "
                    "details are intentionally outside this comparison. Keep every durable "
                    "Hypothesis and Experiment."
                ),
            }
        nodes = {
            str(node["node_id"]): node for node in list(snapshot.get("nodes") or [])
        }
        experiment = nodes.get(str(candidate_id))
        if experiment is None or experiment.get("kind") != NodeKind.EXPERIMENT.value:
            raise ValueError(f"Comparison candidate is not an Experiment: {candidate_id}")
        edges = list(snapshot.get("edges") or [])
        refs_by_node: dict[str, list[dict[str, str]]] = {}
        for ref in list(snapshot.get("refs") or []):
            refs_by_node.setdefault(str(ref["node_id"]), []).append(
                {
                    "ref_kind": str(ref["ref_kind"]),
                    "ref_id": str(ref["ref_id"]),
                }
            )
        hypothesis_ids = [
            str(edge["source_node_id"])
            for edge in edges
            if edge.get("relation") == EdgeRelation.TESTS.value
            and str(edge.get("target_node_id") or "") == candidate_id
        ]
        hypotheses: list[dict[str, Any]] = []
        relevant_result_ids: set[str] = set()
        for hypothesis_id in hypothesis_ids:
            hypothesis = nodes.get(hypothesis_id)
            if hypothesis is None:
                continue
            hypotheses.append(
                {
                    "node_id": hypothesis_id,
                    "title": str(hypothesis.get("title") or ""),
                    "claim": str(hypothesis.get("body", {}).get("claim") or ""),
                    "rationale": str(
                        hypothesis.get("body", {}).get("rationale") or ""
                    ),
                    "observable_predictions": list(
                        hypothesis.get("body", {}).get("predictions") or []
                    ),
                    "source_refs": refs_by_node.get(hypothesis_id, []),
                }
            )
            relevant_result_ids.update(
                str(edge["source_node_id"])
                for edge in edges
                if str(edge.get("target_node_id") or "") == hypothesis_id
                and edge.get("relation")
                in {
                    EdgeRelation.SUPPORTS.value,
                    EdgeRelation.OPPOSES.value,
                    EdgeRelation.INCONCLUSIVE.value,
                }
            )
        results: list[dict[str, Any]] = []
        for result_id in sorted(relevant_result_ids):
            result = nodes.get(result_id)
            if result is None or result.get("kind") != NodeKind.RESULT.value:
                continue
            judgments = [
                {
                    "hypothesis_node_id": str(edge["target_node_id"]),
                    "relation": str(edge["relation"]),
                }
                for edge in edges
                if str(edge.get("source_node_id") or "") == result_id
                and str(edge.get("target_node_id") or "") in set(hypothesis_ids)
                and edge.get("relation")
                in {
                    EdgeRelation.SUPPORTS.value,
                    EdgeRelation.OPPOSES.value,
                    EdgeRelation.INCONCLUSIVE.value,
                }
            ]
            results.append(
                {
                    "node_id": result_id,
                    "title": str(result.get("title") or ""),
                    "summary": str(result.get("body", {}).get("summary") or ""),
                    "judgments": judgments,
                    "source_refs": refs_by_node.get(result_id, []),
                }
            )
        dependency_ids = [
            str(edge["target_node_id"])
            for edge in edges
            if edge.get("relation") == EdgeRelation.DEPENDS_ON.value
            and str(edge.get("source_node_id") or "") == candidate_id
        ]
        dependencies = [
            {
                "node_id": dependency_id,
                "title": str(nodes[dependency_id].get("title") or ""),
                "state": str(nodes[dependency_id].get("state") or ""),
                "objective": str(
                    nodes[dependency_id].get("body", {}).get("objective") or ""
                ),
            }
            for dependency_id in dependency_ids
            if dependency_id in nodes
        ]
        body = dict(experiment.get("body") or {})
        return {
            "candidate_id": candidate_id,
            "kind": "experiment",
            "title": str(experiment.get("title") or ""),
            "objective": str(body.get("objective") or ""),
            "scientific_plan": str(body.get("plan_summary") or ""),
            "decision_rule": str(body.get("decision_rule") or ""),
            "execution_lane": str(body.get("execution_lane") or ""),
            "estimated_compute_cost": str(body.get("estimated_compute_cost") or ""),
            "blocking_reason": str(body.get("blocking_reason") or ""),
            "tested_hypotheses": hypotheses,
            "relevant_results": results,
            "dependencies": dependencies,
            "source_refs": refs_by_node.get(candidate_id, []),
        }

    @classmethod
    def build_pair_comparison_prompt(
        cls,
        snapshot: dict[str, Any],
        *,
        candidate_a_id: str,
        candidate_b_id: str,
    ) -> str:
        graph = dict(snapshot.get("graph") or {})
        packet = {
            "research_question": str(graph.get("question") or ""),
            "completion_criterion": str(graph.get("completion_criterion") or ""),
            "decision_preferences": str(graph.get("decision_preferences") or ""),
            "candidate_A": cls._candidate_comparison_packet(
                snapshot, candidate_a_id
            ),
            "candidate_B": cls._candidate_comparison_packet(
                snapshot, candidate_b_id
            ),
        }
        invariant = (
            "Ignore every numerical or categorical assessment produced by another "
            "reviewer, proposer, planner, search engine, ranking system, or earlier "
            "comparison. Preserve genuine experimental and computational measurements. "
            "Judge A and B from the user-authored goal, canonical H/E/R state, original "
            "sources, and the scientific decision consequences of the two candidates. "
            "Candidate labels and order carry no preference. Persistent Research is "
            "authorized to explore autonomously. Do not choose wait merely because an "
            "Experiment leaves its computational representation, method, parameters, "
            "sampling, or implementation unspecified. Those details are intentionally "
            "outside this comparison. Prefer a bounded ready Experiment whenever its "
            "authorized execution "
            "can test feasibility or materially reduce uncertainty. Choose wait only after "
            "canonical evidence establishes a concrete failure or an external, user, "
            "safety, cost, or authorization boundary prevents every such bounded route."
        )
        return (
            "Run one fresh, isolated comparison for the exact packet below.\n\n"
            f"{invariant}\n\n"
            "Use search or open a durable source only when a specific scientific "
            "premise could reverse the comparison. Search snippets and ranks are "
            "locators, not evidence. When finished, call "
            "`record_research_experiment_comparison` exactly once. Do not propose "
            "new branches, mutate H/E/R state, or execute either candidate.\n\n"
            "# Comparison packet\n"
            + json.dumps(packet, ensure_ascii=False, indent=2)
        )

    def record_pair_comparison(
        self,
        *,
        comparison_thread_id: str,
        outcome: ResearchExperimentPairOutcomeDraft,
    ) -> dict[str, Any]:
        planning = self.store.find_planning_by_comparison_thread(
            comparison_thread_id
        )
        if planning is None:
            raise ValueError("This thread is not an active Experiment comparison.")
        preview = dict(planning.get("preview") or {})
        selection = dict(preview.get("selection") or {})
        active = dict(selection.get("active_comparison") or {})
        if str(active.get("thread_id") or "") != comparison_thread_id:
            raise ValueError("This Experiment comparison is no longer active.")
        graph = self.store.get_graph(str(planning["graph_id"]))
        selection_revision = int(selection.get("revision") or 0)
        if int(graph["revision"]) != selection_revision:
            raise ResearchGraphConflict(
                expected_revision=selection_revision,
                current_revision=int(graph["revision"]),
            )
        record = {
            "sequence": int(active.get("sequence") or 0),
            "purpose": str(active.get("purpose") or ""),
            "candidate_a_id": str(active.get("candidate_a_id") or ""),
            "candidate_b_id": str(active.get("candidate_b_id") or ""),
            **outcome.model_dump(mode="json"),
            "comparison_thread_id": comparison_thread_id,
        }
        records = list(selection.get("pair_records") or [])
        records.append(record)
        selection["pair_records"] = records
        selection["active_comparison"] = {}
        self._apply_pair_record(selection, record)
        preview["selection"] = selection
        self.store.set_planning_preview(
            str(planning["graph_id"]),
            str(planning["planning_id"]),
            start_revision=int(planning["revision"]),
            preview=preview,
        )
        if str(selection.get("status") or "") == "wait":
            self._write_session_state(
                graph_id=str(planning["graph_id"]),
                state="waiting",
                reason=(
                    str(selection.get("reason") or "").strip()
                    or str(selection.get("unresolved_tradeoff") or "").strip()
                    or "No bounded Experiment clearly beat waiting in this revision."
                ),
                token=f"{planning['planning_id']}_{selection_revision}",
                session_root_thread_id=str(
                    planning.get("session_root_thread_id")
                    or comparison_thread_id
                ),
            )
        return {
            "accepted": True,
            "graph_id": str(planning["graph_id"]),
            "planning_id": str(planning["planning_id"]),
            "revision": selection_revision,
            **self._public_selection(selection),
        }

    async def _launch_pair_comparison_child(
        self,
        *,
        graph_id: str,
        planning: dict[str, Any],
    ) -> tuple[bool, Any | None]:
        if self.agent_loop_factory is None:
            return False, None
        preview = dict(planning.get("preview") or {})
        selection = dict(preview.get("selection") or {})
        graph = self.store.get_graph(graph_id)
        if int(selection.get("revision") or 0) != int(graph["revision"]):
            return False, None
        descriptor = self._next_selection_pair(selection)
        if descriptor is None:
            preview["selection"] = selection
            self.store.set_planning_preview(
                graph_id,
                str(planning["planning_id"]),
                start_revision=int(planning["revision"]),
                preview=preview,
            )
            return False, None
        serial = max(1, int(selection.get("next_serial") or 1))
        planning_suffix = str(planning["planning_id"]).removeprefix("planning_")
        comparison_id = f"comparison_{planning_suffix}_{serial}"
        child_thread_id = f"thread_rgcmp_{planning_suffix}_{serial}"
        root_thread_id = str(planning.get("session_root_thread_id") or "")
        active = {
            **descriptor,
            "comparison_id": comparison_id,
            "thread_id": child_thread_id,
            "sequence": len(list(selection.get("pair_records") or [])) + 1,
        }
        selection.update(
            {
                "status": "comparing",
                "active_comparison": active,
                "next_serial": serial + 1,
            }
        )
        preview["selection"] = selection
        self.store.set_planning_preview(
            graph_id,
            str(planning["planning_id"]),
            start_revision=int(planning["revision"]),
            preview=preview,
        )
        try:
            try:
                child = self.thread_store.get_thread(child_thread_id)
            except KeyError:
                child = self.thread_store.create_thread(
                    thread_id=child_thread_id,
                    title="Compare ready Experiments",
                    entrypoint="research",
                    parent_thread_id=root_thread_id,
                    thread_role=ThreadRole.RESEARCH_COMPARISON,
                    meta={
                        "internal_kind": _COMPARISON_THREAD_KIND,
                        "comparison_id": comparison_id,
                    },
                )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id="",
                parent_thread_id=root_thread_id,
                thread_role=ThreadRole.RESEARCH_COMPARISON,
                meta={
                    **child.meta,
                    "internal_kind": _COMPARISON_THREAD_KIND,
                    "comparison_id": comparison_id,
                },
            )
            if not self.thread_store.list_messages(child.thread_id):
                prompt = self.build_pair_comparison_prompt(
                    self.store.get_snapshot(graph_id),
                    candidate_a_id=descriptor["candidate_a_id"],
                    candidate_b_id=descriptor["candidate_b_id"],
                )
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(text=prompt, entrypoint="research"),
                )
            return True, child
        except Exception:
            current = self.store.get_planning(
                graph_id, str(planning["planning_id"])
            )
            retry_preview = dict(current.get("preview") or {})
            retry_selection = dict(retry_preview.get("selection") or {})
            if str(
                dict(retry_selection.get("active_comparison") or {}).get(
                    "thread_id"
                )
                or ""
            ) == child_thread_id:
                retry_selection["active_comparison"] = {}
                retry_selection["status"] = "pending"
                retry_preview["selection"] = retry_selection
                self.store.set_planning_preview(
                    graph_id,
                    str(planning["planning_id"]),
                    start_revision=int(planning["revision"]),
                    preview=retry_preview,
                )
            raise

    def _planning_focus(self, snapshot: dict[str, Any]) -> str:
        nodes = list(snapshot["nodes"])
        if not nodes:
            return ""
        # Mutation-driven planning follows the most recently changed scientific
        # node. An explicit user planning request still supplies its own focus.
        latest = max(
            enumerate(nodes),
            key=lambda item: (
                float(item[1].get("updated_at") or 0.0),
                float(item[1].get("created_at") or 0.0),
                item[0],
            ),
        )
        return str(latest[1]["node_id"])

    @staticmethod
    def _active_frontier_has_new_result(
        snapshot: dict[str, Any],
        planning: dict[str, Any] | None,
        active_launches: list[dict[str, Any]],
    ) -> bool:
        """Whether active work produced evidence after the last decision began.

        Starting an Experiment changes the graph revision because its lifecycle
        state becomes ``running``.  That mechanical mutation must not by itself
        start another scientific planning turn.  A Result created or revised
        after the current planning turn began is a real decision boundary and
        may wake planning while sibling Experiments are still running.
        """

        if not active_launches:
            return False
        if planning is not None:
            decision_started_at = float(planning.get("created_at") or 0.0)
        else:
            decision_started_at = min(
                float(launch.get("created_at") or 0.0)
                for launch in active_launches
            )
        return any(
            str(node.get("kind") or "") == NodeKind.RESULT.value
            and float(node.get("updated_at") or 0.0) > decision_started_at
            for node in snapshot["nodes"]
        )

    async def _launch_planning_child(
        self,
        graph_id: str,
        *,
        revision: int,
        focus_node_id: str = "",
        session_root_thread_id: str = "",
        allow_same_revision_after_no_change: bool = False,
    ) -> tuple[bool, Any | None]:
        if self.agent_loop_factory is None:
            return False, None
        if focus_node_id:
            self.store.get_node(graph_id, focus_node_id)
        claim, claimed = self.store.claim_planning(
            graph_id,
            expected_revision=revision,
            session_root_thread_id=self._session_root_id(
                graph_id, session_root_thread_id
            ),
            allow_same_revision_after_no_change=allow_same_revision_after_no_change,
        )
        if not claimed:
            thread_id = str(claim.get("thread_id") or "")
            if thread_id:
                try:
                    return False, self.thread_store.get_thread(thread_id)
                except KeyError:
                    pass
            return False, None
        planning_id = str(claim["planning_id"])
        root_thread_id = str(claim.get("session_root_thread_id") or "")
        snapshot = self.store.get_snapshot(graph_id)
        focus_node_id = focus_node_id or self._planning_focus(snapshot)
        child_thread_id = f"thread_rg_{planning_id.removeprefix('planning_')}"
        try:
            child = self.thread_store.create_thread(
                thread_id=child_thread_id,
                title=f"Plan next step: {snapshot['graph']['title']}",
                entrypoint="research",
                parent_thread_id=root_thread_id,
                thread_role=ThreadRole.RESEARCH_PLANNING,
                meta={"internal_kind": _PLANNING_THREAD_KIND},
            )
            child = self.thread_store.update_thread(
                child.thread_id,
                active_research_graph_id=graph_id,
                research_focus_node_id=focus_node_id,
                parent_thread_id=root_thread_id,
                thread_role=ThreadRole.RESEARCH_PLANNING,
                meta={**child.meta, "internal_kind": _PLANNING_THREAD_KIND},
            )
            self.store.update_planning(
                graph_id,
                planning_id,
                start_revision=revision,
                status="attached",
                thread_id=child.thread_id,
            )
            if not self.thread_store.list_messages(child.thread_id):
                focus = next(
                    (
                        node
                        for node in snapshot["nodes"]
                        if str(node["node_id"]) == focus_node_id
                    ),
                    None,
                )
                focus_description = (
                    f"The bound focus is {focus['kind']} {focus['title']} "
                    f"({focus['node_id']})."
                    if focus is not None
                    else "No individual focus node is bound."
                )
                planning_mode = (
                    "result-focused revision"
                    if focus is not None
                    and str(focus.get("kind") or "") == NodeKind.RESULT.value
                    else "initial expansion"
                )
                await self.agent_loop_factory(
                    self.workspace, self.workspace_id
                ).submit(
                    thread_id=child.thread_id,
                    payload=ThreadSubmitRequest(
                        text=(
                            f"Run one {planning_mode} scientific planning pass for the bound "
                            f"Research Graph. {focus_description} The supplied graph text "
                            "is an explicitly partial focus snippet; use the narrow graph "
                            "query and evidence-reconciliation skills to inspect canonical "
                            "state and decisive sources as needed. Ask hypothesis_proposer to "
                            "compare the focus, existing predictions, related older Results, "
                            "dependencies, the strongest counterevidence or alternative "
                            "explanation, and the complete runnable frontier. Merge only "
                            "scientifically equivalent drafts. Preserve every distinct branch "
                            "the proposer is willing to submit, with observable predictions and "
                            "discriminating Experiments when justified. Existing Hypotheses may "
                            "already be sufficient; create a new falsifiable explanation only "
                            "for a genuinely unexplained observation. The proposer may publish "
                            "one temporary H/E set and return a concise scientific memo, or "
                            "record no-change without staging. Do not select or execute an "
                            "Experiment in this turn. Submit the complete staged set before "
                            "returning; selection happens only in a later comparison turn."
                        ),
                        entrypoint="research",
                    ),
                )
            return True, child
        except Exception:
            self.store.update_planning(
                graph_id,
                planning_id,
                start_revision=revision,
                status="stale",
                thread_id=child_thread_id,
            )
            raise

    async def plan_next_step(
        self,
        graph_id: str,
        *,
        expected_revision: int,
        focus_node_id: str = "",
        session_root_thread_id: str = "",
    ) -> dict[str, Any]:
        """Send a user-requested scientific reconsideration to the existing owner."""

        if self.agent_loop_factory is None:
            raise RuntimeError("Research service is unavailable.")
        graph = self.store.get_graph(graph_id)
        if int(graph['revision']) != expected_revision:
            raise ResearchGraphConflict(expected_revision=expected_revision, current_revision=graph['revision'])
        if graph['archived']:
            raise ValueError('Restore the graph before requesting research work.')
        root_id = str(graph.get('orchestration_thread_id') or session_root_thread_id or '')
        if root_id:
            root = self.thread_store.get_thread(root_id)
            if root.active_research_graph_id != graph_id:
                raise ValueError('The research session is bound to a different graph.')
        else:
            root = self.thread_store.create_thread(title=graph['title'], entrypoint='research')
            root = self.thread_store.update_thread(root.thread_id, active_research_graph_id=graph_id)
            self.store.set_orchestration_thread(graph_id, root.thread_id)
        if focus_node_id:
            self.store.get_node(graph_id, focus_node_id)
            root = self.thread_store.update_thread(root.thread_id, research_focus_node_id=focus_node_id)
        await self.agent_loop_factory(self.workspace, self.workspace_id).submit(
            thread_id=root.thread_id,
            payload=ThreadSubmitRequest(entrypoint=root.entrypoint, strategy='enqueue',
                llm_config=str((root.meta or {}).get('model_config') or ''),
                text='Reconsider the selected scientific evidence and continue the existing '
                     'research objective within the authorization already given. Choose and '
                     'carry out useful next actions within that scope, preserving scoped '
                     'findings and recording meaningful results or revisions in the bound graph. '
                     'Retain explicit user limits and open premises; complete the requested '
                     'deliverable when its evidence is sufficient.'))
        return {'accepted': True, 'deduplicated': False,
                'thread': root, **self.presentation(graph_id)}

    async def tick(self) -> None:
        """Recover already accepted work without selecting or launching new science."""

        for launch in self.store.active_launches():
            thread_id = str(launch.get("thread_id") or "")
            if not thread_id:
                if float(launch.get("lease_until") or 0) > time.time():
                    continue
                # Deterministic child IDs and the existing-message check make
                # recovery safe after a crash between claim and child binding.
                try:
                    await self._materialize_launch(
                        launch,
                        replicate=str(launch["idempotency_key"]).startswith(
                            "replicate_"
                        ),
                    )
                except (KeyError, ValueError, RuntimeError):
                    continue
            else:
                try:
                    thread = self.thread_store.get_thread(thread_id)
                except KeyError:
                    self.store.update_launch(
                        launch["launch_id"],
                        status="unknown",
                        lease_owner="",
                        lease_until=0,
                    )
                    continue
                if thread.status in {
                    ThreadStatus.ERROR,
                    ThreadStatus.STOPPED,
                    ThreadStatus.IDLE,
                }:
                    self.reconcile_finished_child(
                        child_thread_id=thread_id,
                        terminal_status=thread.status.value,
                        run_id=thread.active_run_id,
                        launch_id=str(launch["launch_id"]),
                    )

        # Existing launches and legacy planner threads may finish, but this
        # recovery sweep never chooses science or starts another coordinator.
        # Native async completions continue the bound ResearchSpecialist root.
        for graph in self.store.list_graphs(include_archived=False):
            root_id = str(graph.get('orchestration_thread_id') or '')
            if not graph['completed'] and root_id and self.agent_loop_factory:
                loop = self.agent_loop_factory(self.workspace, self.workspace_id)
                notify = getattr(loop, 'reconcile_research_graph_updates', None)
                if notify is not None:
                    await notify(graph['graph_id'], root_id)
            planning = self.store.latest_planning_preview(graph["graph_id"], current_revision_only=False)
            if not planning or planning.get("status") not in {"claimed", "attached"}:
                continue
            thread_id = str(planning.get("thread_id") or "")
            if not thread_id:
                continue
            try:
                thread = self.thread_store.get_thread(thread_id)
            except KeyError:
                continue
            if thread.status in {ThreadStatus.ERROR, ThreadStatus.STOPPED, ThreadStatus.IDLE}:
                self.reconcile_finished_child(child_thread_id=thread_id,
                    terminal_status=thread.status.value, run_id=thread.active_run_id)


__all__ = ["ResearchGraphService"]
