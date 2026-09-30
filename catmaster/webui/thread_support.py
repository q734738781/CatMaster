from __future__ import annotations

import asyncio
import json
import logging
import mimetypes
import os
import time
import uuid
from collections.abc import Mapping
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from catmaster.llm.config import LLMProfile
from catmaster.research.knowledge_graph.context import ResearchGraphContextBuilder
from catmaster.research.knowledge_graph.models import (
    DEFAULT_COMPLETION_CRITERION,
    PERSISTENT_RESEARCH_COMPLETION_CRITERION,
)
from catmaster.research.knowledge_graph.store import ResearchGraphStore
from catmaster.runtime.document_reads import native_document_input_decision
from catmaster.runtime.multimodal_blocks import (
    ModelMultimodalCapability,
    PreparedAttachment,
    build_turn_content,
    file_to_content_block,
    guess_mime_type,
    infer_attachment_kind,
    multimodal_prepare_summary,
    parse_data_url,
    text_attachment_block,
)

from .artifact_registry import ArtifactRegistry
from .thread_events import ThreadEventBroker
from .thread_models import (
    ArtifactPart,
    MessagePart,
    ThreadMessage,
    ThreadRole,
    ThreadStatus,
    ThreadSubmitRequest,
    research_automation_paused,
)
from .thread_store import ThreadStore, new_id

logger = logging.getLogger(__name__)

UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
_ACTIVE_RESEARCH_LAUNCH_STATUSES = {"claimed", "submitting", "running", "unknown"}
_TERMINAL_NATIVE_STATUSES = {
    "error",
    "success",
    "timeout",
    "interrupted",
}
_ENTRYPOINT_TO_MODEL_ROLE = {
    "research": "research_lead",
    "persistent_research": "research_lead",
    "research_challenger": "research_challenger",
    "experiment": "director",
    "writing": "write_director",
    "peer_review": "write_reviewer",
    "literature_review": "literature_deep_research",
}
_DEFAULT_RESEARCH_GRAPH_BINDING_ENTRYPOINTS = {
    "research",
    "persistent_research",
    "experiment",
    "literature_review",
}
_RESEARCH_GRAPH_CONTEXT_ENTRYPOINTS = {
    *_DEFAULT_RESEARCH_GRAPH_BINDING_ENTRYPOINTS,
    "writing",
    "research_challenger",
}
def _safe_attachment_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Attachment filename is required.")
    if "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(status_code=400, detail="Attachment filename is invalid.")
    return name


def _plain_dict(value: Any) -> dict[str, Any]:
    if isinstance(value, Mapping):
        return dict(value)
    if hasattr(value, "_asdict"):
        dumped = value._asdict()
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    if hasattr(value, "model_dump"):
        dumped = value.model_dump(mode="json")
        return dict(dumped) if isinstance(dumped, Mapping) else {}
    return {}


def _native_message_text(value: Any) -> str:
    message = _plain_dict(value)
    content = message.get("content_blocks") or message.get("content")
    if isinstance(content, str):
        return content
    if not isinstance(content, list):
        return ""
    chunks: list[str] = []
    for raw in content:
        if isinstance(raw, str):
            chunks.append(raw)
            continue
        block = _plain_dict(raw)
        if str(block.get("type") or "").strip().lower() not in {
            "",
            "text",
            "output_text",
        }:
            continue
        text = block.get("text")
        if isinstance(text, str):
            chunks.append(text)
    return "".join(chunks)


def _utc_iso_seconds() -> str:
    return datetime.now(UTC).strftime("%Y-%m-%dT%H:%M:%SZ")


class ThreadServiceSupport:
    """Attachment, scientific context and HITL projection helpers; no scheduler."""

    def _persistent_automation_enabled(self, thread: Any) -> bool:
        # Ordinary research can finish a newly requested deliverable even when
        # the attached scientific graph describes an already completed stage.
        if self.normalize_entrypoint(thread.entrypoint) != "persistent_research":
            return True
        graph_id = str(thread.active_research_graph_id or "").strip()
        if not graph_id:
            return True
        try:
            graph = ResearchGraphStore(self.workspace).get_graph(graph_id)
        except KeyError:
            return True
        if bool(graph.get("completed")) or bool(graph.get("archived")):
            return False
        return not research_automation_paused(thread, graph)


    def _ensure_default_research_graph_binding(
        self,
        *,
        thread: Any,
        prompt: str,
        entrypoint: str,
        inherited: dict[str, Any] | None,
    ) -> Any:
        if inherited is not None or entrypoint not in _DEFAULT_RESEARCH_GRAPH_BINDING_ENTRYPOINTS:
            return thread
        question = str(prompt or "").strip()
        if not question:
            return thread
        graph_store = ResearchGraphStore(self.workspace)
        bound_graph_id = str(thread.active_research_graph_id or "").strip()
        graph = None
        if bound_graph_id:
            try:
                graph = graph_store.get_graph(bound_graph_id)
            except KeyError:
                pass
            if graph and graph.get("archived"):
                graph = None
        if graph is None:
            # Graph selection belongs to the host. Completed stages remain
            # valid context; editing an older graph does not make it the newest.
            available = graph_store.list_graphs(include_archived=False)
            graph = (
                max(available, key=lambda item: (item["created_at"], item["graph_id"]))
                if available else None
            )
            if graph is None:
                graph = graph_store.create_graph(
                    title="",
                    question=question,
                    completion_criterion=(
                        PERSISTENT_RESEARCH_COMPLETION_CRITERION
                        if entrypoint == "persistent_research"
                        else DEFAULT_COMPLETION_CRITERION
                    ),
                    orchestration_mode="auto" if entrypoint == "persistent_research" else "manual",
                    orchestration_thread_id=thread.thread_id if entrypoint == "persistent_research" else "",
                )
        graph_id = str(graph["graph_id"])
        if graph_id != bound_graph_id:
            thread = self.store.update_thread(
                thread.thread_id, active_research_graph_id=graph_id,
                research_focus_node_id="",
            )
        if entrypoint == "persistent_research":
            if ThreadRole(thread.thread_role) is not ThreadRole.RESEARCH_ROOT:
                thread = self.store.update_thread(thread.thread_id, thread_role=ThreadRole.RESEARCH_ROOT)
            changes: dict[str, Any] = {}
            # Explicit user continuation retains the existing session policy;
            # completed scientific stages are not reopened by attaching a thread.
            if graph.get("orchestration_mode") != "auto" and not graph.get("completed"):
                changes["orchestration_mode"] = "auto"
            if str(graph.get("completion_criterion") or "").strip() == DEFAULT_COMPLETION_CRITERION:
                changes["completion_criterion"] = PERSISTENT_RESEARCH_COMPLETION_CRITERION
            if changes:
                graph_store.update_graph(graph_id, expected_revision=int(graph["revision"]), changes=changes)
            graph_store.set_orchestration_thread(graph_id, thread.thread_id)
        return thread


    def _research_turn_context(
        self,
        *,
        thread: Any,
        entrypoint: str,
        inherited: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        inherited_values = dict(inherited or {})
        graph_value = (
            inherited_values.get("research_graph_id", "")
            if inherited is not None
            else thread.active_research_graph_id
        )
        focus_value = (
            inherited_values.get("research_focus_node_id", "")
            if inherited is not None
            else thread.research_focus_node_id
        )
        graph_id = str(graph_value or "").strip()
        focus_node_id = str(focus_value or "").strip()
        context = {
            "research_graph_id": graph_id,
            "research_focus_node_id": focus_node_id,
            "research_launch_id": "",
        }
        if not graph_id:
            context["research_focus_node_id"] = ""
            return context
        graph_store = ResearchGraphStore(self.workspace)
        graph_store.get_graph(graph_id)
        focus = graph_store.get_node(graph_id, focus_node_id) if focus_node_id else None
        launch = None
        inherited_launch_id = str(inherited_values.get("research_launch_id") or "")
        if inherited is not None and inherited_launch_id:
            launch = graph_store.get_launch(inherited_launch_id)
        elif inherited is None:
            launch = graph_store.find_active_launch_by_thread(thread.thread_id)
        if launch is not None:
            execution_lane = str((focus or {}).get("body", {}).get("execution_lane") or "")
            if (
                str(launch.get("status") or "") in _ACTIVE_RESEARCH_LAUNCH_STATUSES
                and str(launch.get("thread_id") or "") == thread.thread_id
                and str(launch.get("graph_id") or "") == graph_id
                and str(launch.get("experiment_node_id") or "") == focus_node_id
                and str((focus or {}).get("kind") or "") == "experiment"
                and execution_lane == entrypoint
            ):
                context["research_launch_id"] = str(launch["launch_id"])
        return context


    def _research_graph_turn_content(
        self,
        *,
        prompt: str,
        turn_content: str | list[dict[str, Any]] | None,
        entrypoint: str,
        research_context: dict[str, str],
        internal_kind: str = "",
    ) -> str | list[dict[str, Any]] | None:
        delegated = internal_kind in {"research_graph_experiment_comparison", "async_subagent_activity"}
        graph_id = str(research_context.get("research_graph_id") or "")
        if entrypoint not in _RESEARCH_GRAPH_CONTEXT_ENTRYPOINTS:
            if delegated:
                return turn_content if turn_content is not None else prompt
            return turn_content
        focus_node_id = str(research_context.get("research_focus_node_id") or "") if graph_id else ""
        graph_markdown = (
            "# Current Research Graph binding for this turn\n"
            f"Graph query target: {graph_id or 'none (unbound)'}\n"
            f"Focus node: {focus_node_id or 'none'}"
        )
        if graph_id and not delegated:
            context = ResearchGraphContextBuilder(workspace=self.workspace).build(
                graph_id, focus_node_id=focus_node_id,
            )
            graph_markdown += "\n\n" + str(context["markdown"]).strip()
        # Delegates receive the actual binding without the parent's original
        # scientific request or its full focus snippet. Their brief owns scope.
        if isinstance(turn_content, list):
            blocks = [dict(block) for block in turn_content]
            text_index = next(
                (
                    index
                    for index, block in enumerate(blocks)
                    if str(block.get("type") or "") == "text"
                ),
                None,
            )
            if text_index is None:
                blocks.insert(
                    0,
                    {"type": "text", "text": f"{graph_markdown}\n\n# Current user request\n{prompt}"},
                )
            else:
                original = str(blocks[text_index].get("text") or prompt)
                blocks[text_index]["text"] = f"{graph_markdown}\n\n# Current turn\n{original}"
            return blocks
        current = str(turn_content or prompt).strip()
        return f"{graph_markdown}\n\n# Current user request\n{current}"


    def _capability_for_entrypoint(
        self, *, profile: LLMProfile, entrypoint: str
    ) -> ModelMultimodalCapability:
        role = _ENTRYPOINT_TO_MODEL_ROLE.get(entrypoint, "director")
        try:
            cfg = profile.config_for_role(role)
        except Exception:
            cfg = profile.main
        return ModelMultimodalCapability.from_llm_config(cfg)


    def prepare_submit_attachments(
        self,
        thread_id: str,
        attachments: list[dict[str, Any]],
        *,
        capability: ModelMultimodalCapability,
    ) -> list[PreparedAttachment]:
        prepared: list[PreparedAttachment] = []
        seen: set[tuple[str, str, str]] = set()
        for index, row in enumerate(attachments, start=1):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("type") or "file").strip().lower()
            name = _safe_attachment_filename(
                str(row.get("filename") or row.get("name") or f"attachment_{index}")
            )
            mime = str(row.get("mime_type") or "").strip()
            data = str(row.get("data") or "")
            text = str(row.get("text") or "")
            key = (kind, name, data[:120] or text[:120])
            if key in seen:
                continue
            seen.add(key)
            if data.startswith("data:"):
                try:
                    data_mime, blob = parse_data_url(data)
                except ValueError as exc:
                    raise HTTPException(
                        status_code=400,
                        detail=f"Attachment {name} is invalid: {exc}",
                    ) from exc
                prepared.append(
                    self._store_binary_attachment(
                        thread_id=thread_id,
                        index=index,
                        name=name,
                        mime=mime or data_mime,
                        blob=blob,
                        capability=capability,
                    )
                )
            elif text:
                prepared.append(
                    self._store_text_attachment(
                        thread_id=thread_id,
                        index=index,
                        name=name,
                        mime=mime or "text/plain",
                        text=text,
                    )
                )
        return prepared


    def _target_for_attachment(
        self,
        *,
        thread_id: str,
        index: int,
        name: str,
        mime: str,
        default_suffix: str = ".bin",
    ) -> Path:
        suffix = Path(name).suffix or mimetypes.guess_extension(mime or "") or default_suffix
        stem = Path(name).stem or f"attachment_{index}"
        return (
            self.workspace
            / "files"
            / "attachments"
            / thread_id
            / f"{index:03d}_{_safe_attachment_filename(stem + suffix)}"
        )


    def _artifact_part_from_record(self, artifact: Any) -> ArtifactPart:
        return ArtifactPart(
            id=new_id("part_artifact"),
            status="completed",
            artifact_id=artifact.artifact_id,
            renderer=artifact.renderer,
            title=artifact.title,
            summary=artifact.summary,
            path=artifact.path,
            meta=artifact.model_dump(mode="json"),
        )


    def _store_binary_attachment(
        self,
        *,
        thread_id: str,
        index: int,
        name: str,
        mime: str,
        blob: bytes,
        capability: ModelMultimodalCapability,
    ) -> PreparedAttachment:
        if len(blob) > UPLOAD_LIMIT_BYTES:
            raise HTTPException(status_code=413, detail=f"Attachment {name} exceeds upload limit.")
        mime = guess_mime_type(name, mime)
        kind = infer_attachment_kind(name, mime)
        target = self._target_for_attachment(
            thread_id=thread_id, index=index, name=name, mime=mime
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        relative = str(target.relative_to(self.workspace)).replace("\\", "/")
        artifact = self.artifact_registry.register_path(
            relative,
            thread_id=thread_id,
            title=name,
            summary=f"User-submitted {kind} attachment.",
            mime_type=mime,
            meta={"source": "composer_attachment", "original_name": name, "kind": kind},
        )
        warnings: list[str] = []
        current_turn_block: dict[str, Any] | None = None
        if not capability.supports_kind(kind):
            warnings.append(f"configured model capability does not enable {kind} blocks")
        elif kind in {"pdf", "document"} and not native_document_input_decision(target).allowed:
            warnings.append("document stored for bounded read_file pagination")
        elif len(blob) > capability.current_turn_inline_limit_bytes:
            warnings.append("attachment exceeds the current-turn inline limit")
        elif kind in {"image", "audio", "video", "pdf", "document"}:
            current_turn_block = file_to_content_block(
                target, mime_type=mime, kind=kind, filename=name
            )
        elif kind == "text":
            current_turn_block = text_attachment_block(
                blob.decode("utf-8", errors="replace"),
                filename=name,
                workspace_path=artifact.path,
            )
        else:
            warnings.append("unsupported attachment type stored as artifact only")
        return PreparedAttachment(
            artifact_id=artifact.artifact_id,
            workspace_path=artifact.path,
            filename=name,
            mime_type=mime,
            size_bytes=len(blob),
            kind=kind,
            current_turn_block=current_turn_block,
            history_part=self._artifact_part_from_record(artifact),
            warnings=warnings,
        )


    def _store_text_attachment(
        self,
        *,
        thread_id: str,
        index: int,
        name: str,
        mime: str,
        text: str,
    ) -> PreparedAttachment:
        blob = str(text or "").encode("utf-8")
        if len(blob) > UPLOAD_LIMIT_BYTES:
            raise HTTPException(status_code=413, detail=f"Attachment {name} exceeds upload limit.")
        mime = guess_mime_type(name, mime or "text/plain")
        target = self._target_for_attachment(
            thread_id=thread_id,
            index=index,
            name=name,
            mime=mime,
            default_suffix=".txt",
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(blob)
        relative = str(target.relative_to(self.workspace)).replace("\\", "/")
        artifact = self.artifact_registry.register_path(
            relative,
            thread_id=thread_id,
            title=name,
            summary="User-submitted text attachment.",
            mime_type=mime,
            meta={"source": "composer_attachment", "original_name": name, "kind": "text"},
        )
        return PreparedAttachment(
            artifact_id=artifact.artifact_id,
            workspace_path=artifact.path,
            filename=name,
            mime_type=mime,
            size_bytes=len(blob),
            kind="text",
            current_turn_block=text_attachment_block(
                text, filename=name, workspace_path=artifact.path
            ),
            history_part=self._artifact_part_from_record(artifact),
            warnings=[],
        )


    def _current_pending_interrupt_parts(self, thread_id: str) -> list[MessagePart]:
        for message in reversed(self.store.list_messages(thread_id)):
            pending = [
                part
                for part in message.parts
                if part.type == "interrupt" and part.status != "resolved"
            ]
            if pending:
                return pending
        return []


    def _pending_interrupt_reviews(self, thread_id: str) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for part in self._current_pending_interrupt_parts(thread_id):
            meta = dict(part.meta or {})
            payload = _plain_dict(meta.get("payload"))
            raw_interrupts = payload.get("interrupts")
            if isinstance(raw_interrupts, list):
                rows = raw_interrupts
            elif raw_interrupts is not None:
                rows = [raw_interrupts]
            elif payload.get("action_requests") or payload.get("actionRequests"):
                # Read projections written by the first Agent Server adapter,
                # which stored the HITL value without its interrupt envelope.
                rows = [{"value": payload}]
            else:
                rows = []
            for raw in rows:
                row = _plain_dict(raw)
                value = _plain_dict(row.get("value")) or row
                requests = value.get("action_requests") or value.get(
                    "actionRequests"
                )
                configs = value.get("review_configs") or value.get(
                    "reviewConfigs"
                )
                allowed_by_name: dict[str, set[str]] = {}
                if isinstance(configs, list):
                    for raw_config in configs:
                        config = _plain_dict(raw_config)
                        name = str(
                            config.get("action_name")
                            or config.get("actionName")
                            or ""
                        ).strip()
                        allowed = config.get("allowed_decisions") or config.get(
                            "allowedDecisions"
                        )
                        if name and isinstance(allowed, list):
                            allowed_by_name[name] = {
                                str(item).strip().lower()
                                for item in allowed
                                if str(item).strip()
                            }
                if not isinstance(requests, list):
                    continue
                for raw_request in requests:
                    request = _plain_dict(raw_request)
                    if not request:
                        continue
                    name = str(request.get("name") or "").strip()
                    reviews.append(
                        {
                            "action": request,
                            "allowed_decisions": allowed_by_name.get(name)
                            or {"approve", "reject", "respond"},
                        }
                    )
        return reviews


    @staticmethod
    def _coerce_edited_field(original: Any, value: Any, *, field_name: str) -> Any:
        if isinstance(original, bool):
            if isinstance(value, bool):
                return value
            normalized = str(value or "").strip().lower()
            if normalized in {"true", "1", "yes", "on"}:
                return True
            if normalized in {"false", "0", "no", "off"}:
                return False
            raise HTTPException(
                status_code=400, detail=f"{field_name} must be true or false."
            )
        if isinstance(original, int) and not isinstance(original, bool):
            try:
                return int(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"{field_name} must be an integer."
                ) from exc
        if isinstance(original, float):
            try:
                return float(value)
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail=f"{field_name} must be a number."
                ) from exc
        if isinstance(original, str):
            return str(value)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not editable in the ordinary review form.",
        )


    def _decisions_from_public_actions(
        self,
        thread_id: str,
        actions: list[Any],
    ) -> list[dict[str, Any]]:
        pending = self._pending_interrupt_reviews(thread_id)
        if not pending:
            raise HTTPException(
                status_code=409, detail="No pending review action was found."
            )
        submitted: dict[int, Any] = {}
        for item in actions:
            try:
                index = int(str(item.action_id))
            except (TypeError, ValueError) as exc:
                raise HTTPException(
                    status_code=400, detail="Review action id is invalid."
                ) from exc
            if index < 0 or index >= len(pending) or index in submitted:
                raise HTTPException(
                    status_code=400,
                    detail="Review action id is invalid or duplicated.",
                )
            submitted[index] = item
        if set(submitted) != set(range(len(pending))):
            raise HTTPException(
                status_code=400,
                detail="Choose one decision for every pending action before continuing.",
            )

        decisions: list[dict[str, Any]] = []
        for index, review in enumerate(pending):
            item = submitted[index]
            decision_type = str(item.decision or "").strip().lower()
            allowed = set(review.get("allowed_decisions") or ())
            if decision_type not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{decision_type} is not allowed for action {index + 1}."
                    ),
                )
            reason = str(
                item.reason or dict(item.fields or {}).get("reason") or ""
            ).strip()
            if decision_type in {"reject", "respond"} and not reason:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        "A reason or response is required for action "
                        f"{index + 1}."
                    ),
                )
            if decision_type == "approve":
                decisions.append({"type": "approve"})
                continue
            if decision_type == "reject":
                decisions.append({"type": "reject", "message": reason})
                continue
            if decision_type == "respond":
                decisions.append({"type": "respond", "message": reason})
                continue

            original_action = dict(review.get("action") or {})
            original_args = (
                dict(original_action.get("args"))
                if isinstance(original_action.get("args"), Mapping)
                else {}
            )
            edited_args = dict(original_args)
            for field_name, value in dict(item.fields or {}).items():
                if field_name == "reason":
                    continue
                if field_name not in original_args:
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            f"{field_name} is not an editable field for action "
                            f"{index + 1}."
                        ),
                    )
                edited_args[field_name] = self._coerce_edited_field(
                    original_args[field_name],
                    value,
                    field_name=field_name,
                )
            decisions.append(
                {
                    "type": "edit",
                    "edited_action": {
                        "name": str(original_action.get("name") or ""),
                        "args": edited_args,
                    },
                }
            )
        return decisions


    def _normalize_native_decisions(
        self,
        thread_id: str,
        decisions: list[Any],
    ) -> list[dict[str, Any]]:
        pending = self._pending_interrupt_reviews(thread_id)
        if not pending:
            raise HTTPException(
                status_code=409, detail="No pending review action was found."
            )
        if len(decisions) != len(pending):
            raise HTTPException(
                status_code=400,
                detail=(
                    f"Resume requires {len(pending)} decisions for the pending "
                    "tool calls."
                ),
            )
        normalized: list[dict[str, Any]] = []
        for index, (raw, review) in enumerate(zip(decisions, pending)):
            decision = _plain_dict(raw)
            decision_type = str(decision.get("type") or "").strip().lower()
            allowed = set(review.get("allowed_decisions") or ())
            if decision_type not in allowed:
                raise HTTPException(
                    status_code=400,
                    detail=(
                        f"{decision_type or 'The decision'} is not allowed for "
                        f"action {index + 1}."
                    ),
                )
            if decision_type == "approve":
                normalized.append({"type": "approve"})
            elif decision_type == "reject":
                row: dict[str, Any] = {"type": "reject"}
                if decision.get("message") is not None:
                    row["message"] = str(decision.get("message") or "")
                normalized.append(row)
            elif decision_type == "respond":
                message = str(decision.get("message") or "")
                if not message:
                    raise HTTPException(
                        status_code=400,
                        detail=f"A response is required for action {index + 1}.",
                    )
                normalized.append({"type": "respond", "message": message})
            else:
                edited = _plain_dict(decision.get("edited_action"))
                args = edited.get("args")
                if not str(edited.get("name") or "").strip() or not isinstance(
                    args, Mapping
                ):
                    raise HTTPException(
                        status_code=400,
                        detail=(
                            "An edit decision requires edited_action.name and "
                            f"edited_action.args for action {index + 1}."
                        ),
                    )
                normalized.append(
                    {
                        "type": "edit",
                        "edited_action": {
                            "name": str(edited["name"]),
                            "args": dict(args),
                        },
                    }
                )
        return normalized


    def _resolve_projected_interrupts(
        self,
        *,
        thread_id: str,
        resolution: Any,
        resumed_run_id: str,
    ) -> None:
        """Keep the rebuildable WebUI projection aligned after native resume."""

        # A native resume answers the current interrupt checkpoint only. Older
        # unresolved projection rows may survive a crash/reconnect and must not
        # be rewritten as though this decision had answered them too.
        for message in reversed(self.store.list_messages(thread_id)):
            changed = False
            parts: list[MessagePart] = []
            for part in message.parts:
                if part.type != "interrupt" or part.status == "resolved":
                    parts.append(part)
                    continue
                meta = dict(part.meta or {})
                meta["resolution"] = resolution
                meta["resumed_run_id"] = resumed_run_id
                parts.append(
                    part.model_copy(
                        update={"status": "resolved", "meta": meta}
                    )
                )
                changed = True
            if changed:
                self.store.update_message(
                    thread_id,
                    message.id,
                    parts=parts,
                )
                return
