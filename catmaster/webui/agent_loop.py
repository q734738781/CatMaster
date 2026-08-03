from __future__ import annotations

import asyncio
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from catmaster.llm.config import LLMProfile
from catmaster.research.knowledge_graph.context import ResearchGraphContextBuilder
from catmaster.runtime import RunControl
from catmaster.runtime.document_access import MAX_DOCUMENT_TEXT_CHARS, read_document
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
from catmaster.tools.base import workspace_root

from .artifact_registry import ArtifactRegistry
from .thread_events import ThreadEventBroker
from .thread_models import ArtifactPart, MessagePart, ThreadMessage, ThreadStatus
from .thread_store import ThreadStore, new_id

logger = logging.getLogger(__name__)

UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024
_ENTRYPOINT_TO_MODEL_ROLE = {
    "research": "research_lead",
    "experiment": "task_runner",
    "writing": "write_director",
    "peer_review": "write_reviewer",
    "literature_review": "literature_deep_research",
}
_RESEARCH_GRAPH_CONTEXT_ENTRYPOINTS = {
    "research",
    "experiment",
    "literature_review",
    "writing",
}


def _safe_attachment_filename(filename: str) -> str:
    name = Path(str(filename or "").replace("\\", "/")).name.strip()
    if not name or name in {".", ".."}:
        raise HTTPException(status_code=400, detail="Attachment filename is required.")
    if "/" in name or "\\" in name or "\x00" in name:
        raise HTTPException(status_code=400, detail="Attachment filename is invalid.")
    return name


class ThreadAgentLoopService:
    """Thread-native WebUI service boundary for submit/resume/stop orchestration."""

    def __init__(
        self,
        *,
        workspace: Path,
        workspace_id: str,
        store: ThreadStore,
        broker: ThreadEventBroker,
        artifact_registry: ArtifactRegistry,
        thread_tasks: dict[str, asyncio.Task[Any]],
        thread_stop_flags: set[str],
        build_runner: Callable[..., Any],
        streaming_runner_cls: type,
        permission_mode_for_thread: Callable[[Any, Any], str],
        interrupt_on_for_permission_mode: Callable[[Any], dict[str, Any]],
        normalize_entrypoint: Callable[[str], str],
        should_stop: Callable[[str], bool],
        on_turn_finished: Callable[..., Any] | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.workspace_id = str(workspace_id or self.workspace.name)
        self.store = store
        self.broker = broker
        self.artifact_registry = artifact_registry
        self.thread_tasks = thread_tasks
        self.thread_stop_flags = thread_stop_flags
        self.build_runner = build_runner
        self.streaming_runner_cls = streaming_runner_cls
        self.permission_mode_for_thread = permission_mode_for_thread
        self.interrupt_on_for_permission_mode = interrupt_on_for_permission_mode
        self.normalize_entrypoint = normalize_entrypoint
        self.should_stop = should_stop
        self.on_turn_finished = on_turn_finished

    def _research_graph_turn_content(
        self,
        *,
        thread: Any,
        prompt: str,
        turn_content: str | list[dict[str, Any]] | None,
        entrypoint: str,
    ) -> str | list[dict[str, Any]] | None:
        graph_id = str(
            getattr(thread, "active_research_graph_id", "") or ""
        ).strip()
        if entrypoint not in _RESEARCH_GRAPH_CONTEXT_ENTRYPOINTS or not graph_id:
            return turn_content
        focus_node_id = str(
            getattr(thread, "research_focus_node_id", "") or ""
        ).strip()
        context = ResearchGraphContextBuilder(
            workspace=self.workspace
        ).build(
            graph_id,
            focus_node_id=focus_node_id,
        )
        graph_markdown = str(context["markdown"]).strip()
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
                    {
                        "type": "text",
                        "text": (
                            f"{graph_markdown}\n\n# Current user request\n"
                            f"{prompt}"
                        ),
                    },
                )
            else:
                original = str(blocks[text_index].get("text") or prompt)
                blocks[text_index]["text"] = (
                    f"{graph_markdown}\n\n# Current turn\n{original}"
                )
            return blocks
        current = (
            str(turn_content).strip()
            if isinstance(turn_content, str) and turn_content.strip()
            else str(prompt or "").strip()
        )
        return f"{graph_markdown}\n\n# Current user request\n{current}"

    def _capability_for_entrypoint(
        self,
        *,
        profile: LLMProfile,
        entrypoint: str,
    ) -> ModelMultimodalCapability:
        role = _ENTRYPOINT_TO_MODEL_ROLE.get(str(entrypoint or "").strip(), "task_runner")
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
        if not attachments:
            return []
        prepared: list[PreparedAttachment] = []
        attachment_root = self.workspace / "files" / "attachments" / thread_id
        attachment_root.mkdir(parents=True, exist_ok=True)
        seen: set[tuple[str, str, str]] = set()
        for index, row in enumerate(attachments, start=1):
            if not isinstance(row, dict):
                continue
            kind = str(row.get("type") or "file").strip().lower()
            name = _safe_attachment_filename(str(row.get("filename") or row.get("name") or f"attachment_{index}"))
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
                    raise HTTPException(status_code=400, detail=f"Attachment {name} is invalid: {exc}") from exc
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
        stem = Path(name).stem or f"image_{index}"
        return self.workspace / "files" / "attachments" / thread_id / f"{index:03d}_{_safe_attachment_filename(stem + suffix)}"

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

    def _document_attachment_block(
        self,
        path: Path,
        *,
        filename: str,
        workspace_path: str,
        warnings: list[str],
    ) -> dict[str, Any] | None:
        files_root = (self.workspace / "files").resolve()
        try:
            relative = path.resolve().relative_to(files_root)
        except ValueError:
            warnings.append("Document attachment is outside the workspace files root")
            return None
        virtual_path = "/" + str(relative).replace("\\", "/")
        extracted = read_document(files_root, file_path=virtual_path)
        if extracted.startswith("Error reading document:"):
            warnings.append(extracted)
            return None
        return text_attachment_block(
            extracted,
            filename=filename,
            workspace_path=workspace_path,
            limit=MAX_DOCUMENT_TEXT_CHARS,
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
        target = self._target_for_attachment(thread_id=thread_id, index=index, name=name, mime=mime)
        target.write_bytes(blob)
        rel_path = str(target.relative_to(self.workspace)).replace("\\", "/")
        artifact = self.artifact_registry.register_path(
            rel_path,
            thread_id=thread_id,
            title=name,
            summary=f"User-submitted {kind} attachment.",
            mime_type=mime,
            meta={"source": "composer_attachment", "original_name": name, "kind": kind, "size_bytes": len(blob)},
        )
        warnings: list[str] = []
        current_turn_block: dict[str, Any] | None = None
        if kind in {"pdf", "document"}:
            current_turn_block = self._document_attachment_block(
                target,
                filename=name,
                workspace_path=artifact.path,
                warnings=warnings,
            )
        elif not capability.supports_kind(kind):
            warnings.append(f"configured model capability does not enable {kind} blocks")
        elif len(blob) > capability.current_turn_inline_limit_bytes:
            warnings.append(
                f"attachment exceeds current-turn inline limit ({capability.current_turn_inline_limit_bytes} bytes)"
            )
        elif kind in {"image", "audio", "video"}:
            current_turn_block = file_to_content_block(target, mime_type=mime, kind=kind, filename=name)
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
        target = self._target_for_attachment(thread_id=thread_id, index=index, name=name, mime=mime, default_suffix=".txt")
        target.write_bytes(blob)
        rel_path = str(target.relative_to(self.workspace)).replace("\\", "/")
        artifact = self.artifact_registry.register_path(
            rel_path,
            thread_id=thread_id,
            title=name,
            summary="User-submitted text attachment.",
            mime_type=mime,
            meta={"source": "composer_attachment", "original_name": name, "kind": "text", "size_bytes": len(blob)},
        )
        return PreparedAttachment(
            artifact_id=artifact.artifact_id,
            workspace_path=artifact.path,
            filename=name,
            mime_type=mime,
            size_bytes=len(blob),
            kind="text",
            current_turn_block=text_attachment_block(text, filename=name, workspace_path=artifact.path),
            history_part=self._artifact_part_from_record(artifact),
            warnings=[],
        )

    def rebuild_prepared_attachments(
        self,
        metadata: list[dict[str, Any]],
        *,
        capability: ModelMultimodalCapability,
    ) -> list[PreparedAttachment]:
        prepared: list[PreparedAttachment] = []
        for row in metadata:
            if not isinstance(row, dict):
                continue
            rel_path = str(row.get("workspace_path") or "").strip()
            filename = _safe_attachment_filename(str(row.get("filename") or Path(rel_path).name or "attachment"))
            mime = guess_mime_type(filename, str(row.get("mime_type") or ""))
            kind = str(row.get("kind") or infer_attachment_kind(filename, mime)).strip().lower()
            path = self.workspace.joinpath(*Path(rel_path).parts)
            warnings = [str(item) for item in list(row.get("warnings") or []) if str(item).strip()]
            current_turn_block: dict[str, Any] | None = None
            size_bytes = int(row.get("size_bytes") or 0)
            if not path.exists():
                warnings.append("stored attachment file is missing")
            elif kind in {"pdf", "document"}:
                current_turn_block = self._document_attachment_block(
                    path,
                    filename=filename,
                    workspace_path=rel_path,
                    warnings=warnings,
                )
            elif not capability.supports_kind(kind):
                warnings.append(f"configured model capability does not enable {kind} blocks")
            elif size_bytes > capability.current_turn_inline_limit_bytes:
                warnings.append(
                    f"attachment exceeds current-turn inline limit ({capability.current_turn_inline_limit_bytes} bytes)"
                )
            elif kind in {"image", "audio", "video"}:
                current_turn_block = file_to_content_block(path, mime_type=mime, kind=kind, filename=filename)
            elif kind == "text":
                current_turn_block = text_attachment_block(
                    path.read_text(encoding="utf-8", errors="replace"),
                    filename=filename,
                    workspace_path=rel_path,
                )
            prepared.append(
                PreparedAttachment(
                    artifact_id=str(row.get("artifact_id") or ""),
                    workspace_path=rel_path,
                    filename=filename,
                    mime_type=mime,
                    size_bytes=size_bytes,
                    kind=kind,
                    current_turn_block=current_turn_block,
                    history_part=None,
                    warnings=warnings,
                )
            )
        return prepared

    def append_user_message(
        self,
        thread_id: str,
        text: str,
        *,
        attachment_parts: list[MessagePart] | None = None,
        meta: dict[str, Any] | None = None,
        structured_sidecar: dict[str, Any] | None = None,
    ) -> ThreadMessage:
        parts = [MessagePart(id=new_id("part_text"), type="text", text=str(text or ""), status="completed")]
        parts.extend(attachment_parts or [])
        message = ThreadMessage(
            id=new_id("msg"),
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=parts,
            meta=dict(meta or {}),
            structured_sidecar=dict(structured_sidecar or {}),
        )
        self.store.append_message(message)
        self.broker.emit(thread_id, "message.created", message_id=message.id, status="completed", data={"message": message.model_dump(mode="json")})
        return message

    def append_assistant_message(self, thread_id: str, *, run_id: str = "", meta: dict[str, Any] | None = None) -> tuple[ThreadMessage, str]:
        text_part_id = new_id("part_text")
        message_meta = dict(meta or {})
        if run_id:
            message_meta["run_id"] = str(run_id or "")
        message = ThreadMessage(
            id=new_id("msg"),
            thread_id=thread_id,
            role="assistant",
            status="streaming",
            parts=[MessagePart(id=text_part_id, type="text", text="", status="streaming")],
            meta=message_meta,
        )
        self.store.append_message(message)
        self.broker.emit(thread_id, "message.created", message_id=message.id, status="streaming", data={"message": message.model_dump(mode="json")})
        self.broker.emit(thread_id, "message.part.created", message_id=message.id, status="streaming", data={"message_id": message.id, "part_id": text_part_id, "type": "text"})
        return message, text_part_id

    def normalize_resume_decisions(self, thread_id: str, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending_count = self._pending_interrupt_action_count(thread_id)
        if pending_count <= 1:
            return decisions
        if len(decisions) == pending_count:
            return decisions
        if len(decisions) != 1:
            raise HTTPException(status_code=400, detail=f"Resume requires {pending_count} decisions for the pending tool calls.")
        decision = dict(decisions[0])
        if str(decision.get("type") or "").strip() == "edit":
            raise HTTPException(status_code=400, detail=f"Edit resume requires one edited_action per pending tool call ({pending_count}).")
        return [dict(decision) for _ in range(pending_count)]

    def _pending_interrupt_action_count(self, thread_id: str) -> int:
        count = 0
        for row in self.pending_interrupt_action_requests(thread_id):
            count += 1 if isinstance(row, dict) else 0
        return count

    def pending_interrupt_action_requests(self, thread_id: str) -> list[dict[str, Any]]:
        return [item["action"] for item in self.pending_interrupt_reviews(thread_id)]

    def pending_interrupt_reviews(self, thread_id: str) -> list[dict[str, Any]]:
        reviews: list[dict[str, Any]] = []
        for message in self.store.list_messages(thread_id):
            for part in message.parts:
                if part.type != "interrupt" or part.status == "resolved":
                    continue
                raw = (part.meta or {}).get("payload", {}).get("interrupts")
                rows = raw if isinstance(raw, list) else [raw]
                for row in rows:
                    value = row.get("value") if isinstance(row, dict) else None
                    requests = value.get("action_requests") if isinstance(value, dict) else None
                    configs = value.get("review_configs") if isinstance(value, dict) else None
                    allowed_by_name: dict[str, set[str]] = {}
                    if isinstance(configs, list):
                        for config in configs:
                            if not isinstance(config, dict):
                                continue
                            action_name = str(config.get("action_name") or "").strip()
                            allowed = config.get("allowed_decisions")
                            if action_name and isinstance(allowed, list):
                                allowed_by_name[action_name] = {
                                    str(item).strip().lower()
                                    for item in allowed
                                    if str(item).strip()
                                }
                    if isinstance(requests, list):
                        for request in requests:
                            if isinstance(request, dict):
                                action = dict(request)
                                name = str(action.get("name") or "").strip()
                                reviews.append(
                                    {
                                        "action": action,
                                        "allowed_decisions": allowed_by_name.get(name)
                                        or {"approve", "reject", "respond"},
                                    }
                                )
                    elif isinstance(row, str) and "action_requests" in row:
                        for _ in re.findall(r"['\"]name['\"]\s*:", row):
                            reviews.append(
                                {
                                    "action": {},
                                    "allowed_decisions": {"approve", "reject", "respond"},
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
            raise HTTPException(status_code=400, detail=f"{field_name} must be true or false.")
        if isinstance(original, int) and not isinstance(original, bool):
            try:
                return int(value)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"{field_name} must be an integer.") from exc
        if isinstance(original, float):
            try:
                return float(value)
            except Exception as exc:
                raise HTTPException(status_code=400, detail=f"{field_name} must be a number.") from exc
        if isinstance(original, str):
            return str(value)
        raise HTTPException(
            status_code=400,
            detail=f"{field_name} is not editable in the ordinary review form.",
        )

    def decisions_from_public_actions(
        self,
        thread_id: str,
        actions: list[Any],
    ) -> list[dict[str, Any]]:
        pending = self.pending_interrupt_reviews(thread_id)
        if not pending:
            raise HTTPException(status_code=409, detail="No pending review action was found.")
        submitted: dict[int, Any] = {}
        for item in actions:
            try:
                index = int(str(item.action_id))
            except Exception as exc:
                raise HTTPException(status_code=400, detail="Review action id is invalid.") from exc
            if index < 0 or index >= len(pending) or index in submitted:
                raise HTTPException(status_code=400, detail="Review action id is invalid or duplicated.")
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
                    detail=f"{decision_type} is not allowed for action {index + 1}.",
                )
            reason = str(item.reason or item.fields.get("reason") or "").strip()
            if decision_type in {"reject", "respond"} and not reason:
                raise HTTPException(
                    status_code=400,
                    detail=f"A reason or response is required for action {index + 1}.",
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
                if isinstance(original_action.get("args"), dict)
                else {}
            )
            edited_args = dict(original_args)
            for field_name, value in dict(item.fields or {}).items():
                if field_name == "reason":
                    continue
                if field_name not in original_args:
                    raise HTTPException(
                        status_code=400,
                        detail=f"{field_name} is not an editable field for action {index + 1}.",
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

    def resume_tool_inputs_from_decisions(self, thread_id: str, decisions: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pending_actions = self.pending_interrupt_action_requests(thread_id)
        resume_inputs: list[dict[str, Any]] = []
        for action, decision in zip(pending_actions, decisions):
            decision_type = str(decision.get("type") or "").strip()
            if decision_type == "edit":
                selected = decision.get("edited_action")
            elif decision_type == "approve":
                selected = action
            else:
                continue
            if not isinstance(selected, dict):
                continue
            name = str(selected.get("name") or selected.get("tool") or "").strip()
            args = selected.get("args") if "args" in selected else selected.get("input")
            if not name or not isinstance(args, dict):
                continue
            resume_inputs.append({"name": name, "args": dict(args), "source": "interrupt_review"})
        return resume_inputs

    def resolve_pending_interrupt_parts(self, thread_id: str, decisions: list[dict[str, Any]]) -> list[dict[str, str]]:
        resolved: list[dict[str, str]] = []
        resolution = {"decisions": decisions, "resolved_at": time.time()}
        for message in self.store.list_messages(thread_id):
            for part in message.parts:
                if part.type != "interrupt" or part.status == "resolved":
                    continue
                meta = dict(part.meta or {})
                meta["status"] = "resolved"
                meta["resolution"] = resolution
                meta["resolved_at"] = resolution["resolved_at"]
                try:
                    self.store.update_part(thread_id, message.id, part.id, status="resolved", meta=meta)
                except KeyError:
                    continue
                resolved.append({"message_id": message.id, "part_id": part.id})
        return resolved

    async def submit(self, *, thread_id: str, payload: Any) -> dict[str, Any]:
        text = str(payload.text or "").strip()
        thread = self.store.get_thread(thread_id)
        normalized_entrypoint = self.normalize_entrypoint(payload.entrypoint or thread.entrypoint)
        llm_profile = LLMProfile.from_env_or_file(payload.llm_config or None)
        capability = self._capability_for_entrypoint(profile=llm_profile, entrypoint=normalized_entrypoint)
        prepared_attachments = self.prepare_submit_attachments(
            thread_id,
            list(payload.attachments or []),
            capability=capability,
        )
        attachment_parts = [item.history_part for item in prepared_attachments if item.history_part is not None]
        attachment_sidecar = [item.sidecar() for item in prepared_attachments]
        if not text and not attachment_parts:
            raise HTTPException(status_code=400, detail="Message text is required.")
        prompt_text = text or "User submitted attachments."
        turn_content = build_turn_content(prompt_text, prepared_attachments)
        permission_mode = self.permission_mode_for_thread(thread, getattr(payload, "permission_mode", "") or dict(thread.meta or {}).get("permission_mode"))
        current_permission_mode = self.permission_mode_for_thread(thread, "")
        if permission_mode != current_permission_mode:
            thread = self.store.update_thread(thread_id, meta={**dict(thread.meta or {}), "permission_mode": permission_mode})
        task = self.thread_tasks.get(thread_id)
        if thread.status == ThreadStatus.STOPPING:
            raise HTTPException(status_code=409, detail="Thread is stopping. Wait for it to stop before sending another message.")
        running = bool(task and not task.done()) or thread.status == ThreadStatus.RUNNING
        if running:
            user_message = self.append_user_message(
                thread_id,
                text,
                attachment_parts=attachment_parts,
                meta={"kind": "steering"},
                structured_sidecar={"attachments": attachment_sidecar},
            )
            pending = list(thread.pending_steering or [])
            pending.append(
                {
                    "text": prompt_text,
                    "entrypoint": normalized_entrypoint,
                    "model_config": payload.llm_config,
                    "permission_mode": permission_mode,
                    "message_id": user_message.id,
                    "attachments": attachment_sidecar,
                    "created_at": time.time(),
                }
            )
            thread = self.store.update_thread(thread_id, pending_steering=pending)
            self.broker.emit(thread_id, "thread.updated", status=str(thread.status.value), data={"thread": thread.model_dump(mode="json"), "steering_queued": True})
            return {"accepted": True, "queued": True, "thread": thread, "message": user_message}
        user_message = self.append_user_message(
            thread_id,
            text,
            attachment_parts=attachment_parts,
            structured_sidecar={"attachments": attachment_sidecar},
        )
        assistant_message = await self.launch_turn(
            thread_id=thread_id,
            prompt=prompt_text,
            entrypoint=normalized_entrypoint,
            model_config=payload.llm_config,
            permission_mode=permission_mode,
            turn_content=turn_content,
            attachment_metadata=attachment_sidecar,
            existing_user_message_id=user_message.id,
        )
        return {"accepted": True, "queued": False, "thread": self.store.get_thread(thread_id), "message": user_message, "assistant_message": assistant_message}

    async def resume(self, *, thread_id: str, payload: Any, validate_decisions: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> dict[str, Any]:
        requested_decisions = (
            self.decisions_from_public_actions(thread_id, list(payload.actions or []))
            if getattr(payload, "actions", None)
            else payload.decisions
        )
        try:
            decisions = validate_decisions(requested_decisions)
        except ValueError as exc:
            raise HTTPException(status_code=400, detail=str(exc)) from exc
        thread = self.store.get_thread(thread_id)
        task = self.thread_tasks.get(thread_id)
        if task and not task.done():
            raise HTTPException(status_code=409, detail="Thread is already running.")
        decisions = self.normalize_resume_decisions(thread_id, decisions)
        resume_tool_inputs = self.resume_tool_inputs_from_decisions(thread_id, decisions)
        resolved_parts = self.resolve_pending_interrupt_parts(thread_id, decisions)
        self.broker.emit(thread_id, "interrupt.resolved", status="running", data={"decisions": decisions, "resolved_parts": resolved_parts})
        assistant_message = await self.launch_turn(
            thread_id=thread_id,
            prompt=str(payload.text or ""),
            entrypoint=thread.entrypoint,
            resume_decisions=decisions,
            resume_tool_inputs=resume_tool_inputs,
        )
        return {"accepted": True, "assistant_message": assistant_message, "thread": self.store.get_thread(thread_id)}

    async def stop(self, *, thread_id: str, payload: Any) -> dict[str, Any]:
        task = self.thread_tasks.get(thread_id)
        if task is None or task.done():
            thread = self.store.update_thread(
                thread_id,
                status=ThreadStatus.STOPPED,
                active_message_id="",
                active_run_id="",
                pending_steering=[],
            )
            self.broker.emit(thread_id, "thread.status", status="stopped", data={"status": "stopped", "reason": payload.reason})
            return {"accepted": True, "status": "stopped", "thread": thread}
        self.thread_stop_flags.add(thread_id)
        thread = self.store.update_thread(
            thread_id,
            status=ThreadStatus.STOPPING,
            pending_steering=[],
        )
        self.broker.emit(thread_id, "thread.status", status="stopping", data={"status": "stopping", "emergency": payload.emergency, "reason": payload.reason})
        task.cancel()
        try:
            await task
        except asyncio.CancelledError:
            pass
        current = self.thread_tasks.get(thread_id)
        if current is task:
            self.thread_tasks.pop(thread_id, None)
        self.thread_stop_flags.discard(thread_id)
        thread = self.store.get_thread(thread_id)
        if thread.status == ThreadStatus.STOPPING:
            thread = self.store.update_thread(
                thread_id,
                status=ThreadStatus.STOPPED,
                active_message_id="",
                active_run_id="",
                pending_steering=[],
            )
            self.broker.emit(
                thread_id,
                "thread.status",
                status="stopped",
                data={"status": "stopped", "reason": payload.reason},
            )
        elif thread.pending_steering:
            thread = self.store.update_thread(thread_id, pending_steering=[])
        return {"accepted": True, "status": "stopped", "thread": thread}

    async def launch_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        entrypoint: str,
        model_config: str = "",
        permission_mode: str = "",
        turn_content: str | list[dict[str, Any]] | None = None,
        attachment_metadata: list[dict[str, Any]] | None = None,
        resume_decisions: list[dict[str, Any]] | None = None,
        resume_tool_inputs: list[dict[str, Any]] | None = None,
        existing_user_message_id: str = "",
    ) -> ThreadMessage:
        thread = self.store.get_thread(thread_id)
        normalized_entrypoint = self.normalize_entrypoint(entrypoint)
        if thread.entrypoint != normalized_entrypoint:
            thread = self.store.update_thread(thread_id, entrypoint=normalized_entrypoint)
        turn_permission_mode = self.permission_mode_for_thread(thread, permission_mode)
        prior_assistant_message_id = self.store.latest_message_id(
            thread_id,
            role="assistant",
        )
        assistant_message, text_part_id = self.append_assistant_message(
            thread_id,
            meta={"permission_mode": turn_permission_mode, "entrypoint": normalized_entrypoint},
        )
        attachment_rows = [dict(item) for item in list(attachment_metadata or []) if isinstance(item, dict)]
        if turn_content is None and attachment_rows:
            profile = LLMProfile.from_env_or_file(model_config or None)
            capability = self._capability_for_entrypoint(profile=profile, entrypoint=normalized_entrypoint)
            rebuilt_attachments = self.rebuild_prepared_attachments(attachment_rows, capability=capability)
            turn_content = build_turn_content(prompt, rebuilt_attachments)
            attachment_rows = [item.sidecar() for item in rebuilt_attachments]
        turn_content = self._research_graph_turn_content(
            thread=thread,
            prompt=prompt,
            turn_content=turn_content,
            entrypoint=normalized_entrypoint,
        )
        event_attachments: list[PreparedAttachment] = []
        if attachment_rows:
            event_attachments = [
                PreparedAttachment(
                    artifact_id=str(row.get("artifact_id") or ""),
                    workspace_path=str(row.get("workspace_path") or ""),
                    filename=str(row.get("filename") or ""),
                    mime_type=str(row.get("mime_type") or ""),
                    size_bytes=int(row.get("size_bytes") or 0),
                    kind=str(row.get("kind") or ""),
                    current_turn_block={"type": str(row.get("sent_as") or "content_block")}
                    if bool(row.get("sent_to_model"))
                    else None,
                    warnings=[str(item) for item in list(row.get("warnings") or []) if str(item).strip()],
                )
                for row in attachment_rows
            ]

        async def _execute() -> None:
            active_run_context = None
            terminal_status = ""
            try:
                llm_profile = LLMProfile.from_env_or_file(model_config or None)
                built = self.build_runner(
                    workspace=self.workspace,
                    llm_profile=llm_profile,
                    reporter=None,
                    run_control=RunControl(),
                    project_id=self.workspace_id or self.workspace.name,
                    preferred_entrypoint=normalized_entrypoint,
                    interrupt_on=self.interrupt_on_for_permission_mode(turn_permission_mode),
                )
                active_run_context = built.run_context
                self.store.update_message(
                    assistant_message.thread_id,
                    assistant_message.id,
                    meta={"run_id": built.run_context.run_id, "permission_mode": turn_permission_mode},
                )
                if event_attachments:
                    self.broker.emit(
                        thread_id,
                        "multimodal.prepared",
                        message_id=assistant_message.id,
                        status="completed",
                        data={
                            "run_id": built.run_context.run_id,
                            "thread_id": thread_id,
                            "message_id": assistant_message.id,
                            "entrypoint": normalized_entrypoint,
                            **multimodal_prepare_summary(event_attachments),
                        },
                    )
                streaming_runner = self.streaming_runner_cls(
                    runner=built.runner,
                    thread_store=self.store,
                    event_broker=self.broker,
                    artifact_registry=self.artifact_registry,
                    should_stop=self.should_stop,
                    should_steer=lambda active_thread_id: bool(
                        self.store.get_thread(active_thread_id).pending_steering
                    ),
                )
                if resume_decisions is not None:
                    run_result = await streaming_runner.aresume(
                        decisions=resume_decisions,
                        entrypoint=normalized_entrypoint,
                        thread_id=thread_id,
                        message_id=assistant_message.id,
                        text_part_id=text_part_id,
                        deepagent_thread_id=thread.deepagent_thread_id,
                        resume_tool_inputs=resume_tool_inputs or [],
                    )
                else:
                    run_result = await streaming_runner.arun_turn(
                        prompt=prompt,
                        content=turn_content,
                        entrypoint=normalized_entrypoint,
                        thread_id=thread_id,
                        message_id=assistant_message.id,
                        text_part_id=text_part_id,
                        deepagent_thread_id=thread.deepagent_thread_id,
                    )
                terminal_status = str(
                    run_result.get("status") if isinstance(run_result, dict) else ""
                ).strip() or "done"
            except asyncio.CancelledError:
                terminal_status = "stopped"
                self.store.update_thread(thread_id, status=ThreadStatus.STOPPED, active_message_id="", active_run_id="")
                self.broker.emit(thread_id, "thread.status", message_id=assistant_message.id, status="stopped", data={"status": "stopped"})
            except Exception as exc:
                terminal_status = "error"
                self.store.update_thread(thread_id, status=ThreadStatus.ERROR, active_message_id="", active_run_id="")
                self.broker.emit(thread_id, "error", message_id=assistant_message.id, status="error", data={"error": str(exc)})
            finally:
                if self.on_turn_finished is not None:
                    try:
                        self.on_turn_finished(
                            workspace=self.workspace,
                            workspace_id=self.workspace_id,
                            thread_id=thread_id,
                            message_id=str(existing_user_message_id or ""),
                            prior_assistant_message_id=prior_assistant_message_id,
                            assistant_message_id=assistant_message.id,
                            run_id=(
                                active_run_context.run_id
                                if active_run_context is not None
                                else ""
                            ),
                            run_dir=(
                                active_run_context.run_dir
                                if active_run_context is not None
                                else ""
                            ),
                            entrypoint=normalized_entrypoint,
                            terminal_status=terminal_status or "unknown",
                            model_config=model_config,
                        )
                    except Exception:
                        logger.exception("Failed to enqueue self-evolution job for thread %s", thread_id)
                current = self.thread_tasks.get(thread_id)
                task = asyncio.current_task()
                if current is task:
                    self.thread_tasks.pop(thread_id, None)
                self.thread_stop_flags.discard(thread_id)
                try:
                    latest = self.store.get_thread(thread_id)
                    if (
                        terminal_status in {"done", "steered"}
                        and latest.status == ThreadStatus.IDLE
                        and latest.pending_steering
                    ):
                        pending = list(latest.pending_steering)
                        steering = pending.pop(0)
                        self.store.update_thread(thread_id, pending_steering=pending, status=ThreadStatus.IDLE)
                        await self.launch_turn(
                            thread_id=thread_id,
                            prompt=str(steering.get("text") or ""),
                            entrypoint=str(steering.get("entrypoint") or entrypoint),
                            model_config=str(steering.get("model_config") or model_config),
                            permission_mode=str(steering.get("permission_mode") or self.permission_mode_for_thread(latest, "")),
                            attachment_metadata=[
                                dict(item) for item in list(steering.get("attachments") or []) if isinstance(item, dict)
                            ],
                            existing_user_message_id=str(steering.get("message_id") or ""),
                        )
                except Exception:
                    logger.exception("Failed to apply queued steering for thread %s", thread_id)

        task = asyncio.create_task(_execute())
        self.thread_tasks[thread_id] = task
        self.store.update_thread(thread_id, status=ThreadStatus.RUNNING, active_message_id=assistant_message.id)
        self.broker.emit(thread_id, "thread.status", message_id=assistant_message.id, status="running", data={"status": "running"})
        return assistant_message


__all__ = ["ThreadAgentLoopService"]
