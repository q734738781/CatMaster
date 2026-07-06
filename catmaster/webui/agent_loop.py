from __future__ import annotations

import asyncio
import base64
import logging
import mimetypes
import re
import time
from pathlib import Path
from typing import Any, Callable

from fastapi import HTTPException

from catmaster.llm.config import LLMProfile
from catmaster.runtime import RunControl

from .artifact_registry import ArtifactRegistry
from .thread_events import ThreadEventBroker
from .thread_models import ArtifactPart, MessagePart, ThreadMessage, ThreadStatus
from .thread_store import ThreadStore, new_id

logger = logging.getLogger(__name__)

UPLOAD_LIMIT_BYTES = 512 * 1024 * 1024


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

    def prepare_submit_attachments(self, thread_id: str, attachments: list[dict[str, Any]]) -> tuple[list[MessagePart], str]:
        if not attachments:
            return [], ""
        parts: list[MessagePart] = []
        prompt_rows: list[str] = []
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
            if kind == "image" and data.startswith("data:"):
                part, prompt_row = self._store_image_attachment(
                    thread_id=thread_id,
                    index=index,
                    name=name,
                    mime=mime,
                    data=data,
                )
                if part is not None:
                    parts.append(part)
                    prompt_rows.append(prompt_row)
            elif text:
                excerpt = text[:20_000]
                parts.append(
                    MessagePart(
                        id=new_id("part_text"),
                        type="text",
                        status="completed",
                        text=excerpt,
                        meta={
                            "source": "composer_attachment",
                            "name": name,
                            "mime_type": mime,
                            "truncated": len(text) > len(excerpt),
                        },
                    )
                )
                prompt_rows.append(f"- text attachment `{name}` included inline")
        if not prompt_rows:
            return parts, ""
        return parts, "\n\nAttached user files:\n" + "\n".join(prompt_rows)

    def _store_image_attachment(self, *, thread_id: str, index: int, name: str, mime: str, data: str) -> tuple[MessagePart | None, str]:
        header, _, encoded = data.partition(",")
        if not encoded:
            return None, ""
        mime_from_header = header[5:].split(";", 1)[0].strip() if header.startswith("data:") else ""
        mime = mime or mime_from_header
        try:
            blob = base64.b64decode(encoded, validate=True)
        except Exception:
            return None, ""
        if len(blob) > UPLOAD_LIMIT_BYTES:
            raise HTTPException(status_code=413, detail=f"Attachment {name} exceeds upload limit.")
        suffix = Path(name).suffix or mimetypes.guess_extension(mime or "") or ".png"
        stem = Path(name).stem or f"image_{index}"
        target = self.workspace / "files" / "attachments" / thread_id / f"{index:03d}_{_safe_attachment_filename(stem + suffix)}"
        target.write_bytes(blob)
        rel_path = str(target.relative_to(self.workspace)).replace("\\", "/")
        artifact = self.artifact_registry.register_path(
            rel_path,
            thread_id=thread_id,
            title=name,
            summary="User-submitted image attachment.",
            mime_type=mime,
            meta={"source": "composer_attachment", "original_name": name},
        )
        part = ArtifactPart(
            id=new_id("part_artifact"),
            status="completed",
            artifact_id=artifact.artifact_id,
            renderer=artifact.renderer,
            title=artifact.title,
            summary=artifact.summary,
            path=artifact.path,
            meta=artifact.model_dump(mode="json"),
        )
        return part, f"- image attachment `{name}` saved as `{artifact.path}`"

    def append_user_message(
        self,
        thread_id: str,
        text: str,
        *,
        attachment_parts: list[MessagePart] | None = None,
        meta: dict[str, Any] | None = None,
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
        actions: list[dict[str, Any]] = []
        for message in self.store.list_messages(thread_id):
            for part in message.parts:
                if part.type != "interrupt" or part.status == "resolved":
                    continue
                raw = (part.meta or {}).get("payload", {}).get("interrupts")
                rows = raw if isinstance(raw, list) else [raw]
                for row in rows:
                    value = row.get("value") if isinstance(row, dict) else None
                    requests = value.get("action_requests") if isinstance(value, dict) else None
                    if isinstance(requests, list):
                        for request in requests:
                            if isinstance(request, dict):
                                actions.append(dict(request))
                    elif isinstance(row, str) and "action_requests" in row:
                        for _ in re.findall(r"['\"]name['\"]\s*:", row):
                            actions.append({})
        return actions

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
        attachment_parts, attachment_prompt = self.prepare_submit_attachments(thread_id, list(payload.attachments or []))
        if not text and not attachment_parts:
            raise HTTPException(status_code=400, detail="Message text is required.")
        prompt_text = text or "User submitted attachments."
        if attachment_prompt:
            prompt_text = f"{prompt_text}{attachment_prompt}"
        thread = self.store.get_thread(thread_id)
        permission_mode = self.permission_mode_for_thread(thread, getattr(payload, "permission_mode", "") or dict(thread.meta or {}).get("permission_mode"))
        current_permission_mode = self.permission_mode_for_thread(thread, "")
        if permission_mode != current_permission_mode:
            thread = self.store.update_thread(thread_id, meta={**dict(thread.meta or {}), "permission_mode": permission_mode})
        task = self.thread_tasks.get(thread_id)
        running = bool(task and not task.done()) or thread.status in {ThreadStatus.RUNNING, ThreadStatus.STOPPING}
        if running:
            user_message = self.append_user_message(thread_id, text, attachment_parts=attachment_parts, meta={"kind": "steering"})
            pending = list(thread.pending_steering or [])
            pending.append(
                {
                    "text": prompt_text,
                    "entrypoint": self.normalize_entrypoint(payload.entrypoint or thread.entrypoint),
                    "model_config": payload.model_config,
                    "permission_mode": permission_mode,
                    "message_id": user_message.id,
                    "created_at": time.time(),
                }
            )
            thread = self.store.update_thread(thread_id, pending_steering=pending)
            self.broker.emit(thread_id, "thread.updated", status=str(thread.status.value), data={"thread": thread.model_dump(mode="json"), "steering_queued": True})
            return {"accepted": True, "queued": True, "thread": thread, "message": user_message}
        user_message = self.append_user_message(thread_id, text, attachment_parts=attachment_parts)
        assistant_message = await self.launch_turn(
            thread_id=thread_id,
            prompt=prompt_text,
            entrypoint=self.normalize_entrypoint(payload.entrypoint or thread.entrypoint),
            model_config=payload.model_config,
            permission_mode=permission_mode,
        )
        return {"accepted": True, "queued": False, "thread": self.store.get_thread(thread_id), "message": user_message, "assistant_message": assistant_message}

    async def resume(self, *, thread_id: str, payload: Any, validate_decisions: Callable[[list[dict[str, Any]]], list[dict[str, Any]]]) -> dict[str, Any]:
        try:
            decisions = validate_decisions(payload.decisions)
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
            thread = self.store.update_thread(thread_id, status=ThreadStatus.STOPPED, active_message_id="", active_run_id="")
            self.broker.emit(thread_id, "thread.status", status="stopped", data={"status": "stopped", "reason": payload.reason})
            return {"accepted": True, "status": "stopped", "thread": thread}
        self.thread_stop_flags.add(thread_id)
        thread = self.store.update_thread(thread_id, status=ThreadStatus.STOPPING)
        self.broker.emit(thread_id, "thread.status", status="stopping", data={"status": "stopping", "emergency": payload.emergency, "reason": payload.reason})
        if payload.emergency:
            task.cancel()
        return {"accepted": True, "status": "stopping", "thread": thread}

    async def launch_turn(
        self,
        *,
        thread_id: str,
        prompt: str,
        entrypoint: str,
        model_config: str = "",
        permission_mode: str = "",
        resume_decisions: list[dict[str, Any]] | None = None,
        resume_tool_inputs: list[dict[str, Any]] | None = None,
        existing_user_message_id: str = "",
    ) -> ThreadMessage:
        thread = self.store.get_thread(thread_id)
        normalized_entrypoint = self.normalize_entrypoint(entrypoint)
        if thread.entrypoint != normalized_entrypoint:
            thread = self.store.update_thread(thread_id, entrypoint=normalized_entrypoint)
        turn_permission_mode = self.permission_mode_for_thread(thread, permission_mode)
        assistant_message, text_part_id = self.append_assistant_message(
            thread_id,
            meta={"permission_mode": turn_permission_mode, "entrypoint": normalized_entrypoint},
        )

        async def _execute() -> None:
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
                self.store.update_message(
                    assistant_message.thread_id,
                    assistant_message.id,
                    meta={"run_id": built.run_context.run_id, "permission_mode": turn_permission_mode},
                )
                streaming_runner = self.streaming_runner_cls(
                    runner=built.runner,
                    thread_store=self.store,
                    event_broker=self.broker,
                    artifact_registry=self.artifact_registry,
                    should_stop=self.should_stop,
                )
                if resume_decisions is not None:
                    await streaming_runner.aresume(
                        decisions=resume_decisions,
                        entrypoint=normalized_entrypoint,
                        thread_id=thread_id,
                        message_id=assistant_message.id,
                        text_part_id=text_part_id,
                        deepagent_thread_id=thread.deepagent_thread_id,
                        resume_tool_inputs=resume_tool_inputs or [],
                    )
                else:
                    await streaming_runner.arun_turn(
                        prompt=prompt,
                        entrypoint=normalized_entrypoint,
                        thread_id=thread_id,
                        message_id=assistant_message.id,
                        text_part_id=text_part_id,
                        deepagent_thread_id=thread.deepagent_thread_id,
                    )
            except asyncio.CancelledError:
                self.store.update_thread(thread_id, status=ThreadStatus.STOPPED, active_message_id="", active_run_id="")
                self.broker.emit(thread_id, "thread.status", message_id=assistant_message.id, status="stopped", data={"status": "stopped"})
            except Exception as exc:
                self.store.update_thread(thread_id, status=ThreadStatus.ERROR, active_message_id="", active_run_id="")
                self.broker.emit(thread_id, "error", message_id=assistant_message.id, status="error", data={"error": str(exc)})
            finally:
                current = self.thread_tasks.get(thread_id)
                task = asyncio.current_task()
                if current is task:
                    self.thread_tasks.pop(thread_id, None)
                self.thread_stop_flags.discard(thread_id)
                try:
                    latest = self.store.get_thread(thread_id)
                    if latest.pending_steering and latest.status in {ThreadStatus.IDLE, ThreadStatus.STOPPED, ThreadStatus.ERROR}:
                        pending = list(latest.pending_steering)
                        steering = pending.pop(0)
                        self.store.update_thread(thread_id, pending_steering=pending, status=ThreadStatus.IDLE)
                        await self.launch_turn(
                            thread_id=thread_id,
                            prompt=str(steering.get("text") or ""),
                            entrypoint=str(steering.get("entrypoint") or entrypoint),
                            model_config=str(steering.get("model_config") or model_config),
                            permission_mode=str(steering.get("permission_mode") or self.permission_mode_for_thread(latest, "")),
                            existing_user_message_id=str(steering.get("message_id") or ""),
                        )
                except Exception:
                    logger.exception("Failed to apply queued steering for thread %s", thread_id)

        _ = existing_user_message_id
        task = asyncio.create_task(_execute())
        self.thread_tasks[thread_id] = task
        self.store.update_thread(thread_id, status=ThreadStatus.RUNNING, active_message_id=assistant_message.id)
        self.broker.emit(thread_id, "thread.status", message_id=assistant_message.id, status="running", data={"status": "running"})
        return assistant_message


__all__ = ["ThreadAgentLoopService"]
