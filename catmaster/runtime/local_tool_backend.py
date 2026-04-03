from __future__ import annotations

import json
import logging
import inspect
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from langchain_core.messages import ToolMessage

from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.tool_observation_projection import project_tool_observation
from catmaster.runtime.tool_output_adapter import (
    CatMasterToolExecutionError,
    adapt_tool_return,
    tool_error_to_message,
)
from catmaster.runtime.trace_store import TraceStore
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.runtime.tool_executor import ToolExecutor
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.registry import ToolRegistry


class LocalToolBackend(ToolBackend):
    def __init__(
        self,
        *,
        registry: ToolRegistry,
        tool_executor: ToolExecutor,
        artifact_store: ArtifactStore,
        trace_store: TraceStore,
        role: str = "tool_backend",
        workspace: Optional[Path | str] = None,
    ) -> None:
        self.registry = registry
        self.tool_executor = tool_executor
        self.artifact_store = artifact_store
        self.trace_store = trace_store
        self.role = role
        self.workspace = Path(workspace).expanduser().resolve() if workspace is not None else None
        self.logger = logging.getLogger(__name__)
        self._active_lock = threading.Lock()
        self._active_calls: Dict[str, Dict[str, Any]] = {}

    def list_function_tools(self) -> list[dict]:
        return self.registry.as_openai_tools()

    def call(
        self,
        name: str,
        arguments_json: str,
        *,
        toolcall_key: str,
        call_id: str | None = None,
    ) -> ToolMessage:
        raw_params = self._parse_arguments(arguments_json)
        validation_key = self._validation_key(toolcall_key, name)
        validation = self.tool_executor.validate(name, raw_params, toolcall_key=validation_key)

        refs = self.artifact_store.toolcall_refs(toolcall_key)
        validated_params = validation.get("validated_params") if validation.get("ok") else None
        validation_status = "validated" if validation.get("ok") else "validation_failed"

        self.artifact_store.write_input(toolcall_key, {
            "raw_params": raw_params,
            "validated_params": validated_params,
            "tool_name": name,
            "toolcall_id": toolcall_key,
            "call_id": call_id,
            "status": validation_status,
            "input_ref": refs["input_ref"],
        })

        if not validation.get("ok"):
            attempt = int(validation.get("attempt_count") or 0)
            max_attempts = int(validation.get("max_attempts") or 0)
            exc = CatMasterToolExecutionError(
                tool_name=name,
                public_message=str(validation.get("error_digest") or "validation failed"),
                artifact={
                    "validation_errors": validation.get("errors") or [],
                    "attempt_count": attempt,
                    "max_attempts": max_attempts,
                    "next_step": validation.get("next_step") or "",
                    "raw_params": raw_params,
                },
                retryable=bool(max_attempts and attempt < max_attempts),
                error_code="validation_error",
            )
            message = tool_error_to_message(
                exc=exc,
                tool_name=name,
                tool_call_id=toolcall_key,
            )
        else:
            func = self.registry.get_tool_function(name)
            try:
                payload = validated_params or {}
                self._set_active_call(
                    toolcall_key=toolcall_key,
                    tool_name=name,
                    call_id=call_id,
                )
                with toolcall_context(toolcall_key, run_dir=str(self.artifact_store.run_dir)):
                    if self.workspace is not None:
                        with workspace_scope(self.workspace):
                            raw_output = func(payload)
                    else:
                        raw_output = func(payload)
                if inspect.isawaitable(raw_output):
                    if inspect.iscoroutine(raw_output):
                        raw_output.close()
                    raise RuntimeError(
                        f"Tool {name} returned an awaitable, but LocalToolBackend is sync-only. "
                        "Use the async specialist runtime/agent ainvoke path or provide a sync tool function."
                    )
                content, artifact = adapt_tool_return(
                    tool_name=name,
                    raw_result=raw_output,
                    tool_args=payload,
                    workspace_files_root=workspace_root(self.workspace),
                )
                message = ToolMessage(
                    content=content,
                    artifact=artifact,
                    tool_call_id=toolcall_key,
                    name=name,
                    status="success",
                )
            except Exception as exc:
                message = tool_error_to_message(
                    exc=exc,
                    tool_name=name,
                    tool_call_id=toolcall_key,
                )
            finally:
                self._clear_active_call(toolcall_key)

        projection = project_tool_observation(message, tool_name=name)
        tool_status = str(projection.get("tool_status") or message.status)
        error_text = str(projection.get("error") or "")

        self.artifact_store.write_output(toolcall_key, {
            "status": tool_status,
            "validation_status": validation_status,
            "raw_output": message.model_dump(mode="json"),
            "projection": projection,
            "tool_status": tool_status,
            "tool_name": name,
        })

        record = {
            "role": self.role,
            "tool_name": name,
            "validated_params": validated_params,
            "status": tool_status,
            "error": error_text,
            "toolcall_id": toolcall_key,
            "call_id": call_id,
            "input_ref": refs.get("input_ref"),
            "output_ref": refs.get("output_ref"),
        }
        self.trace_store.append_toolcall(record)
        return message

    def cancel_active_call(self, toolcall_key: str) -> bool:
        with self._active_lock:
            active = dict(self._active_calls.get(toolcall_key) or {})
        if not active:
            return False
        return False

    def _set_active_call(
        self,
        *,
        toolcall_key: str,
        tool_name: str,
        call_id: str | None,
    ) -> None:
        with self._active_lock:
            self._active_calls[toolcall_key] = {
                "tool_name": tool_name,
                "call_id": call_id or "",
            }

    def _clear_active_call(self, toolcall_key: str) -> None:
        with self._active_lock:
            self._active_calls.pop(toolcall_key, None)

    @staticmethod
    def _parse_arguments(arguments: Any) -> Any:
        if arguments is None:
            return {}
        if isinstance(arguments, dict):
            return arguments
        if not isinstance(arguments, str):
            return arguments
        if not arguments.strip():
            return {}
        try:
            return json.loads(arguments)
        except Exception:
            return arguments

    @staticmethod
    def _validation_key(toolcall_key: str, tool_name: str) -> str:
        task_id = ""
        if toolcall_key:
            marker = "_s"
            idx = toolcall_key.find(marker)
            if idx > 0:
                task_id = toolcall_key[:idx]
        if task_id:
            return f"{task_id}:{tool_name}"
        if toolcall_key:
            return f"{toolcall_key}:{tool_name}"
        return tool_name


__all__ = ["LocalToolBackend"]
