from __future__ import annotations

import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from catmaster.runtime.artifact_store import ArtifactStore
from catmaster.runtime.trace_store import TraceStore
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.runtime.tool_executor import ToolExecutor
from catmaster.runtime.tool_backend import ToolBackend
from catmaster.tools.base import workspace_scope
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
    ) -> dict:
        raw_params = self._parse_arguments(arguments_json)
        validation_key = self._validation_key(toolcall_key, name)
        validation = self.tool_executor.validate(name, raw_params, toolcall_key=validation_key)

        refs = self.artifact_store.toolcall_refs(toolcall_key)
        validated_params = validation.get("validated_params") if validation.get("ok") else None
        status = "validated" if validation.get("ok") else "validation_failed"

        self.artifact_store.write_input(toolcall_key, {
            "raw_params": raw_params,
            "validated_params": validated_params,
            "tool_name": name,
            "toolcall_id": toolcall_key,
            "call_id": call_id,
            "status": status,
            "input_ref": refs["input_ref"],
        })

        if not validation.get("ok"):
            tool_output = validation.get("tool_output") or {
                "status": "failed",
                "tool_name": name,
                "data": {},
                "error": validation.get("error_digest", "validation failed"),
            }
            # Normalize validation output to match create_tool_output conventions.
            if isinstance(tool_output, dict):
                if tool_output.get("status") == "error":
                    tool_output["status"] = "failed"
                data = tool_output.setdefault("data", {})
                data.setdefault("error_type", "validation_error")
                data["attempt_count"] = validation.get("attempt_count")
                data["max_attempts"] = validation.get("max_attempts")
                if validation.get("next_step"):
                    data["next_step"] = validation.get("next_step")
                attempt = validation.get("attempt_count") or 0
                max_attempts = validation.get("max_attempts") or 0
                tool_output["retryable"] = bool(max_attempts and attempt < max_attempts)
        else:
            func = self.registry.get_tool_function(name)
            try:
                payload = validated_params or {}
                self._set_active_call(
                    toolcall_key=toolcall_key,
                    tool_name=name,
                    call_id=call_id,
                )
                with toolcall_context(toolcall_key):
                    if self.workspace is not None:
                        with workspace_scope(self.workspace):
                            tool_output = func(payload)
                    else:
                        tool_output = func(payload)
            except Exception as exc:
                tool_output = {
                    "status": "failed",
                    "tool_name": name,
                    "data": {},
                    "error": f"{type(exc).__name__}: {exc}",
                }
            finally:
                self._clear_active_call(toolcall_key)

        if not isinstance(tool_output, dict):
            tool_output = {
                "status": "failed",
                "tool_name": name,
                "data": {},
                "error": f"Tool {name} returned non-dict output",
            }

        self.artifact_store.write_output(toolcall_key, {
            "toolresult": tool_output,
            "full_output": tool_output,
            "status": status,
            "tool_status": tool_output.get("status"),
        })

        record = {
            "role": self.role,
            "tool_name": name,
            "validated_params": validated_params,
            "status": tool_output.get("status"),
            "error": tool_output.get("error"),
            "toolcall_id": toolcall_key,
            "call_id": call_id,
            "input_ref": refs.get("input_ref"),
            "output_ref": refs.get("output_ref"),
        }
        self.trace_store.append_toolcall(record)
        return tool_output

    def cancel_active_call(self, toolcall_key: str) -> bool:
        with self._active_lock:
            active = dict(self._active_calls.get(toolcall_key) or {})
        if not active:
            return False
        tool_name = str(active.get("tool_name") or "")
        if tool_name == "bash_exec":
            try:
                from catmaster.tools.misc.bash_exec import cancel_bash_exec_toolcall

                return bool(cancel_bash_exec_toolcall(toolcall_key))
            except Exception:
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
