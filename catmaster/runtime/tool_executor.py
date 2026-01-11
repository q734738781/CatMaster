#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ToolExecutor performs input validation (Pydantic) and manages attempt limits.
"""
from __future__ import annotations

from typing import Any, Dict, Optional
from pydantic import ValidationError

from catmaster.tools.registry import ToolRegistry


class ToolExecutor:
    def __init__(self, registry: ToolRegistry, *, max_attempts: int = 3):
        self.registry = registry
        self.max_attempts = max_attempts
        self._attempts: Dict[str, int] = {}

    def validate(
        self,
        tool_name: str,
        raw_params: Any,
        *,
        toolcall_key: Optional[str] = None,
    ) -> Dict[str, Any]:
        key = toolcall_key or tool_name
        if not isinstance(raw_params, dict):
            attempt_count = self._bump_attempt(key)
            error = "params must be a JSON object"
            return {
                "ok": False,
                "tool_name": tool_name,
                "raw_params": raw_params,
                "validated_params": None,
                "errors": [{"loc": "", "msg": error, "type": "type_error"}],
                "error_digest": error,
                "tool_output": self._validation_output(tool_name, error, [{"loc": "", "msg": error, "type": "type_error"}]),
                "attempt_count": attempt_count,
                "max_attempts": self.max_attempts,
                "next_step": self._next_step_hint(tool_name, error, attempt_count),
            }

        tool_info = self.registry.get_tool_info(tool_name)
        input_model = tool_info.get("input_model")
        if input_model is None:
            attempt_count = self._bump_attempt(key)
            error = f"unknown tool: {tool_name}"
            return {
                "ok": False,
                "tool_name": tool_name,
                "raw_params": raw_params,
                "validated_params": None,
                "errors": [{"loc": "", "msg": error, "type": "value_error"}],
                "error_digest": error,
                "tool_output": self._validation_output(tool_name, error, [{"loc": "", "msg": error, "type": "value_error"}]),
                "attempt_count": attempt_count,
                "max_attempts": self.max_attempts,
                "next_step": self._next_step_hint(tool_name, error, attempt_count),
            }

        extra_keys = [key for key in raw_params.keys() if key not in input_model.model_fields]
        extra_errors = [
            {"loc": key, "msg": "extra fields not permitted", "type": "extra_forbidden"}
            for key in extra_keys
        ]

        try:
            validated = input_model.model_validate(raw_params)
        except ValidationError as exc:
            errors = self._format_validation_errors(exc) + extra_errors
            error_digest = self._summarize_errors(errors)
            attempt_count = self._bump_attempt(key)
            return {
                "ok": False,
                "tool_name": tool_name,
                "raw_params": raw_params,
                "validated_params": None,
                "errors": errors,
                "error_digest": error_digest,
                "tool_output": self._validation_output(tool_name, error_digest, errors),
                "attempt_count": attempt_count,
                "max_attempts": self.max_attempts,
                "next_step": self._next_step_hint(tool_name, error_digest, attempt_count),
            }

        if extra_errors:
            error_digest = self._summarize_errors(extra_errors)
            attempt_count = self._bump_attempt(key)
            return {
                "ok": False,
                "tool_name": tool_name,
                "raw_params": raw_params,
                "validated_params": None,
                "errors": extra_errors,
                "error_digest": error_digest,
                "tool_output": self._validation_output(tool_name, error_digest, extra_errors),
                "attempt_count": attempt_count,
                "max_attempts": self.max_attempts,
                "next_step": self._next_step_hint(tool_name, error_digest, attempt_count),
            }

        self._reset_attempts(key)
        return {
            "ok": True,
            "tool_name": tool_name,
            "raw_params": raw_params,
            "validated_params": validated.model_dump(),
            "errors": [],
            "error_digest": "",
            "attempt_count": self._attempts.get(key, 0),
            "max_attempts": self.max_attempts,
        }

    def _bump_attempt(self, key: str) -> int:
        self._attempts[key] = self._attempts.get(key, 0) + 1
        return self._attempts[key]

    def _reset_attempts(self, key: str) -> None:
        if key in self._attempts:
            self._attempts[key] = 0

    @staticmethod
    def _format_validation_errors(exc: ValidationError) -> list[dict]:
        errors = []
        for err in exc.errors():
            loc = "/".join(str(part) for part in err.get("loc", []))
            errors.append({
                "loc": loc,
                "msg": err.get("msg", ""),
                "type": err.get("type", ""),
            })
        return errors

    @staticmethod
    def _summarize_errors(errors: list[dict], limit: int = 3) -> str:
        if not errors:
            return ""
        parts = []
        for err in errors[:limit]:
            loc = err.get("loc", "") or "<root>"
            parts.append(f"{loc}: {err.get('msg', '')}")
        if len(errors) > limit:
            parts.append(f"... and {len(errors) - limit} more")
        return "; ".join(parts)

    def _next_step_hint(self, tool_name: str, error_digest: str, attempt_count: int) -> str:
        if attempt_count >= self.max_attempts:
            return (
                f"Max attempts reached for tool {tool_name}. "
                "Ask the user for clarification or choose a safer diagnostic tool (e.g., workspace_list_files/workspace_read_file) before retrying."
            )
        return (
            f"Fix parameters for tool {tool_name} to satisfy the schema. "
            f"Validation errors: {error_digest}. "
            "Return only valid params."
        )

    @staticmethod
    def _validation_output(tool_name: str, error_digest: str, errors: list[dict]) -> Dict[str, Any]:
        return {
            "status": "error",
            "tool_name": tool_name,
            "data": {
                "validation_errors": errors,
            },
            "error": error_digest,
        }
