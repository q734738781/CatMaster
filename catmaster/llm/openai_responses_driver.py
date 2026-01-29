from __future__ import annotations

from typing import Any

from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.types import ToolCall, TurnResult

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - optional dependency
    OpenAI = None
    _openai_import_error = exc
else:  # pragma: no cover - optional dependency
    _openai_import_error = None


def _as_dict(item: Any) -> dict:
    """
    Convert OpenAI SDK response output items to plain dicts.

    The official SDK may return Pydantic models (v2) or dicts depending on version.
    We normalize to dict for downstream parsing and for feeding items back into the next turn.
    """
    if isinstance(item, dict):
        return item
    if hasattr(item, "model_dump"):
        try:
            return item.model_dump(mode="json")  # type: ignore[attr-defined]
        except TypeError:
            return item.model_dump()  # type: ignore[attr-defined]
    if hasattr(item, "dict"):
        return item.dict()  # type: ignore[attr-defined]
    if hasattr(item, "__dict__"):
        return dict(getattr(item, "__dict__"))
    raise TypeError(f"Unsupported output item type: {type(item).__name__}")


def _parse_output_items(output_items: list[dict]) -> TurnResult:
    tool_calls: list[ToolCall] = []
    output_text_parts: list[str] = []
    for item in output_items:
        item_type = item.get("type")
        if item_type == "function_call":
            tool_calls.append(ToolCall(
                name=item.get("name", ""),
                call_id=item.get("call_id") or item.get("id", ""),
                arguments=item.get("arguments", ""),
                raw=item,
            ))
            continue
        if item_type == "message":
            content = item.get("content") or []
            if isinstance(content, list):
                for part in content:
                    if not isinstance(part, dict):
                        continue
                    part_type = part.get("type")
                    if part_type in ("output_text", "input_text"):
                        text = part.get("text")
                        if text:
                            output_text_parts.append(text)
            continue
        if item_type == "output_text":
            text = item.get("text")
            if text:
                output_text_parts.append(text)
    return TurnResult(
        output_text="".join(output_text_parts),
        tool_calls=tool_calls,
        output_items_raw=output_items,
    )


class OpenAIResponsesDriver(ToolCallingDriver):
    def __init__(
        self,
        *,
        client: Any | None = None,
        model: str = "gpt-4.1-mini",
        api_key: str | None = None,
        base_url: str | None = None,
        default_headers: dict | None = None,
    ):
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIResponsesDriver. Install with `pip install openai`"
            ) from _openai_import_error
        if client is not None:
            self.client = client
        else:
            kwargs: dict[str, Any] = {}
            if api_key:
                kwargs["api_key"] = api_key
            if base_url:
                kwargs["base_url"] = base_url
            if default_headers:
                kwargs["default_headers"] = default_headers
            self.client = OpenAI(**kwargs)
        self.model = model

    def create_turn(
        self,
        *,
        input_items: list[dict],
        tools: list[dict] | None = None,
        model: str | None = None,
        temperature: float | None = None,
        max_output_tokens: int | None = None,
        metadata: dict | None = None,
        **kwargs: Any,
    ) -> TurnResult:
        payload: dict[str, Any] = {
            "model": model or self.model,
            "input": input_items,
        }
        if tools is not None:
            payload["tools"] = tools
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_output_tokens"] = max_output_tokens
        if metadata is not None:
            payload["metadata"] = metadata
        payload.update(kwargs)

        # --- Compatibility shim: LangChain-style -> Responses API style ---
        effort = payload.pop("reasoning_effort", None)
        if effort is not None:
            reasoning = payload.get("reasoning")
            if isinstance(reasoning, dict):
                reasoning.setdefault("effort", effort)
            else:
                payload["reasoning"] = {"effort": effort}

        # Map max_tokens (if present) to Responses API max_output_tokens.
        if "max_tokens" in payload and "max_output_tokens" not in payload:
            payload["max_output_tokens"] = payload.pop("max_tokens")
        else:
            payload.pop("max_tokens", None)

        # Drop unsupported penalties for Responses API.
        payload.pop("frequency_penalty", None)
        payload.pop("presence_penalty", None)

        # GPT-5.x: sampling params only supported when reasoning.effort == "none".
        model_name = str(payload.get("model") or "")
        effort_value = None
        reasoning = payload.get("reasoning")
        if isinstance(reasoning, dict):
            effort_value = reasoning.get("effort")
        if model_name.startswith(("gpt-5.2", "gpt-5.1")) and effort_value not in (None, "none"):
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            payload.pop("logprobs", None)

        response = self.client.responses.create(**payload)
        raw_output_items = list(getattr(response, "output", []) or [])
        output_items = [_as_dict(item) for item in raw_output_items]
        return _parse_output_items(output_items)
