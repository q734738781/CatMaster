from __future__ import annotations

from typing import Any, Optional, Dict

from catmaster.llm.driver import ToolCallingDriver
from catmaster.llm.types import ToolCall, TurnResult

try:
    from openai import OpenAI
except Exception as exc:  # pragma: no cover - optional dependency
    OpenAI = None
    _openai_import_error = exc
else:  # pragma: no cover - optional dependency
    _openai_import_error = None


def _extract_text_from_message_item(item: dict) -> str:
    content = item.get("content") or []
    if isinstance(content, str):
        return content
    parts: list[str] = []
    if isinstance(content, list):
        for part in content:
            if isinstance(part, dict) and part.get("text"):
                parts.append(part["text"])
    return "".join(parts)


def _input_items_to_chat_messages(input_items: list[dict]) -> list[dict]:
    messages: list[dict] = []
    pending_tool_calls: list[dict] = []

    def flush_tool_calls() -> None:
        nonlocal pending_tool_calls
        if pending_tool_calls:
            messages.append({"role": "assistant", "content": None, "tool_calls": pending_tool_calls})
            pending_tool_calls = []

    for item in input_items:
        item_type = item.get("type")
        if item_type == "message":
            role = item.get("role", "user")
            text = _extract_text_from_message_item(item)
            if role == "assistant" and pending_tool_calls:
                messages.append({"role": "assistant", "content": text or None, "tool_calls": pending_tool_calls})
                pending_tool_calls = []
            else:
                flush_tool_calls()
                messages.append({"role": role, "content": text})
            continue

        if item_type == "function_call":
            call_id = item.get("call_id") or item.get("id") or ""
            pending_tool_calls.append({
                "id": call_id,
                "type": "function",
                "function": {
                    "name": item.get("name", ""),
                    "arguments": item.get("arguments", "") or "",
                },
            })
            continue

        if item_type == "function_call_output":
            flush_tool_calls()
            messages.append({
                "role": "tool",
                "tool_call_id": item.get("call_id", ""),
                "content": item.get("output", "") or "",
            })
            continue

        if item_type == "output_text":
            flush_tool_calls()
            messages.append({"role": "assistant", "content": item.get("text", "")})
            continue

    flush_tool_calls()
    return messages


def _responses_style_tools_to_chat_tools(tools: list[dict]) -> list[dict]:
    out: list[dict] = []
    for tool in tools:
        if tool.get("type") != "function":
            continue
        func = {
            "name": tool.get("name", ""),
            "description": tool.get("description", ""),
            "parameters": tool.get("parameters", {}),
        }
        if "strict" in tool:
            func["strict"] = tool["strict"]
        out.append({"type": "function", "function": func})
    return out


def _extract_chat_content(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("text"):
                parts.append(block.get("text") or "")
        return "".join(parts)
    return str(content) if content is not None else ""


def _turnresult_from_chat_response(choice_message: Any) -> TurnResult:
    output_items: list[dict] = []
    tool_calls = None
    if isinstance(choice_message, dict):
        tool_calls = choice_message.get("tool_calls")
    else:
        tool_calls = getattr(choice_message, "tool_calls", None)
    tool_calls = tool_calls or []
    for call in tool_calls:
        if isinstance(call, dict):
            func = call.get("function") or {}
            call_id = call.get("id") or ""
            name = func.get("name") or ""
            arguments = func.get("arguments") or ""
        else:
            call_id = getattr(call, "id", "") or ""
            func = getattr(call, "function", None)
            name = getattr(func, "name", "") if func is not None else ""
            arguments = getattr(func, "arguments", "") if func is not None else ""
        output_items.append({
            "type": "function_call",
            "call_id": call_id,
            "name": name,
            "arguments": arguments,
        })

    content = choice_message.get("content") if isinstance(choice_message, dict) else getattr(choice_message, "content", None)
    content_text = _extract_chat_content(content)
    if content_text:
        output_items.append({
            "type": "message",
            "role": "assistant",
            "content": [{"type": "output_text", "text": content_text}],
        })

    parsed_tool_calls: list[ToolCall] = []
    output_text_parts: list[str] = []
    for item in output_items:
        if item.get("type") == "function_call":
            parsed_tool_calls.append(ToolCall(
                name=item.get("name", ""),
                call_id=item.get("call_id", ""),
                arguments=item.get("arguments", ""),
                raw=item,
            ))
        if item.get("type") == "message":
            for part in item.get("content") or []:
                if isinstance(part, dict) and part.get("type") in ("output_text", "input_text") and part.get("text"):
                    output_text_parts.append(part["text"])

    return TurnResult(
        output_text="".join(output_text_parts),
        tool_calls=parsed_tool_calls,
        output_items_raw=output_items,
    )


class OpenAIChatCompletionsDriver(ToolCallingDriver):
    def __init__(
        self,
        *,
        model: str,
        api_key: str,
        base_url: Optional[str] = None,
        default_headers: Optional[Dict[str, str]] = None,
        client: Any | None = None,
    ) -> None:
        if OpenAI is None:
            raise ImportError(
                "openai package is required for OpenAIChatCompletionsDriver. Install with `pip install openai`"
            ) from _openai_import_error
        if client is not None:
            self.client = client
        else:
            kwargs: dict[str, Any] = {"api_key": api_key}
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
        **kwargs: Any,
    ) -> TurnResult:
        messages = _input_items_to_chat_messages(input_items)
        payload: dict[str, Any] = {
            "model": model or self.model,
            "messages": messages,
        }
        if tools is not None:
            chat_tools = _responses_style_tools_to_chat_tools(tools)
            if chat_tools:
                payload["tools"] = chat_tools
                payload["tool_choice"] = "auto"
        if temperature is not None:
            payload["temperature"] = temperature
        if max_output_tokens is not None:
            payload["max_tokens"] = max_output_tokens

        reasoning_effort = kwargs.pop("reasoning_effort", None)
        reasoning_cfg = kwargs.pop("reasoning", None)
        extra_body = kwargs.pop("extra_body", None) or {}
        if not isinstance(extra_body, dict):
            raise TypeError("extra_body must be a dict")
        if reasoning_cfg is not None:
            if not isinstance(reasoning_cfg, dict):
                raise TypeError("reasoning must be a dict")
            merged = dict(extra_body.get("reasoning") or {})
            merged.update(reasoning_cfg)
            extra_body["reasoning"] = merged
        if reasoning_effort is not None:
            merged = dict(extra_body.get("reasoning") or {})
            merged["effort"] = reasoning_effort
            extra_body["reasoning"] = merged
        if "reasoning" in extra_body and isinstance(extra_body.get("reasoning"), dict):
            extra_body["reasoning"].setdefault("exclude", True)
        if extra_body:
            payload["extra_body"] = extra_body
        payload.update(kwargs)

        effort = None
        extra_body_payload = payload.get("extra_body")
        if isinstance(extra_body_payload, dict):
            reasoning = extra_body_payload.get("reasoning")
            if isinstance(reasoning, dict):
                effort = reasoning.get("effort")
        if effort is not None and str(effort).lower() != "none":
            payload.pop("temperature", None)
            payload.pop("top_p", None)
            payload.pop("logprobs", None)

        resp = self.client.chat.completions.create(**payload)
        choice = resp.choices[0]
        return _turnresult_from_chat_response(choice.message)


__all__ = ["OpenAIChatCompletionsDriver"]
