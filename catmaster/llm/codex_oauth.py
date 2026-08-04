from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatResult


_APPLY_PATCH_TOOL_NAME = "apply_patch"


def _custom_tool_call_payload(block: Any) -> dict[str, Any] | None:
    if not isinstance(block, dict):
        return None
    if block.get("type") == "custom_tool_call":
        return block
    if block.get("type") != "non_standard":
        return None
    value = block.get("value")
    if isinstance(value, dict) and value.get("type") == "custom_tool_call":
        return value
    return None


def recover_codex_apply_patch_tool_calls(message: AIMessage) -> AIMessage:
    """Restore scheduler metadata lost by LangChain v3 protocol streaming.

    OpenAI Responses emits ``apply_patch`` as a freeform ``custom_tool_call``.
    LangChain OpenAI parses that call, but LangChain Core 1.5's v3 protocol
    compatibility bridge currently projects the provider block as
    ``non_standard`` without carrying its ``tool_call_chunks`` into the final
    ``AIMessage.tool_calls``. DeepAgents routes tools from ``tool_calls``, so
    recover only this bound custom tool at the model-result boundary.

    The provider content block stays byte-for-byte equivalent for Responses API
    replay. This function is idempotent and becomes a no-op when LangChain
    preserves the tool call itself.
    """
    if not isinstance(message.content, list):
        return message

    existing = list(message.tool_calls)
    existing_ids = {
        str(call.get("id") or "")
        for call in existing
        if isinstance(call, dict) and call.get("id")
    }
    recovered: list[dict[str, Any]] = []

    for block in message.content:
        payload = _custom_tool_call_payload(block)
        if payload is None or payload.get("name") != _APPLY_PATCH_TOOL_NAME:
            continue

        status = str(payload.get("status") or "").strip().lower()
        if status and status != "completed":
            continue
        call_id = str(payload.get("call_id") or "").strip()
        patch = payload.get("input")
        if not call_id or not isinstance(patch, str) or call_id in existing_ids:
            continue

        recovered.append(
            {
                "name": _APPLY_PATCH_TOOL_NAME,
                "args": {"__arg1": patch},
                "id": call_id,
                "type": "tool_call",
            }
        )
        existing_ids.add(call_id)

    if not recovered:
        return message
    return message.model_copy(update={"tool_calls": [*existing, *recovered]})


def recover_codex_apply_patch_chat_result(result: ChatResult) -> ChatResult:
    """Normalize every AI generation while preserving the original result."""
    generations = []
    changed = False
    for generation in result.generations:
        message = generation.message
        normalized = (
            recover_codex_apply_patch_tool_calls(message)
            if isinstance(message, AIMessage)
            else message
        )
        if normalized is message:
            generations.append(generation)
            continue
        generations.append(generation.model_copy(update={"message": normalized}))
        changed = True

    if not changed:
        return result
    return result.model_copy(update={"generations": generations})


__all__ = [
    "recover_codex_apply_patch_chat_result",
    "recover_codex_apply_patch_tool_calls",
]
