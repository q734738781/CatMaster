from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer


def _without_inline_file_blocks(message: BaseMessage) -> BaseMessage:
    """Keep provider file bytes out of durable checkpoints.

    The native ``read_file`` tool records the workspace path on tool messages,
    so a resumed agent can reopen the source instead of carrying a large base64
    payload forever. Current-turn delivery remains native and unchanged.
    """
    if not isinstance(message.content, list):
        return message
    changed = False
    cleaned: list[Any] = []
    for block in message.content:
        if not (
            isinstance(block, dict)
            and str(block.get("type") or "").strip().lower() == "file"
            and (block.get("base64") or block.get("file_data") or block.get("data"))
        ):
            cleaned.append(block)
            continue
        changed = True
        path = str(message.additional_kwargs.get("read_file_path") or "").strip()
        filename = str(block.get("filename") or block.get("name") or "stored file").strip()
        source = f"`{path}`" if path else filename
        cleaned.append(
            {
                "type": "text",
                "text": (
                    f"[Inline file bytes omitted from persisted history: {source}. "
                    "Use the native `read_file` tool with the stored workspace path to reopen it.]"
                ),
            }
        )
    if not changed:
        return message
    kwargs = dict(message.additional_kwargs)
    kwargs["catmaster_inline_file_payload_removed"] = True
    return message.model_copy(update={"content": cleaned, "additional_kwargs": kwargs})


def _without_inline_files(value: Any) -> Any:
    """Remove provider file bytes at the checkpoint persistence boundary."""
    if isinstance(value, BaseMessage):
        return _without_inline_file_blocks(value)
    if isinstance(value, dict):
        changed = False
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            next_item = _without_inline_files(item)
            cleaned[key] = next_item
            changed = changed or next_item is not item
        return cleaned if changed else value
    if isinstance(value, list):
        cleaned = [_without_inline_files(item) for item in value]
        return cleaned if any(new is not old for new, old in zip(cleaned, value)) else value
    if isinstance(value, tuple):
        cleaned = tuple(_without_inline_files(item) for item in value)
        return cleaned if any(new is not old for new, old in zip(cleaned, value)) else value
    return value


class FileSafeCheckpointSerializer:
    """LangGraph serializer that never persists inline provider file payloads."""

    def __init__(self) -> None:
        self._delegate = JsonPlusSerializer()

    def dumps_typed(self, value: Any) -> tuple[str, bytes]:
        return self._delegate.dumps_typed(_without_inline_files(value))

    def loads_typed(self, value: tuple[str, bytes]) -> Any:
        return self._delegate.loads_typed(value)


__all__ = ["FileSafeCheckpointSerializer"]
