from __future__ import annotations

from typing import Any

from langchain_core.messages import BaseMessage
from langgraph.checkpoint.serde.jsonplus import JsonPlusSerializer

from catmaster.runtime.document_access import sanitize_document_message


def _without_inline_documents(value: Any) -> Any:
    """Remove document bytes from values at the checkpoint persistence boundary."""
    if isinstance(value, BaseMessage):
        return sanitize_document_message(value)
    if isinstance(value, dict):
        changed = False
        cleaned: dict[Any, Any] = {}
        for key, item in value.items():
            next_item = _without_inline_documents(item)
            cleaned[key] = next_item
            changed = changed or next_item is not item
        return cleaned if changed else value
    if isinstance(value, list):
        cleaned = [_without_inline_documents(item) for item in value]
        return cleaned if any(new is not old for new, old in zip(cleaned, value)) else value
    if isinstance(value, tuple):
        cleaned = tuple(_without_inline_documents(item) for item in value)
        return cleaned if any(new is not old for new, old in zip(cleaned, value)) else value
    return value


class DocumentSafeCheckpointSerializer:
    """LangGraph serializer that never persists inline PDF or Office payloads."""

    def __init__(self) -> None:
        self._delegate = JsonPlusSerializer()

    def dumps_typed(self, value: Any) -> tuple[str, bytes]:
        return self._delegate.dumps_typed(_without_inline_documents(value))

    def loads_typed(self, value: tuple[str, bytes]) -> Any:
        return self._delegate.loads_typed(value)


__all__ = ["DocumentSafeCheckpointSerializer"]
