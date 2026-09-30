from __future__ import annotations

from typing import Any


def llm_text(resp: Any) -> str:
    """Extract plain text from an LLM response across output versions."""
    text = getattr(resp, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    content = getattr(resp, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and "text" in block:
                parts.append(block.get("text") or "")
        return "".join(parts)

    return str(content)


__all__ = ["llm_text"]
