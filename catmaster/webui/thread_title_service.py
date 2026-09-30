from __future__ import annotations

import asyncio
import json
import re
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage

from catmaster.llm import llm_text
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.observed_invocation import observed_invocation

_TITLE_INPUT_LIMIT = 4_000
_TITLE_OUTPUT_LIMIT = 64
_TITLE_TIMEOUT_SECONDS = 30.0
_GENERIC_TITLES = {
    "new thread",
    "research question",
    "task processing",
    "thread title",
    "user request",
    "任务处理",
    "研究问题",
    "用户请求",
    "新对话",
}
_ENTRYPOINT_LABELS = {
    "research": "Research",
    "persistent_research": "Persistent Research",
    "experiment": "Experiment",
    "writing": "Writing",
    "peer_review": "Peer Review",
    "literature_review": "Literature Review",
}


def attachment_names_from_message(message: Any) -> list[str]:
    sidecar = getattr(message, "structured_sidecar", {})
    attachments = sidecar.get("attachments") if isinstance(sidecar, dict) else None
    if not isinstance(attachments, list):
        return []
    names: list[str] = []
    for item in attachments:
        if not isinstance(item, dict):
            continue
        name = Path(str(item.get("filename") or "")).name.strip()
        if name and name not in names:
            names.append(name)
    return names


def sanitize_generated_thread_title(value: Any) -> str:
    raw = str(value or "").strip()
    lines = [line.strip() for line in raw.splitlines() if line.strip()]
    if len(lines) != 1:
        return ""
    title = lines[0]
    title = re.sub(r"^```(?:[A-Za-z0-9_+.-]+)?\s*", "", title)
    title = re.sub(r"\s*```$", "", title)
    title = re.sub(r"^(?:title|标题)\s*[:：]\s*", "", title, flags=re.IGNORECASE)
    title = re.sub(r"\s+", " ", title).strip()
    title = title.strip("`*_~#\"'“”‘’「」『』《》 ")
    title = title.rstrip("。.!！?？,，;；:：").strip()
    if not title or title.casefold() in _GENERIC_TITLES:
        return ""
    if not any(character.isalnum() for character in title):
        return ""
    if len(title) > _TITLE_OUTPUT_LIMIT:
        title = title[: _TITLE_OUTPUT_LIMIT - 1].rstrip() + "…"
    return title


def _response_text(response: Any) -> str:
    try:
        blocks = getattr(response, "content_blocks", None)
    except Exception:
        blocks = None
    if isinstance(blocks, list):
        parts = [
            str(block.get("text") or "")
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        text = "".join(parts).strip()
        if text:
            return text
    return llm_text(response).strip()


async def generate_semantic_thread_title(
    *,
    workspace: Path,
    thread_id: str,
    question: str,
    attachment_names: list[str],
    entrypoint: str,
    model_config: str = "",
) -> str:
    """Run one explicitly configured, tool-free model call for a thread title."""

    profile = LLMProfile.from_env_or_file(model_config or None)
    if not str(profile.agents.get("thread_title") or "").strip():
        return ""
    model = build_chat_model(profile.config_for_role("thread_title"))
    entry_label = _ENTRYPOINT_LABELS.get(str(entrypoint or "").strip(), "Research")
    payload = {
        "entry": entry_label,
        "question": re.sub(r"\s+", " ", str(question or "")).strip()[:_TITLE_INPUT_LIMIT],
        "attachment_names": [Path(str(name)).name[:200] for name in attachment_names[:8]],
    }
    messages = [
        SystemMessage(
            content=(
                "Create a concise title for one scientific-work thread. Return exactly one plain-text line "
                "in the main language of the question. Chinese titles are usually 8-20 characters; English "
                "titles are usually 3-8 words. Name the research object and requested goal. Do not add quotes, "
                "Markdown, punctuation, explanation, generic labels, paths, credentials, long identifiers, or "
                "claims about results that have not been obtained."
            )
        ),
        HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
    ]
    with observed_invocation(
        workspace=workspace, entrypoint="thread_title", stage="thread_title",
        model_label=profile.label_for_role("thread_title"),
        context={"source_thread_id": thread_id},
    ) as (config, _callback):
        response = await asyncio.wait_for(
            model.ainvoke(messages, config=config),
            timeout=_TITLE_TIMEOUT_SECONDS,
        )
    return sanitize_generated_thread_title(_response_text(response))


__all__ = [
    "attachment_names_from_message",
    "generate_semantic_thread_title",
    "sanitize_generated_thread_title",
]
