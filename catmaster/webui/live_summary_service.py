from __future__ import annotations

import json
import threading
from typing import Any, Dict

from catmaster.agents.llm_utils import llm_text
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_llm_bundle

from .live_state import compact_live_state_for_llm

_LIVE_SUMMARY_LLM: Any = None
_LIVE_SUMMARY_INIT_ATTEMPTED = False
_LIVE_SUMMARY_LOCK = threading.Lock()


def summarize_live_state(
    state: Dict[str, Any],
    *,
    enabled: bool,
    max_events: int,
    max_params_chars: int,
    max_journal_items: int,
    timeout_s: float,
) -> Dict[str, Any]:
    fallback = _rule_live_summary(state)
    if not enabled:
        return fallback
    llm = _get_live_summary_llm()
    if llm is None:
        return fallback

    prompt = {
        "task": "Summarize current run state for a monitoring dashboard.",
        "rules": {
            "output": "JSON only",
            "keys": ["live_headline", "live_summary", "next_expected_step"],
            "max_summary_chars": 320,
            "style": "factual, concise, no speculation",
            "timeout_s": timeout_s,
        },
        "state": compact_live_state_for_llm(
            state,
            max_events=max_events,
            max_params_chars=max_params_chars,
            max_journal_items=max_journal_items,
        ),
    }

    try:
        response = llm.invoke(json.dumps(prompt, ensure_ascii=False))
        text = llm_text(response).strip()
    except Exception:
        return fallback
    parsed = _extract_json_object(text)
    if not parsed:
        return fallback

    headline = str(parsed.get("live_headline") or "").strip()
    summary = str(parsed.get("live_summary") or "").strip()
    next_expected = str(parsed.get("next_expected_step") or "").strip()
    if headline:
        fallback["live_headline"] = headline
    if summary:
        fallback["live_summary"] = summary
    if next_expected:
        fallback["next_expected_step"] = next_expected
    fallback["source"] = "llm"
    return fallback


def _rule_live_summary(state: Dict[str, Any]) -> Dict[str, Any]:
    status = str(state.get("status") or "unknown")
    phase = str(state.get("current_phase") or "")
    task_id = str(state.get("current_task_id") or "")
    task_goal = str(state.get("current_task_goal") or "")
    progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
    completed = int(progress.get("completed", 0))
    total = int(progress.get("total", 0))
    pending = int(progress.get("pending", 0))

    active = state.get("active_toolcall")
    if isinstance(active, dict) and active.get("tool"):
        tool_line = f"Running `{active.get('tool')}`"
    else:
        tool_line = "No active tool call"

    headline_parts = [status]
    if task_id:
        headline_parts.append(task_id)
    if phase:
        headline_parts.append(phase)
    headline = " | ".join(headline_parts)

    summary_bits = [f"Progress {completed}/{total} (pending {pending})."]
    if task_goal:
        summary_bits.append(f"Current goal: {task_goal}")
    summary_bits.append(tool_line + ".")

    next_expected = _next_expected_from_phase(phase=phase, active_tool=bool(isinstance(active, dict) and active.get("tool")))
    return {
        "live_headline": headline,
        "live_summary": " ".join(summary_bits).strip(),
        "next_expected_step": next_expected,
        "source": "rule",
    }


def _next_expected_from_phase(*, phase: str, active_tool: bool) -> str:
    if active_tool:
        return "Wait for the current tool call to finish."
    if phase == "planning":
        return "Wait for planning/proposal to complete."
    if phase == "summarizing":
        return "Wait for task summarizer output."
    if phase == "whiteboard_apply":
        return "Wait for whiteboard ops apply result."
    if phase == "waiting_human":
        return "Provide requested human feedback to continue."
    if phase == "finalizing":
        return "Wait for final report synthesis."
    return "Wait for next task/tool event."


def _extract_json_object(text: str) -> Dict[str, Any]:
    if not text:
        return {}
    try:
        parsed = json.loads(text)
        return parsed if isinstance(parsed, dict) else {}
    except Exception:
        pass
    start = text.find("{")
    end = text.rfind("}")
    if start < 0 or end <= start:
        return {}
    try:
        parsed = json.loads(text[start : end + 1])
    except Exception:
        return {}
    return parsed if isinstance(parsed, dict) else {}


def _get_live_summary_llm() -> Any:
    global _LIVE_SUMMARY_LLM
    global _LIVE_SUMMARY_INIT_ATTEMPTED
    if _LIVE_SUMMARY_LLM is not None:
        return _LIVE_SUMMARY_LLM
    with _LIVE_SUMMARY_LOCK:
        if _LIVE_SUMMARY_LLM is not None:
            return _LIVE_SUMMARY_LLM
        if _LIVE_SUMMARY_INIT_ATTEMPTED:
            return None
        _LIVE_SUMMARY_INIT_ATTEMPTED = True
        try:
            profile = LLMProfile.from_env_or_file()
            bundle = build_llm_bundle(profile)
            _LIVE_SUMMARY_LLM = bundle.summary_llm
        except Exception:
            _LIVE_SUMMARY_LLM = None
        return _LIVE_SUMMARY_LLM


__all__ = ["summarize_live_state"]
