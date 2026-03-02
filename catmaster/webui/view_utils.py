from __future__ import annotations

import json
from datetime import datetime
from html import escape
from typing import Any, Callable, Dict, List, Optional

def truncate(value: Any, max_len: int = 140) -> str:
    text = "" if value is None else str(value)
    text = " ".join(text.split())
    if len(text) <= max_len:
        return text
    return text[: max(0, max_len - 3)] + "..."


def summarize_event(event: Dict[str, Any]) -> str:
    name = event.get("name", "")
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    if name == "RUN_INIT_DONE":
        run_id = event.get("run_id") or payload.get("run_id", "")
        model = payload.get("model_name", "")
        return f"run_id={run_id} model={model}".strip()
    if name == "TASK_START":
        goal = payload.get("goal", "")
        return f"{event.get('task_id','')}: {truncate(goal, 120)}".strip()
    if name == "TASK_END":
        outcome = payload.get("outcome", "")
        return f"{event.get('task_id','')}: outcome={outcome}".strip()
    if name == "TASK_DECISION":
        action = payload.get("action", "")
        method = payload.get("method", "")
        params = payload.get("params_compact", "")
        parts = [action]
        if method:
            parts.append(method)
        if params:
            parts.append(params)
        return " | ".join(parts)
    if name == "TOOL_CALL_START":
        tool = payload.get("tool", "")
        params = payload.get("params_compact", "")
        return f"{tool} {params}".strip()
    if name == "TOOL_CALL_END":
        tool = payload.get("tool", "")
        status = payload.get("status", "")
        return f"{tool} status={status}".strip()
    if name == "TOOL_CALL_INTERRUPTED":
        tool = payload.get("tool", "")
        return f"{tool} interrupted".strip()
    if name == "INTERRUPT_REQUESTED":
        return "interrupt requested"
    if name == "INTERRUPT_ACKED":
        phase = payload.get("phase", "")
        return f"interrupt acked phase={phase}".strip()
    if name == "RUN_PAUSED":
        phase = payload.get("phase", "")
        return f"paused phase={phase}".strip()
    if name == "RUN_WAITING_INPUT":
        interrupt_type = payload.get("interrupt_type", "")
        return f"waiting_input type={interrupt_type}".strip()
    if name == "RUN_INPUT_RECEIVED":
        interrupt_type = payload.get("interrupt_type", "")
        return f"input_received type={interrupt_type}".strip()
    if name == "TASK_SUMMARY":
        outcome = payload.get("outcome", "")
        summary = payload.get("summary_snippet", "")
        return f"{outcome} - {truncate(summary, 120)}".strip()
    if name == "FINAL_SUMMARY_DONE":
        return "Final report generated"
    if name == "RUN_END":
        status = payload.get("status", "")
        return f"status={status}".strip()
    return name


def format_event_line(event: Dict[str, Any]) -> str:
    ts = event.get("ts")
    if ts:
        try:
            stamp = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
        except Exception:
            stamp = "--:--:--"
    else:
        stamp = "--:--:--"
    summary = summarize_event(event)
    return f"{stamp} {event.get('name','')} {summary}".rstrip()


# ---------------------------------------------------------------------------
# Event category for styling
# ---------------------------------------------------------------------------

_EVENT_CATEGORY: Dict[str, str] = {
    "TOOL_CALL_START": "tool",
    "TOOL_CALL_END": "tool",
    "TOOL_CALL_INTERRUPTED": "tool",
    "TOOL_VALIDATE_FAILED": "tool",
    "TASK_START": "task",
    "TASK_END": "task",
    "TASK_SUMMARY": "task",
    "TASK_DECISION": "task",
    "TASK_JOURNAL_APPEND": "task",
    "RUN_INIT_DONE": "run",
    "RUN_END": "run",
    "RUN_PAUSED": "run",
    "RUN_WAITING_INPUT": "run",
    "RUN_INPUT_RECEIVED": "run",
    "INTERRUPT_REQUESTED": "interrupt",
    "INTERRUPT_ACKED": "interrupt",
    "FINAL_SUMMARY_DONE": "run",
    "FINAL_SUMMARY_START": "run",
    "PROPOSAL_START": "run",
    "PROPOSAL_REVIEW_WAIT_INPUT": "run",
    "PROPOSAL_REVIEW_SHOW": "run",
}

_CAT_COLORS = {
    "tool": "#0d9488",
    "task": "#6366f1",
    "run": "#64748b",
    "interrupt": "#d97706",
}


def format_event_html(event: Dict[str, Any]) -> str:
    ts = event.get("ts")
    if ts:
        try:
            stamp = datetime.fromtimestamp(float(ts)).strftime("%H:%M:%S")
        except Exception:
            stamp = "--:--:--"
    else:
        stamp = "--:--:--"
    name = str(event.get("name", ""))
    summary = escape(summarize_event(event))
    cat = _EVENT_CATEGORY.get(name, "run")
    cat_color = _CAT_COLORS.get(cat, "#64748b")
    return (
        f'<div style="display:flex;gap:8px;align-items:baseline;padding:2px 0;font-size:0.82rem;'
        f'border-bottom:1px solid var(--border-color-primary,#e2e8f0);">'
        f'<span style="color:var(--body-text-color-subdued,#64748b);font-family:monospace;min-width:60px;">{stamp}</span>'
        f'<code style="background:{cat_color}18;color:{cat_color};padding:1px 6px;border-radius:4px;'
        f'font-size:0.78rem;white-space:nowrap;">{escape(name)}</code>'
        f'<span style="color:var(--body-text-color,#1e293b);">{summary}</span>'
        f"</div>"
    )


# ---------------------------------------------------------------------------
# Run cards HTML
# ---------------------------------------------------------------------------

_CARD_STATUS_COLORS: Dict[str, str] = {
    "running": "#16a34a",
    "done": "#64748b",
    "error": "#dc2626",
    "failure": "#dc2626",
    "paused": "#d97706",
    "interrupted_paused": "#d97706",
    "awaiting_human_feedback": "#2563eb",
    "needs_intervention": "#ea580c",
}


def render_run_cards_html(
    cards: List[Dict[str, Any]],
    *,
    selected_run: str,
    search_text: str,
    run_link_builder: Optional[Callable[[str], str]] = None,
) -> str:
    needle = (search_text or "").strip().lower()
    filtered: List[Dict[str, Any]] = []
    for card in cards:
        run_name = str(card.get("run_name") or "")
        hay = " ".join(
            [
                run_name,
                str(card.get("headline") or ""),
                str(card.get("summary") or ""),
                str(card.get("status") or ""),
                str(card.get("model_name") or ""),
                str(card.get("project_space") or card.get("workspace") or ""),
            ]
        ).lower()
        if needle and needle not in hay:
            continue
        filtered.append(card)

    if not filtered:
        return (
            '<div style="color:var(--body-text-color-subdued);font-size:0.9rem;'
            'border:1px dashed var(--border-color-primary);border-radius:12px;padding:12px;">'
            "No runs matched current filter.</div>"
        )

    out: List[str] = ['<div style="display:flex;flex-direction:column;gap:10px;">']
    for card in filtered:
        run_name = str(card.get("run_name") or "")
        headline = escape(str(card.get("headline") or run_name))
        summary = escape(str(card.get("summary") or ""))
        status = str(card.get("status") or "unknown")
        status_escaped = escape(status)
        model_name = escape(str(card.get("model_name") or ""))
        source = escape(str(card.get("source") or "rule"))
        is_active = run_name and run_name == selected_run
        next_actions = card.get("next_actions") if isinstance(card.get("next_actions"), list) else []

        badge_color = _CARD_STATUS_COLORS.get(status, "#94a3b8")
        border_style = f"border-color:{badge_color};box-shadow:0 0 0 1px {badge_color} inset;" if is_active else ""

        link_open = ""
        link_close = ""
        if run_link_builder is not None and run_name:
            href = escape(run_link_builder(run_name), quote=True)
            link_open = f'<a style="text-decoration:none;color:inherit;display:block;" href="{href}">'
            link_close = "</a>"

        out.append(link_open)
        out.append(
            f'<article style="border:1px solid var(--border-color-primary,#d6deea);border-radius:12px;'
            f"padding:10px;background:var(--background-fill-primary,#fefefe);"
            f'transition:transform .12s ease,box-shadow .12s ease;{border_style}">'
        )
        out.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'font-size:0.84rem;color:var(--body-text-color-subdued,#5b6b7d);">'
            f"<code>{escape(run_name)}</code>"
            f'<span style="border:1px solid {badge_color};color:{badge_color};'
            f'border-radius:999px;padding:1px 8px;font-size:0.75rem;">{status_escaped}</span>'
            f"</div>"
        )
        out.append(f'<h4 style="margin:0.35rem 0;font-size:0.95rem;">{headline}</h4>')
        out.append(f'<p style="margin:0;font-size:0.85rem;color:var(--body-text-color-subdued,#5b6b7d);">{summary}</p>')
        if next_actions:
            out.append('<ul style="margin:0.55rem 0;padding-left:1rem;">')
            for action in next_actions[:3]:
                out.append(f'<li style="margin:0.15rem 0;font-size:0.83rem;">{escape(str(action))}</li>')
            out.append("</ul>")
        meta_bits = [bit for bit in [model_name, f"source:{source}"] if bit]
        out.append(
            f'<div style="font-size:0.76rem;color:var(--body-text-color-subdued,#5b6b7d);">'
            f'{"  |  ".join(meta_bits)}</div>'
        )
        out.append("</article>")
        out.append(link_close)
    out.append("</div>")
    return "".join(out)


def render_live_tracker_markdown(state: Dict[str, Any]) -> str:
    if not isinstance(state, dict) or not state:
        return "### Live Tracker\nNo active run state."

    summary = state.get("live_summary") if isinstance(state.get("live_summary"), dict) else {}
    headline = str(summary.get("live_headline") or "").strip()
    live_summary = str(summary.get("live_summary") or "").strip()
    next_expected = str(summary.get("next_expected_step") or "").strip()
    source = str(summary.get("source") or "rule")

    progress = state.get("progress") if isinstance(state.get("progress"), dict) else {}
    completed = int(progress.get("completed", 0))
    pending = int(progress.get("pending", 0))
    failed = int(progress.get("failed", 0))
    needs_intervention = int(progress.get("needs_intervention", 0))
    total = int(progress.get("total", 0))

    task_id = str(state.get("current_task_id") or "")
    task_goal = str(state.get("current_task_goal") or "")
    phase = str(state.get("current_phase") or "")
    status = str(state.get("status") or "unknown")

    lines: List[str] = ["### Live Tracker"]
    if headline:
        lines.append(f"**{headline}**")
    lines.append(f"Status: `{status}` | Phase: `{phase or 'n/a'}`")
    lines.append(
        "Progress: "
        f"`{completed}` completed / `{pending}` pending / `{failed}` failed / "
        f"`{needs_intervention}` needs_intervention / total `{total}`"
    )
    if live_summary:
        lines.append("")
        lines.append(live_summary)
    if next_expected:
        lines.append(f"Next: {next_expected}")
    lines.append(f"Summary source: `{source}`")

    if task_id or task_goal:
        lines.append("")
        lines.append("#### Current Task")
        if task_id:
            lines.append(f"- Task ID: `{task_id}`")
        if task_goal:
            lines.append(f"- Goal: {task_goal}")

    active = state.get("active_toolcall")
    lines.append("")
    lines.append("#### Active Tool Call")
    if isinstance(active, dict) and active.get("tool"):
        elapsed = active.get("elapsed_sec")
        elapsed_text = f"{int(elapsed)}s" if isinstance(elapsed, (int, float)) else "n/a"
        lines.append(f"- Tool: `{active.get('tool')}`")
        lines.append(f"- Status: `{active.get('status') or 'running'}` | Elapsed: `{elapsed_text}`")
        if active.get("toolcall_id"):
            lines.append(f"- Toolcall ID: `{active.get('toolcall_id')}`")
        params_full = active.get("params_full")
        if params_full is not None:
            lines.append(_render_json_block(params_full, max_chars=3000))
    else:
        lines.append("- (none)")

    recent = state.get("recent_toolcalls")
    lines.append("")
    lines.append("#### Recent Tool Calls")
    if isinstance(recent, list) and recent:
        for item in recent[-5:]:
            if not isinstance(item, dict):
                continue
            tool = str(item.get("tool") or "")
            status_i = str(item.get("status") or "")
            task_i = str(item.get("task_id") or "")
            duration = item.get("duration_sec")
            duration_text = f"{int(duration)}s" if isinstance(duration, (int, float)) else "n/a"
            details = f"{task_i} | `{tool}` | `{status_i}` | {duration_text}"
            lines.append(f"- {details}")
    else:
        lines.append("- (none)")

    journal = state.get("journal_recent")
    lines.append("")
    lines.append("#### Task Journal")
    if isinstance(journal, list) and journal:
        for item in journal[-5:]:
            if not isinstance(item, dict):
                continue
            task = str(item.get("task_id") or "")
            outcome = str(item.get("outcome") or "")
            summary_snippet = truncate(item.get("summary_snippet"), 180)
            lines.append(f"- `{task}` `{outcome}` {summary_snippet}")
    else:
        lines.append("- (none)")

    return "\n".join(lines).strip()


def _render_json_block(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return "```json\n" + text + "\n```"


__all__ = [
    "format_event_html",
    "format_event_line",
    "render_live_tracker_markdown",
    "render_run_cards_html",
    "summarize_event",
    "truncate",
]
