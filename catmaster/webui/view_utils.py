from __future__ import annotations

import json
from datetime import datetime
from html import escape
from typing import Any, Callable, Dict, List, Optional

from .components import status_color


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
    "tool": "#0ea5e9",
    "task": "#8b5cf6",
    "run": "#6b7280",
    "interrupt": "#f59e0b",
}

_CAT_ICONS = {
    "tool": "\u2699",
    "task": "\u25B6",
    "run": "\u25CF",
    "interrupt": "\u26A0",
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
    cat_color = _CAT_COLORS.get(cat, "#475569")
    icon = _CAT_ICONS.get(cat, "\u25CF")
    return (
        f'<div style="display:flex;gap:8px;align-items:stretch;padding:0;font-size:0.82rem;'
        f'position:relative;">'
        # timeline connector
        f'<div style="width:2px;background:{cat_color}30;flex-shrink:0;'
        f'margin:0 4px;border-radius:1px;"></div>'
        # content row
        f'<div style="display:flex;gap:8px;align-items:baseline;padding:4px 4px;flex:1;'
        f'border-radius:6px;transition:background .12s;cursor:default;" '
        f'onmouseenter="this.style.background=\'{cat_color}08\'" '
        f'onmouseleave="this.style.background=\'transparent\'">'
        f'<span style="color:#9ca3af;font-family:monospace;min-width:56px;font-size:0.78rem;">'
        f'{stamp}</span>'
        f'<span style="color:{cat_color};font-size:0.82rem;width:14px;text-align:center;'
        f'flex-shrink:0;">{icon}</span>'
        f'<code style="background:{cat_color}12;color:{cat_color};padding:1px 7px;'
        f'border-radius:5px;font-size:0.76rem;white-space:nowrap;font-weight:600;">'
        f'{escape(name)}</code>'
        f'<span style="color:var(--body-text-color,#1e293b);font-size:0.8rem;">{summary}</span>'
        f"</div></div>"
    )


# ---------------------------------------------------------------------------
# Run cards HTML (with left accent bar, hover, status icons)
# ---------------------------------------------------------------------------

_CARD_STATUS_ICONS: Dict[str, str] = {
    "running": "\u25B6",
    "starting": "\u25B6",
    "done": "\u2714",
    "error": "\u2716",
    "failure": "\u2716",
    "paused": "\u23F8",
    "interrupted_paused": "\u23F8",
    "awaiting_human_feedback": "\u270B",
    "needs_intervention": "\u26A0",
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
            '<div style="color:#9ca3af;font-size:0.9rem;'
            'border:1px dashed #d1d5db;border-radius:14px;padding:20px;text-align:center;">'
            "No runs matched current filter.</div>"
        )

    out: List[str] = ['<div style="display:flex;flex-direction:column;gap:10px;">']
    for card in filtered:
        run_name = str(card.get("run_name") or "")
        headline = escape(str(card.get("headline") or run_name))
        summary = escape(str(card.get("summary") or ""))
        status_str = str(card.get("status") or "unknown")
        status_escaped = escape(status_str)
        model_name = escape(str(card.get("model_name") or ""))
        source = escape(str(card.get("source") or "rule"))
        is_active = run_name and run_name == selected_run
        next_actions = card.get("next_actions") if isinstance(card.get("next_actions"), list) else []

        badge_color = status_color(status_str)
        icon = _CARD_STATUS_ICONS.get(status_str, "\u25CF")
        active_ring = (
            f"box-shadow:0 0 0 2px {badge_color}40, 0 4px 20px rgba(0,0,0,.06);"
            if is_active
            else "box-shadow:0 2px 12px rgba(0,0,0,.04);"
        )

        link_open = ""
        link_close = ""
        if run_link_builder is not None and run_name:
            href = escape(run_link_builder(run_name), quote=True)
            link_open = f'<a style="text-decoration:none;color:inherit;display:block;" href="{href}">'
            link_close = "</a>"

        out.append(link_open)
        out.append(
            f'<article style="display:flex;border-radius:14px;overflow:hidden;'
            f"background:rgba(255,255,255,.92);backdrop-filter:blur(8px);"
            f"border:1px solid #e5e7eb;"
            f'transition:transform .15s ease,box-shadow .15s ease;{active_ring}" '
            f'onmouseenter="this.style.transform=\'translateY(-2px)\';'
            f"this.style.boxShadow='0 8px 24px rgba(0,0,0,.08)'\" "
            f'onmouseleave="this.style.transform=\'none\';'
            f"this.style.boxShadow='{active_ring.split(';')[0]}';\">"
        )
        # left accent bar
        out.append(
            f'<div style="width:4px;flex-shrink:0;background:{badge_color};'
            f'border-radius:4px 0 0 4px;"></div>'
        )
        # card body
        out.append('<div style="flex:1;padding:12px 14px;">')
        # header row
        out.append(
            f'<div style="display:flex;justify-content:space-between;align-items:center;'
            f'margin-bottom:6px;">'
            f'<code style="font-size:0.8rem;color:#6b7280;background:#f3f4f6;'
            f'padding:1px 8px;border-radius:6px;">{escape(run_name)}</code>'
            f'<span style="display:inline-flex;align-items:center;gap:4px;'
            f"border:1px solid {badge_color}40;color:{badge_color};"
            f'border-radius:999px;padding:2px 10px;font-size:0.75rem;font-weight:600;">'
            f'{icon} {status_escaped}</span>'
            f"</div>"
        )
        out.append(
            f'<div style="font-size:0.93rem;font-weight:600;color:#111827;'
            f'margin-bottom:4px;">{headline}</div>'
        )
        if summary:
            out.append(
                f'<div style="font-size:0.83rem;color:#6b7280;margin-bottom:6px;'
                f'line-height:1.4;">{summary}</div>'
            )
        if next_actions:
            out.append(
                '<div style="margin:6px 0;padding-left:14px;">'
            )
            for action in next_actions[:3]:
                out.append(
                    f'<div style="font-size:0.8rem;color:#4b5563;padding:1px 0;">'
                    f'\u2022 {escape(str(action))}</div>'
                )
            out.append("</div>")
        meta_bits = [bit for bit in [model_name, f"source:{source}"] if bit]
        out.append(
            f'<div style="font-size:0.73rem;color:#9ca3af;margin-top:4px;">'
            f'{"  \u00B7  ".join(meta_bits)}</div>'
        )
        out.append("</div>")  # card body
        out.append("</article>")
        out.append(link_close)
    out.append("</div>")
    return "".join(out)


# ---------------------------------------------------------------------------
# Progress bar HTML
# ---------------------------------------------------------------------------

def render_progress_bar_html(
    completed: int,
    pending: int,
    failed: int,
    needs_intervention: int,
    total: int,
) -> str:
    if total <= 0:
        return (
            '<div style="height:8px;border-radius:4px;background:#e5e7eb;'
            'margin:6px 0;"></div>'
        )
    segments: List[str] = []

    def _seg(count: int, color: str) -> None:
        if count > 0:
            pct = max(1, round(count / total * 100))
            segments.append(
                f'<div style="width:{pct}%;height:100%;background:{color};'
                f'transition:width .3s ease;" '
                f'title="{count}/{total}"></div>'
            )

    _seg(completed, "#10b981")
    _seg(failed, "#ef4444")
    _seg(needs_intervention, "#f97316")
    _seg(pending, "#d1d5db")

    bar = "".join(segments)
    labels = (
        f'<div style="display:flex;gap:12px;font-size:0.75rem;color:#6b7280;margin-top:3px;flex-wrap:wrap;">'
        f'<span>\u25CF <span style="color:#10b981">{completed}</span> done</span>'
        f'<span>\u25CF <span style="color:#d1d5db">{pending}</span> pending</span>'
        f'<span>\u25CF <span style="color:#ef4444">{failed}</span> failed</span>'
        f'<span>\u25CF <span style="color:#f97316">{needs_intervention}</span> needs attn</span>'
        f"<span>/ {total} total</span>"
        f"</div>"
    )
    return (
        f'<div style="margin:8px 0;">'
        f'<div style="display:flex;height:8px;border-radius:4px;overflow:hidden;'
        f'background:#e5e7eb;gap:1px;">{bar}</div>'
        f"{labels}</div>"
    )


# ---------------------------------------------------------------------------
# Live tracker (markdown + inline HTML progress)
# ---------------------------------------------------------------------------

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
    status_str = str(state.get("status") or "unknown")

    lines: List[str] = ["### Live Tracker"]
    if headline:
        lines.append(f"**{headline}**")
    lines.append(f"Status: `{status_str}` | Phase: `{phase or 'n/a'}`")

    progress_html = render_progress_bar_html(
        completed, pending, failed, needs_intervention, total,
    )
    lines.append(progress_html)

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


def render_cost_card_markdown(summary: Dict[str, Any]) -> str:
    if not isinstance(summary, dict) or not summary:
        return "### Cost\nNot available yet."

    total_cost = _to_float(summary.get("cost_usd"))
    if total_cost is None:
        return "### Cost\nNot available yet."

    exact_cost = _to_float(summary.get("exact_cost_usd")) or 0.0
    estimated_cost = _to_float(summary.get("estimated_cost_usd")) or 0.0
    cost_source = str(summary.get("cost_source") or "unavailable")
    calls = int(summary.get("calls") or 0)
    prompt_tokens = int(summary.get("input_tokens") or 0)
    cache_read_tokens = int(summary.get("input_cached_tokens") or 0)
    cache_write_tokens = int(summary.get("input_cache_write_tokens") or 0)
    completion_tokens = int(summary.get("output_tokens") or 0)
    reasoning_tokens = int(summary.get("reasoning_tokens") or 0)
    missing_cost_calls = int(summary.get("missing_cost_calls") or 0)
    breakdown = summary.get("breakdown_usd") if isinstance(summary.get("breakdown_usd"), dict) else {}
    by_role = summary.get("by_role") if isinstance(summary.get("by_role"), list) else []

    lines: List[str] = ["### Cost", f"**Total**: `${total_cost:.4f}`"]
    lines.append(f"Source: `{cost_source}` | Calls: `{calls}`")
    if cost_source != "exact" and exact_cost > 0:
        lines.append(f"Exact observed: `${exact_cost:.4f}`")
    if cost_source != "exact" and estimated_cost > 0:
        lines.append(f"Estimated supplement: `${estimated_cost:.4f}`")
    if missing_cost_calls:
        lines.append(f"Calls without cost basis: `{missing_cost_calls}`")

    lines.extend(
        [
            "",
            "#### Tokens",
            f"- prompt total: `{prompt_tokens}`",
            f"- cache read: `{cache_read_tokens}`",
            f"- cache write: `{cache_write_tokens}`",
            f"- completion total: `{completion_tokens}`",
            f"- reasoning: `{reasoning_tokens}`",
        ]
    )

    breakdown_lines = []
    for key in ("prompt_uncached", "cache_read", "cache_write", "completion", "internal_reasoning"):
        value = _to_float(breakdown.get(key))
        if value is None or value <= 0:
            continue
        breakdown_lines.append(f"- {key.replace('_', ' ')}: `${value:.4f}`")
    if breakdown_lines:
        lines.extend(["", "#### Estimated Breakdown", *breakdown_lines])

    if by_role:
        top_roles = sorted(by_role, key=lambda item: float(item.get("cost_usd") or 0.0), reverse=True)[:3]
        role_lines = []
        for item in top_roles:
            name = str(item.get("name") or "(unknown)")
            cost = _to_float(item.get("cost_usd")) or 0.0
            role_lines.append(f"- `{name}`: `${cost:.4f}`")
        if role_lines:
            lines.extend(["", "#### Top Roles", *role_lines])

    return "\n".join(lines).strip()


def _render_json_block(value: Any, *, max_chars: int) -> str:
    try:
        text = json.dumps(value, ensure_ascii=False, indent=2)
    except Exception:
        text = str(value)
    if len(text) > max_chars:
        text = text[: max_chars - 3] + "..."
    return "```json\n" + text + "\n```"


def _to_float(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(str(value).strip())
    except Exception:
        return None


__all__ = [
    "render_cost_card_markdown",
    "format_event_html",
    "format_event_line",
    "render_live_tracker_markdown",
    "render_progress_bar_html",
    "render_run_cards_html",
    "summarize_event",
    "truncate",
]
