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
        return '<div class="cm-empty">No runs matched current filter.</div>'

    out: List[str] = ['<div class="cm-card-list">']
    for card in filtered:
        run_name = str(card.get("run_name") or "")
        headline = escape(str(card.get("headline") or run_name))
        summary = escape(str(card.get("summary") or ""))
        status = escape(str(card.get("status") or "unknown"))
        model_name = escape(str(card.get("model_name") or ""))
        source = escape(str(card.get("source") or "rule"))
        is_active = " cm-active" if run_name and run_name == selected_run else ""
        next_actions = card.get("next_actions") if isinstance(card.get("next_actions"), list) else []

        link_open = ""
        link_close = ""
        if run_link_builder is not None and run_name:
            href = escape(run_link_builder(run_name), quote=True)
            link_open = f'<a class="cm-card-link" href="{href}">'
            link_close = "</a>"

        out.append(link_open)
        out.append(f'<article class="cm-card{is_active}">')
        out.append(f'<div class="cm-card-head"><code>{escape(run_name)}</code><span class="cm-badge">{status}</span></div>')
        out.append(f"<h4>{headline}</h4>")
        out.append(f"<p>{summary}</p>")
        if next_actions:
            out.append('<ul class="cm-actions">')
            for action in next_actions[:3]:
                out.append(f"<li>{escape(str(action))}</li>")
            out.append("</ul>")
        meta_bits = [bit for bit in [model_name, f"source:{source}"] if bit]
        out.append(f'<div class="cm-meta">{" | ".join(meta_bits)}</div>')
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
    "format_event_line",
    "render_live_tracker_markdown",
    "render_run_cards_html",
    "summarize_event",
    "truncate",
]
