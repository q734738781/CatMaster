from __future__ import annotations

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
    if name == "PLAN_CREATED":
        n_items = payload.get("n_items")
        return f"{n_items} tasks" if n_items is not None else "Plan created"
    if name == "PLAN_REVIEW_REVISED":
        n_items = payload.get("n_items")
        return f"Plan revised ({n_items} tasks)" if n_items is not None else "Plan revised"
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
                str(card.get("workspace") or ""),
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


__all__ = [
    "format_event_line",
    "render_run_cards_html",
    "summarize_event",
    "truncate",
]
