from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from catmaster.runtime.observability_store import ObservabilityStore

TERMINAL_STATUSES = {
    "done",
    "error",
    "interrupted",
    "interrupted_paused",
    "stopped",
    "blocked",
}

# Provider request envelopes repeat the complete preceding conversation, system
# prompts, and tool schemas on every model call. Token deltas repeat the final
# response one fragment at a time. The events below are the complete semantic
# trajectory: every model result, tool input, tool result, task boundary, and
# terminal result, without those transport-level duplicates.
_TRAJECTORY_EVENT_NAMES = (
    "LLM_RAW_RESPONSE",
    "LLM_CALL_END",
    "LLM_ERROR",
    "TOOL_RAW_INPUT",
    "TOOL_RAW_OUTPUT",
    "TOOL_CALL_END",
    "TASKS_COMPILED",
    "TASK_START",
    "TASK_DECISION",
    "TASK_SUMMARY",
    "TASK_END",
    "RUN_START",
    "RUN_PAUSED",
    "RUN_END",
)


def _read_json(path: Path) -> dict[str, Any]:
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return value if isinstance(value, dict) else {}


def _callback_id(event: dict[str, Any]) -> str:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    return str(
        event.get("callback_run_id")
        or payload.get("callback_run_id")
        or ""
    ).strip()


def _event_ref(run_id: str, event: dict[str, Any]) -> str:
    event_id = int(event.get("id") or 0)
    return f"run:{run_id}#event:{event_id}" if event_id > 0 else f"run:{run_id}"


def _verified_task_outcome(value: Any) -> str:
    text = str(value or "").strip().lower().replace("-", "_").replace(" ", "_")
    if text in {"success", "succeeded", "passed", "pass", "verified_success"}:
        return "verified_success"
    if text in {"failure", "failed", "error", "blocked", "verified_failure"}:
        return "verified_failure"
    return ""


def _read_all_trajectory_events(store: ObservabilityStore) -> list[dict[str, Any]]:
    """Read every selected event page, oldest first."""

    chunks: list[list[dict[str, Any]]] = []
    before_id = 0
    while True:
        page = store.read_events_page(
            limit=5_000,
            before_id=before_id,
            names=_TRAJECTORY_EVENT_NAMES,
            include_legacy_trace_records=True,
        )
        events = [
            item
            for item in list(page.get("events") or [])
            if isinstance(item, dict)
        ]
        if not events:
            break
        chunks.append(events)
        if not bool(page.get("has_more")):
            break
        next_before = int(page.get("min_id") or 0)
        if next_before <= 0 or next_before == before_id:
            break
        before_id = next_before

    ordered: list[dict[str, Any]] = []
    for chunk in reversed(chunks):
        ordered.extend(chunk)
    ordered.sort(key=lambda item: int(item.get("id") or 0))
    return ordered


def _task_outcome_from_events(
    *,
    raw_events: list[dict[str, Any]],
) -> tuple[str, str]:
    """Return only a formal task verdict with an explicit verifier reference."""

    for event in reversed(raw_events):
        name = str(event.get("name") or event.get("event") or "").strip()
        if name not in {"TASK_END", "TASK_SUMMARY"}:
            continue
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if payload.get("verified") is False:
            continue
        raw_outcome: Any = (
            payload.get("task_outcome")
            if "task_outcome" in payload
            else payload.get("outcome")
            if "outcome" in payload
            else payload.get("verdict")
        )
        if raw_outcome is None and isinstance(payload.get("passed"), bool):
            raw_outcome = "success" if payload["passed"] else "failure"
        outcome = _verified_task_outcome(raw_outcome)
        outcome_ref = str(payload.get("outcome_ref") or "").strip()
        if outcome and outcome_ref:
            return outcome, outcome_ref
    return "", ""


def _model_event(
    *,
    run_id: str,
    event: dict[str, Any],
    raw: bool,
) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    generations = (
        list(payload.get("generations") or [])
        if raw
        else [
            {
                "reasoning_text": payload.get("reasoning_text") or "",
                "response_text": payload.get("text_preview") or "",
                "parsed_tool_calls": list(payload.get("tool_calls") or []),
            }
        ]
    )
    return {
        "source_ref": _event_ref(run_id, event),
        "kind": "model",
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "node": str(event.get("node") or payload.get("node") or ""),
        "generations": generations,
    }


def _tool_input_event(*, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    value: Any = (
        payload.get("params_full")
        if "params_full" in payload
        else payload.get("params_compact")
    )
    return {
        "source_ref": _event_ref(run_id, event),
        "kind": "tool_input",
        "tool": str(event.get("tool") or payload.get("tool") or payload.get("tool_name") or "tool"),
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "input": value,
    }


def _tool_output_event(*, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    projection = (
        payload.get("projection")
        if isinstance(payload.get("projection"), dict)
        else {}
    )
    raw_output = payload.get("raw_output")
    content_text = projection.get("content_text")
    if raw_output is not None and raw_output != "":
        result: Any = raw_output
    elif content_text is not None and content_text != "":
        result: Any = content_text
    else:
        result = projection.get("content_preview") or payload.get("result") or ""
    return {
        "source_ref": _event_ref(run_id, event),
        "kind": "tool_output",
        "tool": str(event.get("tool") or payload.get("tool") or payload.get("tool_name") or "tool"),
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "status": str(event.get("status") or payload.get("status") or payload.get("tool_status") or "unknown"),
        "result": result,
        "error": str(projection.get("error") or payload.get("error") or ""),
        "warnings": list(projection.get("warnings") or []),
        "highlights": list(projection.get("highlights") or []),
        "artifact_refs": list(projection.get("offload_refs") or []),
    }


def _normalize_trajectory(
    *,
    run_id: str,
    raw_events: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    raw_llm_callbacks = {
        _callback_id(event)
        for event in raw_events
        if str(event.get("name") or "") == "LLM_RAW_RESPONSE"
        and _callback_id(event)
    }
    raw_tool_callbacks = {
        _callback_id(event)
        for event in raw_events
        if str(event.get("name") or "") == "TOOL_RAW_OUTPUT"
        and _callback_id(event)
    }
    seen_errors: set[str] = set()
    trajectory: list[dict[str, Any]] = []
    for event in raw_events:
        name = str(event.get("name") or event.get("event") or "").strip()
        callback_id = _callback_id(event)
        payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
        if name == "LLM_RAW_RESPONSE":
            trajectory.append(_model_event(run_id=run_id, event=event, raw=True))
            continue
        if name == "LLM_CALL_END":
            if callback_id and callback_id in raw_llm_callbacks:
                continue
            trajectory.append(_model_event(run_id=run_id, event=event, raw=False))
            continue
        if name == "LLM_ERROR":
            error_key = callback_id or str(payload.get("error") or "")
            if error_key in seen_errors:
                continue
            seen_errors.add(error_key)
            trajectory.append(
                {
                    "source_ref": _event_ref(run_id, event),
                    "kind": "model_error",
                    "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
                    "node": str(event.get("node") or payload.get("node") or ""),
                    "error": str(payload.get("error") or "Model call failed."),
                }
            )
            continue
        if name == "TOOL_RAW_INPUT":
            trajectory.append(_tool_input_event(run_id=run_id, event=event))
            continue
        if name == "TOOL_RAW_OUTPUT":
            trajectory.append(_tool_output_event(run_id=run_id, event=event))
            continue
        if name == "TOOL_CALL_END":
            if callback_id and callback_id in raw_tool_callbacks:
                continue
            trajectory.append(_tool_output_event(run_id=run_id, event=event))
            continue
        if name in {
            "TASKS_COMPILED",
            "TASK_START",
            "TASK_DECISION",
            "TASK_SUMMARY",
            "TASK_END",
            "RUN_START",
            "RUN_PAUSED",
            "RUN_END",
        }:
            trajectory.append(
                {
                    "source_ref": _event_ref(run_id, event),
                    "kind": "boundary",
                    "event": name,
                    "payload": payload,
                }
            )
    return trajectory


def _json_text(value: Any) -> str:
    if isinstance(value, str):
        return value
    return json.dumps(value, ensure_ascii=False, indent=2, default=str)


def _event_markdown(event: dict[str, Any], index: int) -> list[str]:
    source_ref = str(event.get("source_ref") or "")
    kind = str(event.get("kind") or "event")
    lines = [f"### {index}. {kind.replace('_', ' ').title()}", "", f"Source: `{source_ref}`", ""]
    if kind == "model":
        if event.get("agent"):
            lines.append(f"Agent: `{event['agent']}`")
        if event.get("node"):
            lines.append(f"Node: `{event['node']}`")
        if event.get("agent") or event.get("node"):
            lines.append("")
        for generation_index, generation in enumerate(
            list(event.get("generations") or []),
            start=1,
        ):
            if not isinstance(generation, dict):
                lines.extend([_json_text(generation), ""])
                continue
            if len(list(event.get("generations") or [])) > 1:
                lines.extend([f"Generation {generation_index}", ""])
            reasoning = str(generation.get("reasoning_text") or "").strip()
            response = str(generation.get("response_text") or "").strip()
            if reasoning:
                lines.extend(["Reasoning:", "", reasoning, ""])
            if response:
                lines.extend(["Response:", "", response, ""])
            raw_content = generation.get("response_content_raw")
            if raw_content not in (None, "", response):
                lines.extend(
                    [
                        "Raw response content:",
                        "",
                        "```json",
                        _json_text(raw_content),
                        "```",
                        "",
                    ]
                )
            parsed_tool_calls = generation.get("parsed_tool_calls") or []
            if parsed_tool_calls:
                lines.extend(
                    [
                        "Parsed tool calls:",
                        "",
                        "```json",
                        _json_text(parsed_tool_calls),
                        "```",
                        "",
                    ]
                )
            raw_tool_calls = generation.get("raw_tool_calls") or []
            if raw_tool_calls:
                lines.extend(
                    [
                        "Raw tool calls:",
                        "",
                        "```json",
                        _json_text(raw_tool_calls),
                        "```",
                        "",
                    ]
                )
            invalid_tool_calls = generation.get("invalid_tool_calls") or []
            if invalid_tool_calls:
                lines.extend(
                    [
                        "Invalid tool calls:",
                        "",
                        "```json",
                        _json_text(invalid_tool_calls),
                        "```",
                        "",
                    ]
                )
    elif kind == "tool_input":
        lines.extend(
            [
                f"Tool: `{event.get('tool') or 'tool'}`",
                f"Agent: `{event.get('agent') or 'not recorded'}`",
                "",
                "Input:",
                "",
                "```json",
                _json_text(event.get("input")),
                "```",
                "",
            ]
        )
    elif kind == "tool_output":
        lines.extend(
            [
                f"Tool: `{event.get('tool') or 'tool'}`",
                f"Agent: `{event.get('agent') or 'not recorded'}`",
                f"Status: `{event.get('status') or 'unknown'}`",
                "",
                "Result:",
                "",
                _json_text(event.get("result")),
                "",
            ]
        )
        if event.get("error"):
            lines.extend(["Error:", "", str(event["error"]), ""])
        if event.get("warnings"):
            lines.extend(["Warnings:", "", _json_text(event["warnings"]), ""])
        if event.get("highlights"):
            lines.extend(["Highlights:", "", _json_text(event["highlights"]), ""])
        if event.get("artifact_refs"):
            lines.extend(["Artifact references:", "", _json_text(event["artifact_refs"]), ""])
    elif kind == "model_error":
        lines.extend([str(event.get("error") or "Model call failed."), ""])
    else:
        lines.extend(
            [
                f"Event: `{event.get('event') or kind}`",
                "",
                "```json",
                _json_text(event.get("payload") or {}),
                "```",
                "",
            ]
        )
    return lines


@dataclass
class TurnTrace:
    run_id: str
    thread_id: str
    entrypoint: str
    status: str
    user_prompt: str
    final_answer: str
    summary: str
    explicit_correction: str = ""
    resume_guidance: str = ""
    source_message_id: str = ""
    prior_assistant_message_id: str = ""
    assistant_message_id: str = ""
    events: list[dict[str, Any]] = field(default_factory=list)
    task_outcome: str = ""
    outcome_ref: str = ""
    artifact_refs: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        return {
            "run_id": self.run_id,
            "thread_id": self.thread_id,
            "entrypoint": self.entrypoint,
            "status": self.status,
            "user_prompt": self.user_prompt,
            "final_answer": self.final_answer,
            "summary": self.summary,
            "explicit_correction": self.explicit_correction,
            "resume_guidance": self.resume_guidance,
            "source_message_id": self.source_message_id,
            "prior_assistant_message_id": self.prior_assistant_message_id,
            "assistant_message_id": self.assistant_message_id,
            "events": self.events,
            "task_outcome": self.task_outcome,
            "outcome_ref": self.outcome_ref,
            "artifact_refs": list(self.artifact_refs),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TurnTrace":
        fields = cls.__dataclass_fields__
        data = {key: value.get(key) for key in fields if key in value}
        if not isinstance(data.get("events"), list):
            data["events"] = []
        if not isinstance(data.get("artifact_refs"), list):
            data["artifact_refs"] = []
        return cls(**data)

    def has_user_content(self) -> bool:
        return bool(
            self.user_prompt.strip()
            or self.resume_guidance.strip()
            or self.explicit_correction.strip()
        )

    def to_markdown(self) -> str:
        """Return the complete semantic agent trajectory and terminal result."""

        lines = [
            "# Complete episode trajectory",
            "",
            (
                "Host-generated trajectory evidence. It contains every recorded "
                "model result, tool input, tool result, and task/run boundary. "
                "Provider request envelopes and streaming deltas are excluded "
                "because they only repeat this same trajectory."
            ),
            "",
            "## Task",
            "",
            f"- Run: `{self.run_id}`",
            f"- Thread: `{self.thread_id or 'not recorded'}`",
            f"- Entrypoint: `{self.entrypoint or 'not recorded'}`",
            "",
            self.user_prompt or "No initial user task was recovered.",
            "",
        ]
        if self.resume_guidance:
            lines.extend(
                [
                    "## Resume guidance",
                    "",
                    self.resume_guidance,
                    "",
                ]
            )
        if self.explicit_correction:
            lines.extend(
                [
                    "## Explicit durable correction",
                    "",
                    self.explicit_correction,
                    "",
                ]
            )
        lines.extend(
            [
                "## Result",
                "",
                f"- Execution status: `{self.status}`",
                f"- Verified task outcome: `{self.task_outcome or 'not recorded'}`",
                f"- Outcome reference: `{self.outcome_ref or 'not recorded'}`",
                "",
            ]
        )
        if self.final_answer:
            lines.extend(["Final answer:", "", self.final_answer, ""])
        if self.summary and self.summary != self.final_answer:
            lines.extend(["Run summary:", "", self.summary, ""])
        if not self.final_answer and not self.summary:
            lines.extend(["No final result text was recovered.", ""])
        if self.artifact_refs:
            lines.extend(
                [
                    "Artifact references:",
                    "",
                    *[f"- `{item}`" for item in self.artifact_refs],
                    "",
                ]
            )
        lines.extend(["## Agent trajectory", ""])
        if not self.events:
            lines.extend(["No model or tool events were recorded.", ""])
        for index, event in enumerate(self.events, start=1):
            lines.extend(_event_markdown(event, index))
        return "\n".join(lines).strip() + "\n"


def collect_turn_trace(
    *,
    run_dir: Path | str,
    fallback: dict[str, Any] | None = None,
) -> TurnTrace:
    run_path = Path(run_dir).expanduser().resolve()
    fallback = dict(fallback or {})
    state = _read_json(run_path / "run_state.json")
    meta = _read_json(run_path / "meta.json")
    run_id = str(
        state.get("run_id")
        or meta.get("run_id")
        or fallback.get("run_id")
        or run_path.name
    ).strip()
    raw_events = _read_all_trajectory_events(ObservabilityStore(run_path))
    task_outcome, outcome_ref = _task_outcome_from_events(raw_events=raw_events)
    fallback_outcome = _verified_task_outcome(fallback.get("task_outcome"))
    fallback_ref = str(fallback.get("outcome_ref") or "").strip()
    if fallback_outcome and fallback_ref:
        task_outcome = fallback_outcome
        outcome_ref = fallback_ref

    artifact_refs: list[str] = []
    for item in list(
        state.get("artifact_ids")
        or fallback.get("artifact_refs")
        or []
    ):
        text = str(item or "").strip()
        if text and text not in artifact_refs:
            artifact_refs.append(text)
    for item in list(state.get("artifacts") or []):
        if not isinstance(item, dict):
            continue
        text = str(item.get("artifact_id") or item.get("path") or "").strip()
        if text and text not in artifact_refs:
            artifact_refs.append(text)

    return TurnTrace(
        run_id=run_id,
        thread_id=str(
            state.get("webui_thread_id")
            or fallback.get("thread_id")
            or state.get("thread_id")
            or ""
        ).strip(),
        entrypoint=str(
            state.get("entrypoint")
            or fallback.get("entrypoint")
            or ""
        ).strip(),
        status=str(
            state.get("status")
            or fallback.get("terminal_status")
            or "unknown"
        ).strip(),
        user_prompt=str(
            state.get("user_prompt")
            or fallback.get("prompt")
            or ""
        ).strip(),
        final_answer=str(state.get("final_answer") or "").strip(),
        summary=str(state.get("summary") or "").strip(),
        explicit_correction=str(fallback.get("note") or "").strip(),
        resume_guidance=str(state.get("resume_guidance") or "").strip(),
        source_message_id=str(fallback.get("message_id") or "").strip(),
        prior_assistant_message_id=str(
            fallback.get("prior_assistant_message_id") or ""
        ).strip(),
        assistant_message_id=str(
            fallback.get("assistant_message_id") or ""
        ).strip(),
        events=_normalize_trajectory(run_id=run_id, raw_events=raw_events),
        task_outcome=task_outcome,
        outcome_ref=outcome_ref,
        artifact_refs=artifact_refs,
    )


__all__ = [
    "TERMINAL_STATUSES",
    "TurnTrace",
    "collect_turn_trace",
]
