from __future__ import annotations

import json
import sqlite3
from contextlib import closing
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

_BOUNDARY_EVENT_NAMES = frozenset(
    {
        "TASKS_COMPILED",
        "TASK_START",
        "TASK_DECISION",
        "TASK_SUMMARY",
        "TASK_END",
        "RUN_START",
        "RUN_PAUSED",
        "RUN_END",
    }
)


def _read_json(path: Path, *, diagnostics: list[str]) -> dict[str, Any]:
    if not path.is_file():
        diagnostics.append(f"FileNotFoundError: trace metadata is missing: {path}")
        return {}
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        diagnostics.append(
            f"{type(exc).__name__}: cannot read trace metadata {path}: {exc}"
        )
        return {}
    if not isinstance(value, dict):
        diagnostics.append(
            f"TypeError: trace metadata {path} must contain a JSON object, "
            f"found {type(value).__name__}"
        )
        return {}
    return value


def read_thread_turn(
    run_dir: Path | str, *, run_id: str, thread_id: str,
) -> dict[str, Any]:
    """Read a local run's bound messages, never the latest conversation state.

    Local execution persists its turn in workspace.sqlite, not run_state.json.
    Open only the existing message store; constructing a WebUI ThreadStore here
    would perform schema/import writes in this background evidence reader.
    """
    run_path = Path(run_dir).expanduser().resolve()
    database = run_path.parent.parent / "workspace.sqlite"
    if not thread_id or run_path.parent.name != "runs" or not database.is_file():
        return {}
    with closing(sqlite3.connect(database.as_uri() + "?mode=ro", uri=True, timeout=30)) as conn:
        if not conn.execute(
            "SELECT 1 FROM sqlite_master WHERE type='table' AND name='thread_messages'"
        ).fetchone():
            return {}
        row = conn.execute(
            """SELECT payload_json FROM thread_messages
               WHERE thread_id = ? AND message_role = 'assistant' AND message_run_id = ?
                 AND json_extract(payload_json, '$.structured_sidecar.execution.run_id') = ?
               ORDER BY row_id DESC LIMIT 1""",
            (thread_id, run_id, run_id),
        ).fetchone()
        if row is None:
            return {}  # Legacy/CLI runs retain their existing run-state reader.
        answer = json.loads(row[0])
        meta = answer.get("meta") or {}
        execution = (answer.get("structured_sidecar") or {}).get("execution") or {}
        input_id = str(meta.get("input_message_id") or execution.get("input_message_id")
                       or meta.get("episode_id") or "")
        source = conn.execute(
            """SELECT payload_json FROM thread_messages
               WHERE thread_id = ? AND message_id = ? AND message_role = 'user'""",
            (thread_id, input_id),
        ).fetchone() if input_id else None
        user = json.loads(source[0]) if source else {}
    def text(message: dict[str, Any]) -> str:
        return "\n".join(str(p.get("text") or "") for p in message.get("parts", [])
                         if p.get("type") == "text")

    # The graph can finish successfully yet fail to deliver a current answer.
    # The finalized message records that failure even if native_status=success.
    status = str(answer.get("status") or meta.get("native_status") or "unknown")
    status = {"success": "done", "completed": "done", "failed": "error", "timeout": "error"}.get(status, status)
    return {
        "run_id": run_id, "thread_id": thread_id,
        "entrypoint": meta.get("entrypoint") or execution.get("entrypoint") or "",
        "status": status, "episode_id": meta.get("episode_id") or input_id or run_id,
        "user_prompt": text(user), "final_answer": text(answer),
        "message_id": input_id, "assistant_message_id": answer["id"],
        "diagnostics": [] if user else [
            f"The bound input message {input_id or '(unrecorded)'} is unavailable in thread {thread_id}."
        ],
        "prior_assistant_message_id": meta.get("prior_assistant_message_id") or "",
        "checkpoint_resume_available": bool(meta.get("checkpoint_resume_available") or status == "interrupted"),
        "artifact_refs": [str(a["workspace_path"]) for a in
                          (user.get("structured_sidecar") or {}).get("attachments", [])
                          if a.get("workspace_path")],
    }


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


def _read_all_trajectory_events(
    store: ObservabilityStore,
    *,
    diagnostics: list[str],
) -> list[dict[str, Any]]:
    """Read every selected event page, oldest first."""

    chunks: list[list[dict[str, Any]]] = []
    before_id = 0
    while True:
        try:
            page = store.read_events_page(
                limit=5_000,
                before_id=before_id,
                names=_TRAJECTORY_EVENT_NAMES,
                include_legacy_trace_records=True,
            )
        except Exception as exc:
            diagnostics.append(
                f"trajectory event read failed: {type(exc).__name__}: {exc}"
            )
            break
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
        if next_before <= 0 or (before_id > 0 and next_before >= before_id):
            diagnostics.append(
                "trajectory event pagination did not advance: "
                f"before_id={before_id}, next_before={next_before}"
            )
            break
        before_id = next_before

    ordered: list[dict[str, Any]] = []
    for chunk in reversed(chunks):
        ordered.extend(chunk)
    ordered.sort(key=lambda item: int(item.get("id") or 0))
    return ordered


def _read_task_outcome_events(store: ObservabilityStore) -> list[dict[str, Any]]:
    """Read only the small terminal records needed for verified outcome metadata."""

    page = store.read_events_page(
        limit=5_000,
        names=("TASK_SUMMARY", "TASK_END"),
        include_legacy_trace_records=True,
    )
    events = [
        item
        for item in list(page.get("events") or [])
        if isinstance(item, dict)
    ]
    events.sort(key=lambda item: int(item.get("id") or 0))
    return events


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


def _without_encrypted_transport(value: Any) -> Any:
    """Remove unreadable provider transport blobs from the semantic surface."""

    if isinstance(value, dict):
        return {
            str(key): _without_encrypted_transport(item)
            for key, item in value.items()
            if str(key) != "encrypted_content"
        }
    if isinstance(value, list):
        return [_without_encrypted_transport(item) for item in value]
    return value


def _canonical_content_blocks(value: Any) -> list[dict[str, Any]]:
    """Project persisted LangChain content onto readable standardized blocks."""

    if isinstance(value, str):
        return [{"type": "text", "text": value}] if value else []
    if not isinstance(value, list):
        return []
    blocks: list[dict[str, Any]] = []
    for raw in value:
        if not isinstance(raw, dict):
            text = str(raw or "")
            if text:
                blocks.append({"type": "text", "text": text})
            continue
        block = _without_encrypted_transport(raw)
        block_type = str(block.get("type") or "").strip()
        if block_type in {"text", "output_text"}:
            text = str(block.get("text") or block.get("content") or "")
            blocks.append({**block, "type": "text", "text": text})
            continue
        if block_type in {"reasoning", "reasoning_content"}:
            text = str(
                block.get("reasoning")
                or block.get("text")
                or block.get("content")
                or ""
            )
            blocks.append({**block, "type": "reasoning", "reasoning": text})
            continue
        # Unknown blocks may carry citations, refusals or hosted-tool results
        # without a text/url field. Preserve their structure for inspection.
        if block:
            blocks.append(block)
    return blocks


def _canonical_tool_calls(value: Any) -> list[dict[str, Any]]:
    calls: list[dict[str, Any]] = []
    for raw in list(value or []):
        if not isinstance(raw, dict):
            continue
        args: Any = raw.get("args")
        if args is None and "args_json" in raw:
            raw_args = raw.get("args_json")
            try:
                args = {} if raw_args is None else json.loads(str(raw_args))
            except Exception as exc:
                args = {
                    "_args_parse_error": {
                        "error_type": type(exc).__name__,
                        "error": str(exc),
                        "raw_args": "" if raw_args is None else str(raw_args),
                    }
                }
        calls.append(
            {
                "id": str(raw.get("id") or ""),
                "name": str(raw.get("name") or ""),
                "args": _without_encrypted_transport(args if args is not None else {}),
            }
        )
    return calls


def semantic_event_name(name: str) -> str:
    """Return the canonical model-visible name for one persisted occurrence."""

    return {
        "LLM_RAW_RESPONSE": "MODEL_RESPONSE",
        "LLM_CALL_END": "MODEL_RESPONSE",
        "TOOL_RAW_INPUT": "TOOL_INPUT",
        "TOOL_RAW_OUTPUT": "TOOL_RESULT",
        "TOOL_CALL_END": "TOOL_RESULT",
    }.get(str(name or ""), str(name or ""))


def semantic_event_payload(name: str, payload: dict[str, Any]) -> dict[str, Any]:
    """Normalize one raw record without changing the canonical observability DB."""

    event_name = str(name or "")
    source = payload if isinstance(payload, dict) else {}
    if event_name == "LLM_RAW_RESPONSE":
        generations: list[dict[str, Any]] = []
        for raw in list(source.get("generations") or []):
            if not isinstance(raw, dict):
                continue
            response_text = str(raw.get("response_text") or "")
            blocks = _canonical_content_blocks(
                raw.get("content_blocks")
                if "content_blocks" in raw
                else raw.get("response_content_raw")
            )
            if not blocks and response_text:
                blocks = [{"type": "text", "text": response_text}]
            generation: dict[str, Any] = {
                "assistant_text": response_text,
                "content_blocks": blocks,
                "tool_calls": _canonical_tool_calls(raw.get("parsed_tool_calls")),
            }
            reasoning = str(raw.get("reasoning_text") or "").strip()
            if reasoning:
                generation["reasoning_text"] = reasoning
            invalid_calls = _without_encrypted_transport(
                list(raw.get("invalid_tool_calls") or [])
            )
            if invalid_calls:
                generation["invalid_tool_calls"] = invalid_calls
            for key in ("additional_kwargs", "response_metadata", "generation_info"):
                if raw.get(key):
                    generation[key] = _without_encrypted_transport(raw[key])
            original_content = _without_encrypted_transport(raw.get("response_content_raw"))
            if original_content is not None and original_content != blocks and original_content != response_text:
                generation["original_content"] = original_content
            generations.append(generation)
        return {
            "generations": generations,
            "usage": _without_encrypted_transport(source.get("usage") or {}),
            **{key: _without_encrypted_transport(source[key])
               for key in ("partial", "status", "error", "llm_output") if key in source},
        }
    if event_name == "LLM_CALL_END":
        response_text = str(source.get("text_preview") or "")
        result: dict[str, Any] = {
            "assistant_text": response_text,
            "content_blocks": (
                [{"type": "text", "text": response_text}]
                if response_text
                else []
            ),
            "tool_calls": [
                {"id": "", "name": str(item or ""), "args": {}}
                for item in list(source.get("tool_calls") or [])
                if str(item or "").strip()
            ],
            "usage": _without_encrypted_transport(source.get("usage") or {}),
        }
        reasoning = str(source.get("reasoning_text") or "").strip()
        if reasoning:
            result["reasoning_text"] = reasoning
        return result
    if event_name == "TOOL_RAW_INPUT":
        return {
            "input": _without_encrypted_transport(
                source.get("params_full")
                if "params_full" in source
                else source.get("params_compact")
            )
        }
    if event_name in {"TOOL_RAW_OUTPUT", "TOOL_CALL_END"}:
        projection = (
            source.get("projection")
            if isinstance(source.get("projection"), dict)
            else {}
        )
        model_visible = projection.get("content_text")
        if model_visible in (None, ""):
            model_visible = projection.get("content_preview")
        if model_visible in (None, ""):
            model_visible = source.get("raw_output", source.get("result", ""))
        return {
            "result": _without_encrypted_transport(model_visible),
            "error": str(projection.get("error") or source.get("error") or ""),
            "warnings": _without_encrypted_transport(list(projection.get("warnings") or [])),
            "highlights": _without_encrypted_transport(list(projection.get("highlights") or [])),
            "artifact_refs": _without_encrypted_transport(list(projection.get("offload_refs") or [])),
        }
    if event_name == "LLM_ERROR":
        return {"error": str(source.get("error") or "Model call failed.")}
    return _without_encrypted_transport(source)


def _model_event(
    *,
    run_id: str,
    event: dict[str, Any],
    raw: bool,
) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    canonical = semantic_event_payload(
        "LLM_RAW_RESPONSE" if raw else "LLM_CALL_END",
        payload,
    )
    return {
        **canonical,
        "source_ref": _event_ref(run_id, event),
        "kind": "model",
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "node": str(event.get("node") or payload.get("node") or ""),
        "generations": list(canonical.get("generations") or [canonical]),
    }


def _tool_input_event(*, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    value = semantic_event_payload("TOOL_RAW_INPUT", payload).get("input")
    return {
        "source_ref": _event_ref(run_id, event),
        "kind": "tool_input",
        "tool": str(event.get("tool") or payload.get("tool") or payload.get("tool_name") or "tool"),
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "input": value,
    }


def _tool_output_event(*, run_id: str, event: dict[str, Any]) -> dict[str, Any]:
    payload = event.get("payload") if isinstance(event.get("payload"), dict) else {}
    canonical = semantic_event_payload(
        str(event.get("name") or "TOOL_CALL_END"),
        payload,
    )
    return {
        "source_ref": _event_ref(run_id, event),
        "kind": "tool_output",
        "tool": str(event.get("tool") or payload.get("tool") or payload.get("tool_name") or "tool"),
        "agent": str(event.get("agent_name") or payload.get("agent_name") or ""),
        "status": str(event.get("status") or payload.get("status") or payload.get("tool_status") or "unknown"),
        "result": canonical.get("result"),
        "error": str(canonical.get("error") or ""),
        "warnings": list(canonical.get("warnings") or []),
        "highlights": list(canonical.get("highlights") or []),
        "artifact_refs": list(canonical.get("artifact_refs") or []),
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
        if name in _BOUNDARY_EVENT_NAMES:
            trajectory.append(
                {
                    "source_ref": _event_ref(run_id, event),
                    "kind": "boundary",
                    "event": name,
                    "payload": semantic_event_payload(name, payload),
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
            response = str(
                generation.get("assistant_text")
                or generation.get("response_text")
                or ""
            ).strip()
            if reasoning:
                lines.extend(["Reasoning:", "", reasoning, ""])
            if response:
                lines.extend(["Response:", "", response, ""])
            content_blocks = generation.get("content_blocks") or []
            if content_blocks and content_blocks != [{"type": "text", "text": response}]:
                lines.extend(
                    [
                        "Content blocks:",
                        "",
                        "```json",
                        _json_text(content_blocks),
                        "```",
                        "",
                    ]
                )
            tool_calls = generation.get("tool_calls") or []
            if tool_calls:
                lines.extend(
                    [
                        "Tool calls:",
                        "",
                        "```json",
                        _json_text(tool_calls),
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
    diagnostics: list[str] = field(default_factory=list)

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
            "diagnostics": list(self.diagnostics),
        }

    @classmethod
    def from_dict(cls, value: dict[str, Any]) -> "TurnTrace":
        fields = cls.__dataclass_fields__
        data = {key: value.get(key) for key in fields if key in value}
        if not isinstance(data.get("events"), list):
            data["events"] = []
        if not isinstance(data.get("artifact_refs"), list):
            data["artifact_refs"] = []
        if not isinstance(data.get("diagnostics"), list):
            data["diagnostics"] = []
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
        if self.diagnostics:
            lines.extend(
                [
                    "Trace diagnostics:",
                    "",
                    *[f"- {item}" for item in self.diagnostics],
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
    include_events: bool = True,
) -> TurnTrace:
    run_path = Path(run_dir).expanduser().resolve()
    fallback = dict(fallback or {})
    diagnostics: list[str] = []
    state = read_thread_turn(
        run_path, run_id=str(fallback.get("run_id") or run_path.name),
        thread_id=str(fallback.get("thread_id") or ""),
    ) or _read_json(run_path / "run_state.json", diagnostics=diagnostics)
    diagnostics.extend(state.get("diagnostics") or [])
    meta = _read_json(run_path / "meta.json", diagnostics=diagnostics)
    run_id = str(
        state.get("run_id")
        or meta.get("run_id")
        or fallback.get("run_id")
        or run_path.name
    ).strip()
    observation_store = ObservabilityStore(run_path)
    outcome_events = _read_task_outcome_events(observation_store)
    task_outcome, outcome_ref = _task_outcome_from_events(raw_events=outcome_events)
    raw_events = (
        _read_all_trajectory_events(observation_store, diagnostics=diagnostics)
        if include_events
        else []
    )
    fallback_outcome = _verified_task_outcome(fallback.get("task_outcome"))
    fallback_ref = str(fallback.get("outcome_ref") or "").strip()
    if fallback_outcome and fallback_ref:
        task_outcome = fallback_outcome
        outcome_ref = fallback_ref

    artifact_refs: list[str] = []
    for item in list(
        state.get("artifact_ids")
        or state.get("artifact_refs")
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
            or fallback.get("execution_status")
            or "unknown"
        ).strip(),
        user_prompt=str(
            state.get("episode_prompt")
            or state.get("user_prompt")
            or fallback.get("episode_prompt")
            or fallback.get("prompt")
            or ""
        ).strip(),
        final_answer=str(state.get("final_answer") or "").strip(),
        summary=str(state.get("summary") or "").strip(),
        explicit_correction=str(fallback.get("note") or "").strip(),
        resume_guidance=str(state.get("resume_guidance") or "").strip(),
        source_message_id=str(state.get("message_id") or fallback.get("message_id") or "").strip(),
        prior_assistant_message_id=str(
            state.get("prior_assistant_message_id") or fallback.get("prior_assistant_message_id") or ""
        ).strip(),
        assistant_message_id=str(
            state.get("assistant_message_id") or fallback.get("assistant_message_id") or ""
        ).strip(),
        events=_normalize_trajectory(run_id=run_id, raw_events=raw_events),
        task_outcome=task_outcome,
        outcome_ref=outcome_ref,
        artifact_refs=artifact_refs,
        diagnostics=diagnostics,
    )


__all__ = [
    "TERMINAL_STATUSES",
    "TurnTrace",
    "collect_turn_trace",
    "semantic_event_name",
    "semantic_event_payload",
]
