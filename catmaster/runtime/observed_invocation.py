"""Observe standalone model and agent invocations through the shared run store."""
from __future__ import annotations

import json
import logging
import threading
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Any, Iterator
from uuid import uuid4

from catmaster.runtime.artifact_callback import ObservabilityCallbackHandler
from catmaster.runtime.usage_stats import summarize_usage_from_metadata, summarize_usage_from_observability

logger = logging.getLogger(__name__)


def _write_json(path: Path, value: dict[str, Any]) -> None:
    # Readers may inspect a live invocation while a child callback updates it.
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")
    temporary.replace(path)


class InvocationObservationCallback(ObservabilityCallbackHandler):
    """Keep full shared debug observations and persist per-call usage."""

    def __init__(self, run_dir: Path, *, run_id: str, stage: str, context: dict[str, Any]) -> None:
        super().__init__(run_dir, run_id=run_id, default_agent_name=stage)
        self.run_dir = run_dir
        self.context = dict(context)
        self._write_lock = threading.RLock()
        self._calls: dict[str, str] = {}
        self._without_usage: set[str] = set()
        self.summary: dict[str, Any] = {}

    def _agent_name(self, ctx: dict[str, Any]) -> str:
        role = super()._agent_name(ctx) or self.default_agent_name
        if role != self.default_agent_name:
            role = f"{self.default_agent_name}/{role}"
        if ctx.get("lc_source") == "summarization":
            role += "/summarization"
        return role

    def _with_context(self, payload: dict[str, Any]) -> dict[str, Any]:
        return {**payload, **self.context,
                "thread_id": str(self.context.get("source_thread_id") or "")}

    def _record_raw(self, name: str, *, payload: dict[str, Any], **kwargs: Any) -> None:
        with self._write_lock:
            call_id = str(payload.get("callback_run_id") or "")
            if name in {"LLM_RAW_RESPONSE", "LLM_ERROR"} and self._calls.get(call_id) in {"completed", "failed"}:
                return
            super()._record_raw(name, payload=self._with_context(payload), **kwargs)

    def _record_semantic(self, name: str, *, payload: dict[str, Any],
                         category: str, task_id: str = "", step_id: int | None = None) -> None:
        is_llm = name in {"LLM_CALL_START", "LLM_CALL_END", "LLM_ERROR"}
        call_id = str(payload.get("callback_run_id") or "")
        with self._write_lock:
            if is_llm and self._calls.get(call_id) in {"completed", "failed"}:
                return
            super()._record_semantic(
                name, payload=self._with_context(payload), category=category,
                task_id=task_id, step_id=step_id,
            )
            if not is_llm:
                return
            if name == "LLM_CALL_START":
                self._calls[call_id] = "pending"
            elif name == "LLM_CALL_END":
                self._calls[call_id] = "completed"
                if not payload.get("usage"):
                    self._without_usage.add(call_id)
            else:
                self._calls[call_id] = "failed"
                self._without_usage.add(call_id)
            self.persist_summary()

    def persist_summary(self) -> dict[str, Any]:
        with self._write_lock:
            summary = {
                **summarize_usage_from_metadata({}, run_dir=self.run_dir),
                **summarize_usage_from_observability(self.run_dir),
            }
            pending = sum(status == "pending" for status in self._calls.values())
            summary.update(
                started_calls=len(self._calls),
                failed_calls=sum(status == "failed" for status in self._calls.values()),
                pending_calls=pending,
                missing_usage_calls=len(self._without_usage),
                partial=bool(summary.get("partial") or pending or self._without_usage),
            )
            _write_json(self.run_dir / "usage_summary.json", summary)
            self.summary = summary
            return summary

    def fail_pending(self, error: BaseException) -> None:
        # BaseChatModel.agenerate reports provider exceptions, but cancellation
        # of its outer gather can exit before on_llm_error. Close only calls
        # still pending when the owning invocation itself has failed.
        with self._write_lock:
            for call_id, status in list(self._calls.items()):
                if status == "pending":
                    self.on_llm_error(error if str(error) else RuntimeError(type(error).__name__), run_id=call_id)


@contextmanager
def observed_invocation(*, workspace: Path, entrypoint: str, stage: str, model_label: str,
                     context: dict[str, Any]) -> Iterator[tuple[dict[str, Any], InvocationObservationCallback]]:
    """A separate run per invocation keeps retries and concurrent jobs additive."""
    run_id = f"{entrypoint}_{uuid4().hex}"
    run_dir = workspace / "metadata" / "runs" / run_id
    run_dir.mkdir(parents=True, exist_ok=False)
    state = {**context, "run_id": run_id, "entrypoint": entrypoint,
             "agent_name": stage, "model_label": model_label,
             "status": "running", "started_at": time.time()}
    # Keep accounting metadata separate from executable run_state.json: the
    # legacy resume selector treats any directory with that file as resumable.
    _write_json(run_dir / "meta.json", state)
    callback = InvocationObservationCallback(run_dir, run_id=run_id, stage=stage, context=context)
    config = {
        "configurable": {"thread_id": run_id},
        "metadata": {"lc_agent_name": stage, "catmaster_model_label": model_label},
        "callbacks": [callback],
    }
    callback._record_semantic(
        "RUN_START", payload={"entrypoint": entrypoint, "status": "running"}, category="run",
    )
    error = ""
    try:
        yield config, callback
    except BaseException as exc:
        state["status"] = "error"
        error = str(exc) or type(exc).__name__
        callback.fail_pending(exc)
        raise
    else:
        state["status"] = "done"
    finally:
        state["finished_at"] = time.time()
        callback._record_semantic(
            "RUN_END", payload={"entrypoint": entrypoint, "status": state["status"], "error": error},
            category="run",
        )
        try:
            callback.persist_summary()
            _write_json(run_dir / "meta.json", state)
        except Exception:
            logger.exception("Could not finalize invocation observations at %s", run_dir)
