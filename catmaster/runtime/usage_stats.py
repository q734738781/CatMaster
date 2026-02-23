#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Run-level usage aggregation from event_trace.jsonl.
"""
from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict
import json


def usage_summary_path(run_dir: Path) -> Path:
    return Path(run_dir).expanduser().resolve() / "usage_summary.json"


def load_usage_summary(run_dir: Path) -> Dict[str, Any]:
    path = usage_summary_path(run_dir)
    if not path.exists():
        return {}
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return {}
    return payload if isinstance(payload, dict) else {}


def write_usage_summary(run_dir: Path) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    summary = summarize_usage_from_event_trace(run_path)
    path = usage_summary_path(run_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return summary
    return summary


def summarize_usage_from_event_trace(run_dir: Path) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    event_path = run_path / "event_trace.jsonl"
    calls = 0
    missing_usage_calls = 0
    input_tokens = 0
    input_cached_tokens = 0
    output_tokens = 0
    total_tokens = 0

    if event_path.exists():
        for line in event_path.read_text(encoding="utf-8").splitlines():
            raw = line.strip()
            if not raw:
                continue
            try:
                event = json.loads(raw)
            except Exception:
                continue
            if not isinstance(event, dict):
                continue
            if str(event.get("event") or "") != "LLM_USAGE":
                continue
            payload = event.get("payload")
            if not isinstance(payload, dict):
                continue
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                continue

            calls += 1
            src = str(usage.get("source") or "").strip().lower()
            in_tok = _to_int(usage.get("input_tokens"))
            in_cached_tok = _to_int(usage.get("input_cached_tokens"))
            out_tok = _to_int(usage.get("output_tokens"))
            tot_tok = _to_int(usage.get("total_tokens"))

            if src == "missing" or (
                in_tok is None and in_cached_tok is None and out_tok is None and tot_tok is None
            ):
                missing_usage_calls += 1

            if in_tok is not None:
                input_tokens += in_tok
            if in_cached_tok is not None:
                input_cached_tokens += in_cached_tok
            if out_tok is not None:
                output_tokens += out_tok
            if tot_tok is not None:
                total_tokens += tot_tok
            elif in_tok is not None and out_tok is not None:
                total_tokens += in_tok + out_tok

    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path),
        "calls": calls,
        "missing_usage_calls": missing_usage_calls,
        "input_tokens": input_tokens,
        "input_cached_tokens": input_cached_tokens,
        "output_tokens": output_tokens,
        "total_tokens": total_tokens,
    }


def _to_int(value: Any) -> int | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return int(value)
    if isinstance(value, int):
        return value
    try:
        return int(value)
    except Exception:
        return None


__all__ = [
    "load_usage_summary",
    "summarize_usage_from_event_trace",
    "usage_summary_path",
    "write_usage_summary",
]
