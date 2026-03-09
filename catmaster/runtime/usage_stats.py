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

from catmaster.runtime.openrouter_pricing import pricing_cost, resolve_model_pricing


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
    missing_cost_calls = 0
    input_tokens = 0
    input_cached_tokens = 0
    input_cache_write_tokens = 0
    output_tokens = 0
    reasoning_tokens = 0
    total_tokens = 0
    exact_cost_usd = 0.0
    estimated_cost_usd = 0.0
    exact_cost_calls = 0
    estimated_cost_calls = 0
    breakdown_usd = {
        "prompt_uncached": 0.0,
        "cache_read": 0.0,
        "cache_write": 0.0,
        "completion": 0.0,
        "internal_reasoning": 0.0,
    }
    exact_breakdown_usd = {
        "prompt": 0.0,
        "completion": 0.0,
    }
    by_role: dict[str, dict[str, Any]] = {}
    by_model: dict[str, dict[str, Any]] = {}

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
            role = str(payload.get("role") or "").strip() or "(unknown)"
            model_name = str(payload.get("model") or "").strip() or "(unknown)"
            src = str(usage.get("source") or "").strip().lower()
            in_tok = _to_int(usage.get("input_tokens"))
            in_cached_tok = _to_int(usage.get("input_cached_tokens"))
            out_tok = _to_int(usage.get("output_tokens"))
            tot_tok = _to_int(usage.get("total_tokens"))
            input_details = usage.get("input_token_details")
            output_details = usage.get("output_token_details")
            if not isinstance(input_details, dict):
                input_details = {}
            if not isinstance(output_details, dict):
                output_details = {}
            cache_write_tok = _to_int(input_details.get("cache_write"))
            reasoning_tok = _to_int(output_details.get("reasoning"))
            if cache_write_tok is None:
                prompt_details = usage.get("prompt_tokens_details")
                if isinstance(prompt_details, dict):
                    cache_write_tok = _to_int(prompt_details.get("cache_write_tokens"))
            if reasoning_tok is None:
                completion_details = usage.get("completion_tokens_details")
                if isinstance(completion_details, dict):
                    reasoning_tok = _to_int(completion_details.get("reasoning_tokens"))
            exact_cost = _to_float(usage.get("cost"))
            cost_details = usage.get("cost_details") if isinstance(usage.get("cost_details"), dict) else {}
            prompt_cost_exact = _to_float(cost_details.get("upstream_inference_prompt_cost"))
            completion_cost_exact = _to_float(cost_details.get("upstream_inference_completions_cost"))

            if src == "missing" or (
                in_tok is None and in_cached_tok is None and out_tok is None and tot_tok is None
            ):
                missing_usage_calls += 1

            if in_tok is not None:
                input_tokens += in_tok
            if in_cached_tok is not None:
                input_cached_tokens += in_cached_tok
            if cache_write_tok is not None:
                input_cache_write_tokens += cache_write_tok
            if out_tok is not None:
                output_tokens += out_tok
            if reasoning_tok is not None:
                reasoning_tokens += reasoning_tok
            if tot_tok is not None:
                total_tokens += tot_tok
            elif in_tok is not None and out_tok is not None:
                total_tokens += in_tok + out_tok

            call_summary = _call_cost_summary(
                model_name=model_name,
                input_tokens=in_tok,
                input_cached_tokens=in_cached_tok,
                input_cache_write_tokens=cache_write_tok,
                output_tokens=out_tok,
                reasoning_tokens=reasoning_tok,
                exact_cost=exact_cost,
            )
            if call_summary["exact_cost"] is not None:
                exact_cost_usd += float(call_summary["exact_cost"])
                exact_cost_calls += 1
            elif call_summary["estimated_cost"] is not None:
                estimated_cost_usd += float(call_summary["estimated_cost"])
                estimated_cost_calls += 1
            else:
                missing_cost_calls += 1

            if prompt_cost_exact is not None:
                exact_breakdown_usd["prompt"] += prompt_cost_exact
            if completion_cost_exact is not None:
                exact_breakdown_usd["completion"] += completion_cost_exact
            for key, value in (call_summary.get("estimated_breakdown_usd") or {}).items():
                if key in breakdown_usd:
                    breakdown_usd[key] += float(value or 0.0)

            _accumulate_bucket(
                by_role,
                key=role,
                call_summary=call_summary,
                input_tokens=in_tok,
                input_cached_tokens=in_cached_tok,
                input_cache_write_tokens=cache_write_tok,
                output_tokens=out_tok,
                reasoning_tokens=reasoning_tok,
            )
            _accumulate_bucket(
                by_model,
                key=model_name,
                call_summary=call_summary,
                input_tokens=in_tok,
                input_cached_tokens=in_cached_tok,
                input_cache_write_tokens=cache_write_tok,
                output_tokens=out_tok,
                reasoning_tokens=reasoning_tok,
            )

    total_cost_usd = exact_cost_usd + estimated_cost_usd
    if total_cost_usd > 0 and missing_cost_calls == 0 and estimated_cost_calls == 0:
        cost_source = "exact"
    elif total_cost_usd > 0 and exact_cost_calls > 0 and estimated_cost_calls > 0:
        cost_source = "mixed"
    elif total_cost_usd > 0 and estimated_cost_calls > 0:
        cost_source = "estimated"
    else:
        cost_source = "unavailable"
    return {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(run_path),
        "calls": calls,
        "missing_usage_calls": missing_usage_calls,
        "missing_cost_calls": missing_cost_calls,
        "input_tokens": input_tokens,
        "input_cached_tokens": input_cached_tokens,
        "input_cache_write_tokens": input_cache_write_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
        "total_tokens": total_tokens,
        "cost_source": cost_source,
        "cost_usd": round(total_cost_usd, 8),
        "exact_cost_usd": round(exact_cost_usd, 8),
        "estimated_cost_usd": round(estimated_cost_usd, 8),
        "exact_cost_calls": exact_cost_calls,
        "estimated_cost_calls": estimated_cost_calls,
        "breakdown_usd": {k: round(float(v), 8) for k, v in breakdown_usd.items()},
        "exact_breakdown_usd": {k: round(float(v), 8) for k, v in exact_breakdown_usd.items()},
        "by_role": _bucket_list(by_role),
        "by_model": _bucket_list(by_model),
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


def _to_float(value: Any) -> float | None:
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


def _call_cost_summary(
    *,
    model_name: str,
    input_tokens: int | None,
    input_cached_tokens: int | None,
    input_cache_write_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
    exact_cost: float | None,
) -> dict[str, Any]:
    uncached_input_tokens = max(
        0,
        int(input_tokens or 0) - int(input_cached_tokens or 0) - int(input_cache_write_tokens or 0),
    )
    internal_reasoning_tokens = max(0, int(reasoning_tokens or 0))
    visible_completion_tokens = max(0, int(output_tokens or 0) - internal_reasoning_tokens)

    pricing_model, pricing = resolve_model_pricing(model_name)
    estimated_breakdown_usd = {
        "prompt_uncached": pricing_cost(pricing.get("prompt"), uncached_input_tokens),
        "cache_read": pricing_cost(pricing.get("input_cache_read"), int(input_cached_tokens or 0)),
        "cache_write": pricing_cost(pricing.get("input_cache_write"), int(input_cache_write_tokens or 0)),
        "completion": pricing_cost(pricing.get("completion"), visible_completion_tokens),
        "internal_reasoning": pricing_cost(pricing.get("internal_reasoning"), internal_reasoning_tokens),
    }
    estimated_cost = sum(estimated_breakdown_usd.values()) if pricing else None
    return {
        "model_name": model_name,
        "pricing_model": pricing_model,
        "exact_cost": exact_cost,
        "estimated_cost": estimated_cost,
        "estimated_breakdown_usd": estimated_breakdown_usd,
    }


def _new_bucket() -> dict[str, Any]:
    return {
        "calls": 0,
        "input_tokens": 0,
        "input_cached_tokens": 0,
        "input_cache_write_tokens": 0,
        "output_tokens": 0,
        "reasoning_tokens": 0,
        "exact_cost_usd": 0.0,
        "estimated_cost_usd": 0.0,
        "breakdown_usd": {
            "prompt_uncached": 0.0,
            "cache_read": 0.0,
            "cache_write": 0.0,
            "completion": 0.0,
            "internal_reasoning": 0.0,
        },
    }


def _accumulate_bucket(
    buckets: dict[str, dict[str, Any]],
    *,
    key: str,
    call_summary: dict[str, Any],
    input_tokens: int | None,
    input_cached_tokens: int | None,
    input_cache_write_tokens: int | None,
    output_tokens: int | None,
    reasoning_tokens: int | None,
) -> None:
    bucket = buckets.setdefault(key, _new_bucket())
    bucket["calls"] += 1
    bucket["input_tokens"] += int(input_tokens or 0)
    bucket["input_cached_tokens"] += int(input_cached_tokens or 0)
    bucket["input_cache_write_tokens"] += int(input_cache_write_tokens or 0)
    bucket["output_tokens"] += int(output_tokens or 0)
    bucket["reasoning_tokens"] += int(reasoning_tokens or 0)
    bucket["exact_cost_usd"] += float(call_summary.get("exact_cost") or 0.0)
    bucket["estimated_cost_usd"] += float(call_summary.get("estimated_cost") or 0.0)
    for name, value in (call_summary.get("estimated_breakdown_usd") or {}).items():
        if name in bucket["breakdown_usd"]:
            bucket["breakdown_usd"][name] += float(value or 0.0)


def _bucket_list(buckets: dict[str, dict[str, Any]]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for key, payload in buckets.items():
        row = {"name": key}
        row.update(payload)
        row["cost_usd"] = round(float(row.get("exact_cost_usd", 0.0)) + float(row.get("estimated_cost_usd", 0.0)), 8)
        row["exact_cost_usd"] = round(float(row.get("exact_cost_usd", 0.0)), 8)
        row["estimated_cost_usd"] = round(float(row.get("estimated_cost_usd", 0.0)), 8)
        row["breakdown_usd"] = {
            k: round(float(v), 8) for k, v in (row.get("breakdown_usd") or {}).items()
        }
        rows.append(row)
    rows.sort(key=lambda item: float(item.get("cost_usd") or 0.0), reverse=True)
    return rows


__all__ = [
    "load_usage_summary",
    "summarize_usage_from_event_trace",
    "usage_summary_path",
    "write_usage_summary",
]
