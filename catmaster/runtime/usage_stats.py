#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Run-level usage aggregation from LangChain usage metadata."""
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


def write_usage_summary_from_metadata(
    run_dir: Path,
    *,
    usage_metadata: Dict[str, Any],
    call_counts_by_model: Dict[str, int] | None = None,
    usage_metadata_by_role: Dict[str, Any] | None = None,
    call_counts_by_role: Dict[str, int] | None = None,
    append: bool = True,
) -> Dict[str, Any]:
    run_path = Path(run_dir).expanduser().resolve()
    merged_usage = _json_safe_usage(usage_metadata if isinstance(usage_metadata, dict) else {})
    merged_counts = _coerce_call_counts(call_counts_by_model)
    merged_usage_by_role = _json_safe_usage_by_role(usage_metadata_by_role if isinstance(usage_metadata_by_role, dict) else {})
    merged_role_counts = _coerce_call_counts(call_counts_by_role)
    if append:
        existing = load_usage_summary(run_path)
        merged_usage = _merge_usage_metadata(existing.get("raw_usage_metadata"), merged_usage)
        merged_counts = _merge_call_counts(existing.get("call_counts_by_model"), merged_counts)
        merged_usage_by_role = _merge_usage_metadata_by_role(existing.get("raw_usage_metadata_by_role"), merged_usage_by_role)
        merged_role_counts = _merge_call_counts(existing.get("call_counts_by_role"), merged_role_counts)
    summary = summarize_usage_from_metadata(
        merged_usage,
        run_dir=run_path,
        call_counts_by_model=merged_counts,
        usage_metadata_by_role=merged_usage_by_role,
        call_counts_by_role=merged_role_counts,
    )
    summary["raw_usage_metadata"] = merged_usage
    summary["call_counts_by_model"] = merged_counts
    summary["raw_usage_metadata_by_role"] = merged_usage_by_role
    summary["call_counts_by_role"] = merged_role_counts
    path = usage_summary_path(run_path)
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(summary, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        return summary
    return summary


def summarize_usage_from_metadata(
    usage_metadata: Dict[str, Any],
    *,
    run_dir: Path | None = None,
    call_counts_by_model: Dict[str, int] | None = None,
    usage_metadata_by_role: Dict[str, Any] | None = None,
    call_counts_by_role: Dict[str, int] | None = None,
) -> Dict[str, Any]:
    usage_payload = _json_safe_usage(usage_metadata if isinstance(usage_metadata, dict) else {})
    counts = _coerce_call_counts(call_counts_by_model)
    usage_by_role_payload = _json_safe_usage_by_role(usage_metadata_by_role if isinstance(usage_metadata_by_role, dict) else {})
    role_counts = _coerce_call_counts(call_counts_by_role)
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
    by_model: dict[str, dict[str, Any]] = {}
    by_role: dict[str, dict[str, Any]] = {}

    for model_name, usage in usage_payload.items():
        if not isinstance(usage, dict):
            continue
        in_tok = _to_int(usage.get("input_tokens"))
        out_tok = _to_int(usage.get("output_tokens"))
        tot_tok = _to_int(usage.get("total_tokens"))
        input_details = usage.get("input_token_details") if isinstance(usage.get("input_token_details"), dict) else {}
        output_details = usage.get("output_token_details") if isinstance(usage.get("output_token_details"), dict) else {}
        in_cached_tok = _to_int(input_details.get("cache_read"))
        if in_cached_tok is None:
            in_cached_tok = _to_int(input_details.get("cached_tokens"))
        cache_write_tok = _to_int(input_details.get("cache_write"))
        if cache_write_tok is None:
            cache_write_tok = _to_int(input_details.get("cache_creation"))
        reasoning_tok = _to_int(output_details.get("reasoning"))
        call_count = max(1, int(counts.get(str(model_name), 0) or 0))

        calls += call_count
        if in_tok is None and in_cached_tok is None and out_tok is None and tot_tok is None:
            missing_usage_calls += call_count
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
            model_name=str(model_name),
            input_tokens=in_tok,
            input_cached_tokens=in_cached_tok,
            input_cache_write_tokens=cache_write_tok,
            output_tokens=out_tok,
            reasoning_tokens=reasoning_tok,
            exact_cost=None,
        )
        if call_summary["exact_cost"] is not None:
            exact_cost_usd += float(call_summary["exact_cost"])
            exact_cost_calls += call_count
        elif call_summary["estimated_cost"] is not None:
            estimated_cost_usd += float(call_summary["estimated_cost"])
            estimated_cost_calls += call_count
        else:
            missing_cost_calls += call_count
        for key, value in (call_summary.get("estimated_breakdown_usd") or {}).items():
            if key in breakdown_usd:
                breakdown_usd[key] += float(value or 0.0)
        _accumulate_bucket(
            by_model,
            key=str(model_name),
            call_summary=call_summary,
            input_tokens=in_tok,
            input_cached_tokens=in_cached_tok,
            input_cache_write_tokens=cache_write_tok,
            output_tokens=out_tok,
            reasoning_tokens=reasoning_tok,
            call_count=call_count,
        )
    for role_name, role_usage in usage_by_role_payload.items():
        if not isinstance(role_usage, dict):
            continue
        role_call_count = max(1, int(role_counts.get(str(role_name), 0) or 0))
        role_input = 0
        role_input_cached = 0
        role_input_cache_write = 0
        role_output = 0
        role_reasoning = 0
        call_summary_for_role: dict[str, Any] | None = None
        for model_name, usage in role_usage.items():
            if not isinstance(usage, dict):
                continue
            in_tok = _to_int(usage.get("input_tokens"))
            out_tok = _to_int(usage.get("output_tokens"))
            input_details = usage.get("input_token_details") if isinstance(usage.get("input_token_details"), dict) else {}
            output_details = usage.get("output_token_details") if isinstance(usage.get("output_token_details"), dict) else {}
            in_cached_tok = _to_int(input_details.get("cache_read"))
            if in_cached_tok is None:
                in_cached_tok = _to_int(input_details.get("cached_tokens"))
            cache_write_tok = _to_int(input_details.get("cache_write"))
            if cache_write_tok is None:
                cache_write_tok = _to_int(input_details.get("cache_creation"))
            reasoning_tok = _to_int(output_details.get("reasoning"))
            role_input += int(in_tok or 0)
            role_input_cached += int(in_cached_tok or 0)
            role_input_cache_write += int(cache_write_tok or 0)
            role_output += int(out_tok or 0)
            role_reasoning += int(reasoning_tok or 0)
            item_call_summary = _call_cost_summary(
                model_name=str(model_name),
                input_tokens=in_tok,
                input_cached_tokens=in_cached_tok,
                input_cache_write_tokens=cache_write_tok,
                output_tokens=out_tok,
                reasoning_tokens=reasoning_tok,
                exact_cost=None,
            )
            if call_summary_for_role is None:
                call_summary_for_role = item_call_summary
            else:
                merged_breakdown = {
                    key: float(call_summary_for_role["estimated_breakdown_usd"].get(key, 0.0)) + float(item_call_summary["estimated_breakdown_usd"].get(key, 0.0))
                    for key in set(call_summary_for_role["estimated_breakdown_usd"]) | set(item_call_summary["estimated_breakdown_usd"])
                }
                call_summary_for_role = {
                    "model_name": str(call_summary_for_role.get("model_name") or ""),
                    "pricing_model": str(call_summary_for_role.get("pricing_model") or ""),
                    "exact_cost": (call_summary_for_role.get("exact_cost") or 0.0) + (item_call_summary.get("exact_cost") or 0.0),
                    "estimated_cost": (call_summary_for_role.get("estimated_cost") or 0.0) + (item_call_summary.get("estimated_cost") or 0.0),
                    "estimated_breakdown_usd": merged_breakdown,
                }
        if call_summary_for_role is None:
            continue
        _accumulate_bucket(
            by_role,
            key=str(role_name),
            call_summary=call_summary_for_role,
            input_tokens=role_input,
            input_cached_tokens=role_input_cached,
            input_cache_write_tokens=role_input_cache_write,
            output_tokens=role_output,
            reasoning_tokens=role_reasoning,
            call_count=role_call_count,
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
        "run_dir": str(Path(run_dir).expanduser().resolve()) if run_dir is not None else "",
        "source": "langchain_usage_metadata",
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
    call_count: int = 1,
) -> None:
    bucket = buckets.setdefault(key, _new_bucket())
    bucket["calls"] += max(1, int(call_count or 1))
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


def _json_safe_usage(raw: Any) -> dict[str, dict[str, Any]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, Any]] = {}
    for model_name, payload in raw.items():
        if not isinstance(model_name, str) or not isinstance(payload, dict):
            continue
        out[model_name] = json.loads(json.dumps(payload, ensure_ascii=False, default=str))
    return out


def _json_safe_usage_by_role(raw: Any) -> dict[str, dict[str, dict[str, Any]]]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, dict[str, dict[str, Any]]] = {}
    for role_name, payload in raw.items():
        if not isinstance(role_name, str):
            continue
        out[role_name] = _json_safe_usage(payload)
    return out


def _coerce_call_counts(raw: Any) -> dict[str, int]:
    if not isinstance(raw, dict):
        return {}
    out: dict[str, int] = {}
    for key, value in raw.items():
        if not isinstance(key, str):
            continue
        coerced = _to_int(value)
        if coerced is None:
            continue
        out[key] = max(0, coerced)
    return out


def _merge_call_counts(base: Any, update: Any) -> dict[str, int]:
    merged = _coerce_call_counts(base)
    for key, value in _coerce_call_counts(update).items():
        merged[key] = int(merged.get(key, 0)) + int(value or 0)
    return merged


def _merge_usage_metadata(base: Any, update: Any) -> dict[str, dict[str, Any]]:
    merged = _json_safe_usage(base)
    incoming = _json_safe_usage(update)
    for model_name, payload in incoming.items():
        current = merged.get(model_name)
        if not isinstance(current, dict):
            merged[model_name] = payload
            continue
        merged[model_name] = _merge_usage_dict(current, payload)
    return merged


def _merge_usage_metadata_by_role(base: Any, update: Any) -> dict[str, dict[str, dict[str, Any]]]:
    merged = _json_safe_usage_by_role(base)
    incoming = _json_safe_usage_by_role(update)
    for role_name, payload in incoming.items():
        current = merged.get(role_name)
        if not isinstance(current, dict):
            merged[role_name] = payload
            continue
        merged[role_name] = _merge_usage_metadata(current, payload)
    return merged


def _merge_usage_dict(base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
    merged = dict(base)
    for key, value in update.items():
        if isinstance(value, dict):
            current = merged.get(key)
            if isinstance(current, dict):
                merged[key] = _merge_usage_dict(current, value)
            else:
                merged[key] = dict(value)
            continue
        if isinstance(value, bool):
            merged[key] = int(bool(merged.get(key, 0))) + int(value)
            continue
        if isinstance(value, int):
            merged[key] = int(_to_int(merged.get(key)) or 0) + value
            continue
        if isinstance(value, float):
            merged[key] = float(_to_float(merged.get(key)) or 0.0) + value
            continue
        if value is not None:
            merged[key] = value
    return merged


__all__ = [
    "load_usage_summary",
    "summarize_usage_from_metadata",
    "usage_summary_path",
    "write_usage_summary_from_metadata",
]
