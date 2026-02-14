#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

# Ensure repository root is importable when running via:
# `python scripts/prompt_cache_retention_probe.py`
REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catmaster.llm.openai_chat_completions_driver import OpenAIChatCompletionsDriver
from catmaster.runtime.conversation_state import ConversationState


def _utc_now() -> str:
    return datetime.utcnow().isoformat(timespec="seconds") + "Z"


def _build_long_prefix(target_tokens_hint: int = 1400) -> str:
    # OpenAI prompt cache starts working at >=1024 prompt tokens.
    # This is a conservative overbuild using short repeated terms.
    unit = "cache"
    n_words = max(target_tokens_hint * 2, 3000)
    words = [f"{unit}_{i % 200}" for i in range(n_words)]
    body = " ".join(words)
    return (
        "You are a cache probe assistant.\n"
        "Rules:\n"
        "1) Reply with exactly ACK.\n"
        "2) Do not add extra text.\n"
        "3) Keep behavior deterministic.\n\n"
        "Stable long prefix block:\n"
        f"{body}\n"
    )


def _usage_dict(turn: Any) -> dict[str, Any]:
    usage = getattr(turn, "usage", None)
    if usage is None:
        return {}
    if hasattr(usage, "to_dict"):
        return usage.to_dict(include_raw=True)  # type: ignore[attr-defined]
    if isinstance(usage, dict):
        return usage
    return {}


def _print_call_result(call_index: int, phase: str, usage: dict[str, Any], output_text: str) -> None:
    in_tok = usage.get("input_tokens")
    in_cached = usage.get("input_cached_tokens")
    out_tok = usage.get("output_tokens")
    total = usage.get("total_tokens")
    print(
        f"[{_utc_now()}] call={call_index} phase={phase} "
        f"input={in_tok} cached={in_cached} output={out_tok} total={total} "
        f"reply={output_text!r}",
        flush=True,
    )


def _sleep_with_heartbeat(seconds: int) -> None:
    if seconds <= 0:
        return
    print(f"[{_utc_now()}] sleeping {seconds}s before post-sleep calls...", flush=True)
    step = 60
    remaining = seconds
    while remaining > 0:
        chunk = step if remaining > step else remaining
        time.sleep(chunk)
        remaining -= chunk
        print(f"[{_utc_now()}] sleep remaining: {remaining}s", flush=True)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Probe prompt cache behavior with 1 + 3 + sleep + 3 calls, "
            "appending full message history every turn."
        )
    )
    parser.add_argument("--model", default=os.getenv("CATMASTER_LLM_MODEL", "openai/gpt-5.2"))
    parser.add_argument(
        "--base-url",
        default=os.getenv("CATMASTER_BASE_URL", "https://openrouter.ai/api/v1"),
        help="API base URL, e.g. https://openrouter.ai/api/v1 or OpenAI base URL",
    )
    parser.add_argument("--api-key", default=os.getenv("OPENROUTER_API_KEY") or os.getenv("OPENAI_API_KEY"))
    parser.add_argument(
        "--retention",
        default="24h",
        choices=["in_memory", "24h"],
        help="Prompt cache retention to pass through",
    )
    parser.add_argument(
        "--sleep-seconds",
        type=int,
        default=3600,
        help="Sleep between call 4 and call 5. Use 0 for quick smoke run.",
    )
    parser.add_argument(
        "--between-calls-sleep-seconds",
        type=int,
        default=8,
        help="Short sleep between adjacent calls (except right before the long sleep boundary).",
    )
    parser.add_argument(
        "--pre-repeat",
        type=int,
        default=3,
        help="Number of repeated calls before sleep (calls 2..N).",
    )
    parser.add_argument(
        "--post-repeat",
        type=int,
        default=3,
        help="Number of repeated calls after sleep.",
    )
    parser.add_argument(
        "--target-prefix-tokens-hint",
        type=int,
        default=1400,
        help="Hint to construct a long stable prefix block.",
    )
    parser.add_argument(
        "--pin-openai-provider",
        action="store_true",
        default=False,
        help="When using OpenRouter, send provider pinning block to prefer OpenAI only.",
    )
    parser.add_argument(
        "--output",
        default="",
        help="Output JSONL path. Default: devdocs/cache_probe_<timestamp>.jsonl",
    )
    args = parser.parse_args()

    if not args.api_key:
        print("Missing API key. Set --api-key or OPENROUTER_API_KEY / OPENAI_API_KEY.", file=sys.stderr)
        return 2

    out_path = Path(args.output) if args.output else Path("devdocs") / f"cache_probe_{int(time.time())}.jsonl"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    driver = OpenAIChatCompletionsDriver(
        model=args.model,
        api_key=args.api_key,
        base_url=args.base_url,
    )

    state = ConversationState()
    state.append_input_message("system", _build_long_prefix(args.target_prefix_tokens_hint))
    state.append_input_message(
        "user",
        "Cache probe call-1. Reply exactly ACK.",
    )

    extra_body: dict[str, Any] = {"prompt_cache_retention": args.retention}
    if args.pin_openai_provider and "openrouter.ai" in (args.base_url or ""):
        extra_body["provider"] = {
            "order": ["openai"],
            "allow_fallbacks": False,
            "require_parameters": True,
        }

    plan: list[tuple[int, str]] = [(1, "initial")]
    for idx in range(args.pre_repeat):
        plan.append((2 + idx, "pre_sleep_repeat"))
    for idx in range(args.post_repeat):
        plan.append((2 + args.pre_repeat + idx, "post_sleep_repeat"))

    first_post_sleep_call = 2 + args.pre_repeat
    records: list[dict[str, Any]] = []

    for call_index, phase in plan:
        if call_index == first_post_sleep_call:
            _sleep_with_heartbeat(args.sleep_seconds)
        turn = driver.create_turn(
            input_items=state.input_items,
            tools=None,
            temperature=0.0,
            extra_body=extra_body,
        )
        usage = _usage_dict(turn)
        output_text = (turn.output_text or "").strip()
        _print_call_result(call_index, phase, usage, output_text)
        record = {
            "ts": _utc_now(),
            "call_index": call_index,
            "phase": phase,
            "usage": usage,
            "output_text": output_text,
            "extra_body": extra_body,
            "model": args.model,
            "base_url": args.base_url,
        }
        records.append(record)
        with out_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")

        state.append_model_output_items(turn.output_items_raw)
        if call_index != plan[-1][0]:
            state.append_input_message("user", "Next probe call. Reply exactly ACK.")
            next_call_index = call_index + 1
            if next_call_index != first_post_sleep_call and args.between_calls_sleep_seconds > 0:
                print(
                    f"[{_utc_now()}] short sleep {args.between_calls_sleep_seconds}s before next call...",
                    flush=True,
                )
                time.sleep(args.between_calls_sleep_seconds)

        if call_index == 1:
            in_tok = usage.get("input_tokens")
            if isinstance(in_tok, int) and in_tok < 1024:
                print(
                    f"[{_utc_now()}] WARNING: call-1 input_tokens={in_tok} < 1024; cache behavior may not trigger.",
                    flush=True,
                )

    cached_values = [r.get("usage", {}).get("input_cached_tokens") for r in records]
    print(f"[{_utc_now()}] done. output={out_path}", flush=True)
    print(f"[{_utc_now()}] cached_tokens_series={cached_values}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
