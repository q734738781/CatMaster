"""Live Codex OAuth GPT-5.6 apply_patch protocol and concurrency acceptance test.

This test intentionally makes real model calls. Run it from the repository root:

    env -u ALL_PROXY PYTHONPATH=. \
      /home/chenhh/miniconda3/envs/catmaster/bin/python \
      tests/manual/codex_oauth_apply_patch_live.py --workers 3
"""

from __future__ import annotations

import argparse
import asyncio
import json
import tempfile
import time
from pathlib import Path
from typing import Any

from langchain.agents import create_agent
from langchain_core.messages import HumanMessage, ToolMessage

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.native_apply_patch import build_native_apply_patch_tool
from catmaster.specialists.runtime import SpecialistRunner


def _tool_outputs(messages: list[Any]) -> list[str]:
    outputs: list[str] = []
    for message in messages:
        if not isinstance(message, ToolMessage):
            continue
        for block in message.content if isinstance(message.content, list) else []:
            if isinstance(block, dict) and block.get("type") == "custom_tool_call_output":
                outputs.append(str(block.get("output") or ""))
    return outputs


def _tool_call_count(messages: list[Any]) -> int:
    return sum(
        len(list(getattr(message, "tool_calls", None) or []))
        for message in messages
    )


async def _run_worker(index: int, cfg: Any) -> dict[str, Any]:
    root = Path(tempfile.mkdtemp(prefix=f"catmaster-oauth-patch-{index:02d}-"))
    started = time.perf_counter()
    model = build_chat_model(cfg)
    agent = create_agent(
        model=model,
        tools=[build_native_apply_patch_tool(files_root=root)],
        middleware=SpecialistRunner._build_default_middleware(),
        system_prompt=(
            "Use exactly one apply_patch call for each requested edit batch. "
            "After its result, answer only DONE or report the exact tool error."
        ),
    )

    try:
        first = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": (
                            f"Batch {index}: in one apply_patch call create base/a.txt with "
                            f"alpha-{index}, base/b.txt with beta-{index}, and base/c.txt "
                            f"with gamma-{index}. Each file must end with one newline."
                        ),
                    }
                ]
            },
            config={"recursion_limit": 8},
        )
        first_messages = list(first["messages"])

        second = await agent.ainvoke(
            {
                "messages": [
                    *first_messages,
                    HumanMessage(
                        content=(
                            "In one apply_patch call: change base/a.txt from alpha-"
                            f"{index} to alpha-{index}-updated; move base/b.txt to "
                            "moved/b.txt without changing its content; delete base/c.txt; "
                            f"and add base/d.txt containing delta-{index}. Preserve one "
                            "trailing newline in every remaining file."
                        )
                    ),
                ]
            },
            config={"recursion_limit": 8},
        )
        messages = list(second["messages"])
    except Exception as exc:
        return {
            "worker": index,
            "root": str(root),
            "elapsed_s": round(time.perf_counter() - started, 3),
            "passed": False,
            "error_type": type(exc).__name__,
            "error": str(exc),
        }
    elapsed = time.perf_counter() - started

    expected = {
        "base/a.txt": f"alpha-{index}-updated\n",
        "base/d.txt": f"delta-{index}\n",
        "moved/b.txt": f"beta-{index}\n",
    }
    actual = {
        str(path.relative_to(root)): path.read_text(encoding="utf-8")
        for path in sorted(root.rglob("*.txt"))
    }
    outputs = _tool_outputs(messages)
    return {
        "worker": index,
        "root": str(root),
        "elapsed_s": round(elapsed, 3),
        "tool_call_count": _tool_call_count(messages),
        "tool_outputs": outputs,
        "actual_files": actual,
        "expected_files": expected,
        "passed": actual == expected
        and len(outputs) == 2
        and all(output.startswith("Done!") for output in outputs),
    }


async def _main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument("--config", default="configs/llm.yaml")
    args = parser.parse_args()
    if args.workers < 1:
        parser.error("--workers must be positive")

    profile = LLMProfile.from_env_or_file(args.config)
    cfg = profile.config_for_role("task_runner")
    if str(cfg.provider).strip().lower() != "codex_oauth":
        raise RuntimeError(
            f"task_runner must use provider=codex_oauth, got {cfg.provider!r}"
        )

    started = time.perf_counter()
    results = await asyncio.gather(
        *(_run_worker(index, cfg) for index in range(args.workers))
    )
    elapsed = time.perf_counter() - started

    payload = {
        "provider": cfg.provider,
        "model": cfg.model,
        "workers": args.workers,
        "wall_elapsed_s": round(elapsed, 3),
        "passed": all(result["passed"] for result in results),
        "results": results,
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0 if payload["passed"] else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(_main()))
