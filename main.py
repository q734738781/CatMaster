#!/usr/bin/env python3
"""
User-friendly entry point for CatMaster LLM runs.
"""
from __future__ import annotations

import argparse
import logging
import os
import shutil
import sys
from pathlib import Path

from langchain_openai import ChatOpenAI

from catmaster.agents.orchestrator import Orchestrator
from catmaster.ui import create_reporter


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise SystemExit("Provide --prompt or --prompt-file.")


def _build_llm(args: argparse.Namespace, model: str) -> ChatOpenAI:
    llm_kwargs: dict = {"model": model}
    if args.reasoning_effort is not None:
        llm_kwargs["reasoning_effort"] = args.reasoning_effort
    return ChatOpenAI(**llm_kwargs)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatMaster entry point")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="User prompt as a string")
    prompt_group.add_argument("--prompt-file", help="Path to a text file containing the prompt")

    # LLM API calls
    parser.add_argument("--model", required=True, help="LLM model name")
    parser.add_argument("--reasoning-effort", default=None)

    # Workspace
    parser.add_argument("--workspace", default=None, help="Workspace root (or set CATMASTER_WORKSPACE)")
    parser.add_argument("--clean", action="store_true", help="Delete workspace before running")
    parser.add_argument("--resume", action="store_true", help="Resume from existing workspace")

    # Orchestrator detail settings
    parser.add_argument("--max-steps", type=int, default=100)
    parser.add_argument("--max-tool-attempts", type=int, default=10)
    parser.add_argument("--max-plan-attempts", type=int, default=10)
    parser.add_argument("--patch-repair-attempts", type=int, default=1)
    parser.add_argument("--summary-repair-attempts", type=int, default=1)
    parser.add_argument("--no-plan-review", action="store_true", help="Disable plan review")

    # Logging. Default disabled
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-llm", action="store_true", help="Log LLM prompts/responses")

    # Proxy and UI
    parser.add_argument("--proxy", default=None, help="Proxy as <host>:<port> for LLM API calls")
    parser.add_argument("--ui", choices=["rich", "plain", "off"], default=None)
    parser.add_argument("--ui-debug", action="store_true")
    parser.add_argument("--no-splash", action="store_true")

    args = parser.parse_args()

    if args.clean and args.resume:
        raise SystemExit("--clean and --resume cannot be used together.")

    prompt = _load_prompt(args)

    workspace = None
    if args.workspace:
        workspace = Path(args.workspace).expanduser().resolve()
    elif os.environ.get("CATMASTER_WORKSPACE"):
        workspace = Path(os.environ["CATMASTER_WORKSPACE"]).expanduser().resolve()
    else:
        raise SystemExit("Provide --workspace or set CATMASTER_WORKSPACE.")

    if args.proxy:
        host, port = args.proxy.split(":", 1)
        os.environ["HTTP_PROXY"] = f"http://{host}:{port}"
        os.environ["HTTPS_PROXY"] = f"http://{host}:{port}"
        os.environ["SOCKS_PROXY"] = f"socks5://{host}:{port}"

    handlers: list[logging.Handler] = []
    ui_mode = args.ui or ("rich" if sys.stdout.isatty() else "plain")
    if ui_mode == "off":
        handlers.append(logging.StreamHandler())
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "log.log"))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers or None,
    )

    reporter = create_reporter(
        ui_mode,
        ui_debug=args.ui_debug,
        show_splash=not args.no_splash,
        is_tty=sys.stdout.isatty(),
    )

    if args.clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["CATMASTER_WORKSPACE"] = str(workspace)

    llm = _build_llm(args, args.model)

    orch_kwargs: dict = {
        "llm": llm,
        "reporter": reporter,
        "log_llm_console": ui_mode == "off",
        "resume": args.resume,
    }
    if args.log_dir:
        orch_kwargs["llm_log_path"] = str(Path(args.log_dir).expanduser().resolve() / "orchestrator_llm.jsonl")
    if args.max_steps is not None:
        orch_kwargs["max_steps"] = args.max_steps
    if args.max_tool_attempts is not None:
        orch_kwargs["max_tool_attempts"] = args.max_tool_attempts
    if args.max_plan_attempts is not None:
        orch_kwargs["max_plan_attempts"] = args.max_plan_attempts
    if args.patch_repair_attempts is not None:
        orch_kwargs["patch_repair_attempts"] = args.patch_repair_attempts
    if args.summary_repair_attempts is not None:
        orch_kwargs["summary_repair_attempts"] = args.summary_repair_attempts

    orch = Orchestrator(**orch_kwargs)

    orch.run(
        prompt,
        log_llm=args.log_llm,
        plan_review=not args.no_plan_review,
    )


if __name__ == "__main__":
    main()
