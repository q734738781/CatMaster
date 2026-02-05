#!/usr/bin/env python3
"""
User-friendly entry point for CatMaster LLM runs.
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import shutil
from pathlib import Path

from catmaster.agents.orchestrator import Orchestrator
from catmaster.llm.config import LLMProfile
from catmaster.ui import NullReporter


def _load_prompt(args: argparse.Namespace) -> str:
    if args.prompt_file:
        return Path(args.prompt_file).read_text(encoding="utf-8")
    if args.prompt:
        return args.prompt
    raise SystemExit("Provide --prompt or --prompt-file.")


def main() -> None:
    parser = argparse.ArgumentParser(description="CatMaster entry point")
    prompt_group = parser.add_mutually_exclusive_group(required=True)
    prompt_group.add_argument("--prompt", help="User prompt as a string")
    prompt_group.add_argument("--prompt-file", help="Path to a text file containing the prompt")

    # LLM config (loaded from YAML/env by default)
    parser.add_argument("--llm-config", default=None, help="Path to LLM config YAML (default: configs/llm.yaml)")

    # Workspace
    parser.add_argument("--workspace", default=None, help="Workspace root (or set CATMASTER_WORKSPACE)")
    parser.add_argument("--clean", action="store_true", help="Delete workspace before running")
    parser.add_argument("--resume", action="store_true", help="Resume from existing workspace")

    # Orchestrator detail settings
    parser.add_argument("--max-steps", type=int, default=200)
    parser.add_argument("--max-tool-attempts", type=int, default=3)
    parser.add_argument("--max-plan-steps", type=int, default=50)
    parser.add_argument("--patch-repair-attempts", type=int, default=1)
    parser.add_argument("--summary-repair-attempts", type=int, default=1)
    parser.add_argument("--no-plan-review", action="store_true", help="Disable plan review (required for non-interactive CLI)")
    parser.add_argument("--lane", choices=["fast", "standard"], required=True, help="Execution lane")
    parser.add_argument("--full-auto-major", action="store_true", help="Auto-accept major proposal revisions (standard lane)")

    # Logging. Default disabled
    parser.add_argument("--log-level", default="INFO")
    parser.add_argument("--log-dir", default=None)
    parser.add_argument("--log-llm", action="store_true", help="Log LLM prompts/responses")

    # Proxy
    parser.add_argument("--proxy", default=None, help="Proxy as <host>:<port> for LLM API calls")

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

    handlers: list[logging.Handler] = [logging.StreamHandler()]
    if args.log_dir:
        log_dir = Path(args.log_dir).expanduser().resolve()
        log_dir.mkdir(parents=True, exist_ok=True)
        handlers.append(logging.FileHandler(log_dir / "log.log"))

    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers or None,
    )

    reporter = NullReporter()

    if args.clean and workspace.exists():
        shutil.rmtree(workspace)
    workspace.mkdir(parents=True, exist_ok=True)
    os.environ["CATMASTER_WORKSPACE"] = str(workspace)

    llm_profile = LLMProfile.from_env_or_file(args.llm_config)

    orch_kwargs: dict = {
        "llm_profile": llm_profile,
        "reporter": reporter,
        "log_llm_console": False,
        "resume": args.resume,
    }
    if args.resume:
        active_runs_path = workspace / ".catmaster" / "active_runs.json"
        if active_runs_path.exists():
            try:
                active_runs = json.loads(active_runs_path.read_text(encoding="utf-8"))
            except Exception:
                active_runs = {}
            if isinstance(active_runs, dict):
                lane_run = active_runs.get(args.lane)
                if lane_run:
                    candidate = Path(lane_run)
                    if not candidate.is_absolute():
                        candidate = (workspace / ".catmaster" / lane_run).resolve()
                    orch_kwargs["resume_dir"] = str(candidate)
    if args.log_dir:
        orch_kwargs["llm_log_path"] = str(Path(args.log_dir).expanduser().resolve() / "orchestrator_llm.jsonl")
    if args.max_steps is not None:
        orch_kwargs["max_steps"] = args.max_steps
    if args.max_tool_attempts is not None:
        orch_kwargs["max_tool_attempts"] = args.max_tool_attempts
    if args.max_plan_steps is not None:
        orch_kwargs["max_plan_steps"] = args.max_plan_steps
    if args.patch_repair_attempts is not None:
        orch_kwargs["patch_repair_attempts"] = args.patch_repair_attempts
    if args.summary_repair_attempts is not None:
        orch_kwargs["summary_repair_attempts"] = args.summary_repair_attempts

    orch = Orchestrator(**orch_kwargs)

    try:
        sys_root = workspace / ".catmaster"
        sys_root.mkdir(parents=True, exist_ok=True)
        active_runs_path = sys_root / "active_runs.json"
        try:
            active_runs = json.loads(active_runs_path.read_text(encoding="utf-8"))
        except Exception:
            active_runs = {}
        if not isinstance(active_runs, dict):
            active_runs = {}
        try:
            rel_run = orch.run_context.run_dir.relative_to(sys_root)
            active_runs[args.lane] = str(rel_run)
        except Exception:
            active_runs[args.lane] = str(orch.run_context.run_dir)
        active_runs_path.write_text(json.dumps(active_runs, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

    if not args.no_plan_review:
        raise SystemExit("Console UI is removed. Use WebUI for plan review or pass --no-plan-review.")

    orch.run(
        prompt,
        log_llm=args.log_llm,
        plan_review=False,
        lane=args.lane,
        full_auto_major=args.full_auto_major,
    )


if __name__ == "__main__":
    main()
