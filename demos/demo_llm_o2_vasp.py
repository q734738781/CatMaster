#!/usr/bin/env python3
"""
LLM-driven O2-in-the-box workflow demo using GPT-5.1 + DPDispatcher.

Flow (file-first):
1) Let the LLM create an initial POSCAR (or equivalent) using available tools.
2) Let the orchestrator plan/execute using registered tools:
   - relax_prepare (local)
   - vasp_execute (DPDispatcher)
3) Report final energy per atom and O–O bond distance.

Use --run to actually submit to the configured cluster; default is dry-run
stub for vasp_execute.
"""
# Add parent dir to sys.path
from __future__ import annotations
import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parents[1]))
import argparse
import logging
import os
import shutil
from catmaster.agents.orchestrator import Orchestrator
from catmaster.ui import create_reporter



def main() -> None:
    parser = argparse.ArgumentParser(description="LLM DPDispatcher demo: O2 VASP run")
    parser.add_argument("--workspace", default="workspace/demo_llm_o2_vasp", help="Workspace root")
    parser.add_argument("--run", action="store_true", help="Actually submit vasp_execute; otherwise quit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO or DEBUG)")
    parser.add_argument("--log-dir", default=None, help="Directory to store logs (log.log + orchestrator_llm.jsonl)")
    parser.add_argument("--proxy", default=None, help="Proxy server address expressed as <host>:<port>")
    parser.add_argument("--resume", action="store_true", help="Resume from existing workspace (do not clear)")
    parser.add_argument("--ui", choices=["rich", "plain", "off"], default=None, help="UI mode (default: rich if TTY else plain)")
    parser.add_argument("--ui-debug", action="store_true", help="Show UI debug panel with LLM snippets/paths")
    parser.add_argument("--no-splash", action="store_true", help="Disable splash screen")
    args = parser.parse_args()

    ui_mode = args.ui or ("rich" if sys.stdout.isatty() else "plain")
    handlers = []
    if ui_mode == "off":
        handlers.append(logging.StreamHandler())
    if args.proxy:
        print(f"Using proxy: {args.proxy}")
        host, port = args.proxy.split(":")
        os.environ["HTTP_PROXY"] = f"http://{host}:{port}"
        os.environ["HTTPS_PROXY"] = f"http://{host}:{port}"
        os.environ["SOCKS_PROXY"] = f"socks5://{host}:{port}"
    
    log_dir_path = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    if log_dir_path:
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_path = log_dir_path / "log.log"
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )
    reporter = create_reporter(
        ui_mode,
        ui_debug=args.ui_debug,
        show_splash=not args.no_splash,
        is_tty=sys.stdout.isatty(),
    )

    root = Path(args.workspace).resolve()
    # Export to CATMASTER_WORKSPACE (tools should respect this workspace)
    os.environ["CATMASTER_WORKSPACE"] = str(root)
    # Clean and Make dir (skip if resuming)
    if not args.resume:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    user_request = (
        "Compute O2 in a box: prepare VASP inputs from scratch,"
        "DO perform VASP calculation to get the results, and report final energy per atom and O–O bond distance from vasp results."
    )

    orch = Orchestrator(
        max_steps=100,
        llm_log_path=str(log_dir_path / "orchestrator_llm.jsonl") if log_dir_path else None,
        log_llm_console=ui_mode == "off",
        resume=args.resume,
        reporter=reporter,
    )

    if not args.run:
        print("--run not supplied; refusing to submit. Inspect generated POSCAR and rerun with --run when ready.")
        return

    result = orch.run(
        user_request,
        log_llm=True,
    )

    _ = result


if __name__ == "__main__":
    main()
