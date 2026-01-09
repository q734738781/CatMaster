#!/usr/bin/env python3
"""
LLM-driven O2-in-the-box workflow demo using GPT-5.1 + DPDispatcher.

Flow (file-first):
1) Let the LLM create an initial POSCAR (or equivalent) using available tools.
2) Let the orchestrator plan/execute using registered tools:
   - relax_prepare (local)
   - vasp_execute (DPDispatcher)
   - vasp_summarize (local)
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
from langchain_openai import ChatOpenAI
from catmaster.agents.orchestrator import Orchestrator


def main() -> None:
    parser = argparse.ArgumentParser(description="LLM demo: O2 Fe adsorption energy calculation")
    parser.add_argument("--workspace", default="workspace/demo_llm_fe_ads_energy", help="Workspace root")
    parser.add_argument("--run", action="store_true", help="Actually submit vasp_execute; otherwise quit")
    parser.add_argument("--log-level", default="INFO", help="Logging level (INFO or DEBUG)")
    parser.add_argument("--log-dir", default=None, help="Directory to store logs (log.log + orchestrator_llm.jsonl)")
    parser.add_argument("--proxy", default=None, help="Proxy server address expressed as <host>:<port>")
    parser.add_argument("--resume", default=None, help="Run directory to resume from")
    args = parser.parse_args()

    handlers = [logging.StreamHandler()]
    if args.proxy:
        print(f"Using proxy: {args.proxy}")
        
        host, port = args.proxy.split(":")
        os.environ["HTTP_PROXY"] = f"http://{host}:{port}"
        os.environ["HTTPS_PROXY"] = f"http://{host}:{port}"
        os.environ["SOCKS_PROXY"] = f"socks5://{host}:{port}"
    
    log_dir_path = Path(args.log_dir).expanduser().resolve() if args.log_dir else None
    if log_dir_path:
        # Remove the directory if it exists
        if log_dir_path.exists():
            shutil.rmtree(log_dir_path)
        log_dir_path.mkdir(parents=True, exist_ok=True)
        log_path = log_dir_path / "log.log"
        handlers.append(logging.FileHandler(log_path))
    logging.basicConfig(
        level=getattr(logging, args.log_level.upper(), logging.INFO),
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        handlers=handlers,
    )

    root = Path(args.workspace).resolve()
    # Export to CATMASTER_WORKSPACE (tools should respect this workspace)
    os.environ["CATMASTER_WORKSPACE"] = str(root)
    # Clean and Make dir for workspace
    if not args.resume:
        if root.exists():
            shutil.rmtree(root)
        root.mkdir(parents=True, exist_ok=True)

    user_request = (
        "Compute CO adsorption energy on BCC Fe surface for different surfaces and adsorption sites:"
        "Download the mp-150 structure from Materials Project (Fe bcc phase) and use it as the initial structure."
        "Use a 2x2 supercell for the slab, with 10A slab thickness and 15A vacuum thickness, fix the bottom 3 layers of the slab."
        "Enumerate the adsorption sites on the slab's (100), (110), and (111) surfaces, and place the CO molecule on each possible site."
        "Use the following formula to calculate the adsorption energy: E_ads = E_(Fe-CO) - E_(Fe) - E_(CO)"
        "Perform VASP calculation to get the results. Summarize the results in a markdown file, report the adsorption energy for each adsorption site on different surfaces and report the most stable adsorption site."
    )

    llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0,
        # model_kwargs={"response_format": {"type": "json_object"}},
        reasoning_effort="high",

    )
    summary_llm = ChatOpenAI(
        model="gpt-5.2",
        temperature=0,
        reasoning_effort="medium",
    ) # Use natural language response format

    orch = Orchestrator(
        llm=llm,
        summary_llm=summary_llm,
        max_steps=300,
        llm_log_path=str(log_dir_path / "orchestrator_llm.jsonl") if log_dir_path else None,
        log_llm_console=True,
        resume_dir=args.resume,
    )

    if not args.run:
        print("--run not supplied; refusing to submit. Inspect generated POSCAR and rerun with --run when ready.")
        return

    result = orch.run(
        user_request,
        log_llm=True,
    )

    print("\n=== Summary ===")
    print(result.get("summary", "No summary"))
    if "final_answer" in result:
        print("\n=== LLM Final Answer ===")
        print(result.get("final_answer", "No final answer"))
    print("\nObservations:")
    for obs in result.get("observations", []):
        print(obs)
    if log_dir_path:
        print(f"\nLogs saved to: {log_dir_path}")
        print("  - app log: log.log")
        print("  - LLM trace: orchestrator_llm.jsonl")


if __name__ == "__main__":
    main()
