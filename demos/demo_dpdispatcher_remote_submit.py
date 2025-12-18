#!/usr/bin/env python3
"""
Minimal DPDispatcher submission demo.
- Submits a trivial command (hostname + date) to the configured machine/resource.
- Shows how to poll status and locate downloaded files.
- Safe by default; only runs when invoked as a script.
"""
from __future__ import annotations

from pathlib import Path
from pprint import pprint

from catmaster.tools.execution.dpdispatcher_runner import DispatchRequest, dispatch_task, make_work_base


def build_demo_request() -> DispatchRequest:
    """Build a simple demo request that just records hostname/date on the remote."""
    local_root = Path.cwd() / "workspace" / "demo_workspace"
    local_root.mkdir(parents=True, exist_ok=True)

    work_base = make_work_base("demo")
    task_work_path = "."

    # command writes a marker file we can fetch back
    command = "bash -lc 'hostname && date > demo_status.txt'"

    return DispatchRequest(
        machine="cpu_hpc",
        resources="vasp_cpu",
        command=command,
        work_base=work_base,
        task_work_path=task_work_path,
        forward_files=[],
        backward_files=["demo_status.txt"],
        local_root=str(local_root),
        check_interval=20,
    )


def main() -> None:
    req = build_demo_request()
    print("Submitting demo task via DPDispatcher…")
    pprint(req.model_dump())

    result = dispatch_task(req)
    print("\nSubmission finished. Result:")
    pprint(result.model_dump())
    print("\nDownloaded files will be under:", result.output_dir)


if __name__ == "__main__":
    main()
