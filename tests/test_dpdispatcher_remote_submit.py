#!/usr/bin/env python3
"""
DPDispatcher deployment health check.
- Auto-resolves machine from resource when machine is not provided.
- Runs a minimal remote command and downloads a marker file.
- Dry-run by default; add --run to actually submit.
"""
from __future__ import annotations

import argparse
from pathlib import Path
from pprint import pprint
import time

from catmaster.tools.execution.dpdispatcher_runner import DispatchRequest, dispatch_task, make_work_base
from catmaster.tools.execution.machine_registry import MachineRegister


def _resolve_machine(resources: str, machine: str | None) -> str:
    reg = MachineRegister()
    if machine:
        reg.get_machine(machine)
        return machine
    res_cfg = reg.get_resources(resources)
    resolved = res_cfg.get("machine")
    if not resolved:
        raise KeyError(f"Resources '{resources}' does not bind to any machine")
    reg.get_machine(str(resolved))
    return str(resolved)


def build_demo_request(*, resources: str, machine: str | None, workspace: str, check_interval: int) -> tuple[DispatchRequest, str, str]:
    """Build a health-check request that writes hostname/date into a marker file."""
    local_root = Path(workspace).expanduser().resolve()
    local_root.mkdir(parents=True, exist_ok=True)

    resolved_machine = _resolve_machine(resources, machine)
    work_base = make_work_base("dp_health")
    task_work_path = "."
    marker = f"dpdispatcher_health_{int(time.time())}.txt"

    # Write a small marker file so deployment health can be verified by download.
    command = f"bash -lc 'set -e; hostname > {marker}; date -Is >> {marker}; echo DPDISPATCHER_OK >> {marker}'"

    req = DispatchRequest(
        machine=resolved_machine,
        resources=resources,
        command=command,
        work_base=work_base,
        task_work_path=task_work_path,
        forward_files=[],
        backward_files=[marker],
        local_root=str(local_root),
        check_interval=check_interval,
    )
    return req, resolved_machine, marker


def main() -> None:
    parser = argparse.ArgumentParser(description="DPDispatcher deployment health check")
    parser.add_argument("--resource", default="vasp_cpu", help="Resource key in configs/dpdispatcher/resources.yaml")
    parser.add_argument("--machine", default=None, help="Optional machine override")
    parser.add_argument("--workspace", default="workspace/test_dpdispatcher_remote_submit", help="Local workspace root")
    parser.add_argument("--check-interval", type=int, default=20, help="Polling interval seconds")
    parser.add_argument("--run", action="store_true", help="Actually submit; default prints payload only")
    args = parser.parse_args()

    req, resolved_machine, marker = build_demo_request(
        resources=args.resource,
        machine=args.machine,
        workspace=args.workspace,
        check_interval=args.check_interval,
    )
    print(f"Resolved machine: {resolved_machine}")
    print("Planned request:")
    pprint(req.model_dump())

    if not args.run:
        print("Dry-run only. Use --run to submit via DPDispatcher.")
        return

    print("\nSubmitting health-check task via DPDispatcher...")
    result = dispatch_task(req)
    marker_path = Path(result.output_dir) / marker
    if not marker_path.exists():
        raise RuntimeError(f"Health marker not downloaded: {marker_path}")

    print("\nSubmission finished. Result:")
    pprint(result.model_dump())
    print(f"\nHealth marker downloaded: {marker_path}")
    print("Marker content:")
    print(marker_path.read_text(encoding="utf-8", errors="ignore"))


if __name__ == "__main__":
    main()
