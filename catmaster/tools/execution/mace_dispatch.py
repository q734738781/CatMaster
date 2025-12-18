from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import dispatch_task
from catmaster.tools.execution.resource_router import ResourceRouter
from catmaster.tools.execution.task_payloads import build_mace_relax_request


def mace_relax(payload: Dict[str, Any]) -> Dict[str, Any]:
    from catmaster.tools.execution import MaceRelaxInput

    params = MaceRelaxInput(**payload)
    router = ResourceRouter()
    route = router.route("mace_relax")

    model = params.model or route.defaults.get("model")
    if not model:
        raise ValueError("model is required; set in payload or router defaults")

    structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
    work_dir = resolve_workspace_path(params.work_dir)
    work_dir.mkdir(parents=True, exist_ok=True)
    dest_structure = work_dir / structure_path.name
    if dest_structure.resolve() != structure_path:
        dest_structure.write_bytes(structure_path.read_bytes())

    params.model = model
    dispatch_req = build_mace_relax_request(params, route=route)
    result = dispatch_task(dispatch_req)

    return create_tool_output(
        tool_name="mace_relax",
        success=True,
        data={
            "task_states": result.task_states,
            "work_base": result.work_base,
            "relaxed_structure_rel": workspace_relpath(work_dir / "CONTCAR"),
            "trajectory_rel": workspace_relpath(work_dir / "opt.traj"),
            "optimization_log_rel": workspace_relpath(work_dir / "opt.log"),
            "summary_json_rel": workspace_relpath(work_dir / "summary.json"),
        },
        execution_time=result.duration_s,
    )


__all__ = ["mace_relax"]
