from __future__ import annotations

from pathlib import Path
from typing import Any, Dict

from catmaster.tools.base import create_tool_output
from catmaster.tools.execution.dpdispatcher_runner import dispatch_task
from catmaster.tools.execution.resource_router import ResourceRouter
from catmaster.tools.execution.task_payloads import build_vasp_execute_request


def vasp_execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    from catmaster.tools.execution import VaspExecuteInput

    params = VaspExecuteInput(**payload)
    router = ResourceRouter()
    route = router.route("vasp_execute")

    dispatch_req = build_vasp_execute_request(params, route=route)
    result = dispatch_task(dispatch_req)

    return create_tool_output(
        tool_name="vasp_execute",
        success=True,
        data={
            "task_states": result.task_states,
            "download_path": result.output_dir,
            "submission_dir": result.submission_dir,
            "work_base": result.work_base,
        },
        execution_time=result.duration_s,
    )


__all__ = ["vasp_execute"]
