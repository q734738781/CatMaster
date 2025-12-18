from __future__ import annotations

"""LLM-facing adapter that hides resource selection and submission mechanics."""

from pathlib import Path
from typing import Any, Dict

from catmaster.tools.execution.dpdispatcher_runner import dispatch_task
from catmaster.tools.execution.resource_router import ResourceRouter
from catmaster.tools.execution.task_payloads import build_mace_relax_request, build_vasp_execute_request


class DPDispatcherAdapter:
    def __init__(self, router: ResourceRouter | None = None):
        self.router = router or ResourceRouter()

    def submit(self, task_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task_name == "mace_relax":
            return self._submit_mace(payload)
        if task_name == "vasp_execute":
            return self._submit_vasp(payload)
        raise ValueError(f"Unsupported task {task_name}")

    def _submit_mace(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from catmaster.tools.execution import MaceRelaxInput
        params = MaceRelaxInput(**payload)
        route = self.router.route("mace_relax")
        params.model = params.model or route.defaults.get("model")
        req = build_mace_relax_request(params, route=route)
        res = dispatch_task(req)
        return {
            "task_states": res.task_states,
            "download_path": res.output_dir,
            "submission_dir": res.submission_dir,
            "work_base": res.work_base,
        }

    def _submit_vasp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        from catmaster.tools.execution import VaspExecuteInput
        params = VaspExecuteInput(**payload)
        route = self.router.route("vasp_execute")
        req = build_vasp_execute_request(params, route=route)
        res = dispatch_task(req)
        return {
            "task_states": res.task_states,
            "download_path": res.output_dir,
            "submission_dir": res.submission_dir,
            "work_base": res.work_base,
        }


__all__ = ["DPDispatcherAdapter"]
