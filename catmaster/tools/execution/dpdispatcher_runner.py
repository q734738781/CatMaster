from __future__ import annotations

import os
import time
import uuid
from pathlib import Path
from typing import Dict, List, Optional

import yaml
from pydantic import BaseModel, Field

from dpdispatcher import Machine, Resources, Task, Submission
from catmaster.tools.execution.machine_registry import MachineRegister


class DispatchRequest(BaseModel):
    """Parameters required to submit a task via DPDispatcher."""

    machine: str = Field(..., description="Machine key in dpdispatcher config")
    resources: str = Field(..., description="Resource preset key in config")
    command: str = Field(..., description="Command to execute on remote host")
    work_base: str = Field(..., description="Work base relative to machine local_root/remote_root")
    task_work_path: str = Field(".", description="Task work path inside work_base")
    forward_files: List[str] = Field(default_factory=list, description="Files to upload for this task")
    backward_files: List[str] = Field(default_factory=list, description="Files to fetch back after completion")
    forward_common_files: List[str] = Field(default_factory=list, description="Common files uploaded for all tasks")
    backward_common_files: List[str] = Field(default_factory=list, description="Common files downloaded after completion")
    local_root: str = Field(..., description="Local root directory used by DPDispatcher")
    wait: bool = Field(True, description="Wait for task completion")
    clean_remote: bool = Field(False, description="Remove remote work dir after download")
    check_interval: int = Field(30, description="Polling interval seconds when waiting")


class DispatchResult(BaseModel):
    """Result of a dispatch submission."""

    work_base: str
    local_root: str
    output_dir: str
    task_states: List[str]
    submission_dir: str
    duration_s: float


class TaskSpec(BaseModel):
    """Lightweight Task spec for batch submissions."""

    command: str
    task_work_path: str = "."
    forward_files: List[str] = Field(default_factory=list)
    backward_files: List[str] = Field(default_factory=list)


class BatchDispatchRequest(BaseModel):
    """Bundle multiple Tasks into a single Submission."""

    machine: str
    resources: str
    work_base: str
    local_root: str
    tasks: List[TaskSpec]
    forward_common_files: List[str] = Field(default_factory=list)
    backward_common_files: List[str] = Field(default_factory=list)
    clean_remote: bool = Field(False, description="Remove remote work dir after download")
    check_interval: int = Field(30, description="Polling interval seconds when waiting")


def _build_machine(cfg: Dict, local_root: Path) -> Machine:
    cfg = dict(cfg)
    cfg["local_root"] = str(local_root)
    try:
        return Machine.load_from_dict(cfg)
    except AttributeError:
        return Machine(**cfg)


def _build_resources(cfg: Dict, env_setup: Optional[str]) -> Resources:
    cfg = dict(cfg)
    cfg.pop("machine", None)
    prepend = cfg.get("prepend_script", [])
    if isinstance(prepend, str):
        prepend = [prepend]
    if env_setup:
        env_lines = [ln for ln in env_setup.splitlines() if ln.strip()]
        prepend = env_lines + prepend
    if prepend:
        cfg["prepend_script"] = prepend
    try:
        return Resources.load_from_dict(cfg)
    except AttributeError:
        return Resources(**cfg)


def _task_state(task: Task) -> str:
    for attr in ("task_state", "state", "status"):
        if hasattr(task, attr):
            val = getattr(task, attr)
            return str(val)
    return "unknown"


def dispatch_task(request: DispatchRequest, *, config_path: Optional[str] = None, register: Optional[MachineRegister] = None) -> DispatchResult:
    """Submit a single task through DPDispatcher and wait for completion."""

    reg = register or MachineRegister(extra_paths=[Path(config_path)]) if config_path else MachineRegister()

    machine_cfg = reg.get_machine(request.machine)
    res_cfg = reg.get_resources(request.resources)

    local_root = Path(request.local_root).expanduser().resolve()
    env_setup = machine_cfg.get("env_setup")

    machine = _build_machine(machine_cfg, local_root)
    resources = _build_resources(res_cfg, env_setup)

    task = Task(
        command=request.command,
        task_work_path=request.task_work_path,
        forward_files=request.forward_files,
        backward_files=request.backward_files,
    )

    task_list = [task]

    submission = Submission(
        work_base=request.work_base,
        machine=machine,
        resources=resources,
        task_list=task_list,
        forward_common_files=request.forward_common_files,
        backward_common_files=request.backward_common_files,
    )

    t0 = time.time()
    submission.run_submission(clean=request.clean_remote, check_interval=request.check_interval)
    duration = time.time() - t0

    states = [_task_state(t) for t in task_list]
    output_dir = Path(machine.context.init_local_root) / request.work_base / request.task_work_path

    return DispatchResult(
        work_base=request.work_base,
        local_root=str(machine.context.init_local_root),
        output_dir=str(output_dir.resolve()),
        task_states=states,
        submission_dir=str(Path(machine.context.init_local_root) / request.work_base),
        duration_s=duration,
    )


def dispatch_submission(
    batch: BatchDispatchRequest,
    *,
    register: Optional[MachineRegister] = None,
    config_path: Optional[str] = None,
) -> DispatchResult:
    """Submit multiple tasks as a single Submission."""
    reg = register or MachineRegister(extra_paths=[Path(config_path)]) if config_path else MachineRegister()

    machine_cfg = reg.get_machine(batch.machine)
    res_cfg = reg.get_resources(batch.resources)

    local_root = Path(batch.local_root).expanduser().resolve()
    env_setup = machine_cfg.get("env_setup")

    machine = _build_machine(machine_cfg, local_root)
    resources = _build_resources(res_cfg, env_setup)

    task_list = [
        Task(
            command=t.command,
            task_work_path=t.task_work_path,
            forward_files=t.forward_files,
            backward_files=t.backward_files,
        )
        for t in batch.tasks
    ]

    submission = Submission(
        work_base=batch.work_base,
        machine=machine,
        resources=resources,
        task_list=task_list,
        forward_common_files=batch.forward_common_files,
        backward_common_files=batch.backward_common_files,
    )

    t0 = time.time()
    submission.run_submission(clean=batch.clean_remote, check_interval=batch.check_interval)
    duration = time.time() - t0

    states = [_task_state(t) for t in task_list]
    # If all tasks share the same task_work_path, surface that in output_dir
    work_suffixes = {t.task_work_path for t in batch.tasks}
    suffix = work_suffixes.pop() if len(work_suffixes) == 1 else ""
    output_dir = Path(machine.context.init_local_root) / batch.work_base / suffix

    return DispatchResult(
        work_base=batch.work_base,
        local_root=str(machine.context.init_local_root),
        output_dir=str(output_dir.resolve()),
        task_states=states,
        submission_dir=str(Path(machine.context.init_local_root) / batch.work_base),
        duration_s=duration,
    )


def make_work_base(prefix: str = "cm") -> str:
    """Generate a short unique work_base name."""
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
