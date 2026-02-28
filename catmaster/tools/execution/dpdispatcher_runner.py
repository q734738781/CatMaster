from __future__ import annotations

import shlex
import time
import uuid
import json
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
import re
from dpdispatcher import Machine, Resources, Task, Submission
from catmaster.tools.execution.machine_registry import MachineRegister

STATUS_FILE_NAME = ".catmaster_status.json"
STDOUT_FILE_NAME = ".catmaster_stdout.log"
STDERR_FILE_NAME = ".catmaster_stderr.log"

_DP_PATTERNS = [
    re.compile(r"^[0-9a-f]{40}_task_tag_finished$"),
    re.compile(r"^[0-9a-f]{40}_job_tag_finished$"),
    re.compile(r"^[0-9a-f]{40}_job_id$"),
    re.compile(r"^[0-9a-f]{40}_flag_if_job_task_fail$"),
    re.compile(r"^[0-9a-f]{40}_last_err_file$"),
    re.compile(r"^[0-9a-f]{40}_job\.json$"),
    re.compile(r"^[0-9a-f]{40}\.sub$"),
    re.compile(r"^[0-9a-f]{40}\.sub\.run$"),
    re.compile(r"^[0-9a-f]{40}\.json$"),
    re.compile(r"^tag_failure_download_.*$"),
]

def _is_dp_artifact(name: str) -> bool:
    for pat in _DP_PATTERNS:
        if pat.match(name):
            return True
    return False

def cleanup_dpdispatcher_artifacts(work_base_dir: Path) -> None:
    if not work_base_dir.exists():
        return 0

    for p in work_base_dir.rglob("*"):
        if not p.is_file():
            continue
        if not _is_dp_artifact(p.name):
            continue
        try:
            p.unlink()
        except Exception:
            continue


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


def _ensure_status_backward_files(files: List[str]) -> List[str]:
    out = list(files or [])
    for required in (STATUS_FILE_NAME, STDOUT_FILE_NAME, STDERR_FILE_NAME):
        if required not in out:
            out.append(required)
    return out


def _wrap_command_for_dpdispatcher(command: str) -> str:
    status_file_repr = repr(STATUS_FILE_NAME)
    py_script = (
        "import json\n"
        "import os\n"
        "import pathlib\n"
        "import time\n"
        "def _read_tail(path: str, max_chars: int = 4000) -> str:\n"
        "    if not path:\n"
        "        return ''\n"
        "    p = pathlib.Path(path)\n"
        "    if not p.is_file():\n"
        "        return ''\n"
        "    try:\n"
        "        text = p.read_text(encoding='utf-8', errors='replace')\n"
        "    except Exception:\n"
        "        return ''\n"
        "    if len(text) <= max_chars:\n"
        "        return text\n"
        "    return text[-max_chars:]\n"
        "rc_raw = os.environ.get('__CM_RC', '-999').strip()\n"
        "try:\n"
        "    rc = int(rc_raw)\n"
        "except Exception:\n"
        "    rc = -999\n"
        "def _to_float(name: str) -> float:\n"
        "    raw = os.environ.get(name, '').strip()\n"
        "    try:\n"
        "        return float(raw)\n"
        "    except Exception:\n"
        "        return time.time()\n"
        "data = {\n"
        "    'returncode': rc,\n"
        "    'command': os.environ.get('__CM_COMMAND', ''),\n"
        "    'cwd': os.getcwd(),\n"
        "    't_start': _to_float('__CM_T0'),\n"
        "    't_end': _to_float('__CM_T1'),\n"
        "    'stdout_file': os.environ.get('__CM_STDOUT', ''),\n"
        "    'stderr_file': os.environ.get('__CM_STDERR', ''),\n"
        "    'stdout_tail': _read_tail(os.environ.get('__CM_STDOUT', '')),\n"
        "    'stderr_tail': _read_tail(os.environ.get('__CM_STDERR', '')),\n"
        "}\n"
        f"pathlib.Path({status_file_repr}).write_text(json.dumps(data, indent=2) + '\\n', encoding='utf-8')\n"
    )
    script = "\n".join(
        [
            "set -o pipefail",
            f"__CM_COMMAND={shlex.quote(command)}",
            f"__CM_STDOUT={shlex.quote(STDOUT_FILE_NAME)}",
            f"__CM_STDERR={shlex.quote(STDERR_FILE_NAME)}",
            "__CM_T0=$(python -c 'import time; print(time.time())')",
            '( eval "$__CM_COMMAND" ) >"$__CM_STDOUT" 2>"$__CM_STDERR"',
            "__CM_RC=$?",
            "__CM_T1=$(python -c 'import time; print(time.time())')",
            "export __CM_COMMAND __CM_RC __CM_T0 __CM_T1 __CM_STDOUT __CM_STDERR",
            "python - <<'PY' || true",
            py_script,
            "PY",
            "exit 0",
        ]
    )
    return "bash -lc " + shlex.quote(script)


def _read_status_file(path: Path) -> Dict[str, Any] | None:
    if not path.is_file():
        return None
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _status_details_for_task(work_base_dir: Path, *, task_index: int, task_work_path: str) -> Dict[str, Any]:
    work_path = task_work_path or "."
    task_dir = work_base_dir / work_path
    status_path = task_dir / STATUS_FILE_NAME
    rec: Dict[str, Any] = {
        "task_index": task_index,
        "task_work_path": work_path,
        "task_dir": str(task_dir),
        "status_path": str(status_path),
    }
    data = _read_status_file(status_path)
    if not data:
        rec["status_missing_or_invalid"] = True
        return rec
    rec["status_missing_or_invalid"] = False
    rc_raw = data.get("returncode")
    try:
        rc = int(rc_raw)
    except Exception:
        rc = None
    rec["returncode"] = rc
    rec["cwd"] = data.get("cwd")
    rec["command"] = data.get("command")
    rec["stdout_file"] = data.get("stdout_file")
    rec["stderr_file"] = data.get("stderr_file")
    rec["stdout_tail"] = data.get("stdout_tail")
    rec["stderr_tail"] = data.get("stderr_tail")
    return rec


def _assert_remote_success(status_records: List[Dict[str, Any]]) -> None:
    failed = []
    for rec in status_records:
        if rec.get("status_missing_or_invalid"):
            failed.append(rec)
            continue
        if rec.get("returncode") != 0:
            failed.append(rec)
    if not failed:
        return
    first = failed[0]
    if first.get("status_missing_or_invalid"):
        raise RuntimeError(
            "Remote task failed: missing/invalid status file "
            f"(task_index={first.get('task_index')}, task_work_path={first.get('task_work_path')}, "
            f"status_path={first.get('status_path')})"
        )
    stderr_tail = str(first.get("stderr_tail") or "").strip()
    stderr_excerpt = ""
    if stderr_tail:
        one_line = " ".join(stderr_tail.split())
        stderr_excerpt = f", stderr_excerpt={one_line[:240]}"
    raise RuntimeError(
        "Remote task failed with non-zero exit code "
        f"(task_index={first.get('task_index')}, task_work_path={first.get('task_work_path')}, "
        f"returncode={first.get('returncode')}, cwd={first.get('cwd')}, status_path={first.get('status_path')}"
        f"{stderr_excerpt})"
    )


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
        command=_wrap_command_for_dpdispatcher(request.command),
        task_work_path=request.task_work_path,
        forward_files=request.forward_files,
        backward_files=_ensure_status_backward_files(request.backward_files),
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

    work_base_dir = Path(machine.context.init_local_root) / request.work_base
    status_records = [
        _status_details_for_task(work_base_dir, task_index=0, task_work_path=request.task_work_path)
    ]
    _assert_remote_success(status_records)

    states = [_task_state(t) for t in task_list]
    output_dir = Path(machine.context.init_local_root) / request.work_base / request.task_work_path
    cleanup_dpdispatcher_artifacts(output_dir)

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
            command=_wrap_command_for_dpdispatcher(t.command),
            task_work_path=t.task_work_path,
            forward_files=t.forward_files,
            backward_files=_ensure_status_backward_files(t.backward_files),
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

    work_base_dir = Path(machine.context.init_local_root) / batch.work_base
    status_records = [
        _status_details_for_task(work_base_dir, task_index=i, task_work_path=t.task_work_path)
        for i, t in enumerate(batch.tasks)
    ]
    _assert_remote_success(status_records)

    states = [_task_state(t) for t in task_list]
    # If all tasks share the same task_work_path, surface that in output_dir
    work_suffixes = {t.task_work_path for t in batch.tasks}
    suffix = work_suffixes.pop() if len(work_suffixes) == 1 else ""
    output_dir = Path(machine.context.init_local_root) / batch.work_base / suffix
    cleanup_dpdispatcher_artifacts(output_dir)

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
