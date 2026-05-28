from __future__ import annotations

import shlex
import time
import uuid
import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional
import yaml
from pydantic import BaseModel, Field
import re
from dpdispatcher import Machine, Resources, Task, Submission
from dpdispatcher.utils.job_status import JobStatus
from catmaster.tools.base import workspace_relpath, workspace_root
from catmaster.tools.execution.machine_registry import MachineRegister

STATUS_FILE_NAME = "status.json"
STDOUT_FILE_NAME = "stdout.log"
STDERR_FILE_NAME = "stderr.log"

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
    tool_name: str = Field("dpdispatcher", description="CatMaster tool name creating this submission")


class DispatchResult(BaseModel):
    """Result of a dispatch submission."""

    work_base: str
    local_root: str
    output_dir: str
    task_states: List[str]
    task_state_counts: Dict[str, int] = Field(default_factory=dict)
    submission_dir: str
    duration_s: float
    remote_context_id: str = ""
    receipt_rel: str = ""
    submission_hash: str = ""
    jobs: List[Dict[str, Any]] = Field(default_factory=list)
    job_status_counts: Dict[str, int] = Field(default_factory=dict)
    remote_context: Dict[str, Any] = Field(default_factory=dict)


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
    tool_name: str = Field("dpdispatcher", description="CatMaster tool name creating this submission")


class DPDispatcherDispatchError(RuntimeError):
    """DPDispatcher failure carrying the receipt context that reached the agent."""

    def __init__(
        self,
        message: str,
        *,
        remote_context: Dict[str, Any] | None = None,
        cause: Exception | None = None,
    ) -> None:
        self.remote_context = dict(remote_context or {})
        self.cause = cause
        detail_lines = []
        for key in ("remote_context_id", "submission_hash", "receipt_rel"):
            value = self.remote_context.get(key)
            if value not in (None, "", [], {}):
                detail_lines.append(f"{key}={value}")
        if detail_lines:
            message = str(message).rstrip() + "\n" + "\n".join(detail_lines)
        super().__init__(message)


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
            return _status_name(val)
    return "unknown"


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _status_code_and_name(value: Any) -> tuple[int | None, str]:
    if value is None:
        value = JobStatus.unsubmitted
    if isinstance(value, JobStatus):
        return int(value), value.name
    try:
        status = JobStatus(int(value))
        return int(status), status.name
    except Exception:
        text = str(value or "unknown").strip() or "unknown"
        return None, text


def _status_name(value: Any) -> str:
    return _status_code_and_name(value)[1]


def task_state_counts(states: List[Any]) -> Dict[str, int]:
    counts: Dict[str, int] = {}
    for state in states:
        name = _status_name(state)
        counts[name] = counts.get(name, 0) + 1
    return counts


def _job_records(submission: Submission) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    for job in list(getattr(submission, "belonging_jobs", []) or []):
        status_code, status_name = _status_code_and_name(getattr(job, "job_state", None))
        records.append(
            {
                "job_hash": str(getattr(job, "job_hash", "") or ""),
                "job_id": str(getattr(job, "job_id", "") or ""),
                "status_code": status_code,
                "status": status_name,
            }
        )
    return records


def _job_status_counts(jobs: list[dict[str, Any]]) -> dict[str, int]:
    counts: dict[str, int] = {}
    for job in jobs:
        key = str(job.get("status") or "unknown")
        counts[key] = counts.get(key, 0) + 1
    return counts


def _context_id_for_submission(submission_hash: str) -> str:
    stamp = datetime.now().astimezone().strftime("%Y%m%d_%H%M%S")
    token = (submission_hash or uuid.uuid4().hex)[:8]
    return f"dp_{stamp}_{token}"


def _write_remote_receipt(
    *,
    submission: Submission,
    tool_name: str,
    receipt: dict[str, Any] | None = None,
) -> dict[str, Any]:
    submission_hash = str(getattr(submission, "submission_hash", "") or "")
    now = _now_iso()
    context_id = str((receipt or {}).get("context_id") or _context_id_for_submission(submission_hash))
    submitted_at = str((receipt or {}).get("submitted_at") or now)
    receipt_dir = workspace_root() / ".deepagents" / "dpdispatcher" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{context_id}.json"
    jobs = _job_records(submission)
    payload: dict[str, Any] = {
        "context_id": context_id,
        "submitted_at": submitted_at,
        "updated_at": now,
        "tool_name": str(tool_name or "dpdispatcher"),
        "submission_hash": submission_hash,
        "jobs": jobs,
        "job_status_counts": _job_status_counts(jobs),
        "receipt_rel": workspace_relpath(receipt_path),
    }
    receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def write_dispatch_attempt_receipt(
    *,
    tool_name: str,
    work_base: str,
    task_name: str,
    work_dir_rel: str,
    resources: str,
    submitted_at: str = "",
    error: str = "",
) -> dict[str, Any]:
    """Write a CatMaster-side receipt when DPDispatcher fails before a submission hash exists."""

    now = _now_iso()
    context_id = _context_id_for_submission(uuid.uuid4().hex)
    receipt_dir = workspace_root() / ".deepagents" / "dpdispatcher" / "receipts"
    receipt_dir.mkdir(parents=True, exist_ok=True)
    receipt_path = receipt_dir / f"{context_id}.json"
    payload: dict[str, Any] = {
        "context_id": context_id,
        "submitted_at": str(submitted_at or now),
        "updated_at": now,
        "tool_name": str(tool_name or "dpdispatcher"),
        "task_name": str(task_name or ""),
        "work_dir_rel": str(work_dir_rel or ""),
        "work_base": str(work_base or ""),
        "resources": str(resources or ""),
        "submission_hash": "",
        "jobs": [],
        "job_status_counts": {},
        "receipt_rel": workspace_relpath(receipt_path),
    }
    if error:
        payload["dispatch_error"] = str(error)
    receipt_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return payload


def _public_remote_context(
    receipt: dict[str, Any] | None,
    *,
    include_jobs: bool = False,
) -> dict[str, Any]:
    if not receipt:
        return {}
    context = {
        "remote_context_id": receipt.get("context_id") or "",
        "submitted_at": receipt.get("submitted_at") or "",
        "updated_at": receipt.get("updated_at") or "",
        "submission_hash": receipt.get("submission_hash") or "",
        "receipt_rel": receipt.get("receipt_rel") or "",
    }
    if include_jobs:
        context["jobs"] = list(receipt.get("jobs") or [])
        context["job_status_counts"] = dict(receipt.get("job_status_counts") or {})
    return context


def remote_context_from_receipt(receipt: dict[str, Any] | None, *, include_jobs: bool = False) -> dict[str, Any]:
    """Return the agent-facing context fields from a receipt dictionary."""

    return _public_remote_context(receipt, include_jobs=include_jobs)


def remote_context_from_result(result: Any) -> dict[str, Any]:
    """Return the agent-facing remote context from a DispatchResult-like value."""
    context = getattr(result, "remote_context", None)
    if isinstance(context, dict) and context:
        return dict(context)
    out: dict[str, Any] = {}
    for key in (
        "remote_context_id",
        "submitted_at",
        "updated_at",
        "submission_hash",
        "receipt_rel",
    ):
        value = getattr(result, key, None)
        if value not in (None, "", [], {}):
            out[key] = value
    return out


def remote_context_from_exception(exc: Exception | None) -> dict[str, Any]:
    """Return the agent-facing remote context from a dispatch exception."""
    if exc is None:
        return {}
    context = getattr(exc, "remote_context", None)
    if isinstance(context, dict) and context:
        return dict(context)
    return {}


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
        "def _read_text(path: str) -> str:\n"
        "    if not path:\n"
        "        return ''\n"
        "    p = pathlib.Path(path)\n"
        "    if not p.is_file():\n"
        "        return ''\n"
        "    try:\n"
        "        return p.read_text(encoding='utf-8', errors='replace')\n"
        "    except Exception:\n"
        "        return ''\n"
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
        "    'stdout_tail': _read_text(os.environ.get('__CM_STDOUT', '')),\n"
        "    'stderr_tail': _read_text(os.environ.get('__CM_STDERR', '')),\n"
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
        stderr_excerpt = f", stderr_excerpt={one_line}"
    raise RuntimeError(
        "Remote task failed with non-zero exit code "
        f"(task_index={first.get('task_index')}, task_work_path={first.get('task_work_path')}, "
        f"returncode={first.get('returncode')}, cwd={first.get('cwd')}, status_path={first.get('status_path')}"
        f"{stderr_excerpt})"
    )


def dispatch_task(request: DispatchRequest, *, config_path: Optional[str] = None, register: Optional[MachineRegister] = None) -> DispatchResult:
    """Submit a single task through DPDispatcher and wait for completion."""

    reg = register or (MachineRegister(extra_paths=[Path(config_path)]) if config_path else MachineRegister())

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
    if not submission.belonging_jobs:
        submission.generate_jobs()
    receipt = _write_remote_receipt(
        submission=submission,
        tool_name=request.tool_name,
    )

    t0 = time.time()
    try:
        submission.run_submission(clean=request.clean_remote, check_interval=request.check_interval)
    except Exception as exc:
        receipt = _write_remote_receipt(
            submission=submission,
            tool_name=request.tool_name,
            receipt=receipt,
        )
        raise DPDispatcherDispatchError(
            f"{type(exc).__name__}: {exc}",
            remote_context=_public_remote_context(receipt, include_jobs=True),
            cause=exc,
        ) from exc
    duration = time.time() - t0
    receipt = _write_remote_receipt(
        submission=submission,
        tool_name=request.tool_name,
        receipt=receipt,
    )

    work_base_dir = Path(machine.context.init_local_root) / request.work_base
    status_records = [
        _status_details_for_task(work_base_dir, task_index=0, task_work_path=request.task_work_path)
    ]
    try:
        _assert_remote_success(status_records)
    except Exception as exc:
        receipt = _write_remote_receipt(
            submission=submission,
            tool_name=request.tool_name,
            receipt=receipt,
        )
        raise DPDispatcherDispatchError(
            f"{type(exc).__name__}: {exc}",
            remote_context=_public_remote_context(receipt, include_jobs=True),
            cause=exc,
        ) from exc

    states = [_task_state(t) for t in task_list]
    state_counts = task_state_counts(states)
    output_dir = Path(machine.context.init_local_root) / request.work_base / request.task_work_path
    cleanup_dpdispatcher_artifacts(output_dir)
    remote_context = _public_remote_context(receipt)

    return DispatchResult(
        work_base=request.work_base,
        local_root=str(machine.context.init_local_root),
        output_dir=str(output_dir.resolve()),
        task_states=states,
        task_state_counts=state_counts,
        submission_dir=str(Path(machine.context.init_local_root) / request.work_base),
        duration_s=duration,
        remote_context_id=str(remote_context.get("remote_context_id") or ""),
        receipt_rel=str(remote_context.get("receipt_rel") or ""),
        submission_hash=str(remote_context.get("submission_hash") or ""),
        remote_context=remote_context,
    )


def dispatch_submission(
    batch: BatchDispatchRequest,
    *,
    register: Optional[MachineRegister] = None,
    config_path: Optional[str] = None,
) -> DispatchResult:
    """Submit multiple tasks as a single Submission."""
    reg = register or (MachineRegister(extra_paths=[Path(config_path)]) if config_path else MachineRegister())

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
    if not submission.belonging_jobs:
        submission.generate_jobs()
    receipt = _write_remote_receipt(
        submission=submission,
        tool_name=batch.tool_name,
    )

    t0 = time.time()
    try:
        submission.run_submission(clean=batch.clean_remote, check_interval=batch.check_interval)
    except Exception as exc:
        receipt = _write_remote_receipt(
            submission=submission,
            tool_name=batch.tool_name,
            receipt=receipt,
        )
        raise DPDispatcherDispatchError(
            f"{type(exc).__name__}: {exc}",
            remote_context=_public_remote_context(receipt, include_jobs=True),
            cause=exc,
        ) from exc
    duration = time.time() - t0
    receipt = _write_remote_receipt(
        submission=submission,
        tool_name=batch.tool_name,
        receipt=receipt,
    )

    work_base_dir = Path(machine.context.init_local_root) / batch.work_base
    status_records = [
        _status_details_for_task(work_base_dir, task_index=i, task_work_path=t.task_work_path)
        for i, t in enumerate(batch.tasks)
    ]
    try:
        _assert_remote_success(status_records)
    except Exception as exc:
        receipt = _write_remote_receipt(
            submission=submission,
            tool_name=batch.tool_name,
            receipt=receipt,
        )
        raise DPDispatcherDispatchError(
            f"{type(exc).__name__}: {exc}",
            remote_context=_public_remote_context(receipt, include_jobs=True),
            cause=exc,
        ) from exc

    states = [_task_state(t) for t in task_list]
    state_counts = task_state_counts(states)
    # If all tasks share the same task_work_path, surface that in output_dir
    work_suffixes = {t.task_work_path for t in batch.tasks}
    suffix = work_suffixes.pop() if len(work_suffixes) == 1 else ""
    output_dir = Path(machine.context.init_local_root) / batch.work_base / suffix
    cleanup_dpdispatcher_artifacts(output_dir)
    remote_context = _public_remote_context(receipt)

    return DispatchResult(
        work_base=batch.work_base,
        local_root=str(machine.context.init_local_root),
        output_dir=str(output_dir.resolve()),
        task_states=states,
        task_state_counts=state_counts,
        submission_dir=str(Path(machine.context.init_local_root) / batch.work_base),
        duration_s=duration,
        remote_context_id=str(remote_context.get("remote_context_id") or ""),
        receipt_rel=str(remote_context.get("receipt_rel") or ""),
        submission_hash=str(remote_context.get("submission_hash") or ""),
        remote_context=remote_context,
    )


def make_work_base(prefix: str = "cm") -> str:
    """Generate a short unique work_base name."""
    return f"{prefix}_{time.strftime('%Y%m%d_%H%M%S')}_{uuid.uuid4().hex[:6]}"
