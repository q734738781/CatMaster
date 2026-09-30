from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import tempfile
from typing import Iterator, Mapping


_ROOT_BOOKKEEPING_NAMES = {
    "err",
    "log",
    "status.json",
    "stderr.log",
    "stdout.log",
}
_DP_ARTIFACT_RE = re.compile(
    r"^(?:"
    r"[0-9a-f]{40}_(?:task_tag_finished|job_tag_finished|job_id|flag_if_job_task_fail|last_err_file)"
    r"|[0-9a-f]{40}(?:\.sub(?:\.run)?|\.json|\.tar(?:\.gz)?)"
    r"|tag_failure_download_.*"
    r")$"
)
_RSYNC_ROOT_EXCLUDES = (
    "/err",
    "/log",
    "/status.json",
    "/stderr.log",
    "/stdout.log",
    "/????????????????????????????????????????_task_tag_finished",
    "/????????????????????????????????????????_job_tag_finished",
    "/????????????????????????????????????????_job_id",
    "/????????????????????????????????????????_flag_if_job_task_fail",
    "/????????????????????????????????????????_last_err_file",
    "/????????????????????????????????????????.sub",
    "/????????????????????????????????????????.sub.run",
    "/????????????????????????????????????????.json",
    "/????????????????????????????????????????.tar",
    "/????????????????????????????????????????.tar.gz",
    "/tag_failure_download_*",
)


class ScratchStageOutError(RuntimeError):
    """Raised when node-local results cannot be returned to shared storage."""


@dataclass(frozen=True)
class WorkDirectory:
    shared_dir: Path
    run_dir: Path
    node_local: bool


def _allocated_node_count(env: Mapping[str, str]) -> int | None:
    for key in ("SLURM_NNODES", "SLURM_JOB_NUM_NODES"):
        raw = str(env.get(key, "") or "").strip()
        match = re.match(r"^([0-9]+)", raw)
        if not match:
            continue
        value = int(match.group(1))
        if value > 0:
            return value
    return None


def _candidate_roots(env: Mapping[str, str]) -> list[Path]:
    candidates = [
        env.get("CATMASTER_SCRATCH_ROOT", ""),
        env.get("SLURM_TMPDIR", ""),
        env.get("TMPDIR", ""),
        "/tmp",
    ]
    roots: list[Path] = []
    seen: set[str] = set()
    for raw in candidates:
        value = str(raw or "").strip()
        if not value:
            continue
        root = Path(value).expanduser()
        identity = str(root)
        if identity in seen:
            continue
        seen.add(identity)
        roots.append(root)
    return roots


def _same_filesystem(left: Path, right: Path) -> bool:
    return left.stat().st_dev == right.stat().st_dev


def _ignored_root_name(name: str) -> bool:
    return name in _ROOT_BOOKKEEPING_NAMES or bool(_DP_ARTIFACT_RE.match(name))


def _copy_tree_python(source: Path, destination: Path) -> None:
    def _ignore(directory: str, names: list[str]) -> set[str]:
        if Path(directory).resolve() != source.resolve():
            return set()
        return {name for name in names if _ignored_root_name(name)}

    shutil.copytree(
        source,
        destination,
        dirs_exist_ok=True,
        symlinks=True,
        copy_function=shutil.copy2,
        ignore=_ignore,
    )


def _sync_tree(source: Path, destination: Path) -> None:
    rsync = shutil.which("rsync")
    if not rsync:
        _copy_tree_python(source, destination)
        return
    command = [rsync, "-a"]
    for pattern in _RSYNC_ROOT_EXCLUDES:
        command.extend(("--exclude", pattern))
    command.extend((f"{source}/", f"{destination}/"))
    completed = subprocess.run(command, capture_output=True, text=True, check=False)
    if completed.returncode:
        detail = (completed.stderr or completed.stdout or "rsync failed").strip()
        raise OSError(f"rsync returned {completed.returncode}: {detail[-1000:]}")


def _scratch_prefix(engine: str, env: Mapping[str, str]) -> str:
    safe_engine = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(engine or "task")).strip("-.") or "task"
    job_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(env.get("SLURM_JOB_ID", "job") or "job"))
    array_id = re.sub(r"[^A-Za-z0-9_.-]+", "-", str(env.get("SLURM_ARRAY_TASK_ID", "0") or "0"))
    return f"catmaster-{safe_engine}-{job_id}-{array_id}-"


def _prepare_scratch(engine: str, shared_dir: Path, env: Mapping[str, str]) -> Path | None:
    last_error = ""
    for root in _candidate_roots(env):
        try:
            if not root.is_dir() or not os.access(root, os.W_OK | os.X_OK):
                continue
            if _same_filesystem(shared_dir, root):
                continue
            scratch_dir = Path(tempfile.mkdtemp(prefix=_scratch_prefix(engine, env), dir=root))
        except Exception as exc:
            last_error = str(exc)
            continue
        try:
            _sync_tree(shared_dir, scratch_dir)
        except Exception as exc:
            last_error = str(exc)
            shutil.rmtree(scratch_dir, ignore_errors=True)
            continue
        return scratch_dir
    if last_error:
        sys.stderr.write(
            f"[node_local_scratch] staging unavailable for {engine}; using shared stage: {last_error}\n"
        )
    return None


@contextmanager
def node_local_workdir(
    engine: str,
    *,
    env: Mapping[str, str] | None = None,
) -> Iterator[WorkDirectory]:
    """Run a single-node Slurm task in local scratch and return all files."""

    active_env: Mapping[str, str] = os.environ if env is None else env
    shared_dir = Path.cwd().resolve()
    if _allocated_node_count(active_env) != 1:
        yield WorkDirectory(shared_dir=shared_dir, run_dir=shared_dir, node_local=False)
        return

    scratch_dir = _prepare_scratch(engine, shared_dir, active_env)
    if scratch_dir is None:
        yield WorkDirectory(shared_dir=shared_dir, run_dir=shared_dir, node_local=False)
        return

    os.chdir(scratch_dir)
    try:
        yield WorkDirectory(shared_dir=shared_dir, run_dir=scratch_dir, node_local=True)
    finally:
        os.chdir(shared_dir)
        try:
            _sync_tree(scratch_dir, shared_dir)
        except Exception as exc:
            raise ScratchStageOutError(
                f"Failed to copy node-local results from {scratch_dir} to {shared_dir}: {exc}"
            ) from exc
        shutil.rmtree(scratch_dir)


__all__ = ["ScratchStageOutError", "WorkDirectory", "node_local_workdir"]
