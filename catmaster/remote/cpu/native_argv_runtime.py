from __future__ import annotations

import json
import shutil
import subprocess
import tarfile
import time
from pathlib import Path, PurePosixPath
from typing import Any, Iterable

RESULT_ARCHIVE = "catmaster_results.tar.gz"


def _stage_path(raw: str) -> str:
    value = str(raw or "").strip()
    path = PurePosixPath(value)
    if not value or path.is_absolute() or value in {".", ".."} or any(part in {"", ".", ".."} for part in path.parts):
        raise ValueError(f"input_assets entries must be stage-relative without parent traversal: {raw!r}")
    return path.as_posix()


def load_manifest(path: Path) -> tuple[list[str], list[str]]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or set(payload) != {"argv", "input_assets"}:
        raise ValueError("manifest.json must contain only argv and input_assets")
    argv = payload.get("argv")
    assets = payload.get("input_assets")
    if not isinstance(argv, list) or not all(isinstance(token, str) and "\x00" not in token for token in argv):
        raise ValueError("manifest argv must be a list of strings without NUL characters")
    if not isinstance(assets, list) or not all(isinstance(item, str) for item in assets):
        raise ValueError("manifest input_assets must be a list of stage-relative strings")
    normalized = [_stage_path(item) for item in assets]
    if len(set(normalized)) != len(normalized):
        raise ValueError("manifest input_assets contains duplicate paths")
    stage_dir = path.resolve().parent
    missing = [item for item in normalized if not (stage_dir / item).is_file()]
    if missing:
        raise FileNotFoundError("Missing declared input asset(s): " + ", ".join(missing))
    return list(argv), normalized


def resolve_program(requested: str, candidates: Iterable[str]) -> str:
    if requested and requested != "auto":
        return shutil.which(requested) or requested
    for candidate in candidates:
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return next(iter(candidates))


def write_result_archive() -> None:
    archive = Path(RESULT_ARCHIVE)
    with tarfile.open(archive, "w:gz") as handle:
        for path in sorted(Path.cwd().rglob("*")):
            if not path.is_file() or path == archive:
                continue
            handle.add(path, arcname=path.relative_to(Path.cwd()).as_posix(), recursive=False)


def run_native_argv(
    *,
    manifest_path: Path,
    program: str,
    summary_name: str,
    log_name: str,
    normal_markers: Iterable[str],
) -> int:
    started = time.time()
    argv, input_assets = load_manifest(manifest_path)
    command = [program, *argv]
    with Path(log_name).open("w", encoding="utf-8") as log_handle:
        process = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, check=False, shell=False)
    text = Path(log_name).read_text(encoding="utf-8", errors="replace")
    normal_termination = any(marker.lower() in text.lower() for marker in normal_markers)
    not_converged = "NOT_CONVERGED" in text
    execution_state = "completed" if process.returncode == 0 else "failed"
    task_state = "not_converged" if not_converged else ("incomplete" if process.returncode else "unknown")
    payload: dict[str, Any] = {
        "execution_state": execution_state,
        "task_state": task_state,
        "returncode": int(process.returncode),
        "normal_termination": normal_termination,
        "not_converged": not_converged,
        "argv": argv,
        "input_assets": input_assets,
        "log_file": log_name,
        "started_at": started,
        "finished_at": time.time(),
    }
    Path(summary_name).write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    write_result_archive()
    return int(process.returncode)


__all__ = ["RESULT_ARCHIVE", "load_manifest", "resolve_program", "run_native_argv", "write_result_archive"]
