from __future__ import annotations

from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import os
import re
import time
import subprocess
import threading
import uuid

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath, system_root
from catmaster.tools.misc.subprocess_utils import build_no_network_prefix, kill_process_tree
from catmaster.runtime.tool_runtime import current_toolcall_key, current_run_dir


class BashExecInput(BaseModel):
    """
    Execute a multi-line bash script inside the workspace (default) and return stdout/stderr.
    Network access is disabled by default using Linux network namespaces (unshare).
    Symbolic link operations are disabled; use copy/move operations instead.
    Stdout/stderr are returned up to max_output_chars per stream.
    Full streams are persisted to internal metadata audit logs (not for task planning/reading).
    Keep output short in scripts and print one-line summaries when possible.
    """

    script: str = Field(..., description="Bash script to execute (multi-line).")
    cwd: str = Field(".", description="Working directory inside project files root.")
    timeout_s: float = Field(86400.0, ge=0.1, description="Timeout seconds.")
    max_output_chars: int = Field(3000, ge=1000, description="Max chars returned for stdout/stderr each.")
    strict: bool = Field(True, description="Prepend 'set -euo pipefail' for safer scripting.")
    no_network: bool = Field(True, description="Disable network using unshare network namespace.")


def _truncate_text_tail(text: str, limit: int) -> Tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    marker = "\n...[output truncated]...\n"
    if limit <= len(marker):
        return text[-limit:], True
    tail_len = limit - len(marker)
    return marker + text[-tail_len:], True


_FORBIDDEN_SYMLINK_PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    (
        "ln symbolic options",
        re.compile(
            r"(^|[;&|(\n])\s*ln\b[^\n]*(--symbolic\b|\s-(?!-)[A-Za-z]*s[A-Za-z]*\b)",
            flags=re.IGNORECASE,
        ),
    ),
    (
        "cp -s",
        re.compile(
            r"(^|[;&|(\n])\s*cp\b[^\n]*\s-(?!-)[A-Za-z]*s[A-Za-z]*\b",
            flags=re.IGNORECASE,
        ),
    ),
    ("os.symlink()", re.compile(r"\bos\.symlink\s*\(", flags=re.IGNORECASE)),
    ("Path.symlink_to()", re.compile(r"\.symlink_to\s*\(", flags=re.IGNORECASE)),
]


_ACTIVE_PROC_LOCK = threading.Lock()
_ACTIVE_PROCS: dict[str, subprocess.Popen] = {}
_CANCELLED_KEYS: set[str] = set()


def _register_active_proc(toolcall_key: str, proc: subprocess.Popen) -> None:
    if not toolcall_key:
        return
    with _ACTIVE_PROC_LOCK:
        _ACTIVE_PROCS[toolcall_key] = proc


def _unregister_active_proc(toolcall_key: str) -> None:
    if not toolcall_key:
        return
    with _ACTIVE_PROC_LOCK:
        _ACTIVE_PROCS.pop(toolcall_key, None)
        _CANCELLED_KEYS.discard(toolcall_key)


def cancel_bash_exec_toolcall(toolcall_key: str) -> bool:
    key = (toolcall_key or "").strip()
    if not key:
        return False
    with _ACTIVE_PROC_LOCK:
        proc = _ACTIVE_PROCS.get(key)
        if proc is None:
            return False
        _CANCELLED_KEYS.add(key)
    kill_process_tree(proc)
    return True


def _safe_log_token(toolcall_key: str) -> str:
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", (toolcall_key or "").strip())
    token = token.strip("._")
    if token:
        return token[:120]
    return f"manual_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"


def _resolve_audit_logs_dir() -> Path:
    run_dir = (current_run_dir() or "").strip()
    if run_dir:
        try:
            return Path(run_dir).expanduser().resolve() / "audit" / "bash_exec"
        except Exception:
            pass
    return system_root() / "audit" / "bash_exec"


def _write_stream_logs(toolcall_key: str, stdout: str, stderr: str) -> list[str]:
    warnings: list[str] = []
    logs_dir = _resolve_audit_logs_dir()
    token = _safe_log_token(toolcall_key)
    stdout_path = logs_dir / f"{token}.stdout.txt"
    stderr_path = logs_dir / f"{token}.stderr.txt"
    try:
        logs_dir.mkdir(parents=True, exist_ok=True)
        stdout_path.write_text(stdout or "", encoding="utf-8")
        stderr_path.write_text(stderr or "", encoding="utf-8")
    except Exception as exc:
        warnings.append(f"failed to persist bash_exec logs: {type(exc).__name__}: {exc}")
    return warnings


def _detect_forbidden_symlink_usage(script: str) -> Optional[str]:
    for label, pattern in _FORBIDDEN_SYMLINK_PATTERNS:
        match = pattern.search(script)
        if not match:
            continue
        snippet = match.group(0).strip().replace("\n", " ")
        if len(snippet) > 120:
            snippet = snippet[:117] + "..."
        return f"{label}: {snippet}"
    return None


def bash_exec(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = BashExecInput(**payload)
    t0 = time.perf_counter()

    cwd_path = resolve_workspace_path(params.cwd, must_exist=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C")

    script = params.script
    blocked_reason = _detect_forbidden_symlink_usage(script)
    if blocked_reason:
        return create_tool_output(
            "bash_exec",
            False,
            data={
                "stdout": "",
                "stderr": "",
                "exit_code": None,
                "timed_out": False,
                "cmd": [],
                "cwd": workspace_relpath(cwd_path),
                "timeout_s": params.timeout_s,
                "blocked_reason": blocked_reason,
            },
            error=(
                "Symbolic link operations are disabled in bash_exec. "
                "Use copy/move operations (e.g., cp/rsync/mv) instead."
            ),
            execution_time=time.perf_counter() - t0,
        )
    if params.strict:
        script = "set -euo pipefail\n" + script

    base_cmd = ["bash", "-s"]
    cmd = base_cmd
    warnings = []
    if params.no_network:
        prefix = build_no_network_prefix()
        if prefix is None:
            return create_tool_output(
                "bash_exec",
                False,
                error="No network isolation backend found. Install 'unshare' (util-linux) "
                      "and ensure unprivileged user namespaces are enabled, "
                      "or add a bwrap/firejail fallback.",
                execution_time=time.perf_counter() - t0,
            )
        cmd = prefix + base_cmd

    timed_out = False
    cancelled = False
    stdout = ""
    stderr = ""
    exit_code = None
    toolcall_key = current_toolcall_key()

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=str(cwd_path),
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )
        _register_active_proc(toolcall_key, proc)
        try:
            stdout, stderr = proc.communicate(input=script, timeout=params.timeout_s)
        except subprocess.TimeoutExpired as e:
            timed_out = True
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            kill_process_tree(proc)
            try:
                out2, err2 = proc.communicate(timeout=1)
                stdout += out2 or ""
                stderr += err2 or ""
            except Exception:
                pass

        with _ACTIVE_PROC_LOCK:
            cancelled = bool(toolcall_key and toolcall_key in _CANCELLED_KEYS)
        exit_code = proc.returncode

        warnings.extend(_write_stream_logs(toolcall_key, stdout, stderr))

        stdout, cut_out = _truncate_text_tail(stdout, params.max_output_chars)
        stderr, cut_err = _truncate_text_tail(stderr, params.max_output_chars)
        if cut_out:
            warnings.append("stdout too long; truncated")
        if cut_err:
            warnings.append("stderr too long; truncated")

        ok = (not timed_out) and (exit_code == 0)
        data = {
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cancelled": cancelled,
            "cmd": cmd,
            "cwd": workspace_relpath(cwd_path),
            "timeout_s": params.timeout_s,
        }

        if ok:
            return create_tool_output(
                "bash_exec",
                True,
                data=data,
                warnings=warnings,
                execution_time=time.perf_counter() - t0,
            )
        err_msg = (
            "Bash interrupted by user"
            if cancelled
            else (
                f"Bash timed out (>{params.timeout_s}s)"
                if timed_out
                else f"Bash exited with code {exit_code}"
            )
        )
        return create_tool_output(
            "bash_exec",
            False,
            data=data,
            warnings=warnings,
            error=err_msg,
            execution_time=time.perf_counter() - t0,
        )

    except Exception as exc:
        return create_tool_output(
            "bash_exec",
            False,
            error=f"Failed to start/execute bash subprocess: {exc}",
            execution_time=time.perf_counter() - t0,
        )
    finally:
        _unregister_active_proc(toolcall_key)


__all__ = ["bash_exec", "BashExecInput", "cancel_bash_exec_toolcall"]
