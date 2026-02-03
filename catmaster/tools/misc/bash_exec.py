from __future__ import annotations

from typing import Dict, Any, Tuple, Optional
import os
import time
import subprocess

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_view_path, workspace_root
from catmaster.tools.misc.subprocess_utils import build_no_network_prefix, kill_process_tree


class BashExecInput(BaseModel):
    """
    Execute a multi-line bash script inside the workspace (default) and return stdout/stderr.
    Network access is disabled by default using Linux network namespaces (unshare).
    Keep output short; write large logs to files and print a one-line summary.
    """

    script: str = Field(..., description="Bash script to execute (multi-line).")
    cwd: str = Field(".", description="Working directory inside the selected view.")
    view: str = Field("user", description="user or system")
    timeout_s: float = Field(3600.0, ge=0.1, description="Timeout seconds.")
    max_output_chars: int = Field(10000, ge=1000, description="Max chars returned for stdout/stderr each.")
    strict: bool = Field(True, description="Prepend 'set -euo pipefail' for safer scripting.")
    no_network: bool = Field(True, description="Disable network using unshare network namespace.")


def _truncate_text_tail(text: str, limit: int) -> Tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    return "\n...[output truncated]...\n" + text[-limit:], True


def bash_exec(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = BashExecInput(**payload)
    t0 = time.perf_counter()

    cwd_path = resolve_view_path(params.cwd, params.view, must_exist=True)

    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("LC_ALL", "C")
    env["CATMASTER_WORKSPACE"] = str(workspace_root())

    script = params.script
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
    stdout = ""
    stderr = ""
    exit_code = None

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

        exit_code = proc.returncode

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
            "cmd": cmd,
            "cwd": str(cwd_path),
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
            f"Bash timed out (>{params.timeout_s}s)" if timed_out
            else f"Bash exited with code {exit_code}"
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


__all__ = ["bash_exec", "BashExecInput"]
