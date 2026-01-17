from __future__ import annotations

from typing import Dict, Any, Tuple
import os
import sys
import time
import signal
import subprocess

from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, workspace_root


class PythonExecInput(BaseModel):
    """
    Execute inline Python code for expression calculation, result post analysis, or figure drawing when other tools provided can not meet the requirement.
    ALWAYS use `print(...)` to output the final result; only printed output will be returned.
    Basic packages for material science are provided (e.g. numpy, matplotlib, pymatgen, ase, etc.), these packages are encouraged to be used to enhance analysis robustness.
    The working directory is set to the workspace root. Each time the tool is called, a new Python interpreter is started and do not share the same variables with the previous calls.
    """

    code: str = Field(..., description="Python code to execute.")
    timeout_s: float = Field(3600.0, ge=0.1, description="Timeout in seconds for the Python code execution.")
    max_output_chars: int = Field(10000, description="Maximum number of characters to return in the output.")


def _truncate_text(text: str, limit: int) -> Tuple[str, bool]:
    if text is None:
        return "", False
    if len(text) <= limit:
        return text, False
    return text[:limit] + "\n...[output truncated]...\n", True


def _kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        else:
            proc.kill()
    except Exception:
        pass


def python_exec(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = PythonExecInput(**payload)
    t0 = time.perf_counter()

    cwd = str(workspace_root())

    # Inherit current environment with minimal defaults (do not override user settings)
    env = os.environ.copy()
    env.setdefault("PYTHONUTF8", "1")
    env.setdefault("PYTHONIOENCODING", "utf-8")
    env.setdefault("PYTHONUNBUFFERED", "1")
    env.setdefault("PYTHONDONTWRITEBYTECODE", "1")
    env.setdefault("MPLBACKEND", "Agg")

    # Use the same interpreter for consistent package env; execute via stdin to avoid -c quoting issues
    cmd = [sys.executable, "-u", "-B", "-"]

    timed_out = False
    stdout = ""
    stderr = ""
    exit_code = None
    warnings = []

    try:
        proc = subprocess.Popen(
            cmd,
            stdin=subprocess.PIPE,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            cwd=cwd,
            env=env,
            text=True,
            encoding="utf-8",
            errors="replace",
            start_new_session=True,
        )

        try:
            stdout, stderr = proc.communicate(input=params.code, timeout=params.timeout_s)
        except subprocess.TimeoutExpired as e:
            timed_out = True
            # Capture any partial output attached to the timeout exception first
            stdout = (e.stdout or "") if isinstance(e.stdout, str) else ""
            stderr = (e.stderr or "") if isinstance(e.stderr, str) else ""
            _kill_process_tree(proc)
            try:
                out2, err2 = proc.communicate(timeout=1)
                stdout += out2 or ""
                stderr += err2 or ""
            except Exception:
                pass

        exit_code = proc.returncode

        # Truncate output to prevent log/context blowups
        stdout, cut_out = _truncate_text(stdout, params.max_output_chars)
        stderr, cut_err = _truncate_text(stderr, params.max_output_chars)
        if cut_out:
            warnings.append("stdout too long; truncated")
        if cut_err:
            warnings.append("stderr too long; truncated")

        success = (not timed_out) and (exit_code == 0)

        data = {
            "run_result": stdout,   # Backward-compatible field: stdout as primary return
            "stdout": stdout,
            "stderr": stderr,
            "exit_code": exit_code,
            "timed_out": timed_out,
            "cmd": cmd,
            "cwd": cwd,
            "timeout_s": params.timeout_s,
        }

        if success:
            return create_tool_output(
                "python_exec",
                True,
                data=data,
                warnings=warnings,
                execution_time=time.perf_counter() - t0,
            )

        # Provide a direct error message on failure; stderr/traceback are still in data
        if timed_out:
            err_msg = f"Python execution timed out (>{params.timeout_s}s); subprocess terminated"
        else:
            err_msg = f"Python subprocess exited with non-zero code: {exit_code}"

        return create_tool_output(
            "python_exec",
            False,
            data=data,
            warnings=warnings,
            error=err_msg,
            execution_time=time.perf_counter() - t0,
        )

    except Exception as exc:
        return create_tool_output(
            "python_exec",
            False,
            error=f"Failed to start or execute Python subprocess: {exc}",
            execution_time=time.perf_counter() - t0,
        )


__all__ = ["python_exec", "PythonExecInput"]
