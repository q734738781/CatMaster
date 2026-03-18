from __future__ import annotations

import pytest

from catmaster.tools.execution import dpdispatcher_runner as dpr


def test_ensure_status_backward_files_always_includes_status_and_logs() -> None:
    files = dpr._ensure_status_backward_files(["result.txt"])
    assert "result.txt" in files
    assert dpr.STATUS_FILE_NAME in files
    assert dpr.STDOUT_FILE_NAME in files
    assert dpr.STDERR_FILE_NAME in files


def test_wrap_command_captures_stdout_and_stderr() -> None:
    wrapped = dpr._wrap_command_for_dpdispatcher("echo hello")
    assert dpr.STDOUT_FILE_NAME in wrapped
    assert dpr.STDERR_FILE_NAME in wrapped
    assert '>"$__CM_STDOUT" 2>"$__CM_STDERR"' in wrapped


def test_assert_remote_success_raises_for_nonzero_with_excerpt() -> None:
    with pytest.raises(RuntimeError) as exc:
        dpr._assert_remote_success(
            [
                {
                    "task_index": 0,
                    "task_work_path": ".",
                    "status_path": "/tmp/status.json",
                    "status_missing_or_invalid": False,
                    "returncode": 1,
                    "cwd": "/remote/work",
                    "stderr_tail": "ModuleNotFoundError: No module named 'torch_dftd'",
                }
            ]
        )
    text = str(exc.value)
    assert "non-zero exit code" in text
    assert "ModuleNotFoundError" in text


def test_assert_remote_success_raises_for_missing_status() -> None:
    with pytest.raises(RuntimeError) as exc:
        dpr._assert_remote_success(
            [
                {
                    "task_index": 0,
                    "task_work_path": ".",
                    "status_path": "/tmp/status.json",
                    "status_missing_or_invalid": True,
                }
            ]
        )
    assert "missing/invalid status file" in str(exc.value)
