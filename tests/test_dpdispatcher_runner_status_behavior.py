from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.tools.base import workspace_scope
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


def test_remote_receipt_records_dpdispatcher_job_status(tmp_path: Path) -> None:
    class _Job:
        job_hash = "abc123"
        job_id = "3849"
        job_state = dpr.JobStatus.running

    class _Submission:
        submission_hash = "f" * 40
        belonging_jobs = [_Job()]

    with workspace_scope(tmp_path):
        receipt = dpr._write_remote_receipt(
            submission=_Submission(),
            tool_name="vasp_execute_batch",
        )

    assert receipt["context_id"].startswith("dp_")
    assert receipt["submission_hash"] == "f" * 40
    assert receipt["jobs"] == [
        {
            "job_hash": "abc123",
            "job_id": "3849",
            "status_code": 3,
            "status": "running",
        }
    ]
    assert receipt["job_status_counts"] == {"running": 1}
    assert receipt["receipt_rel"].startswith(".deepagents/dpdispatcher/receipts/")
    assert (tmp_path / "files" / receipt["receipt_rel"]).is_file()


def test_task_states_are_reported_as_readable_status_names() -> None:
    class _Task:
        task_state = dpr.JobStatus.finished

    assert dpr._task_state(_Task()) == "finished"
    assert dpr.task_state_counts([5, "5", dpr.JobStatus.running, "finished"]) == {
        "finished": 3,
        "running": 1,
    }


def test_public_remote_context_omits_jobs_unless_requested() -> None:
    receipt = {
        "context_id": "dp_ctx",
        "submitted_at": "2026-05-19T00:00:00+08:00",
        "updated_at": "2026-05-19T00:01:00+08:00",
        "submission_hash": "abc",
        "jobs": [{"job_hash": "jobhash", "job_id": "3849", "status_code": 3, "status": "running"}],
        "job_status_counts": {"running": 1},
        "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_ctx.json",
    }

    success_context = dpr._public_remote_context(receipt)
    error_context = dpr._public_remote_context(receipt, include_jobs=True)

    assert success_context == {
        "remote_context_id": "dp_ctx",
        "submitted_at": "2026-05-19T00:00:00+08:00",
        "updated_at": "2026-05-19T00:01:00+08:00",
        "submission_hash": "abc",
        "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_ctx.json",
    }
    assert error_context["jobs"][0]["job_id"] == "3849"
    assert error_context["job_status_counts"] == {"running": 1}


def test_dpdispatcher_dispatch_error_string_exposes_remote_context() -> None:
    exc = dpr.DPDispatcherDispatchError(
        "Connection reset by peer",
        remote_context={
            "remote_context_id": "dp_ctx",
            "submission_hash": "abc",
            "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_ctx.json",
        },
    )

    text = str(exc)
    assert "Connection reset by peer" in text
    assert "remote_context_id=dp_ctx" in text
    assert "submission_hash=abc" in text
    assert "receipt_rel=.deepagents/dpdispatcher/receipts/dp_ctx.json" in text
