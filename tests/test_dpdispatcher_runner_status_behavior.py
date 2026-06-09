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


def test_wrap_command_can_inject_dispatch_token_for_hash_isolation() -> None:
    wrapped_a = dpr._wrap_command_for_dpdispatcher("echo hello", dispatch_token="run-a:task-0")
    wrapped_b = dpr._wrap_command_for_dpdispatcher("echo hello", dispatch_token="run-b:task-0")

    assert wrapped_a != wrapped_b
    assert "CM_DPDISPATCHER_SUBMISSION_TOKEN=run-a:task-0" in wrapped_a
    assert "CM_DPDISPATCHER_SUBMISSION_TOKEN=run-b:task-0" in wrapped_b


def test_dispatch_token_changes_dpdispatcher_hashes() -> None:
    resources = dpr.Resources(
        number_node=1,
        cpu_per_node=1,
        gpu_per_node=0,
        queue_name="batch",
        group_size=1,
    )
    task_a = dpr.Task(
        command=dpr._wrap_command_for_dpdispatcher("echo hello", dispatch_token="same-stage:a"),
        task_work_path=".",
        forward_files=[],
        backward_files=[],
    )
    task_b = dpr.Task(
        command=dpr._wrap_command_for_dpdispatcher("echo hello", dispatch_token="same-stage:b"),
        task_work_path=".",
        forward_files=[],
        backward_files=[],
    )
    sub_a = dpr.Submission(work_base="same_work_base", resources=resources, task_list=[task_a])
    sub_b = dpr.Submission(work_base="same_work_base", resources=resources, task_list=[task_b])
    sub_a.generate_jobs()
    sub_b.generate_jobs()

    assert task_a.task_hash != task_b.task_hash
    assert sub_a.belonging_jobs[0].job_hash != sub_b.belonging_jobs[0].job_hash
    assert sub_a.submission_hash != sub_b.submission_hash


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


def test_status_details_include_remote_elapsed_seconds(tmp_path: Path) -> None:
    (tmp_path / "status.json").write_text(
        '{"returncode": 0, "command": "echo ok", "t_start": 100.0, "t_end": 145.5}\n',
        encoding="utf-8",
    )

    details = dpr._status_details_for_task(tmp_path, task_index=0, task_work_path=".")

    assert details["returncode"] == 0
    assert details["t_start"] == 100.0
    assert details["t_end"] == 145.5
    assert details["elapsed_seconds"] == 45.5


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


def test_transfer_archive_filter_removes_stale_dpdispatcher_archives() -> None:
    files = dpr._filter_dpdispatcher_transfer_archives(
        [
            "status.json",
            "a" * 40 + ".tar.gz",
            "./" + "b" * 40 + ".tar",
            "result.tar.gz",
        ]
    )

    assert files == ["status.json", "result.tar.gz"]


def test_safe_ssh_get_files_cleans_corrupt_archive_on_error(tmp_path: Path) -> None:
    class _Submission:
        submission_hash = "c" * 40

    class _SFTP:
        def __init__(self) -> None:
            self.removed: list[str] = []

        def remove(self, path: str) -> None:
            self.removed.append(path)

    class _Context:
        submission = _Submission()
        local_root = str(tmp_path)
        remote_root = "/remote/work"

        def __init__(self) -> None:
            self.sftp = _SFTP()

    seen_files: list[str] = []
    archive = tmp_path / ("c" * 40 + ".tar.gz")
    archive.write_bytes(b"stale")

    def _original(context: _Context, files: list[str], *, tar_compress: bool = True) -> None:
        seen_files.extend(files)
        archive.write_bytes(b"")
        raise EOFError("truncated gzip")

    context = _Context()
    with pytest.raises(EOFError):
        dpr._safe_ssh_get_files_call(
            _original,
            context,
            ["status.json", "c" * 40 + ".tar.gz"],
            tar_compress=True,
        )

    assert seen_files == ["status.json"]
    assert not archive.exists()
    assert context.sftp.removed == ["/remote/work/" + "c" * 40 + ".tar.gz"]


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
