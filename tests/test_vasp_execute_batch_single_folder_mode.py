from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution import dpdispatcher_runner as dpr
from catmaster.tools.execution import vasp_dispatch
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskRegistry


def _touch_vasp_inputs(calc_dir: Path) -> None:
    calc_dir.mkdir(parents=True, exist_ok=True)
    for name in ("INCAR", "POTCAR", "POSCAR", "KPOINTS"):
        (calc_dir / name).write_text("x\n", encoding="utf-8")


def test_vasp_execute_batch_accepts_single_calc_folder(monkeypatch, tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        calc_dir = files_root / "runs" / "A"
        _touch_vasp_inputs(calc_dir)

        monkeypatch.setattr(vasp_dispatch, "_resolve_machine_for_resources", lambda _: "dummy_machine")
        def _fake_dispatch(req):
            return SimpleNamespace(
                task_states=["5"],
                submission_dir=str((files_root / "outs" / "_fake_submission").resolve()),
                work_base=req.work_base,
                duration_s=0.01,
            )

        monkeypatch.setattr(vasp_dispatch, "dispatch_submission", _fake_dispatch)

        result = vasp_dispatch.vasp_execute_batch(
            {
                "input_dir": "runs/A",
                "output_dir": "outs",
                "check_interval": 1,
            }
        )

    _, artifact = result
    outputs = (artifact.get("data") or {}).get("outputs") or []
    assert len(outputs) == 1
    assert outputs[0]["input_dir_rel"].endswith("runs/A")
    assert outputs[0]["output_dir_rel"].endswith("outs/A")


def test_vasp_execute_batch_accepts_custom_task_name_for_neb(monkeypatch, tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        calc_dir = files_root / "runs" / "neb_case"
        _touch_vasp_inputs(calc_dir)

        captured = {"resources": "", "task_work_paths": []}

        def _fake_resolve(resources_key: str) -> str:
            captured["resources"] = resources_key
            return "dummy_machine"

        def _fake_dispatch(req):
            captured["task_work_paths"] = [task.task_work_path for task in req.tasks]
            return SimpleNamespace(
                task_states=["5"],
                submission_dir=str((files_root / "outs" / "_fake_submission").resolve()),
                work_base=req.work_base,
                duration_s=0.01,
            )

        monkeypatch.setattr(vasp_dispatch, "_resolve_machine_for_resources", _fake_resolve)
        monkeypatch.setattr(vasp_dispatch, "dispatch_submission", _fake_dispatch)

        _content, artifact = vasp_dispatch.vasp_execute_batch(
            {
                "input_dir": "runs/neb_case",
                "output_dir": "outs",
                "check_interval": 1,
                "task_name": "vasp_execute_neb",
            }
        )

    data = artifact.get("data") or {}
    assert captured["resources"] == "vasp_cpu_neb"
    assert captured["task_work_paths"] == ["neb_case"]
    assert data.get("task_name") == "vasp_execute_neb"
    assert data.get("resources_key") == "vasp_cpu_neb"


def test_vasp_execute_batch_failure_exposes_remote_context(monkeypatch, tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        calc_dir = files_root / "runs" / "A"
        _touch_vasp_inputs(calc_dir)

        monkeypatch.setattr(vasp_dispatch, "_resolve_machine_for_resources", lambda _: "dummy_machine")

        def _fake_dispatch(req):
            raise dpr.DPDispatcherDispatchError(
                "Connection reset by peer",
                remote_context={
                    "remote_context_id": "dp_ctx",
                    "submitted_at": "2026-05-19T00:00:00+08:00",
                    "updated_at": "2026-05-19T00:01:00+08:00",
                    "submission_hash": "abc",
                    "jobs": [{"job_hash": "jobhash", "job_id": "3849", "status_code": 3, "status": "running"}],
                    "job_status_counts": {"running": 1},
                    "receipt_rel": ".deepagents/dpdispatcher/receipts/dp_ctx.json",
                },
            )

        monkeypatch.setattr(vasp_dispatch, "dispatch_submission", _fake_dispatch)

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            vasp_dispatch.vasp_execute_batch(
                {
                    "input_dir": "runs/A",
                    "output_dir": "outs",
                    "check_interval": 1,
                }
            )

    text = str(excinfo.value)
    data = excinfo.value.artifact["data"]
    assert "remote_context_id=dp_ctx" in text
    assert "submission_hash=abc" in text
    assert data["remote_context_id"] == "dp_ctx"
    assert data["submission_hash"] == "abc"
    assert data["jobs"][0]["status"] == "running"
    assert data["job_status_counts"] == {"running": 1}


def test_vasp_execute_batch_rejects_nested_when_root_is_calc_folder(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        root = files_root / "runs" / "A"
        _touch_vasp_inputs(root)
        _touch_vasp_inputs(root / "nested" / "B")

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            vasp_dispatch.vasp_execute_batch(
                {
                    "input_dir": "runs/A",
                    "output_dir": "outs",
                    "check_interval": 1,
                }
            )

    assert "Nested calc folders are not allowed" in str(excinfo.value)


def test_vasp_execute_batch_rejects_output_overlap_before_dispatch(
    monkeypatch, tmp_path: Path
) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        calc_dir = files_root / "runs" / "A"
        _touch_vasp_inputs(calc_dir)

        called = {"dispatch": 0}

        monkeypatch.setattr(vasp_dispatch, "_resolve_machine_for_resources", lambda _: "dummy_machine")
        monkeypatch.setattr(
            vasp_dispatch,
            "dispatch_submission",
            lambda req: called.__setitem__("dispatch", called["dispatch"] + 1),  # should never be called
        )

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            vasp_dispatch.vasp_execute_batch(
                {
                    "input_dir": "runs/A",
                    "output_dir": "runs",
                    "check_interval": 1,
                }
            )

    assert "output_dir mapping overlaps input calc directories" in str(excinfo.value)
    assert called["dispatch"] == 0


def test_vasp_execute_batch_rejects_nested_calc_inside_batch_member(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        root = files_root / "runs"
        _touch_vasp_inputs(root / "A")
        _touch_vasp_inputs(root / "A" / "nested" / "B")
        _touch_vasp_inputs(root / "C")

        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            vasp_dispatch.vasp_execute_batch(
                {
                    "input_dir": "runs",
                    "output_dir": "outs",
                    "check_interval": 1,
                }
            )

    assert "Nested calc folders are not allowed inside submitted calc directories" in str(excinfo.value)


def test_render_task_fields_keeps_recursive_star_for_neb_tree(tmp_path: Path) -> None:
    calc_dir = tmp_path / "neb_case"
    _touch_vasp_inputs(calc_dir)
    (calc_dir / "00").mkdir()
    (calc_dir / "00" / "POSCAR").write_text("x\n", encoding="utf-8")

    cfg = TaskRegistry().get("vasp_execute")
    rendered = render_task_fields(cfg, {"task_name": "vasp_execute"}, calc_dir)

    assert rendered["forward_files"] == ["*"]


def test_render_task_fields_star_supersedes_explicit_entries(tmp_path: Path) -> None:
    calc_dir = tmp_path / "neb_case"
    _touch_vasp_inputs(calc_dir)

    cfg = TaskRegistry().get("vasp_execute").model_copy(
        update={"forward_files": ["*", "INCAR", "task_script/vasp_boot.py"]}
    )
    rendered = render_task_fields(cfg, {"task_name": "vasp_execute"}, calc_dir)

    assert rendered["forward_files"] == ["*"]
