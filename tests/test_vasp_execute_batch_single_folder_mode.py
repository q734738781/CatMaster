from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution import vasp_dispatch


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
        monkeypatch.setattr(
            vasp_dispatch,
            "render_task_fields",
            lambda cfg, payload, stage_dir: {
                "command": "echo run",
                "forward_files": [],
                "backward_files": ["*"],
            },
        )
        monkeypatch.setattr(
            vasp_dispatch,
            "dispatch_submission",
            lambda req: SimpleNamespace(
                task_states=["5"],
                submission_dir=str((files_root / "outs" / "_fake_submission").resolve()),
                work_base=req.work_base,
                duration_s=0.01,
            ),
        )

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
