from __future__ import annotations

from pathlib import Path

from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import MaceRelaxBatchInput, mace_relax_batch
from catmaster.tools.execution.task_registry import TaskRegistry


def test_mace_relax_batch_input_relax_lattice_default_and_override() -> None:
    default_params = MaceRelaxBatchInput(input_dir="inputs", output_root="outputs")
    assert default_params.relax_lattice is False

    override_params = MaceRelaxBatchInput(
        input_dir="inputs",
        output_root="outputs",
        relax_lattice=True,
    )
    assert override_params.relax_lattice is True


def test_mace_relax_batch_accepts_relax_lattice_field(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        result = mace_relax_batch(
            {
                "input_dir": "inputs",
                "output_root": "inputs/outputs",
                "relax_lattice": True,
            }
        )
    assert result["status"] == "failed"
    assert "must not be inside input_dir" in (result.get("error") or "")


def test_mace_relax_dir_task_command_has_relax_lattice_placeholder() -> None:
    cfg = TaskRegistry().get("mace_relax_dir")
    assert "--relax_lattice {relax_lattice}" in cfg.command
