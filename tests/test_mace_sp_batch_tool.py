from __future__ import annotations

from pathlib import Path

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import mace_sp_batch
from catmaster.tools.registry import ToolRegistry


def test_registry_contains_mace_sp_batch() -> None:
    pytest.importorskip("pymatgen")
    registry = ToolRegistry()
    assert "mace_sp_batch" in registry.list_tools()


def test_mace_sp_batch_rejects_output_inside_input(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "POSCAR").write_text("dummy", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_sp_batch(
                {
                    "input_dir": "inputs",
                    "output_root": "inputs/outputs",
                }
            )
    assert "must not be inside input_dir" in str(excinfo.value)


def test_mace_sp_batch_requires_structure_files(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "empty_inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            mace_sp_batch(
                {
                    "input_dir": "empty_inputs",
                    "output_root": "outputs",
                }
            )
    assert "No structure files found" in str(excinfo.value)
