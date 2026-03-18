from __future__ import annotations

import json
from pathlib import Path

import pytest
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

pytest.importorskip("pymatgen")

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.neb_tools import VaspNebPrepareInput, vasp_neb_prepare
from catmaster.tools.geometry_inputs.vasp_inputs import StructWriter


def _write_poscar(path: Path, frac_x: float) -> None:
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["H"],
        coords=[[frac_x, 0.0, 0.0]],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt="poscar")


def _fake_write_vasp_inputs(self, structure, output_dir, **kwargs):
    _ = (self, structure, kwargs)
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    (out / "INCAR").write_text(
        "\n".join(
            [
                "EDIFF = 1E-06",
                "ISMEAR = 0",
                "SIGMA = 0.1",
                "LWAVE = .FALSE.",
                "LCHARG = .FALSE.",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (out / "KPOINTS").write_text("fake kpoints\n", encoding="utf-8")
    (out / "POTCAR").write_text("fake potcar\n", encoding="utf-8")
    return None


def test_user_incar_patch_accepts_element_map_for_magmom() -> None:
    params = VaspNebPrepareInput(
        initial_path="tests/assets/Fe.cif",
        final_path="tests/assets/Fe.cif",
        output_root="tests/test_output/neb_prepare",
        user_incar_patch={"magmom": {"O": 1}, "nupdown": 2},
    )
    assert params.user_incar_patch["MAGMOM"] == {"O": 1}
    assert params.user_incar_patch["NUPDOWN"] == 2


def test_user_incar_patch_rejects_magmom_list() -> None:
    with pytest.raises(
        ValidationError,
        match="MAGMOM must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspNebPrepareInput(
            initial_path="tests/assets/Fe.cif",
            final_path="tests/assets/Fe.cif",
            output_root="tests/test_output/neb_prepare",
            user_incar_patch={"MAGMOM": [1, 1]},
        )


def test_user_incar_patch_rejects_ldauj_symbol_value_form() -> None:
    with pytest.raises(
        ValidationError,
        match="LDAUJ must be an element-map in this tool due to pymatgen constraints",
    ):
        VaspNebPrepareInput(
            initial_path="tests/assets/Fe.cif",
            final_path="tests/assets/Fe.cif",
            output_root="tests/test_output/neb_prepare",
            user_incar_patch={"LDAUJ": [{"symbol": "Fe", "value": 0.1}]},
        )


def test_user_incar_patch_allows_non_element_map_list_for_other_keys() -> None:
    params = VaspNebPrepareInput(
        initial_path="tests/assets/Fe.cif",
        final_path="tests/assets/Fe.cif",
        output_root="tests/test_output/neb_prepare",
        user_incar_patch={"LDAUL": [2, 2]},
    )
    assert params.user_incar_patch == {"LDAUL": [2, 2]}


def test_endpoint_mode_requires_both_initial_and_final_paths() -> None:
    with pytest.raises(ValidationError, match="Endpoint mode requires both initial_path and final_path"):
        VaspNebPrepareInput(
            initial_path="tests/assets/Fe.cif",
            output_root="tests/test_output/neb_prepare",
        )


def test_source_mode_rejects_mixing_endpoint_and_image_tree_inputs() -> None:
    with pytest.raises(ValidationError, match="Provide exactly one NEB source mode"):
        VaspNebPrepareInput(
            initial_path="tests/assets/Fe.cif",
            final_path="tests/assets/Fe.cif",
            images_root="tests/assets",
            output_root="tests/test_output/neb_prepare",
        )


def test_vasp_neb_prepare_from_endpoints_writes_root_and_image_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.0)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.2)

        _content, artifact = vasp_neb_prepare(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_root": "jobs/neb_case",
                "n_images": 3,
                "user_incar_patch": {"EDIFF": 1e-5},
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "jobs" / "neb_case"
    assert data["output_root_rel"] == "jobs/neb_case"
    assert data["num_total_images"] == 5
    assert data["num_intermediate_images"] == 3
    for idx in range(5):
        assert (output_root / f"{idx:02d}" / "POSCAR").is_file()
    incar_text = (output_root / "INCAR").read_text(encoding="utf-8")
    assert "IMAGES = 3" in incar_text
    assert "IBRION = 3" in incar_text
    assert "IOPT = 7" in incar_text
    assert "EDIFF = 1e-05" in incar_text
    diff = json.loads((output_root / "neb_incar_patch.json").read_text(encoding="utf-8"))
    assert diff["IMAGES"]["new"] == "3"
    assert diff["EDIFF"]["new"] == "1e-05"


def test_vasp_neb_prepare_safe_patch_blocks_protected_override(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.0)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.2)
        with pytest.raises(CatMasterToolExecutionError, match="protected NEB INCAR key IBRION"):
            vasp_neb_prepare(
                {
                    "initial_path": "inputs/IS.vasp",
                    "final_path": "inputs/FS.vasp",
                    "output_root": "jobs/neb_case",
                    "user_incar_patch": {"IBRION": 1},
                    "patch_policy": "safe",
                }
            )


def test_vasp_neb_prepare_force_patch_allows_protected_override_from_image_tree(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar(images_root / "00" / "POSCAR", 0.0)
        _write_poscar(images_root / "01" / "POSCAR", 0.1)
        _write_poscar(images_root / "02" / "POSCAR", 0.2)

        _content, artifact = vasp_neb_prepare(
            {
                "images_root": "prepared_images",
                "output_root": "jobs/neb_from_tree",
                "user_incar_patch": {"IBRION": 1, "EDIFF": 2e-6},
                "patch_policy": "force",
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "jobs" / "neb_from_tree"
    assert data["num_total_images"] == 3
    assert data["num_intermediate_images"] == 1
    incar_text = (output_root / "INCAR").read_text(encoding="utf-8")
    assert "IBRION = 1" in incar_text
    assert "IMAGES = 1" in incar_text
    assert "EDIFF = 2e-06" in incar_text
    diff = json.loads((output_root / "neb_incar_patch.json").read_text(encoding="utf-8"))
    assert diff["IBRION"]["new"] == "1"
    assert diff["IMAGES"]["new"] == "1"
    assert diff["EDIFF"]["new"] == "2e-06"
