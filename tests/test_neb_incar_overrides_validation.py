from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure

pytest.importorskip("pymatgen")

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.neb_tools import (
    EstimateNebImageCountInput,
    MakeNebGeometryInput,
    VaspNebPrepareInput,
    estimate_neb_image_count,
    make_neb_geometry,
    vasp_neb_prepare,
)
from catmaster.tools.geometry_inputs.vasp_inputs import StructWriter


def _write_poscar(path: Path, frac_x: float) -> None:
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=["H"],
        coords=[[frac_x, 0.0, 0.0]],
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    structure.to(filename=str(path), fmt="poscar")


def _write_two_atom_poscar(path: Path, species: list[str], frac_xs: list[float]) -> None:
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=species,
        coords=[[frac_xs[0], 0.0, 0.0], [frac_xs[1], 0.5, 0.5]],
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


def test_estimate_neb_image_count_input_accepts_positive_spacing() -> None:
    params = EstimateNebImageCountInput(
        initial_path="tests/assets/Fe.cif",
        final_path="tests/assets/Fe.cif",
        target_spacing_angstrom=0.8,
    )
    assert params.target_spacing_angstrom == 0.8


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


def test_vasp_neb_prepare_copies_endpoint_outcars_when_present(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        initial = tmp_path / "files" / "inputs" / "is_run" / "CONTCAR"
        final = tmp_path / "files" / "inputs" / "fs_run" / "CONTCAR"
        _write_poscar(initial, 0.0)
        _write_poscar(final, 0.2)
        (initial.parent / "OUTCAR").write_text("initial outcar\n", encoding="utf-8")
        (final.parent / "OUTCAR").write_text("final outcar\n", encoding="utf-8")

        _content, _artifact = vasp_neb_prepare(
            {
                "initial_path": "inputs/is_run/CONTCAR",
                "final_path": "inputs/fs_run/CONTCAR",
                "output_root": "jobs/neb_case_with_outcar",
                "n_images": 3,
            }
        )

    output_root = tmp_path / "files" / "jobs" / "neb_case_with_outcar"
    assert (output_root / "00" / "OUTCAR").read_text(encoding="utf-8") == "initial outcar\n"
    assert (output_root / "04" / "OUTCAR").read_text(encoding="utf-8") == "final outcar\n"


def test_make_neb_geometry_writes_flat_vasp_image_tree(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.0)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.2)

        _content, artifact = make_neb_geometry(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_dir": "neb_images",
                "n_images": 3,
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "neb_images"
    assert data["num_total_images"] == 5
    assert data["image_files"][0] == "neb_images/00.vasp"
    assert data["image_files"][-1] == "neb_images/04.vasp"
    for idx in range(5):
        assert (output_root / f"{idx:02d}.vasp").is_file()


def test_make_neb_geometry_uses_minimum_image_when_mic_enabled(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.9)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.1)

        make_neb_geometry(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_dir": "neb_images_mic",
                "n_images": 3,
                "mic": True,
            }
        )

    output_root = tmp_path / "files" / "neb_images_mic"
    scaled = []
    for idx in range(5):
        atoms = ase_read(output_root / f"{idx:02d}.vasp")
        scaled.append(float(atoms.get_scaled_positions(wrap=False)[0, 0]))
    assert np.allclose(scaled, [0.9, 0.95, 1.0, 1.05, 0.1])


def test_make_neb_geometry_uses_in_cell_path_when_mic_disabled(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.9)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.1)

        make_neb_geometry(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_dir": "neb_images_nomic",
                "n_images": 3,
                "mic": False,
            }
        )

    output_root = tmp_path / "files" / "neb_images_nomic"
    scaled = []
    for idx in range(5):
        atoms = ase_read(output_root / f"{idx:02d}.vasp")
        scaled.append(float(atoms.get_scaled_positions(wrap=False)[0, 0]))
    assert np.allclose(scaled, [0.9, 0.7, 0.5, 0.3, 0.1])


def test_estimate_neb_image_count_uses_periodic_minimum_image_when_enabled(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.9)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.1)

        _content, artifact = estimate_neb_image_count(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "mic": True,
            }
        )

    data = artifact["data"]
    assert data["mic"] is True
    assert data["recommended_intermediate_images"] == 2
    assert data["rss_displacement_angstrom"] == pytest.approx(1.0)
    assert data["max_atom_displacement_angstrom"] == pytest.approx(1.0)
    assert data["per_atom_displacements_angstrom"] == pytest.approx([1.0])
    assert "4-8" in data["typical_intermediate_image_range"]


def test_estimate_neb_image_count_uses_in_cell_distance_when_mic_disabled(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_poscar(tmp_path / "files" / "inputs" / "IS.vasp", 0.9)
        _write_poscar(tmp_path / "files" / "inputs" / "FS.vasp", 0.1)

        _content, artifact = estimate_neb_image_count(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "mic": False,
            }
        )

    data = artifact["data"]
    assert data["mic"] is False
    assert data["recommended_intermediate_images"] == 5
    assert data["rss_displacement_angstrom"] == pytest.approx(4.0)
    assert data["max_atom_displacement_angstrom"] == pytest.approx(4.0)
    assert data["per_atom_displacements_angstrom"] == pytest.approx([4.0])


def test_estimate_neb_image_count_rejects_element_sequence_mismatch(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_two_atom_poscar(tmp_path / "files" / "inputs" / "IS.vasp", ["H", "O"], [0.0, 0.2])
        _write_two_atom_poscar(tmp_path / "files" / "inputs" / "FS.vasp", ["O", "H"], [0.1, 0.3])

        with pytest.raises(CatMasterToolExecutionError, match="different element sequences"):
            estimate_neb_image_count(
                {
                    "initial_path": "inputs/IS.vasp",
                    "final_path": "inputs/FS.vasp",
                }
            )


def test_make_neb_geometry_batch_from_task_root(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        batch_root = tmp_path / "files" / "neb_batch_inputs"
        _write_poscar(batch_root / "task0" / "IS.vasp", 0.0)
        _write_poscar(batch_root / "task0" / "FS.vasp", 0.2)
        _write_poscar(batch_root / "task1" / "IS.vasp", 0.1)
        _write_poscar(batch_root / "task1" / "FS.vasp", 0.3)

        _content, artifact = make_neb_geometry(
            {
                "input_root": "neb_batch_inputs",
                "output_root": "neb_batch_outputs",
                "n_images": 2,
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "neb_batch_outputs"
    assert data["task_count"] == 2
    assert (output_root / "batch_summary.json").is_file()
    assert (output_root / "task0" / "00.vasp").is_file()
    assert (output_root / "task0" / "03.vasp").is_file()
    assert (output_root / "task1" / "00.vasp").is_file()
    assert (output_root / "task1" / "03.vasp").is_file()


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
        (images_root / "IS_OUTCAR").write_text("legacy initial\n", encoding="utf-8")
        (images_root / "FS_OUTCAR").write_text("legacy final\n", encoding="utf-8")

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


def test_vasp_neb_prepare_accepts_flat_image_tree_with_required_endpoint_outcars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar(images_root / "00.vasp", 0.0)
        _write_poscar(images_root / "01.vasp", 0.1)
        _write_poscar(images_root / "02.vasp", 0.2)
        (images_root / "IS_OUTCAR").write_text("initial flat outcar\n", encoding="utf-8")
        (images_root / "FS_OUTCAR").write_text("final flat outcar\n", encoding="utf-8")

        _content, artifact = vasp_neb_prepare(
            {
                "images_root": "prepared_images",
                "output_root": "jobs/neb_from_flat_tree",
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "jobs" / "neb_from_flat_tree"
    assert data["num_total_images"] == 3
    assert data["num_intermediate_images"] == 1
    assert (output_root / "00" / "POSCAR").is_file()
    assert (output_root / "02" / "POSCAR").is_file()
    assert (output_root / "00" / "OUTCAR").read_text(encoding="utf-8") == "initial flat outcar\n"
    assert (output_root / "02" / "OUTCAR").read_text(encoding="utf-8") == "final flat outcar\n"


def test_vasp_neb_prepare_image_tree_requires_task_local_outcars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar(images_root / "00.vasp", 0.0)
        _write_poscar(images_root / "01.vasp", 0.1)
        _write_poscar(images_root / "02.vasp", 0.2)

        with pytest.raises(CatMasterToolExecutionError, match="IS_OUTCAR"):
            vasp_neb_prepare(
                {
                    "images_root": "prepared_images",
                    "output_root": "jobs/neb_missing_outcars",
                }
            )


def test_vasp_neb_prepare_batch_from_task_root(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        batch_root = tmp_path / "files" / "prepared_batch"
        _write_poscar(batch_root / "task0" / "00.vasp", 0.0)
        _write_poscar(batch_root / "task0" / "01.vasp", 0.1)
        _write_poscar(batch_root / "task0" / "02.vasp", 0.2)
        (batch_root / "task0" / "IS_OUTCAR").write_text("task0 initial\n", encoding="utf-8")
        (batch_root / "task0" / "FS_OUTCAR").write_text("task0 final\n", encoding="utf-8")
        _write_poscar(batch_root / "task1" / "00.vasp", 0.3)
        _write_poscar(batch_root / "task1" / "01.vasp", 0.4)
        _write_poscar(batch_root / "task1" / "02.vasp", 0.5)
        (batch_root / "task1" / "IS_OUTCAR").write_text("task1 initial\n", encoding="utf-8")
        (batch_root / "task1" / "FS_OUTCAR").write_text("task1 final\n", encoding="utf-8")

        _content, artifact = vasp_neb_prepare(
            {
                "input_root": "prepared_batch",
                "output_root": "jobs/prepared_batch_out",
            }
        )

    data = artifact["data"]
    output_root = tmp_path / "files" / "jobs" / "prepared_batch_out"
    assert data["task_count"] == 2
    assert (output_root / "batch_summary.json").is_file()
    assert (output_root / "task0" / "00" / "POSCAR").is_file()
    assert (output_root / "task0" / "02" / "OUTCAR").read_text(encoding="utf-8") == "task0 final\n"
    assert (output_root / "task1" / "00" / "POSCAR").is_file()
    assert (output_root / "task1" / "02" / "POSCAR").is_file()
