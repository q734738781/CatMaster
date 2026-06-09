from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read as ase_read
from pydantic import ValidationError
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

pytest.importorskip("pymatgen")

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.neb_tools import (
    EstimateNebImageCountInput,
    MakeNebGeometryInput,
    RemapNebEndpointAtomsInput,
    VaspNebPrepareInput,
    estimate_neb_image_count,
    make_neb_geometry,
    remap_neb_endpoint_atoms,
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


def _write_two_atom_cart_poscar(path: Path, species: list[str], positions: list[list[float]]) -> None:
    structure = Structure(
        lattice=Lattice.cubic(8.0),
        species=species,
        coords=positions,
        coords_are_cartesian=True,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure, sort_structure=False).write_file(str(path))


def _write_poscar_with_sd(path: Path, species: list[str], coords: list[list[float]], selective_dynamics: list[list[bool]]) -> None:
    structure = Structure(
        lattice=Lattice.cubic(5.0),
        species=species,
        coords=coords,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure, selective_dynamics=selective_dynamics, sort_structure=False).write_file(str(path))


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


def test_remap_neb_endpoint_atoms_input_accepts_defaults() -> None:
    params = RemapNebEndpointAtomsInput(
        initial_path="tests/assets/Fe.cif",
        final_path="tests/assets/Fe.cif",
    )
    assert params.mic is True
    assert params.lock_if_current_displacement_below_angstrom == 0.5
    assert params.overwrite is False


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


def test_vasp_neb_prepare_warns_how_to_copy_missing_endpoint_outcars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        initial = tmp_path / "files" / "inputs" / "is_run" / "CONTCAR"
        final = tmp_path / "files" / "inputs" / "fs_run" / "CONTCAR"
        _write_poscar(initial, 0.0)
        _write_poscar(final, 0.2)

        _content, artifact = vasp_neb_prepare(
            {
                "initial_path": "inputs/is_run/CONTCAR",
                "final_path": "inputs/fs_run/CONTCAR",
                "output_root": "jobs/neb_case_missing_outcar",
                "n_images": 3,
            }
        )

    warnings = artifact["warnings"]
    assert any("Copy the original relax OUTCAR into jobs/neb_case_missing_outcar/00/OUTCAR" in item for item in warnings)
    assert any("Copy the original relax OUTCAR into jobs/neb_case_missing_outcar/04/OUTCAR" in item for item in warnings)


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


def test_make_neb_geometry_warns_for_short_interatomic_distance(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_two_atom_cart_poscar(tmp_path / "files" / "inputs" / "IS.vasp", ["H", "H"], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        _write_two_atom_cart_poscar(tmp_path / "files" / "inputs" / "FS.vasp", ["H", "H"], [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        _content, artifact = make_neb_geometry(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_dir": "neb_images_overlap",
                "n_images": 1,
                "mic": False,
            }
        )

    warnings = artifact["warnings"]
    geometry_check = artifact["data"]["geometry_check"]
    assert any("minimum interatomic distance below 0.80 Angstrom" in item for item in warnings)
    assert geometry_check["short_distance_count"] == 1
    assert geometry_check["min_pair_distance_image"] == "01"
    assert geometry_check["min_pair_distance_angstrom"] == pytest.approx(0.0)
    assert geometry_check["short_distance_records"][0]["atom_indices_1based"] == [1, 2]


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


def test_make_neb_geometry_rejects_element_sequence_mismatch(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        _write_two_atom_poscar(tmp_path / "files" / "inputs" / "IS.vasp", ["H", "O"], [0.0, 0.2])
        _write_two_atom_poscar(tmp_path / "files" / "inputs" / "FS.vasp", ["O", "H"], [0.1, 0.3])

        with pytest.raises(CatMasterToolExecutionError, match="different element sequences"):
            make_neb_geometry(
                {
                    "initial_path": "inputs/IS.vasp",
                    "final_path": "inputs/FS.vasp",
                    "output_dir": "bad_neb_images",
                    "n_images": 1,
                }
            )


def test_remap_neb_endpoint_atoms_reorders_mobile_same_species_atoms_and_preserves_sd(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        initial = tmp_path / "files" / "inputs" / "IS.vasp"
        final = tmp_path / "files" / "inputs" / "FS.vasp"
        _write_poscar_with_sd(
            initial,
            ["Cu", "Cu", "Cu", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.50, 0.00, 0.00],
                [0.10, 0.50, 0.50],
                [0.90, 0.50, 0.50],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
            ],
        )
        _write_poscar_with_sd(
            final,
            ["Cu", "Cu", "Cu", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.50, 0.00, 0.00],
                [0.88, 0.50, 0.50],
                [0.12, 0.50, 0.50],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
            ],
        )

        _content, artifact = remap_neb_endpoint_atoms(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_path": "mapped/fs_mapped.vasp",
            }
        )

    data = artifact["data"]
    assert data["mobile_atom_indices"] == [3, 4]
    assert data["mobile_atom_indices_source"] == "selective_dynamics"
    assert data["lock_if_current_displacement_below_angstrom"] == pytest.approx(0.5)
    assert data["locked_small_displacement_indices"] == []
    assert data["remap_candidate_indices"] == [3, 4]
    assert data["mapping_changed"] is True
    assert data["rss_displacement_mapped_order_angstrom"] < data["rss_displacement_current_order_angstrom"]
    mapped = Poscar.from_file(str(tmp_path / "files" / "mapped" / "fs_mapped.vasp"))
    sd = np.asarray(mapped.selective_dynamics, dtype=bool)
    assert sd.shape == (5, 3)
    assert np.all(sd[:3] == np.array([[False, False, False]] * 3))
    assert np.all(sd[3:] == np.array([[True, True, True], [True, True, True]]))
    mapped_atoms = ase_read(tmp_path / "files" / "mapped" / "fs_mapped.vasp")
    scaled = mapped_atoms.get_scaled_positions(wrap=False)[:, 0]
    assert scaled[0] == pytest.approx(0.00)
    assert scaled[1] == pytest.approx(0.25)
    assert scaled[2] == pytest.approx(0.50)
    assert scaled[3] == pytest.approx(0.12)
    assert scaled[4] == pytest.approx(0.88)


def test_remap_neb_endpoint_atoms_locks_small_displacement_mobile_atoms(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        initial = tmp_path / "files" / "inputs" / "IS.vasp"
        final = tmp_path / "files" / "inputs" / "FS.vasp"
        _write_poscar_with_sd(
            initial,
            ["Cu", "Cu", "Cu", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.50, 0.00, 0.00],
                [0.10, 0.50, 0.50],
                [0.30, 0.50, 0.50],
                [0.90, 0.50, 0.50],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        )
        _write_poscar_with_sd(
            final,
            ["Cu", "Cu", "Cu", "H", "H", "H"],
            [
                [0.00, 0.00, 0.00],
                [0.25, 0.00, 0.00],
                [0.50, 0.00, 0.00],
                [0.12, 0.50, 0.50],
                [0.88, 0.50, 0.50],
                [0.91, 0.50, 0.50],
            ],
            [
                [False, False, False],
                [False, False, False],
                [False, False, False],
                [True, True, True],
                [True, True, True],
                [True, True, True],
            ],
        )

        _content, artifact = remap_neb_endpoint_atoms(
            {
                "initial_path": "inputs/IS.vasp",
                "final_path": "inputs/FS.vasp",
                "output_path": "mapped/fs_locked.vasp",
            }
        )

    data = artifact["data"]
    assert data["locked_small_displacement_indices"] == [3, 5]
    assert data["remap_candidate_indices"] == [4]
    mapped_atoms = ase_read(tmp_path / "files" / "mapped" / "fs_locked.vasp")
    scaled = mapped_atoms.get_scaled_positions(wrap=False)[:, 0]
    assert scaled[3] == pytest.approx(0.12)
    assert scaled[4] == pytest.approx(0.88)
    assert scaled[5] == pytest.approx(0.91)


def test_remap_neb_endpoint_atoms_refuses_fixed_atom_order_mismatch(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        initial = tmp_path / "files" / "inputs" / "IS.vasp"
        final = tmp_path / "files" / "inputs" / "FS.vasp"
        _write_poscar_with_sd(
            initial,
            ["Cu", "O", "H"],
            [[0.00, 0.00, 0.00], [0.25, 0.00, 0.00], [0.75, 0.50, 0.50]],
            [[False, False, False], [False, False, False], [True, True, True]],
        )
        _write_poscar_with_sd(
            final,
            ["O", "Cu", "H"],
            [[0.26, 0.00, 0.00], [0.01, 0.00, 0.00], [0.77, 0.50, 0.50]],
            [[False, False, False], [False, False, False], [True, True, True]],
        )

        with pytest.raises(CatMasterToolExecutionError, match="Frozen atoms are excluded from remapping"):
            remap_neb_endpoint_atoms(
                {
                    "initial_path": "inputs/IS.vasp",
                    "final_path": "inputs/FS.vasp",
                    "output_path": "mapped/fs_bad.vasp",
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
    warnings = artifact["warnings"]
    assert any("Copy the original relax OUTCAR into jobs/neb_from_tree/00/OUTCAR" in item for item in warnings)
    assert any("Copy the original relax OUTCAR into jobs/neb_from_tree/02/OUTCAR" in item for item in warnings)


def test_vasp_neb_prepare_image_tree_rejects_element_sequence_mismatch(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar_with_sd(
            images_root / "00.vasp",
            ["H", "O"],
            [[0.0, 0.0, 0.0], [0.2, 0.5, 0.5]],
            [[True, True, True], [True, True, True]],
        )
        _write_poscar_with_sd(
            images_root / "01.vasp",
            ["O", "H"],
            [[0.1, 0.0, 0.0], [0.3, 0.5, 0.5]],
            [[True, True, True], [True, True, True]],
        )

        with pytest.raises(CatMasterToolExecutionError, match="different element sequences"):
            vasp_neb_prepare(
                {
                    "images_root": "prepared_images",
                    "output_root": "jobs/bad_neb_from_tree",
                }
            )


def test_vasp_neb_prepare_image_tree_warns_for_short_interatomic_distance(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_two_atom_cart_poscar(images_root / "00.vasp", ["H", "H"], [[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        _write_two_atom_cart_poscar(images_root / "01.vasp", ["H", "H"], [[1.0, 0.0, 0.0], [1.2, 0.0, 0.0]])
        _write_two_atom_cart_poscar(images_root / "02.vasp", ["H", "H"], [[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]])

        _content, artifact = vasp_neb_prepare(
            {
                "images_root": "prepared_images",
                "output_root": "jobs/neb_from_short_tree",
            }
        )

    warnings = artifact["warnings"]
    geometry_check = artifact["data"]["geometry_check"]
    assert any("minimum interatomic distance below 0.80 Angstrom" in item for item in warnings)
    assert geometry_check["short_distance_count"] == 1
    assert geometry_check["min_pair_distance_image"] == "01"
    assert geometry_check["min_pair_distance_angstrom"] == pytest.approx(0.2)


def test_vasp_neb_prepare_accepts_flat_image_tree_and_warns_for_endpoint_outcars(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar(images_root / "00.vasp", 0.0)
        _write_poscar(images_root / "01.vasp", 0.1)
        _write_poscar(images_root / "02.vasp", 0.2)

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
    assert not (output_root / "00" / "OUTCAR").exists()
    assert not (output_root / "02" / "OUTCAR").exists()
    warnings = artifact["warnings"]
    assert any("Copy the original relax OUTCAR into jobs/neb_from_flat_tree/00/OUTCAR" in item for item in warnings)
    assert any("Copy the original relax OUTCAR into jobs/neb_from_flat_tree/02/OUTCAR" in item for item in warnings)


def test_vasp_neb_prepare_image_tree_warns_when_task_local_outcars_missing(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(StructWriter, "write_vasp_inputs", _fake_write_vasp_inputs)

    with workspace_scope(tmp_path):
        images_root = tmp_path / "files" / "prepared_images"
        _write_poscar(images_root / "00.vasp", 0.0)
        _write_poscar(images_root / "01.vasp", 0.1)
        _write_poscar(images_root / "02.vasp", 0.2)

        _content, artifact = vasp_neb_prepare(
            {
                "images_root": "prepared_images",
                "output_root": "jobs/neb_missing_outcars",
            }
        )

    warnings = artifact["warnings"]
    assert any("Copy the original relax OUTCAR into jobs/neb_missing_outcars/00/OUTCAR" in item for item in warnings)
    assert any("Copy the original relax OUTCAR into jobs/neb_missing_outcars/02/OUTCAR" in item for item in warnings)


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
        _write_poscar(batch_root / "task1" / "00.vasp", 0.3)
        _write_poscar(batch_root / "task1" / "01.vasp", 0.4)
        _write_poscar(batch_root / "task1" / "02.vasp", 0.5)

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
    assert (output_root / "task1" / "00" / "POSCAR").is_file()
    assert (output_root / "task1" / "02" / "POSCAR").is_file()
    warnings = artifact["warnings"]
    assert any("task0: initial endpoint OUTCAR not provided for image-tree input." in item for item in warnings)
    assert any("task0: final endpoint OUTCAR not provided for image-tree input." in item for item in warnings)
    assert any("task1: initial endpoint OUTCAR not provided for image-tree input." in item for item in warnings)
    assert any("task1: final endpoint OUTCAR not provided for image-tree input." in item for item in warnings)
