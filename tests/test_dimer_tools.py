from __future__ import annotations

import json
import sys
import types
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp.inputs import Poscar

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.dimer_tools import (
    _build_local_mace_calculator,
    _compute_mace_vibrational_modes,
    _normalize_hessian_2d,
    mace_analyze_frequencies,
    make_dimer_mode_from_mace,
    make_dimer_mode_from_neb,
    vasp_dimer_prepare,
)


def _write_structure(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms, format="vasp", direct=False, vasp5=True)


def _write_poscar_with_selective_dynamics(path: Path, *, selective_dynamics: list[list[bool]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    structure = Structure(
        Lattice.cubic(8.0),
        ["H", "He", "Li"],
        [[0, 0, 0], [0.0, 0.0, 0.2], [0.2, 0.0, 0.0]],
    )
    Poscar(structure, selective_dynamics=selective_dynamics).write_file(path)


def _read_tail_vectors(path: Path, count: int) -> np.ndarray:
    lines = path.read_text(encoding="utf-8").splitlines()
    tail = lines[-count:]
    return np.asarray([[float(token) for token in line.split()] for line in tail], dtype=float)


def test_vasp_dimer_prepare_appends_mass_normalized_vectors(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        atoms = Atoms("HO", positions=[[0, 0, 0], [0.7, 0.0, 0.0]], cell=[10, 10, 10], pbc=[False, False, False])
        _write_structure(files_root / "ts_guess.vasp", atoms)
        (files_root / "mode.txt").write_text("1.0 0.0 0.0\n1.0 0.0 0.0\n", encoding="utf-8")

        _content, artifact = vasp_dimer_prepare(
            {
                "input_path": "ts_guess.vasp",
                "output_root": "dimer_job",
                "mode_text_path": "mode.txt",
                "regime": "gas",
            }
        )

        data = artifact["data"]
        incar_text = (files_root / "dimer_job" / "INCAR").read_text(encoding="utf-8")
        appended = _read_tail_vectors(files_root / "dimer_job" / "POSCAR", 2)

    masses = np.sqrt(np.asarray([1.008, 15.999], dtype=float)).reshape(-1, 1)
    expected = np.asarray([[1.0, 0.0, 0.0], [1.0, 0.0, 0.0]], dtype=float) / masses
    expected /= np.linalg.norm(expected.reshape(-1))
    assert "IBRION = 44" in incar_text
    assert data["ibrion"] == 44
    assert Path(tmp_path / "files" / data["raw_mode_rel"]).is_file()
    assert Path(tmp_path / "files" / data["mass_normalized_mode_rel"]).is_file()
    assert np.allclose(appended, expected, atol=1e-8)


def test_make_dimer_mode_from_neb_uses_adjacent_images(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        images_root = files_root / "neb_images"
        for idx, x in enumerate([0.0, 1.0, 2.0, 3.0, 4.0]):
            atoms = Atoms("H", positions=[[x, 0.0, 0.0]], cell=[10, 10, 10], pbc=[False, False, False])
            _write_structure(images_root / f"{idx:02d}.vasp", atoms)

        _content, artifact = make_dimer_mode_from_neb(
            {
                "images_root": "neb_images",
                "output_root": "neb_mode",
                "use_mic": False,
            }
        )
        data = artifact["data"]
        raw = np.loadtxt(files_root / "neb_mode" / "dimer_mode_raw.txt").reshape(1, 3)
        normalized = np.loadtxt(files_root / "neb_mode" / "dimer_mode_mass_normalized.txt").reshape(1, 3)
        summary = json.loads((files_root / "neb_mode" / "summary.json").read_text(encoding="utf-8"))

    assert data["ts_image_index"] == 2
    assert np.allclose(raw, [[2.0, 0.0, 0.0]])
    assert np.allclose(normalized, [[1.0, 0.0, 0.0]])
    assert summary["neighbor_indices"] == [1, 3]


def test_make_dimer_mode_from_mace_selects_most_imaginary_mode(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_compute(**kwargs):
        atoms = kwargs["atoms"]
        records = [
            {"mode_index": 0, "frequency_cm1": -120.0, "imaginary": True},
            {"mode_index": 1, "frequency_cm1": -40.0, "imaginary": True},
            {"mode_index": 2, "frequency_cm1": 55.0, "imaginary": False},
        ]
        modes = [
            np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.zeros((len(atoms), 3), dtype=float),
        ]
        return records, modes, {"calculator": {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"}}

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes", _fake_compute)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        atoms = Atoms("HH", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[8, 8, 8], pbc=[False, False, False])
        _write_structure(files_root / "ts_guess.vasp", atoms)

        _content, artifact = make_dimer_mode_from_mace(
            {
                "input_path": "ts_guess.vasp",
                "output_root": "mace_mode",
                "model": "mh-1",
            }
        )
        data = artifact["data"]
        raw = np.loadtxt(files_root / "mace_mode" / "dimer_mode_raw.txt").reshape(2, 3)
        summary = json.loads((files_root / "mace_mode" / "summary.json").read_text(encoding="utf-8"))

    assert data["selected_mode_index"] == 0
    assert data["imaginary_mode_count"] == 2
    assert np.allclose(raw[0], [2.0, 0.0, 0.0])
    assert summary["selected_frequency_cm1"] == pytest.approx(-120.0)


def test_mace_analyze_frequencies_exports_all_modes(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_compute(**kwargs):
        atoms = kwargs["atoms"]
        assert kwargs["method"] == "auto"
        records = [
            {"mode_index": 0, "frequency_cm1": -120.0, "imaginary": True},
            {"mode_index": 1, "frequency_cm1": -40.0, "imaginary": True},
            {"mode_index": 2, "frequency_cm1": 55.0, "imaginary": False},
        ]
        modes = [
            np.asarray([[2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.asarray([[0.0, 1.0, 0.0], [0.0, 0.0, 0.0]], dtype=float),
            np.asarray([[0.0, 0.0, 1.0], [0.0, 0.0, 0.0]], dtype=float),
        ]
        return records, modes, {
            "calculator": {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"},
            "active_atom_indices": list(range(len(atoms))),
            "method_used": "finite_difference",
            "fallback_notes": [],
        }

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes", _fake_compute)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        atoms = Atoms("HH", positions=[[0, 0, 0], [0.8, 0, 0]], cell=[8, 8, 8], pbc=[False, False, False])
        _write_structure(files_root / "ts_guess.vasp", atoms)

        _content, artifact = mace_analyze_frequencies(
            {
                "input_path": "ts_guess.vasp",
                "output_root": "mace_freq",
                "model": "mh-1",
            }
        )
        data = artifact["data"]
        summary = json.loads((files_root / "mace_freq" / "summary.json").read_text(encoding="utf-8"))
        csv_text = (files_root / "mace_freq" / "frequencies.csv").read_text(encoding="utf-8")

    assert data["method_used"] == "finite_difference"
    assert data["mode_count"] == 3
    assert data["imaginary_mode_count"] == 2
    assert data["active_atom_indices"] == [0, 1]
    assert data["active_atom_indices_source"] == "all_atoms"
    assert len(data["modes"]) == 3
    assert summary["lowest_frequency_cm1"] == pytest.approx(-120.0)
    assert summary["active_atom_indices"] == [0, 1]
    assert summary["active_atom_indices_source"] == "all_atoms"
    assert summary["modes"][2]["mode_vector_rel"].endswith("mode_0002.txt")
    assert "mode_index,frequency_cm1,imaginary,mode_vector_rel" in csv_text


def test_mace_analyze_frequencies_defaults_to_selective_dynamics_free_atoms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_compute(**kwargs):
        assert kwargs["active_indices"] == [1]
        atoms = kwargs["atoms"]
        records = [{"mode_index": 0, "frequency_cm1": 42.0, "imaginary": False}]
        modes = [np.zeros((len(atoms), 3), dtype=float)]
        return records, modes, {
            "calculator": {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"},
            "active_atom_indices": [1],
            "method_used": "finite_difference",
            "fallback_notes": [],
        }

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes", _fake_compute)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        _write_poscar_with_selective_dynamics(
            files_root / "ts_guess.vasp",
            selective_dynamics=[[False, False, False], [True, True, True], [False, False, False]],
        )

        _content, artifact = mace_analyze_frequencies(
            {
                "input_path": "ts_guess.vasp",
                "output_root": "mace_freq",
                "model": "mh-1",
            }
        )
        data = artifact["data"]
        summary = json.loads((files_root / "mace_freq" / "summary.json").read_text(encoding="utf-8"))

    assert data["active_atom_indices"] == [1]
    assert data["active_atom_indices_source"] == "selective_dynamics"
    assert summary["active_atom_indices"] == [1]
    assert summary["active_atom_indices_source"] == "selective_dynamics"


def test_make_dimer_mode_from_mace_defaults_to_selective_dynamics_free_atoms(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    def _fake_compute(**kwargs):
        assert kwargs["active_indices"] == [1]
        atoms = kwargs["atoms"]
        records = [{"mode_index": 0, "frequency_cm1": -120.0, "imaginary": True}]
        modes = [np.asarray([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [0.0, 0.0, 0.0]], dtype=float)]
        return records, modes, {"calculator": {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"}}

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes", _fake_compute)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        _write_poscar_with_selective_dynamics(
            files_root / "ts_guess.vasp",
            selective_dynamics=[[False, False, False], [True, True, True], [False, False, False]],
        )

        _content, artifact = make_dimer_mode_from_mace(
            {
                "input_path": "ts_guess.vasp",
                "output_root": "mace_mode",
                "model": "mh-1",
            }
        )
        data = artifact["data"]
        summary = json.loads((files_root / "mace_mode" / "summary.json").read_text(encoding="utf-8"))

    assert data["active_atom_indices"] == [1]
    assert data["active_atom_indices_source"] == "selective_dynamics"
    assert summary["active_atom_indices"] == [1]
    assert summary["active_atom_indices_source"] == "selective_dynamics"


def test_compute_mace_vibrational_modes_auto_prefers_finite_difference_for_subset(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[8, 8, 8], pbc=[False, False, False])

    def _fake_build(**kwargs):
        return object(), {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"}

    def _fake_hessian(**kwargs):
        calls.append("hessian")
        return [], []

    def _fake_fd(**kwargs):
        calls.append("finite_difference")
        return [{"mode_index": 0, "frequency_cm1": 42.0, "imaginary": False}], [np.zeros((len(atoms), 3), dtype=float)]

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._build_local_mace_calculator", _fake_build)
    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes_via_hessian", _fake_hessian)
    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes_via_finite_difference", _fake_fd)

    records, modes, info = _compute_mace_vibrational_modes(
        atoms=atoms,
        output_root=Path("/tmp/mace_freq_test"),
        model="mh-1",
        head="omat_pbe",
        dispersion=False,
        device_preference="cpu",
        default_dtype="float64",
        active_indices=[1],
        method="auto",
        delta=0.01,
        nfree=2,
    )

    assert calls == ["finite_difference"]
    assert info["method_used"] == "finite_difference"
    assert len(records) == 1
    assert len(modes) == 1


def test_compute_mace_vibrational_modes_auto_prefers_hessian_for_full_system(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[str] = []
    atoms = Atoms("HO", positions=[[0, 0, 0], [0.7, 0, 0]], cell=[8, 8, 8], pbc=[False, False, False])

    def _fake_build(**kwargs):
        return object(), {"device": "cpu", "source_kind": "pretrained", "source_ref": "mh-1"}

    def _fake_hessian(**kwargs):
        calls.append("hessian")
        return [{"mode_index": 0, "frequency_cm1": -10.0, "imaginary": True}], [np.zeros((len(atoms), 3), dtype=float)]

    def _fake_fd(**kwargs):
        calls.append("finite_difference")
        return [], []

    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._build_local_mace_calculator", _fake_build)
    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes_via_hessian", _fake_hessian)
    monkeypatch.setattr("catmaster.tools.geometry_inputs.dimer_tools._compute_mace_vibrational_modes_via_finite_difference", _fake_fd)

    records, modes, info = _compute_mace_vibrational_modes(
        atoms=atoms,
        output_root=Path("/tmp/mace_freq_test"),
        model="mh-1",
        head="omat_pbe",
        dispersion=False,
        device_preference="cpu",
        default_dtype="float64",
        active_indices=[0, 1],
        method="auto",
        delta=0.01,
        nfree=2,
    )

    assert calls == ["hessian"]
    assert info["method_used"] == "hessian"
    assert len(records) == 1
    assert len(modes) == 1


def test_build_local_mace_calculator_routes_mh1_and_dispersion_through_mace_mp(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    captured: dict[str, object] = {}

    fake_calculators = types.ModuleType("mace.calculators")
    fake_torch = types.ModuleType("torch")
    fake_torch.cuda = types.SimpleNamespace(is_available=lambda: False)

    def _fake_mace_mp(**kwargs):
        captured.update(kwargs)
        return object()

    fake_calculators.mace_mp = _fake_mace_mp
    monkeypatch.setitem(sys.modules, "torch", fake_torch)
    monkeypatch.setitem(sys.modules, "mace.calculators", fake_calculators)

    calc, info = _build_local_mace_calculator(
        model="mh-1",
        head="omat_pbe",
        dispersion=True,
        device_preference="cpu",
        default_dtype="float64",
    )

    assert calc is not None
    assert captured["model"] == "mh-1"
    assert captured["dispersion"] is True
    assert captured["device"] == "cpu"
    assert captured["default_dtype"] == "float64"
    assert captured["head"] == "omat_pbe"
    assert info["source_kind"] == "pretrained"
    assert info["source_ref"] == "mh-1"


def test_normalize_hessian_2d_is_consistent_between_2d_and_4d_inputs() -> None:
    atom_count = 3
    full_hessian = np.arange((atom_count * 3) ** 2, dtype=float).reshape(atom_count * 3, atom_count * 3)
    active_indices = [0, 2]

    from_2d = _normalize_hessian_2d(full_hessian, atom_count=atom_count, active_indices=active_indices)
    from_4d = _normalize_hessian_2d(
        full_hessian.reshape(atom_count, 3, atom_count, 3),
        atom_count=atom_count,
        active_indices=active_indices,
    )

    assert np.array_equal(from_2d, from_4d)


def test_normalize_hessian_2d_matches_manual_cartesian_block_selection() -> None:
    atom_count = 4
    full_hessian = np.arange((atom_count * 3) ** 2, dtype=float).reshape(atom_count * 3, atom_count * 3)
    active_indices = [1, 3]
    manual_cart_indices = np.asarray([3, 4, 5, 9, 10, 11], dtype=int)

    selected = _normalize_hessian_2d(full_hessian, atom_count=atom_count, active_indices=active_indices)
    expected = full_hessian[np.ix_(manual_cart_indices, manual_cart_indices)]

    assert np.array_equal(selected, expected)
