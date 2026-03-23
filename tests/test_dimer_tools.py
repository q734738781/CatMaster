from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest
from ase import Atoms
from ase.io import write as ase_write

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.dimer_tools import (
    make_dimer_mode_from_mace,
    make_dimer_mode_from_neb,
    vasp_dimer_prepare,
)


def _write_structure(path: Path, atoms: Atoms) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    ase_write(str(path), atoms, format="vasp", direct=False, vasp5=True)


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
