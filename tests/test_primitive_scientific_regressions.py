from __future__ import annotations

import importlib
import json
from pathlib import Path

import numpy as np
import pytest
from ase.io import read
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs import crystal_tool as crystal
from catmaster.tools.geometry_inputs import molecular_qchem as molecules
from catmaster.tools.geometry_inputs import slab_tools
from catmaster.tools.geometry_inputs.batch_paths import batch_names


@pytest.fixture
def files(tmp_path):
    with workspace_scope(tmp_path):
        root = tmp_path / "files"
        root.mkdir(exist_ok=True)
        yield root


def write_xyz(path, distances):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(f"2\nfixture\nH 0 0 0\nH 0 0 {distance}\n" for distance in distances))


def nacl():
    return Structure(Lattice.cubic(4), ["Na", "Cl"], [[0, 0, 0], [.5, .5, .5]])


@pytest.mark.parametrize("tool,payload,summary", [
    (crystal.supercell, {"structure_dir": "in", "output_dir": "out", "supercell": [1, 1, 1]}, "batch_supercell.json"),
    (slab_tools.fix_atoms_by_layers, {"structure_dir": "in", "output_dir": "out", "freeze_layers": 1}, "batch_fix_atoms_by_layers.json"),
    (slab_tools.fix_atoms_by_height, {"structure_dir": "in", "output_dir": "out", "z_ranges": [{"z_min": 0, "z_max": 1}]}, "batch_fix_atoms_by_height.json"),
    (slab_tools.fix_atoms_by_indices, {"structure_dir": "in", "output_dir": "out", "indices": [0]}, "batch_fix_atoms_by_indices.json"),
])
def test_batch_preserves_different_inputs_with_same_stem(files, tool, payload, summary):
    (files / "in").mkdir()
    Poscar(nacl()).write_file(files / "in/same.vasp")
    bigger = nacl()
    bigger.scale_lattice(125)
    bigger.to(filename=str(files / "in/same.cif"))
    tool(payload)
    records = json.loads((files / "out" / summary).read_text())["results"]
    paths = [files / row["output_rel"] for row in records]
    assert len(set(paths)) == 2
    assert sorted(round(Structure.from_file(path).volume) for path in paths) == [64, 125]


def test_name_allocator_reserves_existing_suffixes():
    paths = [Path("in/a.cif"), Path("in/a.vasp"), Path("in/a__1.cif")]
    names = batch_names(paths, Path("in"), lambda path: path.stem)
    assert len(set(names.values())) == 3
    assert names[paths[2]] == "a__1"


def test_orca_batch_preserves_flattened_paths(files):
    tool = importlib.import_module("catmaster.tools.geometry_inputs.orca_prepare").orca_prepare
    write_xyz(files / "in/a/b.xyz", [.7])
    write_xyz(files / "in/a_b.xyz", [1.4])
    tool({"input_path": "in", "output_root": "out", "simple_keywords": ["HF"], "charge": 0, "multiplicity": 1})
    rows = json.loads((files / "out/orca_prepare_manifest.json").read_text())["records"]
    assert len({row["stage_path"] for row in rows}) == 2
    assert sorted(read(files / row["coordinate_path"]).get_distance(0, 1) for row in rows) == [.7, 1.4]


def test_manual_phonon_moves_each_representative_species():
    structure = nacl()
    outputs, metadata = crystal._manual_phonon_displacements(structure, supercell=[2, 1, 1], displacement=.01, plus_minus=False, symprec=.01, angle_tolerance=5)
    reference = structure.copy()
    reference.make_supercell([2, 1, 1])
    moved = []
    for _, displaced in outputs:
        indices = np.flatnonzero(np.linalg.norm(displaced.cart_coords - reference.cart_coords, axis=1) > 1e-8)
        assert len(indices) == 1
        moved.append(reference[int(indices[0])].species_string)
    assert moved.count("Na") == moved.count("Cl") == 3
    assert metadata["representative_supercell_sites"] == [0, 2]


def test_phonopy_failure_is_not_silently_replaced(files, monkeypatch):
    import phonopy
    Poscar(nacl()).write_file(files / "input.vasp")
    def fail(*args, **kwargs):
        raise RuntimeError("phonopy failed distinctly")
    monkeypatch.setattr(phonopy, "Phonopy", fail)
    with pytest.raises(Exception, match="phonopy failed distinctly"):
        crystal.generate_phonon_displacements({"structure_file": "input.vasp", "output_dir": "out"})
    assert not list((files / "out").glob("disp*"))


def test_kpath_returns_its_matching_primitive_structure(files):
    structure = Structure.from_spacegroup("Fm-3m", Lattice.cubic(4), ["Cu"], [[0, 0, 0]])
    Poscar(structure).write_file(files / "input.vasp")
    content, result = crystal.generate_kpath({"structure_file": "input.vasp", "output_path": "KPOINTS"})
    paired = Structure.from_file(files / result["data"]["structure_rel"])
    assert len(paired) == 1
    assert np.isclose(paired.volume * 4, structure.volume)
    assert result["data"]["structure_rel"] in content
    assert len(Structure.from_file(files / "input.vasp")) == 4


def test_supercell_accepts_nondiagonal_matrix(files):
    Poscar(nacl()).write_file(files / "input.vasp")
    matrix = [[2, 1, 0], [0, 1, 0], [0, 0, 1]]
    crystal.supercell({"structure_file": "input.vasp", "output_path": "out.vasp", "supercell_matrix": matrix})
    result = Structure.from_file(files / "out.vasp")
    assert len(result) == 4
    assert np.allclose(result.lattice.matrix, np.array(matrix) @ nacl().lattice.matrix)


def test_conformer_energy_window_uses_minimum_reference(files):
    write_xyz(files / "in/a.xyz", [.7])
    write_xyz(files / "in/b.xyz", [1.4])
    (files / "in/conformers.json").write_text(json.dumps({"records": [
        {"structure_rel": "a.xyz", "energy_kcal_mol": 100},
        {"structure_rel": "b.xyz", "energy_kcal_mol": 102},
    ]}))
    _, result = molecules.filter_conformer_ensemble({"input_dir": "in", "output_dir": "out", "rmsd_threshold_angstrom": 0})
    assert result["data"]["count"] == 2
    rows = json.loads((files / "out/filtered_conformers.json").read_text())["records"]
    assert [row["relative_energy_kcal_mol"] for row in rows] == [0, 2]


def test_conformer_frames_and_missing_energy_are_explicit(files):
    write_xyz(files / "in/ensemble.xyz", [.7, 1.4])
    with pytest.raises(ValueError, match="Missing energy"):
        molecules.filter_conformer_ensemble({"input_dir": "in", "output_dir": "out"})
    _, result = molecules.filter_conformer_ensemble({"input_dir": "in", "output_dir": "out", "apply_energy_window": False, "rmsd_threshold_angstrom": 0})
    assert result["data"]["count"] == 2
    with pytest.raises(ValueError, match="fresh empty"):
        molecules.filter_conformer_ensemble({"input_dir": "in", "output_dir": "out", "apply_energy_window": False})


def test_conformer_optimizer_failure_remains_visible(files, monkeypatch):
    from rdkit.Chem import AllChem
    monkeypatch.setattr(AllChem, "MMFFGetMoleculeProperties", lambda *args, **kwargs: None)
    content, result = molecules.enumerate_molecular_conformers({"smiles": "CC", "max_conformers": 1, "output_dir": "out"})
    assert result["data"]["state"] == "partial"
    assert "optimization_failed_count=1" in content
    row = json.loads((files / "out/conformers.json").read_text())["records"][0]
    assert row["optimization_state"] == "failed"
    assert (files / row["structure_rel"]).exists()


def test_extraction_includes_root_runtime_name_and_final_frame(files):
    write_xyz(files / "run/job.runtime.xyz", [.7, 1.4])
    (files / "run/job.out").write_text("THE OPTIMIZATION HAS CONVERGED\nORCA TERMINATED NORMALLY\n")
    _, result = molecules.extract_optimized_molecules({"input_dir": "run", "output_dir": "out"})
    assert result["data"]["count"] == 1
    row = json.loads((files / "out/optimized_molecules.json").read_text())["records"][0]
    assert read(files / row["structure_rel"]).get_distance(0, 1) == 1.4
    assert row["optimization_state"] == "converged"


def test_extraction_does_not_call_process_success_convergence(files):
    write_xyz(files / "run/custom.xyz", [.7, 1.4])
    (files / "run/status.json").write_text('{"returncode":0}')
    _, result = molecules.extract_optimized_molecules({"source_files": ["run/custom.xyz"], "output_dir": "out"})
    assert result["data"]["count"] == 0
    _, result = molecules.extract_optimized_molecules({"source_files": ["run/custom.xyz"], "output_dir": "explicit", "require_converged": False})
    assert result["data"]["count"] == 1
    assert not (files / "run/custom_last.xyz").exists()
