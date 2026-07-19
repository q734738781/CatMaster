from __future__ import annotations

import importlib.util
import subprocess
import sys
from pathlib import Path


def _load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "remote_execution_smoke.py"
    spec = importlib.util.spec_from_file_location("catmaster_remote_execution_smoke", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_execution_smoke_script_lists_cases_without_submitting() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/remote_execution_smoke.py", "--list"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "core: mace_sp, xtb_sp, orca_sp" in proc.stdout
    assert "uma: uma_mol_sp, uma_mol_relax, uma_mat_sp, uma_mat_relax" in proc.stdout
    assert "all: mace_sp, vasp_sp, xtb_sp, orca_sp, cp2k_sp, lammps_min, crest_quick" in proc.stdout
    assert "all: mace_sp, uma_mol_sp" not in proc.stdout
    assert "vasp_sp" in proc.stdout
    assert "uma_mol_sp" in proc.stdout
    assert "uma_mol_relax" in proc.stdout
    assert "cp2k_sp" in proc.stdout
    assert "mlff_si512: si512_mace_sp" in proc.stdout
    assert "si512_orb_md" in proc.stdout
    assert "orb_neb" in proc.stdout
    assert "no_cp2k" not in proc.stdout


def test_si512_acceptance_structure_is_deterministic(tmp_path: Path) -> None:
    from ase.io import read
    import numpy as np

    smoke = _load_smoke_module()
    first = tmp_path / "first.vasp"
    second = tmp_path / "second.vasp"
    smoke._write_si512_poscar(first, displacement_A=0.01)
    smoke._write_si512_poscar(second, displacement_A=0.01)

    atoms = read(first)
    repeated = read(second)
    assert len(atoms) == 512
    assert set(atoms.get_chemical_symbols()) == {"Si"}
    assert np.allclose(atoms.positions, repeated.positions)
