from __future__ import annotations

import subprocess
import sys
from pathlib import Path


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
    assert "all: mace_sp, vasp_sp, xtb_sp, orca_sp, cp2k_sp, lammps_min, crest_quick" in proc.stdout
    assert "vasp_sp" in proc.stdout
    assert "cp2k_sp" in proc.stdout
    assert "no_cp2k" not in proc.stdout
