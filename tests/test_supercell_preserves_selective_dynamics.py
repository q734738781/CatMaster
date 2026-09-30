from __future__ import annotations

from collections import Counter
from pathlib import Path

from pymatgen.io.vasp import Poscar

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.crystal_tool import supercell


def test_supercell_preserves_selective_dynamics_in_single_file_mode(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        source = Path("tests/assets/Fe_hkl111_12A_15AVac_5ARelax.vasp").read_text(encoding="utf-8")
        input_path = tmp_path / "files" / "inputs" / "slab.vasp"
        output_path = tmp_path / "files" / "outputs" / "slab_2x2.vasp"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(source, encoding="utf-8")

        supercell(
            {
                "structure_file": "inputs/slab.vasp",
                "supercell": [2, 2, 1],
                "output_path": "outputs/slab_2x2.vasp",
            }
        )

    poscar = Poscar.from_file(output_path)
    assert poscar.selective_dynamics is not None
    assert len(poscar.selective_dynamics) == 24
    counts = Counter(tuple(flag.tolist()) for flag in poscar.selective_dynamics)
    assert counts[(True, True, True)] == 8
    assert counts[(False, False, False)] == 16
