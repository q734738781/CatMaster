#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry
from pymatgen.core import Structure


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "fix_atoms"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    layers_tool = reg.get_tool_function("fix_atoms_by_layers")
    height_tool = reg.get_tool_function("fix_atoms_by_height")

    structure_ref = "tests/assets/Fe_bcc_111__CONTCAR_h111_t0.vasp"
    struct_path = root / structure_ref
    slab = Structure.from_file(struct_path)
    zmin = 25.581158897299265
    zmax = 30.581158897299265

    res1 = layers_tool(
        {
            "structure_ref": structure_ref,
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_fix_layers.vasp"),
            "freeze_layers": 3,
            "centralize": False,
        }
    )

    res2 = layers_tool(
        {
            "structure_ref": structure_ref,
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_fix_layers_cen.vasp"),
            "freeze_layers": 3,
            "centralize": True,
            "layer_tol": 0.1,
        }
    )

    res3 = height_tool(
        {
            "structure_ref": structure_ref,
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_fix_height.vasp"),
            "z_ranges": [(zmin, zmax)],
            "centralize": True,
        }
    )

    (out_dir / "run.json").write_text(json.dumps({"res1": res1, "res2": res2, "res3": res3}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
