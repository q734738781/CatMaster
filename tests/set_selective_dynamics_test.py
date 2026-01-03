#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "set_selective_dynamics"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    tool = reg.get_tool_function("set_selective_dynamics")

    res1 = tool(
        {
            "structure_ref": "tests/assets/Fe_hkl111_12A_15AVac.vasp",
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_SD.vasp"),
            "relax_thickness": 5.0,
            "centralize": False,
        }
    )

    res2 = tool(
        {
            "structure_ref": "tests/assets/Fe_hkl111_12A_15AVac.vasp",
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_SD_layer.vasp"),
            "freeze_layers": 3,
            "centralize": False,
        }
    )

    res3 = tool(
        {
            "structure_ref": "tests/assets/Fe_hkl111_12A_15AVac.vasp",
            "output_path": str(out_dir / "Fe_hkl111_12A_15AVac_SD_layer_cen.vasp"),
            "freeze_layers": 3,
            "centralize": True,
            "layer_tol": 0.1,
        }
    )

    (out_dir / "run.json").write_text(json.dumps({"res1": res1, "res2": res2, "res3": res3}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
