#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "build_slab"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    tool = reg.get_tool_function("build_slab")

    res1 = tool(
        {
            "bulk_structure": "tests/assets/Fe.cif",
            "miller_index": [1, 1, 1],
            "output_root": str(out_dir / "fe_111_1x1"),
            "slab_thickness": 12.0,
            "vacuum_thickness": 15.0,
            "supercell": [1, 1, 1],
            "get_symmetry_slab": False,
        }
    )

    res2 = tool(
        {
            "bulk_structure": "tests/assets/Fe.cif",
            "miller_index": [1, 1, 1],
            "output_root": str(out_dir / "fe_111_2x2"),
            "slab_thickness": 12.0,
            "vacuum_thickness": 15.0,
            "supercell": [2, 2, 1],
            "get_symmetry_slab": False,
        }
    )

    (out_dir / "run.json").write_text(json.dumps({"1x1": res1, "2x2": res2}, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
