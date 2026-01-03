#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "enumerate_adsorption_sites"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    tool = reg.get_tool_function("enumerate_adsorption_sites")

    res = tool(
        {
            "slab_file": "tests/assets/Fe_hkl111_12A_15AVac_5ARelax.vasp",
            "mode": "all",
            "distance": 2.0,
            "output_json": str(out_dir / "sites.json"),
        }
    )

    (out_dir / "run.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
