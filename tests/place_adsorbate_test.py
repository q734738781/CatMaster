#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "place_adsorbate"
    out_dir.mkdir(parents=True, exist_ok=True)

    slab = "tests/assets/Fe_hkl111_12A_15AVac_5ARelax.vasp"
    ads = "tests/assets/CO.xyz"

    reg = get_tool_registry()
    tool = reg.get_tool_function("place_adsorbate")

    sites = ["auto", "ontop_0", "bridge_0", "hollow_0"]
    results = []
    for site in sites:
        res = tool(
            {
                "slab_file": slab,
                "adsorbate_file": ads,
                "site": site,
                "distance": 2.0,
                "output_poscar": str(out_dir / f"{site}.vasp"),
            }
        )
        results.append({"site": site, "result": res})

    (out_dir / "run.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
