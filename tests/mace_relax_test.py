#!/usr/bin/env python3
import json
import os
import shutil
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "mace_relax"
    out_dir.mkdir(parents=True, exist_ok=True)

    # copy input structure into out_dir
    src = root / "tests" / "assets" / "Fe_hkl111_12A_15AVac_5ARelax.vasp"
    dest = out_dir / src.name
    shutil.copy(src, dest)

    reg = get_tool_registry()
    tool = reg.get_tool_function("mace_relax")

    res = tool({"structure_file": str(dest.relative_to(root)), "fmax": 0.05, "maxsteps": 20})

    (out_dir / "run.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
