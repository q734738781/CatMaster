#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "mp_download_structure"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    tool = reg.get_tool_function("mp_download_structure")

    results = []
    for fmt in ["cif", "poscar"]:
        res = tool({"mp_id": "mp-150", "fmt": fmt, "output_dir": str(out_dir)})
        results.append({"fmt": fmt, "result": res})

    (out_dir / "run.json").write_text(json.dumps(results, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
