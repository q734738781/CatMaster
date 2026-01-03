#!/usr/bin/env python3
import json
import os
from pathlib import Path

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    out_dir = root / "tests" / "test_output" / "create_molecule_from_smiles"
    out_dir.mkdir(parents=True, exist_ok=True)

    reg = get_tool_registry()
    tool = reg.get_tool_function("create_molecule_from_smiles")

    res = tool(
        {
            "smiles": "[C-]#[O+]",
            "output_path": str(out_dir / "co"),
            "box_padding": 10.0,
            "fmt": "both",
        }
    )

    (out_dir / "run.json").write_text(json.dumps(res, indent=2, ensure_ascii=False), encoding="utf-8")


if __name__ == "__main__":
    main()
