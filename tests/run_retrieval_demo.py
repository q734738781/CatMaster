#!/usr/bin/env python3
"""
Stage-1 retrieval demo:
  * performs a literature search and page capture
  * issues a Materials Project query
  * stores outputs beneath the configured workspace root

Run from project root after activating the PC conda environment:
    python test/run_retrieval_demo.py
"""

from __future__ import annotations

import json
import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from catmaster.pc.retrieval.lit_browser import search, capture
from catmaster.pc.retrieval.matdb import query


def main() -> None:
    workspace_root = Path(os.environ.get("CATMASTER_WORKSPACE_ROOT", PROJECT_ROOT / "tmp" / "workspace"))
    demo_root = workspace_root / "retrieval_demo"
    demo_root.mkdir(parents=True, exist_ok=True)

    print(f"[info] writing retrieval outputs to {demo_root}")

    literature = search("ZnCr2O4 surface reconstruction CO2 CO")
    search_path = demo_root / "lit_search.json"
    search_path.write_text(json.dumps(literature, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] literature search results saved to {search_path}")

    page = capture("https://www.example.com")
    page_path = demo_root / "page_capture.json"
    page_path.write_text(json.dumps(page, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] page capture saved to {page_path}")

    structures_dir = demo_root / "structures"
    matdb = query(
        criteria={"formula": "ZnCr2O4"},
        structures_dir=structures_dir,
    )
    matdb_path = demo_root / "matdb_hits.json"
    matdb_path.write_text(json.dumps(matdb, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"[info] matdb query summary saved to {matdb_path}")
    print(f"[info] CIF files stored under {structures_dir}")


if __name__ == "__main__":
    main()
