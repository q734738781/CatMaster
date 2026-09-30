#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from pathlib import Path

from catmaster.research.knowledge_graph.migration import ResearchGraphMigrator


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Dry-run, apply, or roll back the one-way Research Graph migration."
    )
    parser.add_argument("workspace", type=Path)
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--apply",
        action="store_true",
        help="Import valid legacy state and archive all legacy files.",
    )
    mode.add_argument(
        "--rollback",
        type=Path,
        help="Rollback using a generated metadata/legacy_research_state manifest.",
    )
    args = parser.parse_args()
    migrator = ResearchGraphMigrator(args.workspace)
    if args.rollback:
        result = migrator.rollback(args.rollback)
    elif args.apply:
        result = migrator.apply()
    else:
        result = migrator.dry_run()
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
