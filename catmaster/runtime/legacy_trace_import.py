"""Explicit legacy JSONL import utility for old CatMaster run directories."""
from __future__ import annotations

import argparse
from pathlib import Path

from catmaster.runtime.observability_store import ObservabilityStore


def import_legacy_observability(
    run_dir: Path | str,
    *,
    include_ui_events: bool = True,
    include_trace_records: bool = True,
) -> int:
    """Import old UI/trace JSONL files into the canonical ObservabilityStore."""
    return ObservabilityStore(Path(run_dir)).import_legacy_jsonl(
        include_ui_events=include_ui_events,
        include_legacy_trace_records=include_trace_records,
    )


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Import legacy CatMaster JSONL observability files into observability.sqlite.")
    parser.add_argument("run_dir", help="Run directory containing legacy ui_events.jsonl or trace JSONL files.")
    parser.add_argument("--no-ui-events", action="store_true", help="Do not import ui_events.jsonl.")
    parser.add_argument("--no-trace-records", action="store_true", help="Do not import event/tool/patch trace JSONL files.")
    args = parser.parse_args(argv)
    count = import_legacy_observability(
        Path(args.run_dir),
        include_ui_events=not args.no_ui_events,
        include_trace_records=not args.no_trace_records,
    )
    print(f"Imported {count} legacy observation records.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())


__all__ = ["import_legacy_observability", "main"]
