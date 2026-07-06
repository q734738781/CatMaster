from __future__ import annotations

import argparse

from catmaster.runtime.legacy_trace_import import import_legacy_observability


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="python -m catmaster", description="CatMaster command utilities.")
    subparsers = parser.add_subparsers(dest="command", required=True)

    migrate = subparsers.add_parser("migrate-observability", help="Import legacy JSONL observation files into observability.sqlite.")
    migrate.add_argument("run_dir", help="Run directory to migrate.")
    migrate.add_argument("--no-ui-events", action="store_true", help="Do not import ui_events.jsonl.")
    migrate.add_argument("--no-trace-records", action="store_true", help="Do not import event/tool/patch trace JSONL files.")

    args = parser.parse_args(argv)
    if args.command == "migrate-observability":
        count = import_legacy_observability(
            args.run_dir,
            include_ui_events=not args.no_ui_events,
            include_trace_records=not args.no_trace_records,
        )
        print(f"Imported {count} legacy observation records.")
        return 0
    parser.error(f"Unknown command: {args.command}")
    return 2


if __name__ == "__main__":
    raise SystemExit(main())
