from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

try:
    from .native_argv_runtime import resolve_program, run_native_argv, write_result_archive
except ImportError:  # Staged scripts run as top-level files beside the helper.
    from native_argv_runtime import resolve_program, run_native_argv, write_result_archive


def main() -> int:
    parser = argparse.ArgumentParser(description="Run one exact native CREST argv manifest")
    parser.add_argument("--manifest", default="manifest.json")
    parser.add_argument("--program", default="crest")
    parser.add_argument("--log", default="crest_stdout.out")
    args = parser.parse_args()
    try:
        program = resolve_program(args.program, ("crest",))
        return run_native_argv(
            manifest_path=Path(args.manifest),
            program=program,
            summary_name="crest_summary.json",
            log_name=args.log,
            normal_markers=("CREST terminated normally", "normal termination of crest"),
        )
    except Exception as exc:
        Path("crest_summary.json").write_text(
            json.dumps(
                {
                    "execution_state": "failed",
                    "task_state": "incomplete",
                    "returncode": 2,
                    "normal_termination": False,
                    "error": f"{type(exc).__name__}: {exc}",
                    "log_file": args.log,
                    "finished_at": time.time(),
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        try:
            write_result_archive()
        except Exception:
            pass
        sys.stderr.write(f"[crest_boot] {type(exc).__name__}: {exc}\n")
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
