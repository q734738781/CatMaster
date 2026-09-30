"""Run a live writing evaluation through CatMaster's normal specialist lane.

Supply an isolated workspace, its LLM profile, and a common brief. This makes
billable model calls and writes normal runtime state plus benchmark_result.json.
It does not copy source workspaces or modify deployment configuration.
"""

from __future__ import annotations

import argparse
import asyncio
from dataclasses import asdict
from datetime import datetime, timezone
import json
import os
import sqlite3
from pathlib import Path
import time
import traceback

from catmaster.llm.config import LLMProfile
from catmaster.specialists import build_specialist_runner
from catmaster.ui.reporters import Reporter


class FileReporter(Reporter):
    def __init__(self, path: Path) -> None:
        self.path = path

    def emit(self, event) -> None:
        with self.path.open("a", encoding="utf-8") as stream:
            stream.write(json.dumps(asdict(event), ensure_ascii=False, default=str) + "\n")


async def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--workspace", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--capture-request-limit", type=int, default=1000,
                        help="Maximum Codex HTTP request/response pairs to capture; 0 disables")
    args = parser.parse_args()
    if args.capture_request_limit < 0:
        parser.error("--capture-request-limit must be nonnegative")
    ws = args.workspace.resolve()
    config = args.config.resolve()
    prompt = args.brief.resolve().read_text(encoding="utf-8")
    if not (ws / "files").is_dir():
        raise ValueError("Prepare an isolated workspace with files/ first")
    result_path = ws / "benchmark_result.json"
    if result_path.exists():
        raise FileExistsError("Use a fresh copy for another benchmark attempt")
    os.environ["CATMASTER_LLM_CONFIG"] = str(config)
    profile = LLMProfile.from_env_or_file(str(config))
    built = build_specialist_runner(
        workspace=ws,
        llm_profile=profile,
        reporter=FileReporter(ws / "benchmark_events.jsonl"),
        run_control=None,
        project_id=ws.name,
        preferred_entrypoint="writing",
    )
    # Models and observation handlers are constructed inside arun, so select
    # the actual run now, before either snapshots the diagnostic settings.
    if args.capture_request_limit:
        os.environ["CATMASTER_CAPTURE_REQUEST_RUN_IDS"] = built.run_context.run_id
        os.environ["CATMASTER_CAPTURE_REQUEST_LIMIT"] = str(args.capture_request_limit)
        os.environ.pop("CATMASTER_CAPTURE_REQUEST_AGENTS", None)
    else:
        os.environ.pop("CATMASTER_CAPTURE_REQUEST_RUN_IDS", None)
    started = time.monotonic()
    record = {
        "status": "running",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "run_dir": str(built.run_context.run_dir),
        "model": built.run_context.model_name,
        "provider": built.run_context.provider,
    }
    result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2))
    try:
        result = await built.runner.arun(
            prompt,
            entrypoint="writing",
            proposal_review=False,
            thread_id=built.run_context.run_id,
        )
        record.update(result)
    except Exception:
        record.update(status="error", error=traceback.format_exc())
    finally:
        record["elapsed_seconds"] = time.monotonic() - started
        db_path = built.run_context.run_dir / "observability.sqlite"
        if db_path.exists():
            with sqlite3.connect(f"file:{db_path}?mode=ro", uri=True) as db:
                counts = dict(db.execute("SELECT name,COUNT(*) FROM observation_events "
                                         "WHERE name IN ('LLM_PROVIDER_REQUEST','LLM_PROVIDER_RESPONSE') "
                                         "GROUP BY name"))
            record["provider_capture"] = {"limit": args.capture_request_limit, **counts}
        result_path.write_text(json.dumps(record, ensure_ascii=False, indent=2, default=str))
    print(json.dumps({k: record[k] for k in ("status", "model", "run_dir", "elapsed_seconds")}))
    return 0 if record["status"] == "done" else 1


if __name__ == "__main__":
    raise SystemExit(asyncio.run(main()))
