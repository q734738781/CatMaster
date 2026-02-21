from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.orchestrator import Orchestrator


def test_director_observations_view_drops_metadata_paths() -> None:
    raw = [
        {
            "task_id": "task_01",
            "outcome": "success",
            "summary": "done",
            "failure_kind": "max_steps",
            "auto_replan": True,
            "observation_path": "observations/obs_001_task_01.md",
            "event_path": "memory/events.jsonl",
            "key_artifacts": [
                {"path": "results/final.csv", "description": "table", "kind": "report"},
                {"path": "", "description": "invalid", "kind": "report"},
            ],
            "resume_state": {"next_step": 2},
            "interrupted_toolcall": {
                "tool": "bash_exec",
                "status": "failed",
                "highlights": "timeout",
                "cancel_accepted": True,
                "toolcall_id": "task_01_xxx",
            },
        }
    ]

    sanitized = Orchestrator._director_observations_view(raw)
    assert len(sanitized) == 1
    row = sanitized[0]
    assert row["task_id"] == "task_01"
    assert row["outcome"] == "success"
    assert row["summary"] == "done"
    assert row["failure_kind"] == "max_steps"
    assert row["auto_replan"] is True
    assert row["key_artifacts"] == [{"path": "results/final.csv", "description": "table", "kind": "report"}]
    assert "observation_path" not in row
    assert "event_path" not in row
    assert "resume_state" not in row
    assert row["interrupted_toolcall"] == {
        "tool": "bash_exec",
        "status": "failed",
        "highlights": "timeout",
        "cancel_accepted": True,
    }
