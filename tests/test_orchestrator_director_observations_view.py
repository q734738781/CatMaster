from __future__ import annotations

import pytest

pytest.importorskip("langchain_core")

from catmaster.agents.nodes import (
    _director_observations_view,
    _director_task_outcomes_history,
    _render_task_outcomes_history_lines,
)


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

    sanitized = _director_observations_view(raw)
    assert len(sanitized) == 1
    row = sanitized[0]
    assert row["task_id"] == "task_01"
    assert row["outcome"] == "success"
    assert row["summary"] == "done"
    assert row["failure_kind"] == "max_steps"
    assert row["auto_replan"] is True
    assert "key_artifacts" not in row
    assert "observation_path" not in row
    assert "event_path" not in row
    assert "resume_state" not in row
    assert row["interrupted_toolcall"] == {
        "tool": "bash_exec",
        "status": "failed",
        "highlights": "timeout",
        "cancel_accepted": True,
    }


def test_director_task_outcomes_history_merges_task_status_goal_and_counts() -> None:
    tasks = [
        {"task_id": "task_01", "status": "success", "goal": "prepare inputs"},
        {"task_id": "task_02", "status": "failure", "goal": "run vasp"},
    ]
    observations = [
        {
            "task_id": "task_01",
            "outcome": "success",
            "summary": "inputs prepared",
            "key_artifacts": [{"path": "runs/o2/POSCAR"}],
            "decisions": ["use triplet"],
            "next_steps": ["run vasp"],
        }
    ]

    history = _director_task_outcomes_history(tasks, observations)
    assert len(history) == 2

    row1 = history[0]
    assert row1["task_id"] == "task_01"
    assert row1["status"] == "success"
    assert row1["goal"] == "prepare inputs"
    assert row1["outcome"] == "success"
    assert row1["artifact_count"] == 1
    assert row1["decision_count"] == 1

    row2 = history[1]
    assert row2["task_id"] == "task_02"
    assert row2["status"] == "failure"
    assert row2["goal"] == "run vasp"


def test_render_task_outcomes_history_lines_markdown_kv() -> None:
    history = [
        {
            "task_id": "task_01",
            "status": "success",
            "outcome": "success",
            "goal": "prepare inputs",
            "summary": "inputs prepared",
            "artifact_count": 2,
            "decision_count": 1,
            "next_steps": ["run vasp", "collect teten"],
        }
    ]

    rendered = _render_task_outcomes_history_lines(history)
    assert "## Record 1" in rendered
    assert "```md" in rendered
    assert "task_id: task_01" in rendered
    assert "status: success" in rendered
    assert "summary: inputs prepared" in rendered
    assert "artifact_count: 2" in rendered
    assert "next_steps:" in rendered
    assert "  - run vasp" in rendered
