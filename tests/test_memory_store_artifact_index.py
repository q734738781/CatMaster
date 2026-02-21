from __future__ import annotations

from catmaster.runtime.memory_store import MemoryStore


def test_artifact_index_prefers_newest_records(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()

    store.merge_task_result(
        run_id="run_01",
        task_id="task_01",
        outcome="success",
        task_goal="produce first file",
        result={
            "summary": "first",
            "facts": [],
            "files": [{"path": "outputs/first.txt", "description": "first", "kind": "output"}],
            "constraints": [],
            "open_questions": [],
            "decisions": [],
            "next_steps": [],
            "artifacts": [],
        },
    )
    store.merge_task_result(
        run_id="run_02",
        task_id="task_02",
        outcome="success",
        task_goal="produce second file",
        result={
            "summary": "second",
            "facts": [],
            "files": [{"path": "outputs/second.txt", "description": "second", "kind": "output"}],
            "constraints": [],
            "open_questions": [],
            "decisions": [],
            "next_steps": [],
            "artifacts": [],
        },
    )

    records = store.artifact_index(limit=2)
    assert len(records) == 2
    assert records[0]["path"] == "outputs/second.txt"
    assert records[1]["path"] == "outputs/first.txt"
