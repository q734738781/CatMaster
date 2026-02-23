from __future__ import annotations

from catmaster.runtime.memory_store import MemoryStore


def test_default_memory_index_uses_files_root_relative_pointers(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    text = store.read_index()
    assert "Put details in MEMORY/topics/*.md." in text
    assert "Goal / principles: MEMORY/topics/GOAL.md" in text
    assert "Files / artifacts: MEMORY/topics/FILES.md" in text
    assert "files/MEMORY/topics/" not in text


def test_rebuilt_memory_index_uses_files_root_relative_pointers(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    store.merge_task_result(
        run_id="run_01",
        task_id="task_01",
        outcome="success",
        task_goal="demo",
        result={
            "summary": "done",
            "facts": ["fact"],
            "files": [{"path": "outputs/a.txt", "description": "output", "kind": "output"}],
            "constraints": ["constraint"],
            "open_questions": [],
            "decisions": [],
            "next_steps": ["next"],
            "artifacts": [],
        },
    )
    text = store.read_index()
    assert "Put details in MEMORY/topics/*.md." in text
    assert "Goal / principles: MEMORY/topics/GOAL.md" in text
    assert "Files / artifacts: MEMORY/topics/FILES.md" in text
    assert "files/MEMORY/topics/" not in text
