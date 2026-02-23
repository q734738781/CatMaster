#!/usr/bin/env python3
from pathlib import Path
import tempfile

from catmaster.runtime.context_pack import ContextPackBuilder, ContextPackPolicy
from catmaster.runtime.memory_store import MemoryStore


def main() -> None:
    with tempfile.TemporaryDirectory(prefix="cm_ctx_pack_") as tmp:
        root = Path(tmp).resolve()
        store = MemoryStore.create_default(workspace=root)
        store.ensure_exists()
        builder = ContextPackBuilder(store)
        pack = builder.build(
            "demo goal",
            role="task_runner",
            policy=ContextPackPolicy(memory_head_lines=50, max_memory_chars=2000, max_artifacts=20),
        )
        assert "memory_index_excerpt" in pack
        assert "workspace_policy" in pack
        assert "workspace_root" in pack
        assert "constraints" not in pack
        assert "artifact_slice" not in pack
        assert "whiteboard_excerpt" not in pack


def test_context_pack_task_runner_removes_goal_pointer(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    store.index_path.write_text(
        "\n".join(
            [
                "# MEMORY (AUTOLOADED INDEX)",
                "",
                "## Pointers",
                "- Goal / principles: MEMORY/topics/GOAL.md",
                "- Facts / decisions: MEMORY/topics/FACTS.md",
                "",
                "## Active Open Questions (max 5)",
                "1. (empty)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    builder = ContextPackBuilder(store)
    pack = builder.build("demo goal", role="task_runner", policy=ContextPackPolicy())
    memory_excerpt = str(pack.get("memory_index_excerpt") or "")
    assert "Goal / principles: MEMORY/topics/GOAL.md" not in memory_excerpt
    assert "Facts / decisions: MEMORY/topics/FACTS.md" in memory_excerpt


if __name__ == "__main__":
    main()
