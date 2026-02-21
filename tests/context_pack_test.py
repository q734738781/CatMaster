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
        assert "constraints" in pack
        assert "artifact_slice" in pack
        assert "whiteboard_excerpt" not in pack


def test_context_pack_constraints_from_memory_index(tmp_path) -> None:
    store = MemoryStore.create_default(workspace=tmp_path)
    store.ensure_exists()
    store.index_path.write_text(
        "\n".join(
            [
                "# MEMORY (AUTOLOADED INDEX)",
                "",
                "## Top Constraints (max 5)",
                "1. Always keep files-root relative paths",
                "2. Avoid network operations by default",
                "",
                "## Active Open Questions (max 5)",
                "1. (empty)",
                "",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    (store.topics_dir / "CONSTRAINTS.md").write_text(
        "# CONSTRAINTS\n\n## TL;DR\n- stale constraint from head\n",
        encoding="utf-8",
    )

    builder = ContextPackBuilder(store)
    pack = builder.build("demo goal", role="task_runner", policy=ContextPackPolicy())
    constraints = str(pack.get("constraints") or "")
    assert "Always keep files-root relative paths" in constraints
    assert "Avoid network operations by default" in constraints
    assert "stale constraint from head" not in constraints


if __name__ == "__main__":
    main()
