from __future__ import annotations

import pytest

pytest.importorskip("aider")

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.misc.memory_patch_apply import apply_aider_edits


def test_apply_aider_edits_success(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        memory_path = files_root / "MEMORY" / "MEMORY.md"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text("# old\n", encoding="utf-8")

        content, artifact = apply_aider_edits(
            {
                "edits_text": (
                    "MEMORY/MEMORY.md\n"
                    "<<<<<<< SEARCH\n"
                    "# old\n"
                    "=======\n"
                    "# new\n"
                    ">>>>>>> REPLACE\n"
                )
            }
        )

    assert "apply_aider_edits completed" in str(content).lower()
    assert "# new" in memory_path.read_text(encoding="utf-8")
    data = (artifact or {}).get("data", {})
    assert "MEMORY/MEMORY.md" in data.get("applied_files", [])
    assert "diff --git a/MEMORY/MEMORY.md b/MEMORY/MEMORY.md" in str(data.get("diff_text", ""))


def test_apply_aider_edits_rejects_forbidden_path_when_allowlist_is_set(tmp_path) -> None:
    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            apply_aider_edits(
                {
                    "allowed_paths": ["MEMORY/"],
                    "edits_text": (
                        "src/main.py\n"
                        "<<<<<<< SEARCH\n"
                        "a\n"
                        "=======\n"
                        "b\n"
                        ">>>>>>> REPLACE\n"
                    )
                }
            )
    assert "path validation failed" in str(excinfo.value).lower()


def test_apply_aider_edits_allows_general_workspace_paths_by_default(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        note_path = files_root / "notes" / "tmp.md"
        note_path.parent.mkdir(parents=True, exist_ok=True)
        note_path.write_text("", encoding="utf-8")
        apply_aider_edits(
            {
                "edits_text": (
                    "notes/tmp.md\n"
                    "<<<<<<< SEARCH\n"
                    "\n"
                    "=======\n"
                    "x\n"
                    ">>>>>>> REPLACE\n"
                )
            }
        )
        assert note_path.read_text(encoding="utf-8") == "x\n"


def test_apply_aider_edits_rejects_invalid_blocks(tmp_path) -> None:
    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError):
            apply_aider_edits({"edits_text": "not a valid aider edit"})


def test_apply_aider_edits_is_atomic_when_later_block_fails(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        memory_path = files_root / "MEMORY" / "MEMORY.md"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text("# Title\n## Facts\n- old\n", encoding="utf-8")

        with pytest.raises(CatMasterToolExecutionError):
            apply_aider_edits(
                {
                    "edits_text": (
                        "MEMORY/MEMORY.md\n"
                        "<<<<<<< SEARCH\n"
                        "- old\n"
                        "=======\n"
                        "- new\n"
                        ">>>>>>> REPLACE\n"
                        "MEMORY/MEMORY.md\n"
                        "<<<<<<< SEARCH\n"
                        "Topic TL;DR excerpts (JSON):\n"
                        "=======\n"
                        "X\n"
                        ">>>>>>> REPLACE\n"
                    )
                }
            )

        # Atomicity: first block should be rolled back after later failure.
        text = memory_path.read_text(encoding="utf-8")
        assert "- old" in text
        assert "- new" not in text
