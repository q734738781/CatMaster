from __future__ import annotations

import pytest

from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.misc.memory_patch_apply import memory_apply_aider_edits


def test_memory_apply_aider_edits_success(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        memory_path = files_root / "MEMORY" / "MEMORY.md"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text("# old\n", encoding="utf-8")

        out = memory_apply_aider_edits(
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

        assert out["status"] == "success"
        assert "# new" in memory_path.read_text(encoding="utf-8")
        data = out.get("data", {})
        assert "MEMORY/MEMORY.md" in data.get("applied_files", [])
        assert "diff --git a/MEMORY/MEMORY.md b/MEMORY/MEMORY.md" in str(data.get("diff_text", ""))


def test_memory_apply_aider_edits_rejects_forbidden_path(tmp_path) -> None:
    with workspace_scope(tmp_path):
        out = memory_apply_aider_edits(
            {
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

    assert out["status"] == "failed"
    assert "path validation failed" in str(out.get("error") or "")
    data = out.get("data", {})
    assert data.get("error_code") == "path_forbidden"


def test_memory_apply_aider_edits_rejects_invalid_blocks(tmp_path) -> None:
    with workspace_scope(tmp_path):
        out = memory_apply_aider_edits({"edits_text": "not a valid aider edit"})
    assert out["status"] == "failed"
    assert "aider edit blocks" in str(out.get("error") or "") or "no aider edit blocks found" in str(out.get("error") or "")
    data = out.get("data", {})
    assert data.get("error_code") == "no_blocks"


def test_memory_apply_aider_edits_is_atomic_when_later_block_fails(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        memory_path = files_root / "MEMORY" / "MEMORY.md"
        memory_path.parent.mkdir(parents=True, exist_ok=True)
        memory_path.write_text("# Title\n## Facts\n- old\n", encoding="utf-8")

        out = memory_apply_aider_edits(
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

        assert out["status"] == "failed"
        data = out.get("data", {})
        assert data.get("error_code") == "replace_no_match"
        assert data.get("failed_block_index") == 2
        assert data.get("failed_path") == "MEMORY/MEMORY.md"
        assert memory_path.read_text(encoding="utf-8") == "# Title\n## Facts\n- old\n"
