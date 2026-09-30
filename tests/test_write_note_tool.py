from __future__ import annotations

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.misc.memory import write_note


def test_write_note_persists_markdown_note(tmp_path) -> None:
    with workspace_scope(tmp_path):
        content, artifact = write_note(
            {
                "note": "remember to reuse converged ENCUT and KPOINTS settings",
                "tags": ["fast-director", "handoff"],
            }
        )
        note_path = workspace_root(tmp_path) / "notes" / "agent_notes.md"

    assert note_path.exists()
    text = note_path.read_text(encoding="utf-8")
    assert "remember to reuse converged ENCUT and KPOINTS settings" in text
    assert "tags: fast-director, handoff" in text
    assert "write_note completed." in str(content)
    data = (artifact or {}).get("data", {})
    assert data.get("note_file_rel") == "notes/agent_notes.md"
    assert data.get("tags") == ["fast-director", "handoff"]


def test_write_note_rejects_empty_note(tmp_path) -> None:
    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError):
            write_note({"note": "   "})
