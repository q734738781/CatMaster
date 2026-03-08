from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from catmaster.tools.base import workspace_scope


class _FakeModel:
    def invoke(self, messages):
        _ = messages
        return SimpleNamespace(content="\\section{Results}\nPolished academic prose.")


def test_polish_academic_prose_reads_and_overwrites_workspace_file(
    tmp_path: Path,
    monkeypatch,
) -> None:
    tool_module = importlib.import_module("catmaster.tools.analysis.polish_academic_prose")
    files_root = tmp_path / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    manuscript = files_root / "writing" / "write_001" / "manuscript" / "MANUSCRIPT.tex"
    manuscript.parent.mkdir(parents=True, exist_ok=True)
    manuscript.write_text("\\section{Results}\nrough draft", encoding="utf-8")

    monkeypatch.setattr(
        tool_module,
        "_resolve_polisher_config",
        lambda: SimpleNamespace(model="google/gemini-3.1-pro-preview"),
    )
    monkeypatch.setattr(
        tool_module,
        "build_chat_model",
        lambda cfg: _FakeModel(),
    )

    with workspace_scope(tmp_path):
        content, artifact = tool_module.polish_academic_prose(
            {
                "source_path": "writing/write_001/manuscript/MANUSCRIPT.tex",
                "focus": "Tighten the prose for journal submission.",
            }
        )

    assert "Academic polish applied" in content
    assert artifact["data"]["output_path"] == "writing/write_001/manuscript/MANUSCRIPT.tex"
    assert manuscript.read_text(encoding="utf-8") == "\\section{Results}\nPolished academic prose."
