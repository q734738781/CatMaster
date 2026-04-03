from __future__ import annotations

import importlib
from pathlib import Path

from catmaster.tools.base import ensure_project_space_layout, workspace_scope


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self):
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _FakeResponse(
            "Priority 1: tighten title/abstract/conclusion alignment.\n"
            "Priority 2: make the main figure narrative easier to follow.\n"
            "Priority 3: trim defensive filler in the discussion."
        )


def test_review_pdf_manuscript_sends_pdf_as_file_block(tmp_path: Path, monkeypatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    pdf_path = files_root / "manuscript" / "draft.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf bytes\n")

    fake_model = _FakeModel()
    tool_module = importlib.import_module("catmaster.tools.analysis.review_pdf_manuscript")

    monkeypatch.setattr(tool_module, "build_chat_model", lambda cfg: fake_model)

    with workspace_scope(tmp_path):
        content, artifact = tool_module.review_pdf_manuscript(
            {
                "pdf_path": "manuscript/draft.pdf",
                "focus": "publication readiness and figure logic",
                "context_text": "Targeting a chemistry journal submission.",
            }
        )

    assert "Priority 1:" in content
    data = artifact["data"]
    assert data["pdf_path"] == "manuscript/draft.pdf"
    assert data["review_text"] == content
    assert fake_model.messages is not None
    assert fake_model.messages[0]["role"] == "system"
    human_message = fake_model.messages[1]
    assert human_message["role"] == "user"
    assert isinstance(human_message["content"], list)
    assert human_message["content"][0]["type"] == "text"
    assert human_message["content"][1]["type"] == "file"
    assert human_message["content"][1]["mime_type"] == "application/pdf"
    assert human_message["content"][1]["filename"] == "draft.pdf"
