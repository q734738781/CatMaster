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
            "Verdict\nPotentially publishable after revision.\n\n"
            "Major concerns\n- Claim strength outruns the control evidence.\n\n"
            "Minor concerns\n- Figure order is harder to follow than necessary.\n\n"
            "Recommended revision focus\n- Tighten claims and clarify the figure narrative."
        )


def test_peer_review_pdf_manuscript_uses_pdf_file_block_and_default_model(tmp_path: Path, monkeypatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    pdf_path = files_root / "manuscript" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf bytes\n")

    fake_model = _FakeModel()
    tool_module = importlib.import_module("catmaster.tools.analysis.peer_review_pdf_manuscript")

    monkeypatch.setattr(tool_module, "build_chat_model", lambda cfg: fake_model)

    with workspace_scope(tmp_path):
        content, artifact = tool_module.peer_review_pdf_manuscript(
            {
                "pdf_path": "manuscript/paper.pdf",
                "user_request_context": "Write a journal-style manuscript and make the scientific case rigorous.",
                "round_index": 1,
                "max_rounds": 2,
            }
        )

    assert "Verdict" in content
    data = artifact["data"]
    assert data["pdf_path"] == "manuscript/paper.pdf"
    assert data["round_index"] == 1
    assert data["max_rounds"] == 2
    assert data["model_name"] == "google/gemini-3.1-pro"
    assert fake_model.messages is not None
    assert fake_model.messages[0]["role"] == "system"
    human_message = fake_model.messages[1]
    assert human_message["role"] == "user"
    assert isinstance(human_message["content"], list)
    assert human_message["content"][0]["type"] == "text"
    assert human_message["content"][1]["type"] == "file"
    assert human_message["content"][1]["mime_type"] == "application/pdf"
    assert human_message["content"][1]["filename"] == "paper.pdf"
