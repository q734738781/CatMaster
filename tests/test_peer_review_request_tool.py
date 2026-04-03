from __future__ import annotations

import importlib
from pathlib import Path
from types import SimpleNamespace

from catmaster.tools.base import ensure_project_space_layout, workspace_scope


class _FakeResponse:
    def __init__(self, content):
        self.content = content


class _FakeModel:
    def __init__(self, model_name: str):
        self.model_name = model_name
        self.messages = None

    def invoke(self, messages):
        self.messages = messages
        return _FakeResponse(
            "General Comments\nSolid direction but not yet publication-ready.\n\n"
            "Major Comments\n- The control story is incomplete.\n\n"
            "Minor Comments\n- Figure flow can be tightened."
        )


def test_peer_review_request_uses_all_configured_models(tmp_path: Path, monkeypatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    pdf_path = files_root / "manuscript" / "paper.pdf"
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    pdf_path.write_bytes(b"%PDF-1.4\n% fake pdf bytes\n")

    tool_module = importlib.import_module("catmaster.tools.analysis.peer_review_request")
    models: dict[str, _FakeModel] = {}

    def _fake_build_chat_model(cfg):
        model = _FakeModel(str(cfg.model))
        models[str(cfg.model)] = model
        return model

    fake_profile = SimpleNamespace(
        peer_review_models=["peer_reviewer_gemini", "peer_reviewer_claude"],
        models={
            "peer_reviewer_gemini": SimpleNamespace(model="google/gemini-3.1-pro", provider="openrouter", base_url=None),
            "peer_reviewer_claude": SimpleNamespace(model="openrouter/anthropic/claude-sonnet-4", provider="openrouter", base_url=None),
        },
        config_for_role=lambda role: SimpleNamespace(model=f"{role}-model", provider="openrouter", base_url=None),
        label_for_role=lambda role: "peer_reviewer_gemini",
    )

    monkeypatch.setattr(tool_module, "build_chat_model", _fake_build_chat_model)
    monkeypatch.setattr(tool_module.LLMProfile, "from_env_or_file", staticmethod(lambda: fake_profile))

    with workspace_scope(tmp_path):
        content, artifact = tool_module.peer_review_request(
            {
                "pdf_path": "manuscript/paper.pdf",
                "review_request": "Assess whether this meets submission-grade ACS expectations.",
            }
        )

    assert "Reviewer 1" in content
    assert "Reviewer 2" in content
    data = artifact["data"]
    assert data["pdf_path"] == "manuscript/paper.pdf"
    assert data["model_labels"] == ["peer_reviewer_gemini", "peer_reviewer_claude"]
    assert data["models"] == ["google/gemini-3.1-pro", "openrouter/anthropic/claude-sonnet-4"]
    assert len(data["reviews"]) == 2
    first_model = models["google/gemini-3.1-pro"]
    assert first_model.messages is not None
    assert first_model.messages[0]["role"] == "system"
    human_message = first_model.messages[1]
    assert human_message["role"] == "user"
    assert isinstance(human_message["content"], list)
    assert human_message["content"][0]["type"] == "text"
    assert human_message["content"][1]["type"] == "file"
    assert human_message["content"][1]["mime_type"] == "application/pdf"
