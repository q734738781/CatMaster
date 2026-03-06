from __future__ import annotations

import importlib
from pathlib import Path

from PIL import Image

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
            "Answer: The molecule is roughly vertical above the surface.\n"
            "High-confidence: A two-atom molecule is visible above the surface. "
            "The upper atom sits above the lower atom in the side view.\n"
            "Low-confidence: The molecule appears roughly upright rather than strongly tilted."
        )


def test_analyze_images_uses_local_images_and_parses_json(tmp_path: Path, monkeypatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    image_path = files_root / "viz" / "panel.png"
    image_path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (12, 12), color="white").save(image_path)

    fake_model = _FakeModel()
    tool_module = importlib.import_module("catmaster.tools.analysis.analyze_images")

    monkeypatch.setattr(tool_module, "build_chat_model", lambda cfg: fake_model)

    with workspace_scope(tmp_path):
        content, artifact = tool_module.analyze_images(
            {
                "query": "Is the molecule upright?",
                "image_paths": ["viz/panel.png"],
                "context_text": "Rendered slab and adsorbate panel.",
            }
        )

    assert "Answer:" in content
    assert "High-confidence:" in content
    assert "Low-confidence:" in content
    data = artifact["data"]
    assert data["answer"].startswith("Answer:")
    assert data["image_paths"] == ["viz/panel.png"]
    assert data["analysis_text"] == content
    assert fake_model.messages is not None
