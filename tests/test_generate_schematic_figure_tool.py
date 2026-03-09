from __future__ import annotations

import importlib
from pathlib import Path

from catmaster.llm.config import ImageGenerationConfig, LLMConfig
from catmaster.tools.base import ensure_project_space_layout, workspace_scope


_ONE_BY_ONE_PNG = (
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnR0d0AAAAASUVORK5CYII="
)


class _FakeResponse:
    status_code = 200
    text = '{"ok":true}'

    def json(self):
        return {
            "choices": [
                {
                    "message": {
                        "content": "Clean schematic with labeled gas, surface, and adsorption arrow.",
                        "images": [
                            {
                                "image_url": {
                                    "url": f"data:image/png;base64,{_ONE_BY_ONE_PNG}"
                                }
                            }
                        ],
                    }
                }
            ]
        }


class _FakeClient:
    def __init__(self, *args, **kwargs):
        self.calls: list[dict] = []

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def post(self, url, *, headers=None, json=None):
        self.calls.append({"url": url, "headers": headers or {}, "json": json or {}})
        self.last_call = self.calls[-1]
        return _FakeResponse()


class _FakeProfile:
    def __init__(self) -> None:
        self.image_generation = ImageGenerationConfig(image_config={"aspect_ratio": "4:3"})

    def config_for_image_generation(self) -> LLMConfig:
        return LLMConfig(
            provider="openrouter",
            model="google/gemini-2.5-flash-image-preview",
            api_key_env="OPENROUTER_API_KEY",
            base_url="https://openrouter.ai/api/v1",
            default_headers={"X-Title": "CatMaster"},
            provider_options={"openrouter": {"extra_body": {"provider": {"order": ["google"]}}}},
        )


def test_generate_schematic_figure_writes_workspace_image(tmp_path: Path, monkeypatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    tool_module = importlib.import_module("catmaster.tools.analysis.generate_schematic_figure")
    fake_client = _FakeClient()

    monkeypatch.setenv("OPENROUTER_API_KEY", "test-key")
    monkeypatch.setattr(tool_module.httpx, "Client", lambda *args, **kwargs: fake_client)
    monkeypatch.setattr(tool_module.LLMProfile, "from_env_or_file", staticmethod(lambda: _FakeProfile()))

    with workspace_scope(tmp_path):
        content, artifact = tool_module.generate_schematic_figure(
            {
                "prompt": "Schematic of CO approaching Fe(110) with an adsorption arrow and site labels.",
                "output_path": "writing/write_001/figures/adsorption-schematic",
            }
        )

    output_path = tmp_path / "files" / "writing" / "write_001" / "figures" / "adsorption-schematic.png"
    assert output_path.exists()
    assert output_path.read_bytes()
    assert "Generated schematic figure:" in content
    assert artifact["data"]["output_path"] == "writing/write_001/figures/adsorption-schematic.png"
    assert artifact["data"]["mime_type"] == "image/png"
    assert artifact["data"]["image_config"] == {"aspect_ratio": "4:3"}
    assert fake_client.last_call["json"]["modalities"] == ["image", "text"]
    assert fake_client.last_call["json"]["provider"] == {"order": ["google"]}
    assert fake_client.last_call["json"]["image_config"]["aspect_ratio"] == "4:3"
    message_text = fake_client.last_call["json"]["messages"][0]["content"]
    assert "Generate a clean scientific schematic or conceptual manuscript figure." in message_text
    assert "User figure request:" in message_text
    assert "CO approaching Fe(110)" in message_text
