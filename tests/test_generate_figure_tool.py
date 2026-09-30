from __future__ import annotations

import base64
import importlib
import json
from pathlib import Path
from types import SimpleNamespace

import httpx
import pytest

from catmaster.llm.config import ImageGenerationConfig, LLMConfig
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.registry import ToolRegistry


_PNG = "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAusB9WnR0d0AAAAASUVORK5CYII="


@pytest.fixture
def generation(tmp_path, monkeypatch):
    module = importlib.import_module("catmaster.tools.analysis.generate_figure")
    default = LLMConfig(
        provider="openrouter", model="openai/gpt-image-2.5-sunburst",
        api_key_env="FIGURE_TEST_KEY", base_url="https://openrouter.ai/api/v1",
        provider_options={"openrouter": {"extra_body": {"provider": {"order": ["openai"]}}}},
    )
    alternate = LLMConfig(
        provider="openrouter", model="google/gemini-3.1-flash-image",
        api_key_env="FIGURE_ALTERNATE_KEY", base_url="https://image-gateway.test/api/v1",
    )
    profile = SimpleNamespace(
        models={"default": default, "alternate": alternate},
        config_for_image_generation=lambda: default,
        image_generation=ImageGenerationConfig(model_label="default", image_config={"aspect_ratio": "4:3"}),
    )
    state = SimpleNamespace(
        module=module, profile=profile, requests=[], status=200,
        response={"data": [{"b64_json": _PNG, "media_type": "image/png"}]},
    )
    def handle(request):
        state.requests.append(request)
        return httpx.Response(state.status, json=state.response)
    client = httpx.Client
    monkeypatch.setattr(module.httpx, "Client", lambda **kwargs: client(transport=httpx.MockTransport(handle), **kwargs))
    monkeypatch.setattr(module.LLMProfile, "from_env_or_file", staticmethod(lambda: profile))
    monkeypatch.setenv("FIGURE_TEST_KEY", "test-secret")
    monkeypatch.setenv("FIGURE_ALTERNATE_KEY", "alternate-secret")
    ensure_project_space_layout(tmp_path, create=True)
    with workspace_scope(tmp_path):
        yield state


def test_generation_saves_image_and_preserves_prompt(generation, tmp_path):
    content, artifact = generation.module.generate_figure({"prompt": "Dark background with a red sphere.", "output_path": "writing/figure"})
    request = generation.requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://openrouter.ai/api/v1/images"
    assert request.headers["Authorization"] == "Bearer test-secret"
    assert body == {
        "model": "openai/gpt-image-2.5-sunburst", "prompt": "Dark background with a red sphere.",
        "aspect_ratio": "4:3", "provider": {"order": ["openai"]}, "n": 1, "stream": False,
    }
    assert (tmp_path / "files/writing/figure.png").read_bytes() == base64.b64decode(_PNG)
    assert artifact["data"]["output_path"] == "writing/figure.png"
    assert "writing/figure.png" in content


def test_model_label_uses_own_connection_and_attaches_reference(generation, tmp_path):
    source = tmp_path / "files/source.png"
    source.write_bytes(base64.b64decode(_PNG))
    generation.module.generate_figure({
        "prompt": "Retain the object; make the background blue.", "output_path": "writing/edit.png",
        "model": "alternate", "reference_images": ["source.png"],
        "image_options": {"aspect_ratio": "16:9", "quality": "high"},
    })
    request = generation.requests[0]
    body = json.loads(request.content)
    assert str(request.url) == "https://image-gateway.test/api/v1/images"
    assert request.headers["Authorization"] == "Bearer alternate-secret"
    assert body["model"] == "google/gemini-3.1-flash-image"
    assert body["input_references"] == [{"type": "image_url", "image_url": {"url": f"data:image/png;base64,{_PNG}"}}]
    assert body["aspect_ratio"] == "16:9"
    assert body["quality"] == "high"
    assert source.read_bytes() == base64.b64decode(_PNG)


def test_raw_model_override_and_explicit_size(generation):
    generation.module.generate_figure({
        "prompt": "A test sphere", "output_path": "figure.png",
        "model": "openai/gpt-image-2.5-flare", "image_options": {"size": "1024x1024"},
    })
    body = json.loads(generation.requests[0].content)
    assert body["model"] == "openai/gpt-image-2.5-flare"
    assert body["size"] == "1024x1024"
    assert "provider" not in body
    assert "aspect_ratio" not in body


@pytest.mark.parametrize("extra", [
    {"reference_images": ["missing.png"]},
    {"reference_images": ["../outside.png"]},
    {"image_options": {"n": 2}},
    {"image_options": {"output_format": "webp"}},
    {"output_path": "../outside.png"},
])
def test_invalid_local_inputs_fail_before_paid_request(generation, extra):
    with pytest.raises(CatMasterToolExecutionError):
        generation.module.generate_figure({"prompt": "Test", "output_path": "figure.png", **extra})
    assert not generation.requests


def test_missing_media_type_is_inferred(generation):
    generation.response = {"data": [{"b64_json": _PNG}]}
    _, artifact = generation.module.generate_figure({"prompt": "Test", "output_path": "figure"})
    assert artifact["data"]["mime_type"] == "image/png"


def test_provider_failure_is_actionable_and_redacts_key(generation, tmp_path):
    generation.status = 401
    generation.response = {"error": {"message": "Invalid test-secret"}}
    with pytest.raises(CatMasterToolExecutionError, match="HTTP 401") as error:
        generation.module.generate_figure({"prompt": "Test", "output_path": "figure.png"})
    assert "test-secret" not in str(error.value)
    assert not (tmp_path / "files/figure.png").exists()


def test_empty_response_does_not_report_success(generation):
    generation.response = {"data": []}
    with pytest.raises(CatMasterToolExecutionError, match="single requested image"):
        generation.module.generate_figure({"prompt": "Test", "output_path": "figure.png"})


def test_final_tool_schemas_and_old_name_alias():
    registry = ToolRegistry()
    tools = registry.as_openai_tools(allowlist=["generate_figure", "generate_nanobanana_figure"])
    assert [tool["name"] for tool in tools] == ["generate_figure"]
    native = registry.as_langchain_tools(allowlist=["generate_figure"])[0]
    for schema in (tools[0]["parameters"], native.args_schema):
        assert schema["required"] == ["prompt", "output_path"]
        for key, kind in (("model", "string"), ("reference_images", "array"), ("image_options", "object")):
            field = schema["properties"][key]
            assert field["type"] == kind
            assert "anyOf" not in field
    input_model = registry.get_tool_info("generate_figure")["input_model"]
    parsed = input_model(prompt="Test", output_path="figure.png", model=None, reference_images=None, image_options=None)
    assert parsed.model == ""
    assert parsed.reference_images == []
    assert parsed.image_options == {}
    assert registry.get_tool_function("generate_nanobanana_figure") is registry.get_tool_function("generate_figure")
