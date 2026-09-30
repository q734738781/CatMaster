from __future__ import annotations

import asyncio
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def smoke():
    path = Path(__file__).parent / "manual" / "local_skill_retirement_smoke.py"
    spec = importlib.util.spec_from_file_location("local_skill_smoke", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_custom_request_uses_isolated_input_copy(smoke, tmp_path, monkeypatch):
    source = tmp_path / "source.txt"
    source.write_text("existing evidence", encoding="utf-8")
    request = tmp_path / "request.md"
    prompt = "Revise input/source.txt using the supplied feedback."
    request.write_text(prompt, encoding="utf-8")
    workspace = tmp_path / "isolated"
    workspace.mkdir()
    monkeypatch.setattr(smoke.tempfile, "mkdtemp", lambda **_: str(workspace))

    class Runner:
        async def arun(self, actual_prompt, **kwargs):
            assert actual_prompt == prompt
            assert kwargs["entrypoint"] == "writing"
            staged = workspace / "files/input/source.txt"
            assert staged.read_text() == "existing evidence"
            staged.write_text("revised copy", encoding="utf-8")
            return {"status": "done"}

    monkeypatch.setattr(smoke, "build_specialist_runner", lambda **kwargs: SimpleNamespace(
        runner=Runner(), run_context=SimpleNamespace(run_dir=workspace / "metadata/runs/test"),
    ))
    result = asyncio.run(smoke.run_case(
        "progress-slides", object(), prompt_file=request, input_files=[source],
    ))
    assert result["status"] == "done"
    assert source.read_text() == "existing evidence"


def test_input_basename_collision_is_reported_before_copying(smoke, tmp_path):
    workspace = tmp_path / "isolated"
    with pytest.raises(ValueError, match="distinct basenames"):
        smoke.stage_inputs(workspace, [tmp_path / "a/data.txt", tmp_path / "b/data.txt"])
    assert not workspace.exists()


def test_audience_cases_share_evidence_but_have_separate_requests(smoke):
    report_lane, report = smoke.CASES["audience-report"]
    technical_lane, technical = smoke.CASES["audience-technical"]
    assert report_lane == technical_lane == "writing"
    assert report.startswith(smoke.AUDIENCE_EVIDENCE)
    assert technical.startswith(smoke.AUDIENCE_EVIDENCE)
    assert report.removeprefix(smoke.AUDIENCE_EVIDENCE) != technical.removeprefix(smoke.AUDIENCE_EVIDENCE)
