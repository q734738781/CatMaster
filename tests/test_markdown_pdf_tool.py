from __future__ import annotations

import importlib
import subprocess
from pathlib import Path

from catmaster.tools.base import workspace_scope
from catmaster.tools.registry import get_tool_registry


markdown_pdf = importlib.import_module("catmaster.tools.analysis.markdown_pdf")


def test_markdown_pdf_schema_is_non_nullable_and_defaults_to_yahei() -> None:
    schema = next(
        tool["parameters"]
        for tool in get_tool_registry().as_openai_tools(allowlist=["render_markdown_pdf"])
        if tool["name"] == "render_markdown_pdf"
    )
    properties = schema["properties"]

    assert properties["output_path"]["type"] == "string"
    assert properties["output_path"]["default"] == ""
    assert properties["document_title"]["type"] == "string"
    assert properties["font_family"]["default"] == "Microsoft YaHei"
    assert properties["page_size"]["default"] == "A4"
    assert "anyOf" not in properties["output_path"]
    assert "anyOf" not in properties["document_title"]


def test_render_markdown_pdf_preserves_source_and_uses_html_mathml_pipeline(
    tmp_path: Path,
    monkeypatch,
) -> None:
    project = tmp_path / "project"
    files = project / "files"
    source = files / "reports" / "summary.md"
    source.parent.mkdir(parents=True)
    original = "# 中文报告\n\n公式：$E = mc^2$。\n"
    source.write_text(original, encoding="utf-8")
    commands: list[list[str]] = []

    def _fake_executable(env_name: str, _names: tuple[str, ...]) -> Path:
        return Path("/usr/bin/pandoc" if env_name == "CATMASTER_PANDOC_BIN" else "/usr/bin/google-chrome")

    def _fake_run(command: list[str], *, cwd: Path, timeout: int):
        _ = cwd, timeout
        commands.append(command)
        if command[0].endswith("pandoc"):
            output = Path(command[command.index("--output") + 1])
            output.write_text("<html><body>中文报告</body></html>", encoding="utf-8")
        else:
            output_arg = next(item for item in command if item.startswith("--print-to-pdf="))
            Path(output_arg.split("=", 1)[1]).write_bytes(b"%PDF-1.7\nmock")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    monkeypatch.setattr(markdown_pdf, "_resolve_executable", _fake_executable)
    monkeypatch.setattr(
        markdown_pdf,
        "_resolve_font",
        lambda requested: (requested, "Microsoft YaHei,微软雅黑", "/fonts/msyh.ttc", False),
    )
    monkeypatch.setattr(markdown_pdf, "_run_command", _fake_run)
    monkeypatch.setattr(markdown_pdf, "_pdf_fonts", lambda _path: ["MicrosoftYaHei"])
    monkeypatch.setattr(markdown_pdf, "_pdf_page_count", lambda _path: 1)

    with workspace_scope(project):
        content, artifact = markdown_pdf.render_markdown_pdf({"source_path": "reports/summary.md"})

    assert source.read_text(encoding="utf-8") == original
    assert (files / "reports" / "summary.pdf").read_bytes().startswith(b"%PDF-")
    assert artifact["data"]["source_path"] == "reports/summary.md"
    assert artifact["data"]["pdf_path"] == "reports/summary.pdf"
    assert artifact["data"]["font_resolved_family"] == "Microsoft YaHei,微软雅黑"
    assert artifact["data"]["embedded_pdf_fonts"] == ["MicrosoftYaHei"]
    assert "Source preserved" in content
    assert any("--mathml" in command for command in commands)
    assert any(any(item.startswith("--print-to-pdf=") for item in command) for command in commands)
    assert not any("latex" in item.lower() for command in commands for item in command)
