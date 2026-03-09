from __future__ import annotations

import importlib
from pathlib import Path

import pytest

from catmaster.tools.base import workspace_scope

compile_mod = importlib.import_module("catmaster.tools.analysis.agentic_compile_tex")


class _FakeStructuredModel:
    def __init__(self, response):
        self._response = response

    def invoke(self, _messages):
        return self._response


class _FakeBaseModel:
    def __init__(self, response):
        self._response = response

    def with_structured_output(self, _schema):
        return _FakeStructuredModel(self._response)


def test_agentic_compile_tex_fixes_static_tex_issue_without_compiler(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    sections_dir = manuscript_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    sec_tex = sections_dir / "sec_results.tex"
    root_tex.write_text("\\documentclass{article}\n\\begin{document}\n\\input{sections/sec_results.tex}\n\\end{document}\n", encoding="utf-8")
    sec_tex.write_text("\\section{Results\nBody.\n", encoding="utf-8")

    monkeypatch.setattr(
        compile_mod,
        "build_chat_model",
        lambda _cfg: _FakeBaseModel(
            response=type(
                "_Fix",
                (),
                {
                    "files": [
                        type(
                            "_Rewrite",
                            (),
                            {
                                "path": "manuscript/sections/sec_results.tex",
                                "content": "\\section{Results}\nBody.\n",
                            },
                        )()
                    ],
                    "notes": ["balanced section heading braces"],
                },
            )()
        ),
    )
    monkeypatch.setattr(
        compile_mod,
        "_resolve_config",
        lambda: type("_Cfg", (), {"model": "fake-tex-fixer"})(),
    )
    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    calls = {"n": 0}

    def _fake_run(*_args, **_kwargs):
        calls["n"] += 1
        if calls["n"] == 1:
            return _Proc(1, stderr="! Missing } inserted.")
        pdf_path = manuscript_dir / "MANUSCRIPT.pdf"
        pdf_path.write_bytes(b"%PDF-1.4\n")
        return _Proc(0)

    monkeypatch.setattr(compile_mod.shutil, "which", lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    monkeypatch.setattr(compile_mod.subprocess, "run", _fake_run)

    with workspace_scope(tmp_path):
        content, artifact = compile_mod.agentic_compile_tex({"source_path": "manuscript/MANUSCRIPT.tex"})

    assert "Compiler used: pdflatex" in content
    assert artifact["data"]["compiler_available"] is True
    assert artifact["data"]["compiled_ok"] is True
    assert artifact["data"]["remaining_diagnostics"] == []
    assert "manuscript/sections/sec_results.tex" in artifact["data"]["rewritten_files"]
    assert sec_tex.read_text(encoding="utf-8") == "\\section{Results}\nBody.\n"
