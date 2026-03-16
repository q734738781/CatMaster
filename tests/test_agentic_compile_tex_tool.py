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


def test_agentic_compile_tex_runs_bibtex_for_bibliography(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    bib_path = manuscript_dir / "references.bib"
    root_tex.write_text(
        "\\documentclass{article}\n\\begin{document}\nSee~\\cite{foo}.\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    bib_path.write_text("@article{foo, title={Foo}}\n", encoding="utf-8")

    monkeypatch.setattr(
        compile_mod,
        "build_chat_model",
        lambda _cfg: _FakeBaseModel(
            response=type("_Fix", (), {"files": [], "notes": []})()
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

    commands: list[list[str]] = []

    def _fake_run(cmd, *_args, **_kwargs):
        commands.append(list(cmd))
        if cmd[0] == "pdflatex" and len(commands) >= 4:
            (manuscript_dir / "MANUSCRIPT.pdf").write_bytes(b"%PDF-1.4\n")
        return _Proc(0)

    monkeypatch.setattr(
        compile_mod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"pdflatex", "bibtex"} else None,
    )
    monkeypatch.setattr(compile_mod.subprocess, "run", _fake_run)

    with workspace_scope(tmp_path):
        content, artifact = compile_mod.agentic_compile_tex({"source_path": "manuscript/MANUSCRIPT.tex"})

    assert "Compiler used: pdflatex+bibtex" in content
    assert artifact["data"]["compiled_ok"] is True
    assert commands[:4] == [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
        ["bibtex", "MANUSCRIPT"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
    ]
    assert any(cmd == ["bibtex", "MANUSCRIPT"] for cmd in commands)


def test_agentic_compile_tex_restores_bibliography_when_citations_exist(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    sections_dir = manuscript_dir / "sections"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    sections_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    sec_tex = sections_dir / "sec_intro.tex"
    root_tex.write_text(
        "\\documentclass{article}\n"
        "\\renewcommand{\\cite}[2][]{ }\n"
        "\\begin{document}\n"
        "\\input{sections/sec_intro.tex}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )
    sec_tex.write_text("Text with citation~\\cite{foo}.\n", encoding="utf-8")

    monkeypatch.setattr(
        compile_mod,
        "build_chat_model",
        lambda _cfg: _FakeBaseModel(response=type("_Fix", (), {"files": [], "notes": []})()),
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

    monkeypatch.setattr(
        compile_mod.shutil,
        "which",
        lambda name: f"/usr/bin/{name}" if name in {"pdflatex", "bibtex"} else None,
    )
    monkeypatch.setattr(compile_mod.subprocess, "run", lambda *_args, **_kwargs: _Proc(0))

    with workspace_scope(tmp_path):
        compile_mod.agentic_compile_tex({"source_path": "manuscript/MANUSCRIPT.tex"})

    updated = root_tex.read_text(encoding="utf-8")
    assert "\\renewcommand{\\cite}[2][]{ }" not in updated
    assert "\\bibliography{references}" in updated
    assert (manuscript_dir / "references.bib").exists()
