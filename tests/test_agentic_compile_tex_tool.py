from __future__ import annotations

import importlib
from pathlib import Path

from catmaster.tools.base import workspace_scope

compile_mod = importlib.import_module("catmaster.tools.analysis.agentic_compile_tex")


def test_agentic_compile_tex_returns_log_summary_without_rewriting(monkeypatch, tmp_path: Path) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    sections_dir = manuscript_dir / "sections"
    sections_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    sec_tex = sections_dir / "sec_results.tex"
    root_tex.write_text("\\documentclass{article}\n\\begin{document}\n\\input{sections/sec_results.tex}\n\\end{document}\n", encoding="utf-8")
    sec_tex.write_text("\\section{Results\nBody.\n", encoding="utf-8")

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(compile_mod.shutil, "which", lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    monkeypatch.setattr(
        compile_mod.subprocess,
        "run",
        lambda *_args, **_kwargs: _Proc(1, stderr="! Missing } inserted.\nLaTeX Error: File ended while scanning use of \\\\section."),
    )

    with workspace_scope(tmp_path):
        content, artifact = compile_mod.agentic_compile_tex({"source_path": "manuscript/MANUSCRIPT.tex"})

    assert "Compiled cleanly: no" in content
    assert artifact["data"]["compiled_ok"] is False
    assert artifact["data"]["pdf_path"] is None
    assert artifact["data"]["log_excerpt"]
    assert any("Unbalanced braces" in item for item in artifact["data"]["remaining_diagnostics"])
    assert any("LaTeX error:" in item or "LaTeX Error:" in item for item in artifact["data"]["remaining_diagnostics"])
    assert sec_tex.read_text(encoding="utf-8") == "\\section{Results\nBody.\n"


def test_agentic_compile_tex_runs_bibtex_and_reports_artifacts(monkeypatch, tmp_path: Path) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    bib_path = manuscript_dir / "references.bib"
    root_tex.write_text(
        "\\documentclass{article}\n\\begin{document}\nSee~\\cite{foo}.\n\\bibliography{references}\n\\end{document}\n",
        encoding="utf-8",
    )
    bib_path.write_text("@article{foo, title={Foo}}\n", encoding="utf-8")

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
            (manuscript_dir / "MANUSCRIPT.log").write_text("clean compile", encoding="utf-8")
            (manuscript_dir / "MANUSCRIPT.bbl").write_text("% bibliography", encoding="utf-8")
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
    assert artifact["data"]["pdf_path"] == "manuscript/MANUSCRIPT.pdf"
    assert artifact["data"]["bib_paths"] == ["manuscript/references.bib"]
    assert artifact["data"]["bbl_path"] == "manuscript/MANUSCRIPT.bbl"
    assert artifact["data"]["log_path"] == "manuscript/MANUSCRIPT.log"
    assert commands[:4] == [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
        ["bibtex", "MANUSCRIPT"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", "MANUSCRIPT.tex"],
    ]


def test_agentic_compile_tex_flags_inline_bibliography(monkeypatch, tmp_path: Path) -> None:
    manuscript_dir = tmp_path / "files" / "manuscript"
    manuscript_dir.mkdir(parents=True, exist_ok=True)
    root_tex = manuscript_dir / "MANUSCRIPT.tex"
    root_tex.write_text(
        "\\documentclass{article}\n"
        "\\begin{document}\n"
        "Text with citation~\\cite{foo}.\n"
        "\\begin{thebibliography}{9}\n"
        "\\bibitem{foo} Foo.\n"
        "\\end{thebibliography}\n"
        "\\end{document}\n",
        encoding="utf-8",
    )

    class _Proc:
        def __init__(self, returncode: int, stdout: str = "", stderr: str = "") -> None:
            self.returncode = returncode
            self.stdout = stdout
            self.stderr = stderr

    monkeypatch.setattr(compile_mod.shutil, "which", lambda name: "/usr/bin/pdflatex" if name == "pdflatex" else None)
    monkeypatch.setattr(compile_mod.subprocess, "run", lambda *_args, **_kwargs: _Proc(0))

    with workspace_scope(tmp_path):
        _content, artifact = compile_mod.agentic_compile_tex({"source_path": "manuscript/MANUSCRIPT.tex"})

    assert artifact["data"]["compiled_ok"] is False
    assert artifact["data"]["bib_paths"] == []
    assert any("Inline `thebibliography` detected" in item for item in artifact["data"]["remaining_diagnostics"])
