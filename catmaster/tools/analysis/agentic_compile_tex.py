from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class CompileTextInput(BaseModel):
    """[writing/compile] Compile or statically validate a manuscript bundle and return LaTeX diagnostics and artifacts."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        description="Workspace-relative path under files/ to the root manuscript .tex file.",
    )
_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_BIB_RE = re.compile(r"\\bibliography\{([^}]+)\}")
_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")
_MISSING_BIBKEY_RE = re.compile(r'Warning--I didn\'t find a database entry for "([^"]+)"')
_CITE_RE = re.compile(r"\\cite[a-zA-Z*]*\{")
_LATEX_ERROR_RE = re.compile(r"^!\s*(.+)$", re.MULTILINE)


def _strip_comments(text: str) -> str:
    lines: list[str] = []
    for line in text.splitlines():
        buf: list[str] = []
        escaped = False
        for ch in line:
            if ch == "%" and not escaped:
                break
            buf.append(ch)
            escaped = (ch == "\\") and not escaped
            if ch != "\\":
                escaped = False
        lines.append("".join(buf))
    return "\n".join(lines)


def _candidate_with_extensions(base: Path, raw_ref: str, exts: tuple[str, ...]) -> Path | None:
    ref = Path(raw_ref)
    candidates: list[Path] = []
    if ref.suffix:
        candidates.append((base / ref).resolve())
    else:
        candidates.extend(((base / f"{raw_ref}{ext}").resolve() for ext in exts))
        candidates.append((base / raw_ref).resolve())
    for candidate in candidates:
        if candidate.exists():
            return candidate
    return candidates[0] if candidates else None


def _discover_related_files(root_tex: Path) -> tuple[list[Path], list[str]]:
    root = root_tex.resolve()
    bundle_root = root.parent.resolve()
    visited: set[Path] = set()
    ordered: list[Path] = []
    diagnostics: list[str] = []

    def visit(path: Path) -> None:
        resolved = path.resolve()
        if resolved in visited or not resolved.exists():
            return
        visited.add(resolved)
        ordered.append(resolved)
        text = resolved.read_text(encoding="utf-8")
        cleaned = _strip_comments(text)
        base = resolved.parent
        for raw in _INPUT_RE.findall(cleaned):
            child = _candidate_with_extensions(base, raw.strip(), (".tex",))
            if child is None:
                continue
            try:
                child.relative_to(bundle_root)
            except ValueError:
                diagnostics.append(f"Input/include escapes manuscript bundle: {raw}")
                continue
            if not child.exists():
                diagnostics.append(f"Missing input/include target: {workspace_relpath(child)}")
                continue
            visit(child)

    visit(root)
    return ordered, diagnostics


def _brace_balance_issues(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    cleaned = _strip_comments(text)
    depth = 0
    escaped = False
    for ch in cleaned:
        if escaped:
            escaped = False
            continue
        if ch == "\\":
            escaped = True
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth < 0:
                issues.append(f"Unmatched closing brace in {workspace_relpath(path)}")
                depth = 0
                break
    if depth != 0:
        issues.append(f"Unbalanced braces in {workspace_relpath(path)}")
    return issues


def _environment_issues(path: Path, text: str) -> list[str]:
    issues: list[str] = []
    cleaned = _strip_comments(text)
    stack: list[str] = []
    for match in re.finditer(r"\\(?:begin|end)\{([^}]+)\}", cleaned):
        token = match.group(0)
        env = match.group(1)
        if token.startswith("\\begin"):
            stack.append(env)
        else:
            if not stack or stack[-1] != env:
                issues.append(f"Mismatched environment end '{env}' in {workspace_relpath(path)}")
                continue
            stack.pop()
    if stack:
        issues.append(f"Unclosed environments in {workspace_relpath(path)}: {', '.join(stack[:6])}")
    return issues


def _reference_issues(path: Path, text: str, *, bundle_root: Path) -> list[str]:
    issues: list[str] = []
    cleaned = _strip_comments(text)
    base = path.parent
    for raw in _GRAPHICS_RE.findall(cleaned):
        target = _candidate_with_extensions(base, raw.strip(), (".pdf", ".png", ".jpg", ".jpeg", ".eps"))
        if target is None or not target.exists():
            issues.append(f"Missing includegraphics target from {workspace_relpath(path)}: {raw}")
            continue
        try:
            target.resolve().relative_to(bundle_root)
        except ValueError:
            issues.append(f"Graphic reference escapes manuscript bundle from {workspace_relpath(path)}: {raw}")
    for group in _BIB_RE.findall(cleaned):
        for item in [part.strip() for part in group.split(",") if part.strip()]:
            target = _candidate_with_extensions(base, item, (".bib",))
            if target is None or not target.exists():
                issues.append(f"Missing bibliography target from {workspace_relpath(path)}: {item}")
    return issues


def _static_diagnostics(root_tex: Path) -> tuple[list[Path], list[str]]:
    files, diagnostics = _discover_related_files(root_tex)
    bundle_root = root_tex.parent.resolve()
    for path in files:
        text = path.read_text(encoding="utf-8")
        diagnostics.extend(_brace_balance_issues(path, text))
        diagnostics.extend(_environment_issues(path, text))
        diagnostics.extend(_reference_issues(path, text, bundle_root=bundle_root))
    deduped: list[str] = []
    seen: set[str] = set()
    for item in diagnostics:
        if item not in seen:
            seen.add(item)
            deduped.append(item)
    return files, deduped


def _needs_bibtex(files: list[Path]) -> bool:
    for path in files:
        text = _strip_comments(path.read_text(encoding="utf-8"))
        if _BIB_RE.search(text):
            return True
    return False


def _has_real_citations(files: list[Path]) -> bool:
    for path in files:
        text = _strip_comments(path.read_text(encoding="utf-8"))
        if _CITE_RE.search(text):
            return True
    return False


def _bibliography_diagnostics(*, root_tex: Path, files: list[Path]) -> list[str]:
    diagnostics: list[str] = []
    has_citations = _has_real_citations(files)
    cleaned_root = _strip_comments(root_tex.read_text(encoding="utf-8"))
    has_bibliography_cmd = bool(_BIB_RE.search(cleaned_root))
    has_inline_bibliography = "\\begin{thebibliography}" in cleaned_root
    bib_files = sorted(path for path in root_tex.parent.glob("*.bib") if path.is_file())
    if has_inline_bibliography:
        diagnostics.append(
            "Inline `thebibliography` detected. Prefer a separate `.bib` file with `\\bibliography{references}`."
        )
    if has_citations and not has_bibliography_cmd and not has_inline_bibliography:
        diagnostics.append(
            "Citations are present but no bibliography command or inline bibliography was found."
        )
    if has_citations and has_bibliography_cmd and not bib_files:
        diagnostics.append(
            "Citations use `\\bibliography{...}` but no `.bib` file was found in the manuscript bundle."
        )
    return diagnostics


def _compiler_commands(*, root_tex: Path, needs_bibtex: bool) -> tuple[list[list[str]], str] | None:
    name = root_tex.name
    stem = root_tex.stem
    if not shutil.which("pdflatex"):
        return None
    if needs_bibtex and not shutil.which("bibtex"):
        raise ValueError("bibtex not available in PATH for manuscript with bibliography")
    commands: list[list[str]] = [
        ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name],
    ]
    if needs_bibtex:
        commands.extend(
            [
                ["bibtex", stem],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name],
                ["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name],
            ]
        )
        return commands, "pdflatex+bibtex"
    commands.append(["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name])
    return commands, "pdflatex"


def _run_compiler(root_tex: Path, *, needs_bibtex: bool) -> dict[str, Any]:
    resolved = _compiler_commands(root_tex=root_tex, needs_bibtex=needs_bibtex)
    if resolved is None:
        return {"available": False, "name": None, "ok": False, "stdout": "", "stderr": "", "returncode": None}
    commands, name = resolved
    stdout_parts: list[str] = []
    stderr_parts: list[str] = []
    returncode: int | None = None
    for cmd in commands:
        proc = subprocess.run(
            cmd,
            cwd=str(root_tex.parent),
            capture_output=True,
            text=True,
            timeout=120,
        )
        returncode = proc.returncode
        stdout_parts.append(proc.stdout)
        stderr_parts.append(proc.stderr)
        if proc.returncode != 0:
            break
    return {
        "available": True,
        "name": name,
        "ok": returncode == 0,
        "stdout": "\n".join(part for part in stdout_parts if part),
        "stderr": "\n".join(part for part in stderr_parts if part),
        "returncode": returncode,
    }


def _compiler_diagnostics(compiler_result: dict[str, Any]) -> list[str]:
    diagnostics: list[str] = []
    combined = f"{compiler_result.get('stdout') or ''}\n{compiler_result.get('stderr') or ''}"
    missing_keys = sorted(set(_MISSING_BIBKEY_RE.findall(combined)))
    for key in missing_keys:
        diagnostics.append(f"Missing bibliography entry for citation key: {key}")
    if "Empty `thebibliography' environment" in combined:
        diagnostics.append("Bibliography resolved to an empty thebibliography environment.")
    for item in _LATEX_ERROR_RE.findall(combined):
        text = str(item).strip()
        if text:
            diagnostics.append(f"LaTeX error: {text}")
    for line in combined.splitlines():
        stripped = line.strip()
        if stripped.startswith("LaTeX Error:"):
            diagnostics.append(stripped)
        elif stripped.startswith("Package ") and " Error:" in stripped:
            diagnostics.append(stripped)
    return diagnostics


def _compiler_excerpt(compiler_result: dict[str, Any], *, limit: int = 4000) -> str:
    combined = f"{compiler_result.get('stdout') or ''}\n{compiler_result.get('stderr') or ''}".strip()
    if len(combined) <= limit:
        return combined
    return combined[-limit:]


AgenticCompileTexInput = CompileTextInput


def compile_text(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[writing/compile] Run static checks plus an optional TeX compile pass for a manuscript bundle."""
    tool_name = "compile_text"
    try:
        params = CompileTextInput(**payload)
        root_tex = resolve_workspace_path(params.source_path, must_exist=True)
        if root_tex.suffix.lower() != ".tex":
            raise ValueError("source_path must point to a .tex file")
        files, diagnostics = _static_diagnostics(root_tex)
        diagnostics.extend(_bibliography_diagnostics(root_tex=root_tex, files=files))
        needs_bibtex = _needs_bibtex(files)
        compile_result = _run_compiler(root_tex, needs_bibtex=needs_bibtex)
        final_diagnostics = list(
            dict.fromkeys(
                [
                    *diagnostics,
                    *_compiler_diagnostics(compile_result),
                    *(["Compiler reported errors; inspect log excerpt."] if not compile_result.get("ok") else []),
                ]
            )
        )
        compiled_ok = bool(compile_result.get("ok") and not diagnostics)
        pdf_path = root_tex.with_suffix(".pdf")
        log_path = root_tex.with_suffix(".log")
        bib_paths = sorted(workspace_relpath(path) for path in root_tex.parent.glob("*.bib") if path.is_file())
        bbl_path = root_tex.with_suffix(".bbl")
        content_lines = [
            f"TeX compile pass finished for {workspace_relpath(root_tex)}",
            f"Compiler used: {str(compile_result.get('name') or '(none)')}",
            f"Compiled cleanly: {'yes' if compiled_ok else 'no'}",
            f"Separate bibliography files: {len(bib_paths)}",
        ]
        if pdf_path.exists():
            content_lines.append(f"pdf_path={workspace_relpath(pdf_path)}")
        if log_path.exists():
            content_lines.append(f"log_path={workspace_relpath(log_path)}")
        if bbl_path.exists():
            content_lines.append(f"bbl_path={workspace_relpath(bbl_path)}")
        if bib_paths:
            if len(bib_paths) <= 12:
                content_lines.append(f"bib_paths={json.dumps(bib_paths, ensure_ascii=False)}")
            else:
                content_lines.append(f"bib_paths_count={len(bib_paths)}; inspect the source directory")
        if final_diagnostics:
            content_lines.append("Diagnostics summary:")
            content_lines.extend(f"- {item}" for item in final_diagnostics[:8])
        artifact = {
            "tool_name": tool_name,
            "data": {
                "source_path": workspace_relpath(root_tex),
                "compiler_available": bool(compile_result.get("available")),
                "compiler_name": str(compile_result.get("name") or ""),
                "compiled_ok": compiled_ok,
                "pdf_path": workspace_relpath(pdf_path) if pdf_path.exists() else None,
                "bib_paths": bib_paths,
                "bbl_path": workspace_relpath(bbl_path) if bbl_path.exists() else None,
                "log_path": workspace_relpath(log_path) if log_path.exists() else None,
                "remaining_diagnostics": final_diagnostics,
                "log_excerpt": _compiler_excerpt(compile_result),
                "inspected_files": [workspace_relpath(path) for path in files],
            },
        }
        return "\n".join(content_lines).strip(), artifact
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={
                "tool_name": tool_name,
                "data": {
                    "source_path": payload.get("source_path"),
                },
            },
            error_code="compile_text_failed",
        ) from exc


def agentic_compile_tex(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    return compile_text(payload)


__all__ = ["CompileTextInput", "compile_text", "AgenticCompileTexInput", "agentic_compile_tex"]
