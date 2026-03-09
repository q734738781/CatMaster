from __future__ import annotations

import json
import re
import shutil
import subprocess
from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class AgenticCompileTexInput(BaseModel):
    """Compile or statically validate a manuscript bundle and repair LaTeX compile/reference issues without changing scientific meaning."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        description="Workspace-relative path under files/ to the root manuscript .tex file.",
    )


class _RewriteFile(BaseModel):
    model_config = ConfigDict(extra="forbid")

    path: str = Field(...)
    content: str = Field(...)


class _TexFixOutput(BaseModel):
    model_config = ConfigDict(extra="forbid")

    files: list[_RewriteFile] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


_INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
_GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
_BIB_RE = re.compile(r"\\bibliography\{([^}]+)\}")
_BEGIN_RE = re.compile(r"\\begin\{([^}]+)\}")
_END_RE = re.compile(r"\\end\{([^}]+)\}")


def _resolve_config() -> LLMConfig:
    profile = LLMProfile.from_env_or_file()
    return profile.config_for_role("tex_compile_fixer")


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


def _compiler_command(root_tex: Path) -> tuple[list[str], str] | None:
    name = root_tex.name
    if shutil.which("pdflatex"):
        return (["pdflatex", "-interaction=nonstopmode", "-halt-on-error", name], "pdflatex")
    return None


def _run_compiler(root_tex: Path) -> dict[str, Any]:
    resolved = _compiler_command(root_tex)
    if resolved is None:
        return {"available": False, "name": None, "ok": False, "stdout": "", "stderr": "", "returncode": None}
    cmd, name = resolved
    proc = subprocess.run(
        cmd,
        cwd=str(root_tex.parent),
        capture_output=True,
        text=True,
        timeout=120,
    )
    return {
        "available": True,
        "name": name,
        "ok": proc.returncode == 0,
        "stdout": proc.stdout,
        "stderr": proc.stderr,
        "returncode": proc.returncode,
    }


def _build_fix_messages(*, root_tex: Path, diagnostics: list[str], compiler_result: dict[str, Any], files: list[Path]) -> list[Any]:
    payload_files = []
    for path in files[:8]:
        payload_files.append(
            {
                "path": workspace_relpath(path),
                "content": path.read_text(encoding="utf-8"),
            }
        )
    compiler_excerpt = "\n".join(
        [
            f"Compiler available: {'yes' if compiler_result.get('available') else 'no'}",
            f"Compiler name: {compiler_result.get('name') or '(none)'}",
            f"Return code: {compiler_result.get('returncode')}",
            "Compiler stdout/stderr excerpt:",
            (str(compiler_result.get("stdout") or "") + "\n" + str(compiler_result.get("stderr") or ""))[-6000:],
        ]
    ).strip()
    human = "\n".join(
        [
            f"Root manuscript: {workspace_relpath(root_tex)}",
            "Goal: make the manuscript bundle compile cleanly and keep references/graphics/input paths valid.",
            "Allowed edits: LaTeX syntax, path fixes, missing \\input/\\includegraphics reference corrections, environment/bracing fixes, harmless compile-oriented escapes.",
            "Forbidden edits: changing scientific claims, changing numerical conclusions, rewriting substantive interpretation, deleting evidence-backed content just to silence errors.",
            "Return only file rewrites for files that truly need changes.",
            "",
            "Diagnostics:",
            *([f"- {item}" for item in diagnostics] or ["- (none)"]),
            "",
            compiler_excerpt,
            "",
            "Files:",
            json.dumps(payload_files, ensure_ascii=False, indent=2),
        ]
    ).strip()
    return [
        SystemMessage(
            content=(
                "You repair LaTeX manuscript bundles. "
                "Fix compile/reference/syntax/path issues only. "
                "Preserve the author's scientific wording and meaning as much as possible."
            )
        ),
        HumanMessage(content=human),
    ]


def _apply_rewrites(*, root_tex: Path, rewrites: list[_RewriteFile]) -> list[str]:
    bundle_root = root_tex.parent.resolve()
    touched: list[str] = []
    for rewrite in rewrites:
        target = resolve_workspace_path(rewrite.path, must_exist=False)
        try:
            target.relative_to(bundle_root)
        except ValueError as exc:
            raise ValueError(f"rewrite target escapes manuscript bundle: {rewrite.path}") from exc
        target.write_text(rewrite.content, encoding="utf-8")
        touched.append(workspace_relpath(target))
    return touched


def agentic_compile_tex(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "agentic_compile_tex"
    try:
        params = AgenticCompileTexInput(**payload)
        root_tex = resolve_workspace_path(params.source_path, must_exist=True)
        if root_tex.suffix.lower() != ".tex":
            raise ValueError("source_path must point to a .tex file")
        if _compiler_command(root_tex) is None:
            raise ValueError("pdflatex not available in PATH")
        cfg = _resolve_config()
        model = build_chat_model(cfg).with_structured_output(_TexFixOutput)
        touched: list[str] = []
        compiler_name: str | None = None
        final_diagnostics: list[str] = []
        compiled_ok = False
        for _ in range(2):
            files, diagnostics = _static_diagnostics(root_tex)
            compile_result = _run_compiler(root_tex)
            compiler_name = str(compile_result.get("name") or "") or compiler_name
            compile_errors = []
            if not compile_result.get("ok"):
                compile_errors.append("Compiler reported errors; inspect stdout/stderr excerpt.")
            final_diagnostics = list(dict.fromkeys([*diagnostics, *compile_errors]))
            compiled_ok = bool(compile_result.get("ok") and not diagnostics)
            if compiled_ok:
                break
            fix = model.invoke(
                _build_fix_messages(
                    root_tex=root_tex,
                    diagnostics=final_diagnostics,
                    compiler_result=compile_result,
                    files=files,
                )
            )
            if not fix.files:
                break
            touched.extend(_apply_rewrites(root_tex=root_tex, rewrites=fix.files))
        files, diagnostics = _static_diagnostics(root_tex)
        compile_result = _run_compiler(root_tex)
        compiler_name = str(compile_result.get("name") or "") or compiler_name
        final_diagnostics = list(dict.fromkeys([*diagnostics, *(["Compiler reported errors; inspect log excerpt."] if not compile_result.get("ok") else [])]))
        compiled_ok = bool(compile_result.get("ok") and not diagnostics)
        pdf_path = root_tex.with_suffix(".pdf")
        content_lines = [
            f"TeX compile pass finished for {workspace_relpath(root_tex)}",
            f"Compiler used: {compiler_name or '(none)'}",
            f"Compiled cleanly: {'yes' if compiled_ok else 'no'}",
            f"Files rewritten: {len(list(dict.fromkeys(touched)))}",
        ]
        if final_diagnostics:
            content_lines.append("Remaining diagnostics:")
            content_lines.extend(f"- {item}" for item in final_diagnostics[:8])
        artifact = {
            "tool_name": tool_name,
            "data": {
                "source_path": workspace_relpath(root_tex),
                "compiler_available": True,
                "compiler_name": compiler_name,
                "compiled_ok": compiled_ok,
                "pdf_path": workspace_relpath(pdf_path) if pdf_path.exists() else None,
                "rewritten_files": list(dict.fromkeys(touched)),
                "remaining_diagnostics": final_diagnostics,
                "inspected_files": [workspace_relpath(path) for path in files],
                "model_name": cfg.model,
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
            error_code="agentic_compile_tex_failed",
        ) from exc


__all__ = ["AgenticCompileTexInput", "agentic_compile_tex"]
