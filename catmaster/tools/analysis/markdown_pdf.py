from __future__ import annotations

import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath, workspace_root


class RenderMarkdownPdfInput(BaseModel):
    """[writing/compile] Render existing Markdown directly to PDF with deterministic page and CJK font settings."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        description="Workspace-relative path under files/ to an existing .md or .markdown document.",
    )
    output_path: str = Field(
        default="",
        description="Workspace-relative .pdf output path. Leave empty to write beside the source with the same stem.",
    )
    document_title: str = Field(
        default="",
        description="PDF document metadata title. Leave empty to use the source filename without adding a visible heading.",
    )
    font_family: Literal["Microsoft YaHei", "Noto Sans CJK SC"] = Field(
        default="Microsoft YaHei",
        description="Primary body font. Microsoft YaHei is the Chinese-report default; Noto Sans CJK SC is the portable fallback.",
    )
    page_size: Literal["A4", "Letter"] = Field(
        default="A4",
        description="Printed PDF page size.",
    )


def _resolve_executable(env_name: str, names: tuple[str, ...]) -> Path:
    explicit = str(os.getenv(env_name) or "").strip()
    if explicit:
        path = Path(explicit).expanduser().resolve()
        if not path.is_file() or not os.access(path, os.X_OK):
            raise ValueError(f"{env_name} does not point to an executable file: {path}")
        return path
    for name in names:
        resolved = shutil.which(name)
        if resolved:
            return Path(resolved).resolve()
    raise ValueError(f"No executable found for {env_name}; tried: {', '.join(names)}")


def _match_font(query: str) -> tuple[str, str]:
    fc_match = shutil.which("fc-match")
    if not fc_match:
        raise ValueError("fontconfig fc-match is required to verify the PDF font")
    proc = subprocess.run(
        [fc_match, "-f", "%{family}\n%{file}\n", query],
        capture_output=True,
        text=True,
        timeout=20,
    )
    if proc.returncode != 0:
        raise ValueError(f"fc-match failed for {query}: {proc.stderr.strip()}")
    lines = [line.strip() for line in proc.stdout.splitlines() if line.strip()]
    family = lines[0] if lines else ""
    font_file = lines[1] if len(lines) > 1 else ""
    return family, font_file


def _resolve_font(requested: str) -> tuple[str, str, str, bool]:
    family, font_file = _match_font(requested)
    if requested.casefold() in family.casefold():
        return requested, family, font_file, False
    fallback = "Noto Sans CJK SC"
    fallback_family, fallback_file = _match_font(fallback)
    if fallback.casefold() not in fallback_family.casefold():
        raise ValueError(
            f"Neither {requested} nor {fallback} is installed. Install one CJK font before rendering."
        )
    return fallback, fallback_family, fallback_file, True


def _stylesheet(*, font_family: str, page_size: str) -> str:
    fallback_order = (
        '"Microsoft YaHei", "微软雅黑", "Noto Sans CJK SC", "Source Han Sans SC", sans-serif'
        if font_family == "Microsoft YaHei"
        else '"Noto Sans CJK SC", "Source Han Sans SC", "Microsoft YaHei", "微软雅黑", sans-serif'
    )
    return f"""
@page {{ size: {page_size}; margin: 18mm 18mm 20mm; }}
html {{ font-family: {fallback_order}; color: #202124; background: #ffffff; }}
body {{ margin: 0; font-family: inherit; font-size: 10.8pt; line-height: 1.72; letter-spacing: 0; }}
h1, h2, h3, h4, h5, h6 {{
  font-family: inherit; color: #111827; line-height: 1.32; page-break-after: avoid;
  break-after: avoid-page; letter-spacing: 0;
}}
h1 {{ font-size: 22pt; margin: 0 0 14pt; border-bottom: 1px solid #d1d5db; padding-bottom: 7pt; }}
h2 {{ font-size: 16pt; margin: 20pt 0 8pt; }}
h3 {{ font-size: 13pt; margin: 15pt 0 6pt; }}
p {{ margin: 0 0 8pt; orphans: 3; widows: 3; }}
ul, ol {{ margin: 4pt 0 9pt; padding-left: 1.8em; }}
li {{ margin: 2pt 0; }}
blockquote {{ margin: 10pt 0; padding: 7pt 11pt; border-left: 3px solid #6b7280; background: #f6f7f8; }}
code, pre {{ font-family: "Cascadia Mono", "Noto Sans Mono CJK SC", "DejaVu Sans Mono", monospace; }}
code {{ font-size: 0.91em; background: #f2f3f5; padding: 0.08em 0.28em; border-radius: 2px; }}
pre {{ font-size: 8.8pt; line-height: 1.5; background: #f5f6f7; border: 1px solid #dfe1e5;
  padding: 9pt; white-space: pre-wrap; overflow-wrap: anywhere; break-inside: avoid; }}
pre code {{ background: transparent; padding: 0; }}
table {{ width: 100%; border-collapse: collapse; margin: 10pt 0 13pt; font-size: 9.5pt; }}
thead {{ display: table-header-group; }}
tr {{ break-inside: avoid; }}
th, td {{ border: 1px solid #b9bec5; padding: 5pt 6pt; text-align: left; vertical-align: top; }}
th {{ background: #eef0f2; font-weight: 700; }}
img, svg {{ max-width: 100%; height: auto; break-inside: avoid; }}
figure {{ margin: 12pt auto; text-align: center; break-inside: avoid; }}
figcaption {{ margin-top: 5pt; color: #4b5563; font-size: 9pt; }}
a {{ color: #185abc; text-decoration: none; }}
hr {{ border: 0; border-top: 1px solid #c9cdd2; margin: 16pt 0; }}
.math.display {{ display: block; margin: 10pt 0; overflow-wrap: anywhere; }}
@media print {{ html, body {{ print-color-adjust: exact; -webkit-print-color-adjust: exact; }} }}
""".strip() + "\n"


def _run_command(command: list[str], *, cwd: Path, timeout: int) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        command,
        cwd=str(cwd),
        capture_output=True,
        text=True,
        timeout=timeout,
    )


def _pdf_fonts(pdf_path: Path) -> list[str]:
    executable = shutil.which("pdffonts")
    if not executable:
        return []
    proc = subprocess.run([executable, str(pdf_path)], capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        return []
    rows = proc.stdout.splitlines()[2:]
    return sorted({row.split()[0] for row in rows if row.split()})


def _pdf_page_count(pdf_path: Path) -> int:
    executable = shutil.which("pdfinfo")
    if not executable:
        return 0
    proc = subprocess.run([executable, str(pdf_path)], capture_output=True, text=True, timeout=20)
    if proc.returncode != 0:
        return 0
    match = re.search(r"^Pages:\s+(\d+)\s*$", proc.stdout, re.MULTILINE)
    return int(match.group(1)) if match else 0


def render_markdown_pdf(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[writing/compile] Render existing Markdown directly to PDF without converting the source into LaTeX."""
    tool_name = "render_markdown_pdf"
    try:
        params = RenderMarkdownPdfInput(**payload)
        source = resolve_workspace_path(params.source_path, must_exist=True)
        if source.suffix.lower() not in {".md", ".markdown"}:
            raise ValueError("source_path must point to a .md or .markdown file")
        output = (
            resolve_workspace_path(params.output_path)
            if params.output_path
            else source.with_suffix(".pdf")
        )
        if output.suffix.lower() != ".pdf":
            raise ValueError("output_path must end in .pdf")
        output.parent.mkdir(parents=True, exist_ok=True)

        pandoc = _resolve_executable("CATMASTER_PANDOC_BIN", ("pandoc",))
        chrome = _resolve_executable(
            "CATMASTER_CHROME_BIN",
            ("google-chrome", "google-chrome-stable", "chromium", "chromium-browser"),
        )
        selected_font, resolved_family, font_file, used_fallback = _resolve_font(params.font_family)

        with tempfile.TemporaryDirectory(prefix="catmaster-markdown-pdf-") as temp_name:
            temp_dir = Path(temp_name)
            css_path = temp_dir / "report.css"
            html_path = temp_dir / "document.html"
            profile_dir = temp_dir / "chrome-profile"
            css_path.write_text(
                _stylesheet(font_family=selected_font, page_size=params.page_size),
                encoding="utf-8",
            )
            resource_path = os.pathsep.join((str(source.parent), str(workspace_root())))
            title = params.document_title.strip() or source.stem
            pandoc_command = [
                str(pandoc),
                source.name,
                "--from=markdown+tex_math_dollars+pipe_tables+task_lists+strikeout+fenced_code_blocks-raw_html",
                "--to=html5",
                "--standalone",
                "--embed-resources",
                "--mathml",
                f"--resource-path={resource_path}",
                f"--metadata=pagetitle:{title}",
                "--css",
                str(css_path),
                "--output",
                str(html_path),
            ]
            pandoc_result = _run_command(pandoc_command, cwd=source.parent, timeout=120)
            if pandoc_result.returncode != 0 or not html_path.is_file():
                raise ValueError(
                    "Pandoc HTML conversion failed: "
                    + (pandoc_result.stderr.strip() or pandoc_result.stdout.strip())[-2000:]
                )

            if output.exists():
                output.unlink()
            chrome_command = [
                str(chrome),
                "--headless=new",
                "--disable-gpu",
                "--disable-dev-shm-usage",
                "--disable-extensions",
                "--disable-javascript",
                "--no-pdf-header-footer",
                "--run-all-compositor-stages-before-draw",
                f"--user-data-dir={profile_dir}",
                f"--print-to-pdf={output}",
                html_path.as_uri(),
            ]
            chrome_result = _run_command(chrome_command, cwd=source.parent, timeout=120)
            if chrome_result.returncode != 0 or not output.is_file():
                raise ValueError(
                    "Chrome PDF rendering failed: "
                    + (chrome_result.stderr.strip() or chrome_result.stdout.strip())[-2000:]
                )

        if output.stat().st_size < 5 or output.read_bytes()[:5] != b"%PDF-":
            raise ValueError("renderer did not produce a valid PDF file")
        embedded_fonts = _pdf_fonts(output)
        page_count = _pdf_page_count(output)
        content = (
            f"Rendered Markdown PDF: {workspace_relpath(output)}\n"
            f"Source preserved: {workspace_relpath(source)}\n"
            f"Font: {resolved_family}{' (fallback)' if used_fallback else ''}\n"
            f"Pages: {page_count or '(unknown)'}"
        )
        return content, {
            "tool_name": tool_name,
            "data": {
                "source_path": workspace_relpath(source),
                "pdf_path": workspace_relpath(output),
                "page_size": params.page_size,
                "page_count": page_count,
                "font_requested": params.font_family,
                "font_selected": selected_font,
                "font_resolved_family": resolved_family,
                "font_file": font_file,
                "font_fallback_used": used_fallback,
                "embedded_pdf_fonts": embedded_fonts,
                "pandoc_path": str(pandoc),
                "chrome_path": str(chrome),
                "pdf_bytes": output.stat().st_size,
                "source_format": "markdown",
                "intermediate_format": "embedded-html5-mathml",
            },
        }
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
                    "output_path": payload.get("output_path") or "",
                },
            },
            error_code="render_markdown_pdf_failed",
        ) from exc


__all__ = ["RenderMarkdownPdfInput", "render_markdown_pdf"]
