---
name: markdown-pdf-export
description: Render an existing Markdown note, report, or summary directly to a stable PDF without rewriting it as LaTeX; use when the requested source is Markdown and the deliverable is PDF.
license: project-local
compatibility: local
allowed-tools: "render_markdown_pdf"
---
# markdown-pdf-export

## Overview

Use the deterministic Markdown-to-HTML-to-PDF path for format conversion while preserving the authored Markdown source.

## Quick Start

Read the requested Markdown file, confirm the output path, and call `render_markdown_pdf` once. Leave `font_family` at `Microsoft YaHei` for Chinese or mixed Chinese-English reports unless the user requests the portable Noto fallback. Return both source and PDF paths.

## Allowed tools

- `render_markdown_pdf`

## Workflow

### 1. Keep conversion separate from rewriting

If the user asks to turn an existing `.md` or `.markdown` file into PDF, preserve its wording, headings, tables, links, images, code blocks, and formulas. Do not create a `.tex` replacement and do not invoke `compile_text` unless the user explicitly asks for LaTeX or a journal template requires TeX.

### 2. Render through the registered path

Use `render_markdown_pdf` with the existing Markdown path. Leave `output_path` empty to create a sibling PDF with the same stem, or provide an explicit workspace-relative `.pdf` path when requested. The tool uses Pandoc embedded HTML5 with MathML, then headless Chrome PDF printing.

### 3. Verify the returned artifact

Check `pdf_path`, `page_count`, `font_resolved_family`, `font_fallback_used`, and `embedded_pdf_fonts`. A successful command is insufficient if no valid PDF artifact is returned. If Microsoft YaHei is unavailable, report the explicit Noto Sans CJK SC fallback rather than silently changing typography.

## Method-critical defaults

- Preserve the Markdown file as the source of truth; PDF generation is not a content-rewrite task.
- Default page size is A4.
- Default body font is Microsoft YaHei with Noto Sans CJK SC as the explicit fallback.
- Use the tool's fixed print CSS for headings, tables, code blocks, images, and page margins; do not generate ad hoc CSS unless the user asks for a different visual specification.
- Keep formulas in the Markdown source and render them as MathML; do not convert the entire document to TeX just to display equations.

## Output Contract

Return the unchanged Markdown source path, generated PDF path, page count, actual resolved font, whether fallback occurred, and any renderer error. Do not claim completion from an HTML intermediary alone.

## References

- Registered conversion tool: `render_markdown_pdf`
- TeX-only compilation remains owned by `compile_text` and the relevant manuscript/template skill.
