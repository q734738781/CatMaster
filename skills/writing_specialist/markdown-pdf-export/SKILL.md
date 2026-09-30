---
name: markdown-pdf-export
description: Convert an existing Markdown document to PDF while preserving its authored content.
license: project-local
---

# Markdown PDF export

Use `render_markdown_pdf` with the existing Markdown source and requested output
path. The tool schema supplies page and font defaults, including Microsoft
YaHei for Chinese or mixed text and a Noto fallback. Other installed font families
can be requested explicitly.

Preserve the source, wording, headings, tables, citations, figures and formulas.
Conversion does not authorize rewriting the content or replacing it with TeX.
Use a TeX workflow only when explicitly requested or required by the venue.

Use returned diagnostics to address a failed render. Inspect the actual PDF when
layout matters, especially tables, equations and page breaks; tool success alone
does not establish legibility. Return the source and PDF paths, plus a material
unresolved issue or font fallback when it affects the requested typography.

## Tool scope and examples

Use css_path for document-specific CSS overrides and font_family for the requested installed font. Rendering writes a new temporary PDF and replaces an existing output only after success; overwrite=false refuses an existing target. Example: `render_markdown_pdf(source_path="report.md", output_path="release/report.pdf", css_path="styles/report.css")`. Inspect any reported font fallback when typography matters.
