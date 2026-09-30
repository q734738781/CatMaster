---
name: achemso-latex-manuscript
description: Draft or revise ACS LaTeX manuscripts using an existing achemso project or the bundled template.
metadata:
  catmaster-roles: "write_director section_writer write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing latex achemso acs"
---

# ACS LaTeX manuscripts

For an existing manuscript, start from its actual root, section files and
bibliography. For a new ACS draft, inspect `assets/achemso-demo.tex`; consult
`assets/achemso-demo.bib` when its citation conventions are needed. Mounted assets
are read-only references: create or edit the manuscript in the workspace.
Preserve the applicable documentclass, macros and packages unless the requested
change requires adapting them. Template content is illustrative, not evidence.

A section task may write a `.tex` fragment and explain its intended insertion
point. A full-manuscript or integration task assembles the root, sections,
figures and bibliography into the actual document. There is no automatic host
assembly of returned section text. Return workspace paths and any information
needed to integrate the requested artifact in ordinary prose.

Use separate `.tex` and `.bib` files for a cited TeX manuscript. Verify citation
identity; do not invent entries or hide unresolved references in explanatory
BibTeX notes. Resolve material gaps or report them as unfinished citation work.

Insert figures near the paragraphs that discuss them. Use conservative float
placement such as `[htbp]`; use `\FloatBarrier` only when the project's packages
support it and a real placement problem warrants it. Reference actual figures
and tables, preserving values, units and attribution.

For an assembled TeX manuscript, run `compile_text` and repair relevant errors
from its diagnostics. Inspect the rendered PDF for placement, legibility and
missing material; a successful compile does not establish presentation quality.
A section-only task need not fabricate a new wrapper simply to compile a
fragment. Integrate and compile when that is the assigned scope. Retain the PDF
when requested or required downstream; otherwise use a temporary review render.

For specialized sentence revision, read
[style and revision checks](../_references/style-and-revision-checks.md).
For submission preparation, read
[submission and editorial readiness](../_references/submission-and-editorial-readiness.md).

## Tool scope and examples

`compile_text` is a local compiler wrapper. Select engine and bibliography_tool when the document requires them, and output_dir for a separate build tree. Example: `compile_text(source_path="paper/main.tex", engine="xelatex", bibliography_tool="biber", output_dir="paper/build")`. Its diagnostics_path contains complete diagnostics and compiler output; static hints do not override a successful compiler/PDF result.
