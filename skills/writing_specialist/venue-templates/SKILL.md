---
name: venue-templates
description: Select and apply a venue-specific document template or the bundled LaTeX report style when that formatting is needed.
metadata:
    skill-author: K-Dense Inc.
---

# Venue templates

Use the user's target venue or requested report style. A template supplies
formatting, not a reason to change scientific claims, add figures or run another
research workflow. Existing user templates and explicit requirements take priority.

## Select one relevant source

| Document | Bundled guidance |
|---|---|
| Journal article | [Journal formatting](references/journals_formatting.md) |
| Conference paper | [Conference formatting](references/conferences_formatting.md) |
| Scientific poster | [Poster guidelines](references/posters_guidelines.md) |
| Grant proposal | [Grant requirements](references/grants_requirements.md) |
| Explicit professional report/white paper | [Report formatting](references/professional_report_formatting.md) |
| Venue-specific prose | [Writing-style index](references/venue_writing_styles.md), only when that venue/style matters |

Search `assets/` for the matching template. Optional helpers are
`scripts/query_template.py`, `scripts/customize_template.py` and
`scripts/validate_format.py`; consult `--help` for the operation needed rather
than reading every helper or running all of them.

Read the chosen guidance and template, adapt author/title/content fields, compile
and inspect the rendered output. Check current official submission instructions
when compliance with a named venue is required; bundled limits may be outdated.
Apply only requirements relevant to that venue and deliverable. Preserve existing
scientific content, citations and editable sources. Reuse already completed checks
and inspect the affected pages after corrections.

Return the requested source/rendered artifact and any unresolved venue requirement.
General manuscript argument design remains in `publication-launch-writing`;
progress-report and presentation content follows `scientific-communication`.
