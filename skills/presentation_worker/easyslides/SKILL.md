---
name: easyslides
description: Create, edit or reconstruct editable PPTX presentations with EasySlides, including academic talks, defenses and existing PowerPoint templates.
---

# Editable presentations

For scientific content creation or substantive revision, read
`scientific-communication` from the staged skills before selecting layouts. It
guides evidence selection and the relationship between slides and spoken content;
EasySlides supplies the authoring and rendering mechanics. Format-only edits and
faithful reconstruction preserve the approved content.

## Installed runtime

EasySlides is preinstalled with CatMaster. Its scripts, workflows, templates and
reference assets are available through `/.easyslides/`. In shell commands the same
directory is `$CATMASTER_EASYSLIDES_ROOT`; use ordinary `python` from the workspace
environment. These are two paths to the same library, not interchangeable path
names. Put projects and adapted assets in the workspace, leaving the shared
installation intact. Do not reinstall the upstream requirements during a deck job.

Start with `/.easyslides/workflows/routing.md` and read the relevant workflow and
script help. The command hub is:

```bash
python "$CATMASTER_EASYSLIDES_ROOT/scripts/easyslides.py" --help
```

CatMaster defaults to the native editable production scheme. Resolve routine
layout choices from the brief; ask only for missing information that changes the
requested result. Upstream installation steps, mandatory scheme-choice popups,
browser confirmation pages, developer release gates and personal wording defaults
are not CatMaster task requirements. Honor the user's actual wording and format.

## Authoring and editing

- For a new deck, inspect the installed `templates/layouts/` registry and a fitting
  template, or use the user's design. Keep each slide's content in editable SVG
  text/shapes and export through `scripts/svg_to_pptx.py --only native`. This selects
  the editable deck without the additional legacy SVG/PNG slide-image copy.
- For an existing PPTX, prefer native `template-fill` or `enhance` operations;
  preserve its theme, objects and notes. Use `distill` only when a reusable template
  is actually requested. For screenshot reconstruction, rebuild text and structure
  natively and retain complex photographic or illustrative regions as image assets
  unless the user requests full vector reconstruction.
- Keep titles, prose, tables and page layout as native PowerPoint objects. Use
  python-pptx native tables/charts when cell or chart-data editing is required.
  Matplotlib is appropriate for individual data plots, not for painting a slide.
  When creating or revising a data plot, apply the staged
  `publication-data-plotting` skill, including its Origin/NPG style and scientific
  fidelity guidance. Preserve an explicit user style or editable-chart requirement.
- Use `generate_figure` for illustrations or reference-image edits. It saves a
  workspace image; pass the desired model and image options when needed. Translate
  upstream ImageGen examples into this tool's `prompt`, `reference_images`,
  `output_path`, `model` and `image_options` fields. Image generation does not
  produce scientific observations or substitute for data-driven charts.

For a project with `svg_output/*.svg` and optional `notes/*.md`:

```bash
python "$CATMASTER_EASYSLIDES_ROOT/scripts/svg_to_pptx.py" writing/talk \
  --only native -o writing/talk.pptx
```

Keep slide sources and project-specific assets when needed for later edits;
the requested deck is the deliverable. Put disposable render previews under
workspace `tmp/`.

## Inspect the result

Use `scientific-communication` for authored or substantively revised slide text
and scripts, with phrasing examples only when needed. Inspect whether the core scientific evidence and
explanation made it onto the pages, not only into the notes or planning files.
Do not add a separate quality report or preset revision cycle unless requested.

Render the actual PPTX and inspect the slides for clipping, overlapping objects,
missing fonts and legibility. On Linux use LibreOffice with a separate writable
profile per rendering task; concurrent renders must not share the default profile:

```bash
mkdir -p tmp/talk-preview
soffice "-env:UserInstallation=file://$PWD/tmp/talk-preview/lo-profile" \
  --headless --convert-to pdf --outdir tmp/talk-preview writing/talk.pptx
pdftoppm -scale-to 1440 -png tmp/talk-preview/talk.pdf tmp/talk-preview/slide
```

For a deck-wide or repeated visual review, use `task` with `general-purpose`, when
that delegate is available, to inspect coherent page groups in separate contexts.
Choose groups around sections and comparisons, with adjacent pages where a
transition matters; preserve full review coverage without a fixed page quota.
Pass the actual render paths, relevant slide text and notes, audience, scientific
question, necessary source paths and specific acceptance criteria. Give each
check a read-only brief and a stopping condition. The authoring worker remains
the single deck editor. A checker with no delegation surface completes its own
assigned group directly.

Each checker uses `read_file` on its preview PNGs at a readable page size; a
montage alone may hide crowded legends or weak figure-text relationships. Return
the inspected pages, concrete defects with page/figure locations, their effect
on the scientific explanation, suggested corrections and unresolved questions.
Keep source and preview paths reachable; return findings rather than image
blocks or copied tool history. Save detailed findings in a workspace note when
needed and return its path with the actionable findings.

The authoring worker integrates these checks and owns the complete narrative,
scientific coverage and acceptance. Read the slide text and notes across the
whole sequence, resolve contradictions or uncovered pages, and open specific
images whenever a finding needs adjudication. Do not repeat a completed visual
pass by loading every group's images into the authoring context. Fix material
problems, re-render and inspect changed pages and affected transitions in bounded
checks, reusing accepted findings for unchanged content. Apply the reader-facing
guidance in `scientific-communication`. Source figures can be cropped, annotated
or laid out differently when their meaning and necessary labels remain intact;
unchanged source data need no new audit.

Inspect the PPTX with python-pptx or its slide XML to confirm that important text
remains text and the page is not a single picture. Return the PPTX and useful
editing-source paths, plus the temporary preview location so the rendered result
can be inspected without repeating production. State any material unresolved
limitation; a declaration of successful rendering is not content acceptance.
