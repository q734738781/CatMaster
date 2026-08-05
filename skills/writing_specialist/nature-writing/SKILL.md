---
name: nature-writing
description: Draft, restructure, or plan evidence-grounded scientific manuscripts for Nature-family and other journals from author-provided claims, results, figures, notes, or Chinese drafts. Use for titles, abstracts, introductions, related work, methods, results or experiments, discussions, conclusions, full-paper arguments, generic IMRAD structure, and study-design reporting standards such as CONSORT, STROBE, or PRISMA. Also trigger on general academic-writing requests such as writing or rebuilding a paper, SCI manuscript, or section rather than only polishing finished prose.
metadata:
  version: "1.1.0"
  author: Community contribution, refactored into static/dynamic layers
---

# Scientific Manuscript Writing — Router

This skill is split into two layers:

- A **static layer** under `static/` that holds reusable content fragments (core stance and workflow, paper-type playbooks, per-section drafting guidance, language-specific rules, and journal style).
- A **dynamic layer** (this file plus `manifest.yaml`) that detects the request's axes and loads only the fragments needed for the current job.

This is the single general scientific-writing skill in the Writing lane. Nature
is a supported journal style, not a requirement for triggering the skill.

Do not try to apply the drafting logic from memory or from this router. Always load fragments from disk as described below.

## Routing protocol

Follow these five steps every time the skill is invoked.

### 1. Load the manifest and the core layer

Read [manifest.yaml](manifest.yaml). It declares the axes (`paper_type`, `section`, `language`, `journal`), the allowed values, and the file paths each value maps to.

Also read every file listed under `always_load`. These hold the default stance, writing workflow, and output format that apply to every drafting job.

### 2. Detect the axis values for this request

For each axis in the manifest, decide the value using the manifest's `detect:` hint and the user's input:

- `paper_type` — research / methods / hypothesis / algorithmic / review. Default: research.
- `section` — abstract / intro / related-work / method / experiments / discussion / conclusion / title. May be multiple. Ask the user if it is ambiguous and matters for the draft.
- `language` — en or zh-to-en. Detect from the user's notes themselves.
- `journal` — nature / nat-comms / generic. Default: generic. If the user names a Nature subjournal, treat it as `nature`.

Also identify whether the study design carries a formal reporting standard. This
is an on-demand reference decision, not another mandatory axis.

State the detected axis values in one short line to the user before drafting, so they can correct you cheaply.

### 3. Load the matching fragments

For each axis value, Read the file mapped in the manifest. Skip the `section` axis only when the user has explicitly asked for a free-floating argument paragraph with no section context.

Do **not** read every fragment in `static/`. Load only what step 2 selected.

### 4. Draft using the loaded material

Apply the loaded fragments in this priority order:

1. Runtime academic-launch policy plus core stance (`core/stance.md`) — select the strongest evidence-supported publishable value before drafting.
2. Paper-type playbook — argument chain, drafting order.
3. Section-specific drafting rules and structure.
4. Journal-specific framing and constraints.
5. Language-specific sentence and paragraph rules (apply last).

Run the workflow in `core/workflow.md` end-to-end. Do not skip the launch thesis and evidence map just because the user asked for prose immediately.

If decisive evidence is missing, narrow the claim. Use a visible placeholder only when the user requested a scaffold; do not append a defensive assumptions or limitations inventory to finished manuscript prose.

### 5. Reach for references only when needed

The files under `references/` are deep references and the example library, not defaults. Open them on demand per the `references.on_demand` table in the manifest. Typical triggers:

- The user asks for a concrete example or template → `references/examples/index.md`.
- A section's draft has structural problems that the section fragment alone does not explain → the matching `references/<section>.md`.
- The user needs a broad-audience `Nature` abstract opening or asks about a `summary paragraph` → `references/nature-summary-paragraph.md`.
- The user asks "does this paragraph flow?" → `references/paragraph-flow.md`.
- The user asks for a manuscript self-review → use `references/paper-review.md` to strengthen the launch thesis and claim-evidence fit, not to manufacture reviewer attacks.
- The paper is a randomized trial, observational study, systematic review, diagnostic or prediction study, protocol, case report, qualitative study, animal study, quality-improvement study, or economic evaluation → `references/reporting-standards.md`.

Do not duplicate specialist capabilities inside this router. Use the citation
skills for source discovery, verification, BibTeX, and venue citation style; use
`plot_worker` or the figure skills for quantitative plots and scientific
schematics; use `venue-templates` for document templates, including explicitly
requested professional reports. This skill owns the manuscript argument,
section architecture, and integration of those outputs.

## Why this split

- The static layer is versioned and reviewable. Adding a new journal style, paper type, or section is one new file plus one manifest line.
- The dynamic layer keeps each invocation cheap: only the fragments relevant to this draft enter context, instead of the full multi-thousand-line reference set.
- The router itself is short on purpose. Update fragments, not this file, when adding scope.
- This structure mirrors `nature-polishing`; genuinely shared content lives in the existing `_shared/` layer.
