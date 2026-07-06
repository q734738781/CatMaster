# Workflow and output format

## Accepted inputs

The skill may receive: pasted editorial decision or revision-invitation email; editor decision letter; reviewer comments; previous response draft; manuscript change notes; tracked-change summary; line or page numbers; figure, table, and supplement list; author notes in Chinese or English; journal name and article type; manuscript title; author list; manuscript ID; original manuscript text or LaTeX source; requested cover-letter or LaTeX output format.

If reviewer boundaries or comment segmentation are ambiguous, flag the ambiguity instead of inventing reviewer structure.

## Workflow

1. Identify task mode and input readiness: `draft`, `audit`, `revise`, `triage-only`, `cover-letter`, `revision-package`, `latex-template`, or `appeal-like`.
2. If the input is a pasted journal email, automatically extract manuscript title, manuscript ID, journal, decision type, editor instructions, reviewer-report boundaries, required revision files, deadline, and portal-specific constraints before drafting.
3. Identify decision type: minor revision, major revision, revise-and-resubmit, transfer after review, or unclear.
4. Extract editor instructions first and assign IDs such as `E.1`, then split reviewer comments with IDs such as `R1.1`, `R1.2`, and `R2.1`.
5. Classify each item by category, severity, action label, missing input, readiness state, and risk.
6. Create a response strategy summary before drafting prose.
7. Draft responses using preserved reviewer comments unless the mode is `triage-only`, `cover-letter`, or `appeal-like`.
8. For `cover-letter` or `revision-package`, draft a concise editor-facing cover letter that summarizes revision scope and points to the point-by-point response without duplicating it.
9. Map each claimed change to manuscript location, figure, table, supplement, citation, or explicit placeholder.
10. If editing manuscript text, create or instruct use of a backed-up manuscript copy and mark changed text in red. For LaTeX, use `\revised{...}` from `templates/revised-manuscript-redline.tex`.
11. If pasting revised manuscript text after a response, format it in italics. For LaTeX response files, use `\RevisedExcerpt{...}` from `templates/response-to-reviewers.tex`.
12. In LaTeX or print-oriented response letters, start each new reviewer response on a new page. Use `\ReviewerSection{1}`, `\ReviewerSection{2}`, etc. from `templates/response-to-reviewers.tex`.
13. If the user requests LaTeX, use `templates/cover-letter.tex`, `templates/response-to-reviewers.tex`, and/or `templates/revised-manuscript-redline.tex`; preserve visible placeholders for missing facts.
14. Flag missing author input rather than fabricating details.
15. Run QA for completeness, traceability, factuality, tone, unresolved risk, red-marked changes, italic revised excerpts, reviewer page breaks, and LaTeX placeholder visibility.
16. Return the response package with package readiness: `ready_to_submit`, `draft_with_placeholders`, `needs_author_input`, or `blocked`.

## Output format

Unless the user asks for another format, return:

```text
Response strategy summary
- Decision type:
- Task mode:
- Overall posture:
- Major risks:
- Parsed email metadata:
- Suggested ordering:

Comment-response tracker
| ID | Reviewer concern | Type | Severity | Proposed action | Missing author input |
|---|---|---|---|---|---|

Draft point-by-point response letter
[editor-readable English response]

Draft revision cover letter
[only when requested or when returning a revision package]

Marked manuscript changes
- [red-marked changed passages or path to marked backup copy]

LaTeX files
- cover letter: [path or template-filled text when requested]
- response to reviewers: [path or template-filled text when requested]
- red-marked manuscript: [path or template-filled text when requested]

Manuscript change checklist
- [specific manuscript changes or placeholders]

Missing information / risk flags
- [specific unresolved items or "None"]

中文核对
- [when the user writes in Chinese; otherwise omit unless useful]
```
