---
name: nature-literature-pipeline
description: Multi-source literature discovery, reasoned selection, selective reading, digest delivery, and optional archival without paper-quality scoring.
version: 1.1.0
license: MIT
---
# Nature Literature Pipeline

Use this workflow for recurring or one-off literature discovery that needs a
traceable candidate pool, selective source reading, and a compact delivery.

## Workflow

1. Search the configured scholarly sources and record the query and source.
2. Deduplicate by DOI, arXiv ID, OpenAlex ID, then normalized title.
3. Assign each candidate one status: `selected`, `deferred`, or `excluded`.
4. Give a short task-specific reason grounded in topic coverage, method or
   dataset relevance, access depth, duplication, date need, and the user's
   stated scope. Do not calculate a total or component paper score.
5. Read only selected sources deeply enough to support the intended digest.
6. Deliver the digest with stable identifiers and access depth. Archive notes
   only when requested or configured.

`selected` means read or report now. `deferred` means potentially useful but not
needed for this pass, with the missing condition stated. `excluded` means out of
scope, duplicate, inaccessible for a full-text-dependent claim, or otherwise
unsuitable for this task. Venue, citation count, and author identity are search
or user-preference signals, never evidence quality.

## Safeguards

- Preserve every candidate's status and reason so selection is auditable.
- Record access depth separately as metadata, abstract, full text, or
  supplementary/source data.
- Do not infer methods, novelty, or findings from metadata.
- Use exact DOI/arXiv/OpenAlex identifiers for deduplication and stable links.
- If fewer papers are worth selecting, deliver fewer.
- Do not modify a curated wiki or knowledge base without authorization.

## References

- `references/selection-policy.md`
- `references/push-format.md`
- `references/note-template.md`
- `references/gap-analysis.md`
- `references/cron-setup.md`
- `references/review-compilation-workflow.md`
