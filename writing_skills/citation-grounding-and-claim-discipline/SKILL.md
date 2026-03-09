---
name: citation-grounding-and-claim-discipline
description: Review scientific prose for unsupported claims, citation gaps, and overstatement.
compatibility: local
metadata:
  catmaster-roles: "write_reviewer section_writer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing citations review"
  catmaster-suggested-tools: "review_research_context read_research_pack run_literature_research"
---

# citation-grounding-and-claim-discipline

## Overview
Review manuscript text for claim-evidence alignment, missing citations, and places where the prose materially outruns the evidence.

## Quick Start
Use after section drafting or when a section depends heavily on literature-supported context.

## Suggested tools
- `review_research_context`
- `read_research_pack`
- `run_literature_research`

## Workflow
1. Highlight major claims that need direct support.
2. Classify each claim: direct observation, literature comparison, field background, mechanistic explanation, or recommendation.
3. Check whether each claim has the right support type: citation, artifact, figure/table, or explicit uncertainty language.
4. Mark overbroad language and recommend narrower wording when it would materially improve accuracy.
5. Preserve unresolved gaps explicitly in the review output, but do not block a draft over minor stylistic or borderline wording issues.

### What normally requires support

- numerical findings
- comparisons with prior work
- claims of novelty or importance
- mechanistic explanations
- statements about generality, transferability, or practical impact

Common knowledge may not need a citation, but domain-specific facts usually do.

### Overstatement control

Downgrade wording when the evidence is limited:

- from "demonstrates" to "supports" or "is consistent with"
- from "proves" to "shows"
- from "general" to "within the systems studied"
- from "causes" to "may contribute to" unless causality is directly established

Do this for sentences that would actually mislead a scientific reader, not for every line that could be phrased more cautiously.

### Integrity checks

- numbers in prose match figures, tables, and cited sources
- the cited paper actually supports the sentence it is attached to
- no paragraph is too close to a source's wording or sentence structure
- self-plagiarism risk is flagged when prose looks recycled without adaptation

## Approval standard

- Approve when the section is scientifically faithful, readable, and only has minor wording or citation improvements left.
- Request revision when a central claim lacks support, a major quantitative statement is unsupported, or the prose would materially mislead a domain reader.

## Output Contract
Return revision notes, unsupported claims, and missing citation findings, focusing on substantive rather than cosmetic issues.

## References
- A polished section with weak evidence is still a failed draft, but a good manuscript section does not need forensic-level line-by-line policing.
- Read [`../_references/style-and-revision-checks.md`](../_references/style-and-revision-checks.md) for the final consistency and plagiarism-oriented passes.
- Read [`../_references/submission-and-editorial-readiness.md`](../_references/submission-and-editorial-readiness.md) before approving a journal-facing manuscript.
