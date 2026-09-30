---
name: publication-launch-writing
description: Develop or revise a scientific manuscript's contribution and argument from existing evidence. Use for paper writing, not general literature reports or progress notes.
license: project-local
---

# Manuscript argument

Identify the contribution the evidence supports and why it matters for the
paper's scientific question. It may be a mechanism, capability, result,
applicability regime or meaningful tradeoff. Choose the emphasis from evidence,
not the original project plan or the amount of work spent on each activity.

Explain the problem, relevant prior work, contribution and decisive evidence in
an order readers can follow. Titles, abstract, results and conclusion should
refer to the same scientific argument. State supported advantages clearly with
the conditions and comparisons that establish them; do not leave the reader to
infer the contribution from a table alone.

When evidence changes the framing, revise the claim or argument within the
user's editing scope. Narrowing a claim, explaining a tradeoff, changing the
comparison or reorganizing the manuscript are editorial options, not an ordered
recipe. Retain adverse findings and alternative explanations needed to assess
the claim. Do not hide results, change a metric merely to manufacture an
advantage, or omit a required disclosure. Scientific uncertainty can be stated
directly without a general verdict on the worth of the work.

Keep decisive evidence in the main argument; place extended methods, auxiliary
figures and exhaustive data in supporting content when useful. Explain each
display's finding near its use. For the manuscript-review path, the review target
is one PDF: include supporting text after the references in that review copy
when it is part of the requested review; supporting data may remain separate.
Follow explicit venue requirements for the delivered submission files.

Write as the author of the scientific work. Include research chronology or
computational details when they explain the science, not incidental drafting
history. Agents, prompts, tools and workflows are legitimate content when they
are study objects, relevant methods or required disclosures. Bibliographies
contain verified publication metadata, not workspace-processing notes.

Use supplied author samples to match voice without importing their claims.
Routine wording and structure choices belong to the writer. Ask only when an
unresolved user-controlled choice or missing decisive evidence prevents the
requested work. Language-only editing preserves scientific claims; an authorized
argument rewrite may change their organization and supported scope.

For publication-readiness work, inspect the actual manuscript and rendered form,
repair material scientific, citation and presentation defects, and stop when
the requested artifact is usable. Formal external peer review is a separate
requested workflow, not a prerequisite for ordinary drafting. Use the available
comment-only manuscript review when the task calls for publication-readiness
assessment or a concrete unresolved editorial issue warrants it.

Read [study-design reporting standards](references/reporting-standards.md) when
the study design or target venue requires a formal reporting guideline. Use the
applicable venue/template skill for format-specific work. These resources do not
expand the authorized experiments or impose a fixed revision count.

## Tool scope and examples

When available, `review_pdf_manuscript` reviews one local PDF in one model call and returns comments without editing the manuscript. Use `focus` and `context_text` for the requested editorial question; the tool does not receive other workspace context automatically.

For a requested review episode with `peer_review_request` available, `model_labels` selects configured reviewers and an empty list uses configured defaults. `review_request` sets scientific focus, venue and presentation requirements. Each review or failure is retained at `results_path`; retry only failed labels when needed. Do not add review rounds beyond the requested scope.
