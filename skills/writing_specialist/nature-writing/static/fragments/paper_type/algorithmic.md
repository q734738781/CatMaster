# Paper type: algorithmic or device

A procedure, tool, or system is proposed; the paper must show it performs reliably and advantageously.

## Argument chain

`task definition + target value -> what the system is -> why it works -> evaluation of the target advantage -> ablations isolating the contribution -> meaningful cost or applicability tradeoff -> implication`

## Drafting rules

- Separate "what the system is" from "why it works" from "how well it works." Do not braid them. A common failure is mixing design rationale with evaluation results in the same paragraph.
- Every performance claim must specify dataset, metric, baseline, and conditions. Bare numbers do not survive review.
- Avoid marketing verbs (`leverages`, `enables`, `empowers`) unless they carry concrete information.
- Do not create a generic failure-mode or limitations section. Discuss a constraint only when it materially conditions the stated target advantage; otherwise keep the Discussion centered on why the method works and where that capability matters.

## Module / pipeline writing

For module-level writing (each component of a pipeline), open `references/method.md` for:

- the three-element pattern (motivation, mechanism, evidence)
- module-motivation templates
- overview-template for the pipeline figure caption + first paragraph
