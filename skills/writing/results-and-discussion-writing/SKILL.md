---
name: results-and-discussion-writing
description: Write Results and Discussion sections that separate observations from interpretation and preserve uncertainty.
metadata:
  catmaster-roles: "section_writer write_reviewer"
  catmaster-lanes: "writing"
  catmaster-tags: "writing results discussion"
---

# results-and-discussion-writing

## Overview
Organize numerical or computational findings into a Results/Discussion section that reads like a paper section, not a run report.

## Quick Start
Use when the section must explain computed structures, trends, figures, or comparisons without overclaiming.

## Allowed tools
- `read_file`
- `execute`

## Workflow
1. Decide whether the subsection belongs in Results, Discussion, or a deliberate hybrid. Do not mix them by accident.
2. In Results, state the observation before the interpretation and anchor it to a figure, table, or artifact.
3. In Discussion, answer what the observation means, how it compares with prior work, and where uncertainty remains.
4. Separate robust trends from tentative mechanisms or background explanations.
5. End the local argument with remaining uncertainties or scope boundaries when appropriate, without turning every paragraph into defensive hedging.

### Results paragraph pattern

Use this pattern by default:

1. question or setup
2. observation with quantity, comparison, or visual reference
3. local takeaway

Do not turn Results into a raw log of every output file. Select the observations that move the paper's argument.
Prefer 2-4 strong observations over exhaustive reporting.

### Discussion paragraph pattern

Use this pattern by default:

1. interpretation or answer
2. support from the current evidence
3. comparison with literature or competing explanations
4. limitation or scope boundary

### Tense and claim discipline

- Use past tense for what was observed or computed.
- Use present tense for stable interpretations or established literature only when warranted.
- Prefer "is consistent with", "suggests", or "may reflect" when the mechanism is not directly demonstrated.
- Do not let mechanistic speculation read like fact.
- Do not over-correct into lifeless prose; a normal scientific interpretive sentence is acceptable when it stays within evidentiary bounds.

## Output Contract
Return section text that clearly distinguishes observations, interpretation, and limitations while still reading as continuous manuscript prose.

## References
- Avoid collapsing unsupported mechanistic speculation into factual statements.
- Read [`../_references/section-patterns-and-story.md`](../_references/section-patterns-and-story.md) for results/discussion structural patterns.
- Read [`../_references/style-and-revision-checks.md`](../_references/style-and-revision-checks.md) for numerical consistency and anti-clutter revision passes.
