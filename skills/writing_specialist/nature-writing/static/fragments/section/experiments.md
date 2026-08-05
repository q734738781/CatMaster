# Section: Experiments / Results (writing)

## Default evidence ladder

`core method validation -> strongest result -> claim-relevant comparison -> ablation or mechanism evidence -> target application or generalization`

Each subsection has a claim-first opening, then data support.

## Drafting rules

- Stay mainly in past tense.
- Report what was observed, under what conditions, with what quantitative support.
- Use statistics correctly and sparingly. Every test needs a stated hypothesis.
- Use supplementary data sparingly. If a result belongs in the main text, do not hide it in supplements.
- **Each major claim needs the evidence appropriate to that claim.** Select comparisons, ablations, mechanism tests, or target-setting demonstrations because they prove the launch thesis, not because a generic checklist asks for every experiment type.

## Results syntax (vs Discussion)

Results sentences usually report:

- `was detected` / `increased` / `showed` / `enabled` / `achieved`

Do not drift into Discussion syntax (`may reflect`, `suggests`, `is likely due to`) unless the transition is intentional.

## Common failure modes when drafting

- Mixing observation and interpretation in the same paragraph.
- Citing supplementary data when the result should be in the main text.
- Vague comparisons (`higher than control`) without effect size, sample size, or test.
- Per-paragraph claims without per-paragraph evidence.

## Deeper reference

For ML/conference-style experiment sections — baselines, ablations, metrics, tables, figures — open `references/experiments.md`.
