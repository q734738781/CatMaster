# Six-Dimension Scoring System

Use this LATS candidate-selection score in the coarse-filtering stage. It ranks
which papers are most useful to inspect next; it is not a grade of scientific
truth, evidence strength, or confidence in a claim. Keep the six component
scores, total, rationale, and inspected access depth together so the composite
remains auditable.

## Dimensions

| # | Dimension | Weight | What It Measures |
|---|-----------|:------:|------------------|
| 1 | Topic Match | 35 | How closely the paper aligns with core research questions |
| 2 | Novelty / Contribution | 20 | What is new or newly useful relative to the current candidate set and known work |
| 3 | Method Quality | 15 | Quality and applicability of the experimental, computational, or analytical method |
| 4 | Source / Author Signal | 10 | Relevant venue, author, institution, collaboration, or tracked-source signal; never a proxy for truth |
| 5 | Applied/Engineering Value | 10 | Practical utility: protocols, datasets, benchmarks, engineering insights |
| 6 | Archival Value | 10 | Long-term reference value: review potential, foundational status, teaching utility |

## Scoring Rules

1. Cap each dimension at its weight; do not overshoot.
2. Recalculate the total from all six components instead of trusting model arithmetic.
3. Return the component vector with the total and a short rationale; a total-only score is not auditable.
4. Use Topic Match as the gate: reject papers below 10/35 regardless of other scores.
5. Record access depth separately as metadata, abstract, full text, or SI/source data. If the inspected material cannot support a novelty or method judgment, mark the score `provisional` and say which components remain under-informed rather than treating missing information as poor evidence.
6. Use this score only for triage, reading order, or archival priority. Assess scientific claims separately through evidence attributes such as modality, claim relationship, condition fit, and provenance.

## Subagent Prompt Template

```
Score each candidate paper for selection priority on six dimensions (0-100 total):

1. Topic Match (max 35): [insert your research questions]
2. Novelty / Contribution (max 20)
3. Method Quality (max 15)
4. Source / Author Signal (max 10): [insert relevant tracked sources, authors, or institutions]
5. Practical Value (max 10)
6. Archive Value (max 10)

For each paper, output:
{
  "title": "...",
  "scores": {"topic": X, "novelty": X, "method": X, "source_signal": X, "practical": X, "archive": X},
  "total": X,
  "access_depth": "metadata | abstract | full_text | supporting_information",
  "provisional": true | false,
  "rationale": "one sentence"
}

Return the configured number of selected papers by total score, sorted descending.
Do not describe the total as evidence strength or confidence.
```

## Common Pitfalls

- Models may inflate Topic Match for well-known papers that are not actually relevant; verify the rationale against the configured research question.
- Do not infer novelty or method quality from a title, venue, citation count, or author reputation. Mark the score provisional when access is too shallow.
- Source / Author Signal is a routing and attention signal, not scientific authority and not evidence strength.
- Do not let the composite replace the claim-level evidence attributes used during synthesis.

## Calibration

After 2-3 pipeline runs, review the score distribution with the user:
- If top papers consistently score 90+, the rubric is too loose
- If no paper breaks 60, keywords may be too narrow or the field is sparse
- Adjust weights based on user feedback (e.g., if applied value matters more than journal prestige for their work)
