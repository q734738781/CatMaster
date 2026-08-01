# Changelog

This file records notable behavior changes from this point forward. It is not a
reconstruction of earlier development history. The manuals and technical
references describe the current system.

## Unreleased

### Agent Prompting

- Writing system prompts no longer prescribe claim counts or fixed review,
  polishing, and compilation pass counts. Conditional planning guidance now
  lives in writing skills, while runtime prompts retain qualitative completion
  conditions and hard safety or transaction limits.

### Literature Review

- Literature Review now keeps numerical candidate-pool guidance in its search
  skill instead of the specialist system prompt. Explicit user limits control
  discovery breadth, while broad reviews expand by coverage gaps and stop at
  saturation. Candidate discovery remains shallow until papers are selected for
  deeper evidence extraction.

### Research Graph

- Research planning now lets the proposer choose the number of scientifically
  distinct temporary branches instead of exposing fixed 12-Hypothesis and
  24-Experiment quotas. Temporary experiments may remain drafts until their
  execution plan and decision rule are known.

### Documentation

- Removed migration progress, completed implementation checklists, and
  future-removal notes from the DeepAgents reference documents.
- Established the repository rule that manuals describe current capabilities,
  configuration, limits, and verification. Notable behavior changes belong in
  this file.
