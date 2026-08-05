---
name: publication-launch-writing
description: Use this skill when planning, drafting, restructuring, revising, or compressing a scientific paper, or when organizing its experiments and displays around the strongest publishable advantage instead of a project chronology, defensive audit, or exhaustive result inventory.
license: project-local
---

# publication-launch-writing

## Overview

Build the paper as an academic launch: select the strongest evidence-supported value, make it the organizing thesis, and use every section and display to prove that thesis.

## Quick Start

1. Write one sentence naming the paper's strongest publishable advantage and the condition in which it matters.
2. Map only the claims and evidence needed to establish that advantage.
3. Rebuild the title, abstract, introduction, results order, displays, and conclusion around that map.
4. Remove project chronology, defensive prose, irrelevant comparisons, and non-scientific implementation detail before polishing sentences.

## Allowed tools

No additional tool surface is required. Use the current writing lane's file, document, citation, compilation, and review capabilities only when they serve the bounded writing task.

## Workflow

### 1. Select the launch thesis

Identify what is genuinely leading, distinctive, or hard to replace in the available evidence. Candidate advantages include a new capability, question, mechanism, viewpoint, applicability regime, lower cost, higher efficiency, better scalability, or a meaningful tradeoff. State the winning condition precisely. Do not choose a thesis merely because it matches the project's original plan or consumed the most work.

Define the paper as a claim-evidence argument rather than a record of activity:

- important unresolved problem;
- decisive gap in the appropriate existing approaches;
- distinctive solution or scientific insight;
- strongest result and its meaning.

### 2. Reconstruct the story around the strongest evidence

Retain the final scientific logic. Remove attempt order, failed branches, abandoned implementations, and "first we did...then we tried..." narration unless a sequence is itself the scientific result.

Allow full reconstruction when the evidence no longer supports the original framing. Change the problem definition, contribution order, title, abstract, introduction, result sequence, display hierarchy, and conclusion as needed. The story follows the strongest evidence rather than the initial hypothesis or project plan.

Make supported advantages explicit. For each core result, explain the condition in which it is strongest, why the advantage appears, which scientific or practical problem it resolves, and why the capability matters relative to the right prior art. Never rely on a table alone to announce the contribution.

### 3. Control comparisons without defensive writing

Do not make a non-advantageous dimension the paper's central contest. Choose the task definition, evaluation dimension, constraint, target scenario, or comparison scope that represents the actual value of the work. Seek claim-evidence alignment, not universal dominance across every dataset and metric.

When material appears unfavorable, decide in this order:

1. remove it when it does not bear on the core claim;
2. narrow the claim to what the evidence establishes;
3. use the evaluation dimension that reflects the intended value;
4. explain a genuine difference in objective or a meaningful tradeoff;
5. recenter the experiment and display hierarchy;
6. redefine the paper's story;
7. qualify the result only when the condition cannot be avoided and materially changes the core conclusion.

Do not use phrases that volunteer a verdict such as "the method is weaker," "still substantially lags," "performance deteriorates," "only achieves," or "has serious limitations." If a qualification is scientifically necessary, state the exact condition once in neutral language and calibrate the claim around it.

### 4. Make experiments and displays perform argumentative work

Assign every experiment, figure, and table one job: demonstrate the core method, isolate the mechanism behind the advantage, prove value in the target setting, or exclude a plausible alternative explanation. Delete, weaken, relocate to Supporting Information, or redesign material that does not strengthen the main line.

Keep claim-critical evidence in the main text and make it visually dominant. Supporting content can carry extended methods, robustness checks, exhaustive tables, secondary comparisons, and reproducibility material that readers may need but that should not compete with the launch thesis.

### 5. Open and close on the memory point

The abstract and introduction should establish the unresolved problem, gap, distinctive solution, most important result, and consequence quickly. Do not spend the opening on implementation chronology, generic background, or limitations before the contribution is established.

The conclusion should reinforce what the paper solved, introduced, demonstrated, and why it matters. Do not reopen the paper as a self-assessment or add a new list of weaknesses in the final paragraph.

## Method-critical defaults

- The launch framing never permits fabrication, selective alteration of data, concealment of evidence needed to assess the stated claim, or omission of a venue-required disclosure.
- Claim strength follows the evidence. Narrow or reposition a claim instead of defending an unsupported broad one.
- Mention a limitation only when it materially conditions interpretation of the core claim. State the condition precisely without generalizing it into a verdict on the whole work.
- Keep hardware, accelerator, launcher, scheduler, software-build, platform, and performance trivia out of the narrative. Put genuinely required reproducibility detail in Methods or Supporting Information; retain an operational detail in the scientific story only when it is causally result-changing.
- Use an authorial scientific voice. Do not mention agents, prompts, workspaces, runs, tools, or the process used to assemble the draft.

## Output Contract

Return manuscript-ready prose or the requested revised artifact, not a project report. The final structure must expose one clear launch thesis, a bounded set of supported claims, an evidence order that proves them, and a conclusion that reinforces the same scientific memory point. Report only unresolved conditions that change that argument or require an author decision.

## References

The runtime `Academic-launch writing policy` is the always-on authority for WritingSpecialist, writing_worker_agent, and writing_polisher_agent. This skill supplies the detailed restructuring workflow without replacing that system contract.
