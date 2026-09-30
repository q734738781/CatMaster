---
name: orca-optfreq-thermochemistry
description: Use this skill to choose a task-appropriate molecular electronic-structure method and compose native ORCA single-point, optimization, frequency, thermochemistry, TDDFT, or NMR stages from an explicit electronic state.
allowed-tools: "orca_prepare remote_submission remote_submission_batch analyze_orca_results extract_optimized_molecules execute"
---

# ORCA opt/frequency and properties

## Overview

Choose the molecular model and electronic method for the requested property, then prepare and analyze one native ORCA stage without importing unrelated settings from examples.

## Quick Start

1. Preserve explicit user, project, or literature constraints; otherwise choose the method from the target system and property.
2. Read [ORCA electronic-method selection](references/orca_method_selection.md) before choosing an unprescribed method.
3. Add only the requested operation fragment from [ORCA task-keyword fragments](references/orca_native_examples.md), then prepare and submit the stage.
4. Accept the dedicated analyzer result unless a requested field, error, or result-changing warning requires deeper inspection.

## Allowed tools

- `orca_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_orca_results`
- `extract_optimized_molecules`
- `execute`

## Workflow

### 1. Define the scientific model

Select the structure or conformer set, environment, charge, multiplicity, and requested observable. Keep a Kohn-Sham orbital gap distinct from a fundamental or optical gap, and keep shielding distinct from a referenced NMR shift.

### 2. Select method and operation independently

Choose the electronic method, basis, dispersion, and solvation from the property and system. Then select only the requested operation: a single point has no task token, while `Opt`, `Freq`, `NumFreq`, `NMR`, TDDFT, scan, TS, and IRC controls are separate choices. A method recommendation does not authorize an optimization or frequency stage.

### 3. Prepare and execute

Compose complete ordered `simple_keywords` and native `input_blocks`, then call `orca_prepare` with explicit charge and multiplicity. Submit one prepared stage through `remote_submission` with task `orca_execute`; use `remote_submission_batch` for independent same-config stages. It selects first-level children by default; use relative `stage_paths` for a selected subset or nested stages.

Batch preparation takes one structure per source file and returns the source-to-stage mapping. Use the returned paths instead of deriving stage names from basenames, which may repeat across inputs.

### 4. Analyze and hand off

Use `analyze_orca_results` to distinguish process completion, task convergence, energy, frequencies, and NMR shielding state. Extract an accepted optimized structure only when it is needed downstream.

`analyze_orca_results` accepts a result tree, one output file, or explicit `result_files`; select explicitly when a directory contains multiple scientific logs. For a single run, `geometry_file`/`input_file` select supplementary files. Extracted final geometry is written under the analysis output directory. Before collecting optimized molecules, read the [extraction scope](../conformer-search-and-preopt/SKILL.md#tool-scope-and-examples): it uses the last frame and requires optimization convergence by default.

## Method-critical defaults

### Electronic method selection

- For an unprescribed routine main-group ground-state geometry or requested harmonic-frequency stage, use `r2SCAN-3c` as the normal efficient starting candidate. It is a composite method with its own tailored basis and corrections; do not append a separate basis or dispersion token. Do not treat it as universal for difficult spin states, strong multireference character, electronically diffuse states, or a property with a better validated method.
- For unprescribed main-group relative energies, reaction barriers, conformational energies, and noncovalent energies, `WB97M-V/def2-TZVPP` is a strong routine final-DFT starting candidate, often as a single point on a suitable lower-cost geometry. Use `WB97X-V` for established continuity, a supporting domain benchmark, or a justified cost/robustness tradeoff. Neither is a universal spectroscopy, excited-state, redox, transition-metal, or multireference method.
- Do not choose B3LYP as the unprescribed routine default, with `def2-SVP` or otherwise, merely because it is familiar or inexpensive. Preserve B3LYP when the user, an established project comparison, or property-specific evidence requires it; otherwise choose a modern task-matched method.
- Escalate beyond routine DFT only when the requested accuracy or decision warrants it. A double hybrid, `DLPNO-CCSD(T)`/basis-set treatment, canonical coupled cluster for a small system, or a multireference method can be appropriate, but only after checking reference-state suitability, property support, basis requirements, and cost. Do not turn this into an automatic publication ladder.
- For orbital energies or gaps, ionization/electron attachment, optical excitations, NMR/EPR, transition-metal spin energetics, bond breaking, and near-degenerate states, follow the property-specific branch in the method-selection reference rather than borrowing the routine energy candidate. Add diffuse functions when the target state or density requires them.

### Keyword and numerical composition

- Keep electronic-method selection separate from task syntax. Task examples supply only an operation fragment, never an unrelated functional, basis, dispersion model, or solvent.
- `orca_prepare` writes `simple_keywords` verbatim; successful preparation is not an ORCA keyword-validity check. Use the documented native or LibXC spelling for the chosen functional. For SCAN in the managed ORCA build, use `LibXC(SCAN)`; `r2SCAN` is a native keyword. Before composing an unfamiliar functional, read the two method-token examples in [functional token spelling](references/orca_method_selection.md#functional-token-spelling); do not wrap every functional in LibXC.
- Put detailed controls in native blocks such as `%scf`, `%geom`, `%tddft`, `%freq`, `%irc`, or `%cpcm`; the tool writes them verbatim.
- `WB97M-V` and `WB97X-V` already contain VV10 nonlocal correlation. Do not append `D3` or `D4`. Start ordinary work with ORCA's default `DEFGRID2`; increase the grid only for a demonstrated accuracy or sensitivity requirement.
- ORCA's normal SCF setting is sufficient for ordinary single points and optimizations. Do not add `TightSCF`, `VeryTightSCF`, `TightOpt`, or `VeryTightOpt` merely because a result is final, reliable, reproducible, publication-facing, or QC. An actually requested frequency calculation is a documented task-specific case where ORCA advises `TightSCF` to reduce frequency noise; do not infer a frequency stage from those labels.
- Ordinary `Opt` already applies ORCA's optimization-specific SCF handling. SCF and geometry convergence controls remain independent.

### Scientific interpretation

- A normal process exit is not proof that an optimization converged.
- Missing frequency output is not zero imaginary modes.
- A Kohn-Sham HOMO-LUMO eigenvalue difference is not automatically an optical excitation or fundamental gap.
- ORCA NMR output is isotropic shielding unless a reference conversion to chemical shift is explicitly performed.

## Output Contract

Return the molecular state, environment, actual method and operation keywords, requested result and its interpretation, task-convergence state, and the accepted structure or property artifacts.

## References

- [ORCA electronic-method selection](references/orca_method_selection.md)
- [ORCA task-keyword fragments](references/orca_native_examples.md)
- [ORCA general method recommendations](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)
- [ORCA DFT and composite methods](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/3cmethods.html)
- [SCF convergence](https://www.faccts.de/docs/orca/6.1/manual/contents/essentialelements/scf.html)
- [Vibrational frequencies](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/frequencies.html)
