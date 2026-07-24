---
name: conformer-search-and-preopt
description: Use this skill for one molecular conformer-search episode that starts from a SMILES string or one seed structure, expands into a conformer ensemble, prunes it, and produces xTB-optimized candidates for downstream ORCA work.
allowed-tools: "create_molecule_from_smiles enumerate_molecular_conformers filter_conformer_ensemble xtb_prepare remote_submission remote_submission_batch analyze_xtb_results extract_optimized_molecules"
---

# conformer-search-and-preopt

## Overview
Use this skill when the task is a bounded molecular preoptimization episode before higher-level ORCA work.

## Quick Start
1. If the input is a SMILES string, build a first 3D seed with `create_molecule_from_smiles`.
2. Use `enumerate_molecular_conformers` for an RDKit ETKDG seed ensemble.
3. Use `remote_submission_batch` with `task_name="crest_run"` when broad conformer-rotamer exploration is needed.
4. Use `filter_conformer_ensemble` to enforce one explicit energy/RMSD pruning policy.
5. Prepare the accepted ensemble with `xtb_prepare(mode="opt")`, then run its first-level stages with `task_name="xtb_execute"`.
6. Use `extract_optimized_molecules` when the next stage needs a clean set of optimized XYZ files.

## Allowed tools
- `create_molecule_from_smiles`
- `enumerate_molecular_conformers`
- `filter_conformer_ensemble`
- `xtb_prepare`
- `remote_submission`
- `remote_submission_batch`
- `analyze_xtb_results`
- `extract_optimized_molecules`

## Workflow

### 1. Keep one explicit ensemble contract
- Do not mix raw RDKit seeds, CREST ensembles, and xTB-optimized structures without a manifest.
- After every filtering step, emit the retained structure directory and the summary file path.

### 2. Separate exploration from refinement
- Use RDKit or CREST for ensemble generation.
- Use xTB for preoptimization, low-cost ranking, and quick Hessian checks.
- Do not skip the pruning step before higher-level work unless the task explicitly needs the full ensemble.

### 3. Hand off cleanly
- If the next stage is ORCA, pass a directory of optimized XYZ files rather than a mixed results tree.

## Method-critical defaults
- Keep charge, unpaired-electron count, GFN family, and solvation identical across conformers being compared.
- Use `xtb_prepare` as the only place where xTB scientific settings are chosen; submit `xtb_execute` without scientific overrides.
- Preserve the prepared `manifest.json` and optional `xtb.inp` alongside each optimized conformer.

## Output Contract
Return:
- retained ensemble directory
- xTB prepare manifest
- xTB summary path
- extracted optimized-molecule directory when generated

## References
- `/.deepagents/skills/execution/remote-stage-layouts/SKILL.md#xtb_execute`
