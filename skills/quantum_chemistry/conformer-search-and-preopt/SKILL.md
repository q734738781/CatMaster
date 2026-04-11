---
name: conformer-search-and-preopt
description: Use this skill for one molecular conformer-search episode that starts from a SMILES string or one seed structure, expands into a conformer ensemble, prunes it, and produces xTB-optimized candidates for downstream ORCA work.
---

# conformer-search-and-preopt

## Overview
Use this skill when the task is a bounded molecular preoptimization episode before higher-level ORCA work.

## Quick Start
1. If the input is a SMILES string, build a first 3D seed with `create_molecule_from_smiles`.
2. Use `enumerate_molecular_conformers` for an RDKit ETKDG seed ensemble.
3. Use `crest_conformer_search` when broad conformer-rotamer exploration is needed.
4. Use `filter_conformer_ensemble` to enforce one explicit energy/RMSD pruning policy.
5. Run `xtb_run_batch(mode="opt")` on the accepted ensemble and summarize it with `analyze_xtb_results`.
6. Use `extract_optimized_molecules` when the next stage needs a clean set of optimized XYZ files.

## Allowed tools
- `create_molecule_from_smiles`
- `enumerate_molecular_conformers`
- `filter_conformer_ensemble`
- `crest_conformer_search`
- `xtb_run_batch`
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

## Output Contract
Return:
- retained ensemble directory
- xTB summary path
- extracted optimized-molecule directory when generated

