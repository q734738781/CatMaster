---
name: mace-screening-and-relaxation
description: Use this skill for MACE-based rapid screening and relaxation loops before DFT, including candidate pruning and handoff criteria.
compatibility: Designed for CatMaster local tools and project-space relative-path execution.
metadata:
  catmaster-suggested-tools: "mace_relax_batch mace_sp_batch"
---

# mace-screening-and-relaxation

## Overview
Use this skill to run cheap MACE screening on a structure batch before spending VASP resources.

## Quick Start
1. Prepare a clean structure batch under `input_dir`.
2. Choose `mace_relax_batch` for geometry cleanup or `mace_sp_batch` for static ranking.
3. Keep `output_root` outside `input_dir`.
4. Use the collected outputs and batch-state files to decide which candidates advance to VASP.

## Suggested tools
- mace_relax_batch
- mace_sp_batch

## Workflow

### 1. Choose relax vs single-point deliberately
- `mace_relax_batch` needs a `model`; it can also toggle `head`, `dispersion`, and `relax_lattice`.
- `mace_sp_batch` is for energy evaluation only and does not relax geometry.
- Do not compare relax and SP outputs as if they were the same screening stage.

### 2. Keep input and output trees separate
- Both tools reject `output_root` inside `input_dir`.
- The runtime stages a temporary batch tree under `output_root`, dispatches remotely, collects outputs, then removes the staging tree.

### 3. Use collected evidence, not launch success alone
- Returned metadata includes `batch_state_rel`, collected stdout/stderr/status files, and any `batch_summary_rel`.
- On dispatch failure, the tool still tries to collect partial outputs; inspect those before deciding to rerun.

## Output Contract
Return:
- chosen MACE stage (`relax` or `sp`)
- `output_root_rel`
- `batch_state_rel`
- shortlist or keep/drop rule for downstream VASP handoff

## References
- Use `vasp-input-preparation` only after a MACE shortlist exists; do not send the whole raw candidate pool forward by default.
