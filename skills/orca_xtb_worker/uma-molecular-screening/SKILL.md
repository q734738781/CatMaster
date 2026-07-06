---
name: uma-molecular-screening
description: Use this skill for FairChem UMA OMOL single-point screening or pre-relaxation of molecules, clusters, and polymer-like nonperiodic structures before xTB/ORCA validation.
license: project-local
compatibility: local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task remote_submission remote_submission_batch"
---

# uma-molecular-screening

Use this skill when `orca_xtb_worker` needs a fast ML-potential screening or preoptimization step for molecules, clusters, or polymer-like nonperiodic structures. UMA does not replace ORCA for final molecular quantum-chemistry evidence.

## Required workflow
1. Read `remote-stage-layouts` for `uma_sp_dir` or `uma_relax_dir`.
2. Build one clean stage directory with `input/` containing `.xyz`, `.extxyz`, or other ASE-readable molecular structures.
3. Use `uma_task=omol`.
4. Set the correct `charge` and `spin` for the molecule. FairChem OMOL examples use `spin=1` for singlet and `spin=3` for triplet.
5. Submit with `remote_submission` and return the receipt context and `output/batch_summary.json`.

## When to use
- Pre-rank many conformers or cluster geometries before expensive ORCA calculations.
- Pre-relax rough molecular structures before xTB/ORCA preparation.
- Compare spin-state candidates only as a fast screening step.

## Guardrails
- Do not treat UMA as a final replacement for ORCA single-point, frequency, TS/IRC, TDDFT, NMR, or thermochemistry workflows.
- Do not use catalyst/materials UMA tasks (`oc20`, `oc22`, `oc25`, `omat`, `odac`, `omc`) for molecular charge/spin screening.
- Do not pass `charge` or `spin` through remote config. They belong in task params or `params/uma_metadata.json`; secrets and tokens never belong in params.
- Keep `relax_cell=false` for molecular jobs.

## Stage example

```text
stage/
  input/
    conf_001.xyz
    conf_002.xyz
```

Single point params:

```json
{"uma_task": "omol", "charge": 0, "spin": 1, "model": "uma-s-1p2"}
```

Relaxation params:

```json
{"uma_task": "omol", "charge": 0, "spin": 1, "fmax": 0.03, "steps": 300}
```
