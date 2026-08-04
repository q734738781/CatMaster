---
name: mlff-molecular-screening
description: Use this skill for MLFF single-point screening or pre-relaxation of molecules, conformers, and clusters before xTB or ORCA validation.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch"
---

# mlff-molecular-screening

## Overview

Use an enabled molecular-capable MLFF backend for fast triage without replacing quantum-chemistry evidence.

## Quick Start

1. Put uniquely named molecular structures directly under a clean `input/`.
2. Query `mlff_sp` or `mlff_relax` for the selected molecular-capable backend in one call, using `template_overrides={"backend": "<enabled-backend>"}` and `detail="full"`; do not infer provider fields from another backend's schema.
3. For FairChem UMA, use `omol` and set charge plus multiplicity-style spin for every structure.
4. Submit a stage or batch of complete stages, then inspect all per-input summaries/errors.
5. Send the shortlist to xTB/ORCA with charge, multiplicity, geometry, and provenance intact.

## Allowed tools

- `ls`
- `read_file`
- `write_file`
- `edit_file`
- `execute`
- `get_avail_remote_task`
- `get_remote_task_spec`
- `remote_submission`
- `remote_submission_batch`

## Workflow

### 1. Choose a molecular-capable backend

- FairChem UMA `omol` is the initially validated molecular route. Do not use periodic UMA domains for isolated molecules.
- Query current availability; do not assume every remote deployment installs the same providers.

### 2. Preserve physical metadata

- Put shared UMA metadata in `backend_config.defaults` and per-file exceptions in `backend_config.items`.
- Use exact paths relative to `input/`. Preserve charge and spin when handing selected geometries to xTB/ORCA.

### 3. Group only comparable structures

- One stage reuses model loading. Group conformers or similarly sized candidates, and split heterogeneous or large relaxations into balanced stages.
- Use SP for ranking unchanged geometries and relax only when preoptimization is intended.

### 4. Escalate to quantum chemistry

- MLFF does not replace ORCA frequencies, TS/IRC, thermochemistry, excited states, NMR, or final spin-state energetics.
- Report the MLFF shortlist criterion and the reference method still required.

## Method-critical defaults

- UMA `omol` requires explicit charge and multiplicity-style spin; common values include spin 1 for singlet and spin 3 for triplet.
- Keep backend/model and relaxation settings fixed across candidates being ranked.
- Treat MLFF energy gaps as screening values until checked at the requested xTB/ORCA level.

## Output Contract

Return backend/model, operation, charge/spin mapping, stage/batch path, batch summary, shortlist, and the planned xTB/ORCA validation. Keep receipt/context and platform details in runtime records unless failure recovery needs them; provide them whenever the user explicitly asks to inspect, compare, record, or report them.

## References

- Canonical stage layout: `skills/execution/remote-stage-layouts/SKILL.md#mlff_sp-and-mlff_relax`
- Full UMA domain notes: `skills/materials_worker/mlff-screening-and-relaxation/references/fairchem_uma.md`
