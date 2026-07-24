---
name: mlff-transition-state-refinement
description: Use this skill to refine one TS-like structure with constrained managed MLFF RS-pRFO and validate its saddle order.
license: project-local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task get_remote_task_spec remote_submission remote_submission_batch"
---

# mlff-transition-state-refinement

## Overview

Refine a chemically informed TS guess. This is local transition-state refinement, not an open-ended pathway search.

## Workflow

1. Start from one TS-like geometry, normally a high-energy NEB image, dimer result, scan maximum, or chemically constructed guess.
2. Put exactly one POSCAR/VASP or extxyz structure under `stage/input/`. Encode fixed atoms or components in that file.
3. Query `get_remote_task_spec(task_name="mlff_ts", template_overrides={"backend": "<enabled-backend>"}, detail="full")`. Select backend molecular metadata explicitly when required, such as UMA `omol` charge and multiplicity-style spin.
4. Submit one candidate with `remote_submission`. Use `remote_submission_batch` only for independent candidates sharing the same backend and task configuration.
5. Treat `converged` and `validated_first_order_saddle` as separate results. A validated result must converge and have exactly one significant imaginary mode.

## Hessian behavior

- Keep `hessian_method=auto` unless a diagnostic comparison requires an override.
- Auto uses a calculator's public analytic Hessian when available. Otherwise it uses a constraint-projected finite-difference Hessian for small systems and iterative Sella diagonalization for larger systems.
- Final saddle-order validation is independent of the optimizer's approximate Hessian. For a backend without an analytic Hessian, it uses finite differences in the unconstrained subspace.
- `hessian_delta` is a Cartesian displacement in angstrom. Tightening it without a force-noise check can make the Hessian less reliable.

## Constraints

- POSCAR/VASP Selective Dynamics and extxyz `move_mask` are the source of truth.
- Cartesian and scaled-coordinate components are projected separately. Fixed components are excluded from finite-displacement directions.
- The cell is not optimized during order-one saddle refinement.
- Do not pass a second fixed-atom list or rewrite the staged runner.

## Output contract

Report:

- input stage, backend/model/task metadata, fmax, steps, and resolved Hessian method;
- convergence, maximum projected force, constraint drift, and final energy;
- significant imaginary-mode count, lowest frequency, and `validated_first_order_saddle`;
- final `ts.vasp`, `ts.xyz`, or `ts.extxyz`, plus the shared `vibrations.npz`, `frequencies.csv`, `modes.extxyz`, and `reaction_mode.txt`;
- receipt/context identifiers and any per-stage error.

An optimizer-completed structure with zero or multiple significant imaginary modes is still a useful result, but it is not a validated first-order saddle.
