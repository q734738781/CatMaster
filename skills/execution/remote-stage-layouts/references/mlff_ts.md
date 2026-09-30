# mlff_ts

Prepare exactly one TS-like structure directly under `input/`:

```text
stage/
  input/
    ts_guess.vasp or ts_guess.extxyz
```

The runner performs fixed-cell, order-one constrained RS-pRFO refinement. Put constraints in the structure itself: POSCAR/VASP Selective Dynamics may fix scaled-coordinate components, while extxyz uses ASE `move_mask` and preserves Cartesian component constraints in `ts.extxyz`. Do not add a second constraint file or a duplicate atom-index override.

One stage contains one TS candidate. Use separate stages for independent candidates, selected from first-level children by default or through explicit relative `stage_paths`. Inspect both `converged` and `validated_first_order_saddle`: validation additionally requires exactly one frequency below the configured negative-frequency threshold. Retain `vibrations.npz`, `frequencies.csv`, `modes.extxyz`, `reaction_mode.txt`, and the final `ts.*` structure.
