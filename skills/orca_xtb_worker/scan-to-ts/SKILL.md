---
name: scan-to-ts
description: Use this skill to compose an ORCA relaxed scan and subsequent OptTS refinement through the generic native ORCA preparation surface.
allowed-tools: "orca_prepare remote_submission remote_submission_batch analyze_orca_results execute"
---

# ORCA scan to TS

## Workflow

1. Prepare the relaxed scan with `orca_prepare`: include the method/basis and `Opt` in `simple_keywords`, and place the complete `%geom Scan ... end` section in `input_blocks`.
2. Submit with `orca_execute` and inspect the scan outputs/profile with a focused parser plus `analyze_orca_results`.
3. Select the TS-side geometry and prepare a new generic ORCA stage with `OptTS` and the intended `%geom` Hessian controls.
4. Validate the accepted TS with the requested frequency and, when needed, IRC stages.

## Scientific choices

- Define the scanned internal coordinate, range, number of points, and relaxed constraints explicitly.
- `OptTS`, SCF convergence, geometry convergence, and Hessian controls are separate native choices.
- Keep scan, TS refinement, frequency validation, and final energy levels identifiable; do not mix their energies without stating the level.

## References

- [ORCA transition-state searches](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations_TS.html)

## Tool scope and examples

Pass the complete native ORCA block as one input_blocks string; the wrapper does not supply missing geom/Scan terminators. Atom indices are zero-based. For a bond scan:

```text
%geom
  Scan
    B 0 1 = 1.4, 2.6, 9
  end
end
```

Keep Opt with the intended method/basis in simple_keywords. This is ORCA geom Scan syntax, not an xTB `$scan` block or a standalone SCAN keyword. See the [ORCA surface-scan manual](https://www.faccts.de/docs/orca/6.1/manual/contents/structurereactivity/optimizations_scans.html).
