# Geometry validation and recovery contract

## Distance ratio

For an atom pair `i,j`, the checker reports

```text
q_ij = d_ij / (r_cov,i + r_cov,j)
```

using ASE covalent radii and a PBC-aware neighbor list. Defaults are deliberately conservative:

- `d < 0.50 Å` or `q < 0.60`: `FAIL`;
- `0.60 <= q < 0.75`: `WARN`;
- `0.75 <= q < 0.85`: `REVIEW`;
- otherwise no short-distance flag.

These bands detect geometric collisions; they do not decide bonding, oxidation state, coordination chemistry, transition-state validity, or energetic stability. Metal contacts, compressed phases, unusual bonding, and forming/breaking bonds may require a documented pair-specific interpretation. Keep the raw distance, elements, indices, image shift, and ratio visible when overriding a warning.

## Checker inputs and exit behavior

```bash
python "$CATMASTER_SKILLS_ROOT/atomistic/atomic-structure-validation-and-recovery/scripts/check_atomic_structure.py" \
  structures/candidate.extxyz \
  --context structures/candidate.context.json \
  --output analysis/candidate_geometry.json
```

The optional context is written by the rigid-assembly helper and contains expected contacts plus body index ranges. The checker validates those target ranges independently and labels short pairs as inter- or intra-body. It does not suppress a short pair merely because it was intended.

Default exit behavior is nonzero only for `FAIL`; `WARN` and `REVIEW` remain machine-readable in the JSON. Use `--fail-on warn` only when the current workflow intentionally treats every warning as a blocking gate.

The console gives status, counts, the worst example for each failed distance criterion, cell issues, and the largest expected-contact violation. The JSON preserves all flagged pairs, expected contacts, coordination counts, and the requested shortest pairs (`--top-pairs`, default 20). Start with the console; query specific JSON fields when needed instead of printing the whole report. For batch checks, retain each report and return each structure's status and report path. Custom callers should use `format_summary(report, report_path)` for diagnostics rather than raising or printing the full report dictionary.

Console atom indices are one-based; JSON includes both zero-based `i,j` and explicit `i_1based,j_1based`. Assembly specifications use zero-based body-local indices.

Add `--verbose` only when the full JSON is wanted on the console as well as in the report file. It does not change the checks or exit status.

The checker also reports non-finite coordinates, invalid periodic cell vectors, periodic self-image contacts, atoms outside the primary cell, and heuristic neighbor counts. Outside-cell and coordination information is diagnostic; unwrapped coordinates and unusual coordination are not automatically wrong.

## Failure triage

Evidence favoring a construction failure includes:

- a hard short-distance failure in the submitted input;
- non-finite or grossly repulsive energy/force at the first evaluation;
- a fragment that moves by a large amount immediately or is expelled from the model;
- loss of the requested anchor, ring, interface order, or site identity before a physical basin is established;
- the same failure across reasonable optimizers or physical models from the same starting coordinates.

Evidence favoring a later numerical/model problem includes a valid coherent start followed by late oscillation, an electronic convergence error without a geometric pathology, force noise near the requested tolerance, or a documented model-domain limitation. The categories can coexist; repair the geometry first, then diagnose what remains.

## Force precheck interpretation

For an authorized, domain-compatible `mlff_sp`, set `task_config.document_extxyz=true`. The summary reports the largest raw atomic force norm as `max_force_eVA`. The same-basename `sp.extxyz` preserves the full per-atom force vectors. Sort their norms, then map the largest-force atoms back to the numerical geometry report and the intended contact list.

Treat a force outlier as supporting evidence for reconstruction when it is localized on a suspicious short pair, wrong interface contact, buried fragment, or periodic-image collision. If the high-force region has no geometric pathology, investigate model coverage, electronic metadata, or unusual chemistry before changing the structure. Do not introduce one fixed eV/angstrom threshold as a chemistry-independent acceptance gate, and do not let a modest MLFF force override a numerical geometry failure.

Do not use minimization, MD, looser SCF, a larger step limit, or a stronger physical potential as the first response to an overlapped assembly. Rebuild from the last chemically valid components.
