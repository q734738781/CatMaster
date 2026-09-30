---
name: atomic-structure-validation-and-recovery
description: Use this skill after atomic coordinates are created or changed, before DFT, MLFF, force-field, or MD execution, and when optimization, SCF, energy, or force behavior suggests that the starting geometry may need to be rebuilt.
license: project-local
---

# Atomic-structure validation and recovery

## Scope

Separate geometric feasibility from physical relaxation. Numerical distance checks are the hard first gate; rendered multimodal inspection checks semantic 3D arrangement; a cheap physical calculation is a later diagnostic, not a collision-removal method.

ExperimentSpecialist should use this skill to classify a failure and delegate one reconstruction objective. The worker that owns the structure performs the checks and rebuild. A dynamics worker may validate its starting structure, but general slab, adsorbate, defect, interface, or pathway reconstruction returns to `materials_worker`.

## When it applies

Run the gate after adsorption placement, fragment merge, pore or interstitial insertion, interface construction, atom substitution with large size change, random perturbation, endpoint remapping or path interpolation, TS/precomplex construction, and any other material coordinate edit. Run it again on the accepted post-relaxation structure.

## Numerical gate

1. Read [the validation and recovery contract](references/validation-contract.md).
2. Execute [the PBC-aware checker](scripts/check_atomic_structure.py) directly using the command in the contract. No source read, copy, or wrapper is needed for ordinary checks.
3. Inspect its console summary first. For warnings, failures, or a specific scientific question, query the relevant JSON fields: flagged pairs, periodic image shifts, fragment scope, cell/PBC issues, or expected-contact errors. Normal shortest-pair tables and per-atom counts stay in the report unless needed.
4. Treat an absolute distance below `0.50 Å` or a default normalized ratio below `0.60` as a hard failure. Treat the higher ratio bands as review ranges, not universal chemical laws. Use project- or method-specific values when evidence justifies them.
5. Do not waive a hard failure because a renderer looks normal. Coincident atoms can render as one sphere, and a single cell view can hide a periodic collision.

## Multimodal gate

For important constructed models, inspect top, side or crystallographic-axis, and perspective views after the numerical gate. Ask only concrete questions:

- Are fragments or layers interpenetrating or ordered on the wrong side?
- Is an adsorbate or guest buried in an unintended layer or cavity wall?
- Is the requested anchor/site/motif visible and oriented as intended?
- Are there isolated atoms, reversed fragments, duplicated layers, or periodic-image collisions?
- Does the structure preserve the requested vacancy, substitution, interface, or reaction-center topology?

For `materials_worker`, use `structure-visual-inspection` when VESTA rendering is appropriate. Numerical evidence remains authoritative for distances; visual evidence catches whole-fragment and topology errors that pair distances alone may miss.

## Optional MLFF force precheck

After the numerical gate passes, a compatible MLFF single point can expose a local problem that distance thresholds missed. Use it only when an appropriate backend is available and the calculation is authorized. Query the current `mlff_sp` schema, keep the candidate geometry unchanged, and set `task_config.document_extxyz=true`. The ordinary structure output remains available, while the same-basename `sp.extxyz` records energy, stress, and atom-resolved raw forces.

Start with `max_force_eVA`, then calculate force-vector norms from `sp.extxyz` and locate the largest-force atoms. Compare those atoms with the shortest-pair report, expected contacts, fragment boundaries, periodic image shifts, and local coordination. A large force concentrated on a short unintended pair or an interpenetrating region is strong evidence to rebuild that local assembly. Do not ask relaxation to repair it.

A large MLFF force is not proof of overlap by itself. If geometry and topology remain coherent, check whether the model covers the elements and bonding regime, whether charge or spin is correct, and whether the structure contains reaction-center, high-pressure, magnetic, or other out-of-domain chemistry. Use model-specific experience or a same-model candidate distribution rather than a universal force cutoff. An unavailable or unsuitable MLFF does not block a structure that has passed the numerical and semantic checks.

## Recovery before optimizer tuning

When a relaxation, single point, SCF, MD start, or TS refinement fails, first compare the failing input against the last chemically valid components:

1. Stop repeated submission or tolerance escalation.
2. Check the exact starting geometry. Immediate non-finite values, extreme repulsion, very short pairs, large first-step displacements, fragment ejection, or a motif that collapses before meaningful optimization strongly implicate construction.
3. If the geometry is wrong, discard the distorted attempt as a seed and reconstruct from the authoritative host/fragments with `constraint-guided-atomic-assembly`.
4. Preserve chemical intent: atom identity/order, charge and spin, cell/PBC, intended contacts, reaction coordinate, and fixed/movable groups.
5. Generate several rigid candidates. If none is feasible, revise the site, conformer, cell/cavity, atom mapping, or reaction motif rather than forcing the optimizer.
6. After numerical and visual acceptance, use the optional MLFF force precheck above or another same-model cheap single point. Use constrained pre-relaxation only after the unchanged geometry has been diagnosed. Interpret force or energy outliers against the relevant model and comparison set; do not turn one universal force threshold into a chemistry-independent law.
7. Revalidate the resulting structure before DFT, MLFF relaxation, MD, pathway execution, or scientific reporting.

## Rebuild completely when

- slabs, layers, or fragments interpenetrate;
- the adsorbate/guest is on the wrong side or in the wrong region;
- the requested anchor or TS topology is absent;
- periodic repetition creates a collision;
- early optimization changes connectivity or ejects a fragment before reaching a meaningful basin;
- the only way to continue would be to weaken clash thresholds broadly or increase iteration limits blindly.

Late-stage numerical oscillation on an already valid, chemically coherent structure may instead be an optimizer, electronic, force-field, or model-domain problem. Report that distinction explicitly.

## Handoff

Return the accepted structure path, checker report, concrete visual finding when images were used, expected-contact result, and whether recovery reused the original components or a justified revised structural model. Do not return a large audit inventory or operational metadata unrelated to the scientific decision.
