---
name: constraint-guided-atomic-assembly
description: Use this skill when independently generated atomic fragments must be combined into an adsorbate, pore guest, supported cluster, interface, reactive complex, or transition-state guess without relying on a physical optimizer to repair the initial geometry.
license: project-local
---

# Constraint-guided atomic assembly

## Scope

Choose the chemical contacts and fragment mobility first, then solve the 3D pose in rigid-fragment space. This skill owns initial assembly or complete reconstruction; it does not replace a later MLFF, xTB, force-field, or DFT relaxation.

The transferable part of IDPP is its use of pair-distance penalties. The bundled assembly script already applies that idea to one structure with a PBC-aware, one-sided soft-sphere penalty on unintended inter-fragment contacts. It does not instantiate ASE IDPP: an assembly has no endpoint-derived target-distance matrix, and atoms that are already outside the exclusion radius should not be pulled toward an invented target. IDPP itself remains useful for improving an interpolated NEB band after both endpoints are valid.

If you are ExperimentSpecialist, use this skill only to recognize an assembly problem and pass a bounded reconstruction brief to the worker that owns the structure. Periodic, surface, interface, and materials pathway models belong to `materials_worker`; molecular precomplexes and finite-cluster TS guesses belong to `orca_xtb_worker`.

## Invariant

When structures came from different sources, do not concatenate their coordinates and proceed directly to physical relaxation. Identify intended inter-fragment contacts, preserve each accepted fragment, remove unintended contacts in rigid-body space, and validate the assembled candidates before invoking a physical potential.

## Assembly brief

Record only chemically meaningful inputs:

- the authoritative host and fragment files;
- fragment identity and atom indexing;
- intended anchor pairs and target distance ranges;
- any required axis, angle, ring, plane, or region relationship;
- which bodies are fixed, rigid, or later semi-rigid;
- cell and PBC semantics inherited from the host;
- alternatives that remain chemically plausible.

Do not ask the language model to invent final Cartesian coordinates. If the binding atom, adsorption side, reactive topology, charge/spin state, or atom mapping is ambiguous and changes the chemical model, preserve separate candidates or return that scientific choice instead of hiding it inside geometry optimization.

## Workflow

1. Start again from the last chemically valid component files, not from a badly distorted failed relaxation.
2. Normalize atom order and choose anchors. Treat normal bonded geometry inside each component as fixed during the rigid stage.
3. Read [the assembly specification](references/assembly-spec.md) and execute [the rigid-fragment script](scripts/rigid_fragment_assembly.py) directly as shown there. Copy source to the workspace only when the required constraints need implementation changes; leave mounted originals intact.
4. Use deterministic multi-start translation and rotation around a chemically informed seed. Rank by unintended-contact penetration plus anchor, orientation, and region violations. Retain several genuinely different feasible poses when the chemistry does not select one.
5. Run `atomic-structure-validation-and-recovery` on every retained candidate. A low geometric objective is not itself structural acceptance.
6. For an important model, inspect orthogonal and perspective renders after the numerical gate. Check layer order, burial, reversed fragments, wrong site identity, periodic repetition, and unintended isolated atoms.
7. Only then perform a cheap physical single point or constrained pre-relaxation. When a compatible managed MLFF is available and execution is authorized, `mlff_sp` with `task_config.document_extxyz=true` keeps atom-resolved forces in `sp.extxyz` for local diagnosis. Keep the reaction center or intended contacts restrained while relaxing nearby atoms when a fully rigid model cannot reach a useful pre-TS geometry.

## Rigid versus semi-rigid

- Use a rigid stage for adsorption placement, pore insertion, supported clusters, initial interface registry, reactant encounter complexes, and the first construction of cyclic TS motifs.
- Use a semi-rigid second stage when reaction-center angles or neighboring bonds must adapt. Release the reaction center and nearby atoms, restrain remote skeletons, and retain the key forming/breaking-coordinate definition.
- Do not use a full unconstrained relaxation as an assembly algorithm. It may eject a fragment, destroy the intended motif, or enter an extreme repulsive region before the chosen potential is meaningful.

## Failure decisions

If no feasible rigid pose survives multi-start, do not merely reduce exclusion radii or increase optimizer iterations. Reconsider the site, fragment conformer, supercell or cavity size, anchor mapping, interface registry, or proposed reaction motif. An infeasible assembly is useful evidence that the current structural model may be wrong.

The reference script assumes that the host supplies the output cell and that interface lattices are already compatible. It optimizes rigid poses; it does not perform lattice matching, conformer generation, bond rearrangement, or physical energy ranking.

## Handoff

Return the accepted candidate paths, source component paths, intended contacts, any unresolved chemical alternatives, and the validation result. Keep rejected candidates only when their failure explains a scientific modeling decision.
