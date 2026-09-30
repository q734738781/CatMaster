# Rigid-fragment assembly specification

Read this reference when the model combines two or more independently generated structures. The bundled script is a reusable starting implementation, not a universal chemistry engine.

## Minimal shape

```json
{
  "host": {"path": "structures/slab.vasp"},
  "fragments": [
    {
      "name": "co",
      "path": "structures/co.xyz",
      "placement": {
        "fragment_atom": 0,
        "target": {"body": "host", "atom": 18},
        "distance": 1.85,
        "direction": [0.0, 0.0, 1.0]
      },
      "align": {
        "axis_atoms": [0, 1],
        "target_vector": [0.0, 0.0, 1.0]
      }
    }
  ],
  "anchors": [
    {
      "a": {"body": "host", "atom": 18},
      "b": {"body": "co", "atom": 0},
      "target": 1.85,
      "tolerance": 0.10,
      "weight": 100.0
    }
  ],
  "orientations": [
    {
      "body": "co",
      "axis_atoms": [0, 1],
      "target_vector": [0.0, 0.0, 1.0],
      "target_angle_deg": 0.0,
      "tolerance_deg": 10.0,
      "weight": 10.0
    }
  ],
  "settings": {
    "starts": 32,
    "keep": 3,
    "seed": 20260826,
    "clash_scale": 0.75,
    "absolute_clash": 0.50,
    "maxiter": 500,
    "translation_jitter": 0.35,
    "rotation_jitter_deg": 35.0,
    "diversity_rmsd": 0.35,
    "output_format": "extxyz"
  }
}
```

All atom indices are zero-based within their source body. `host` is reserved; fragment names must be unique. A fragment `placement` puts `fragment_atom` at `target + distance * direction` after optional axis alignment. The placement target may instead be `{"point": [x, y, z]}`. This seed expresses the chemical side of an adsorption site or encounter geometry; random starts perturb it rather than replacing it with uninformed packing.

`align` sets the initial orientation. An entry in `orientations` keeps that relationship in the objective. Use both when orientation is chemically meaningful. `initial_translation` and extrinsic `initial_rotation_deg` are available when a prealigned source is a better seed.

Several anchors between the same bodies can define a cyclic or transition-state contact motif while preserving each fragment internally. The reference implementation supports distance anchors, one-body axis orientations, and bounded regions. If the construction requires an explicit angle, dihedral, plane, or bond-length combination, extend the objective for that named coordinate or carry it into the semi-rigid constrained stage; do not pretend that an unsupported constraint was enforced.

## Optional spatial region

For a pore, interface gap, or bounded insertion region:

```json
{
  "body": "guest",
  "reference_atom": 0,
  "lower": [0.2, 0.2, 0.2],
  "upper": [0.8, 0.8, 0.8],
  "coordinate_type": "fractional",
  "weight": 50.0
}
```

Add such objects under `regions`. Omit `reference_atom` to constrain the fragment centroid. Cartesian bounds use angstrom. Fractional bounds require a valid host cell and are useful for a periodic pore, but the bounds do not discover cavities; choose them from the actual host geometry.

## Intended contacts and clash handling

Anchored pairs are removed from the generic inter-body repulsion because their chemical target controls the distance. They are still written into each validation context and must pass the independent structure checker. Do not exempt an entire pair of fragments or a broad atom set.

The exclusion distance is a configurable multiple of the sum of covalent radii, with an absolute lower floor. It is a geometry feasibility heuristic rather than a force field or a transferable bond-length model. If a legitimate metal, high-pressure, or transition-state contact falls in a warning range, examine that pair explicitly; do not globally weaken the gate to make one candidate pass.

For each non-anchored pair between different bodies, the current script uses

```text
R_ij = max(clash_scale * (r_cov,i + r_cov,j), absolute_clash)
p_ij = max(0, R_ij - d_ij) / R_ij
E_clash = clash_weight * sum(p_ij^2)
```

Distances use the minimum-image convention when the host is periodic. This is already a single-frame overlap penalty. It differs from ASE IDPP, which compares every pair distance with an endpoint-derived target matrix and weights the mismatch by an inverse power of distance. That target matrix is meaningful for NEB interpolation but absent in fragment assembly. The one-sided penalty leaves separated atoms alone and combines cleanly with anchor, orientation, and region terms.

The soft penalty guides rigid-body optimization; `max_penetration_A` remains the acceptance quantity. Multi-start sampling matters because a symmetric exact coincidence can be a poor numerical starting point for any distance-only gradient. The independent checker still decides whether the emitted frame contains a collision.

## Running the reference implementation

Execute the bundled resource directly; relative paths inside the specification use the shell's current workspace directory:

```bash
python "$CATMASTER_SKILLS_ROOT/atomistic/constraint-guided-atomic-assembly/scripts/rigid_fragment_assembly.py" \
  assembly.json --output-dir structures/assembled
```

The script writes accepted candidates, one validation-context JSON beside each candidate, and `assembly_report.json`. If no pose satisfies the explicit clash/anchor/orientation/region contract, it writes `best_infeasible.extxyz`, records the violated terms, and exits nonzero. Treat that as a request to revise the structural hypothesis, not as an invitation to hide the failure.

Use the console status, maximum violations, and report path for triage. Read the relevant clash, anchor, orientation, or region rows from the report when revising that constraint. If a custom objective is necessary, preserve detailed metrics in a workspace JSON and use `format_metrics_summary(metrics)` in error messages; do not raise or print the entire metrics dictionary.

`--verbose` additionally prints the complete saved report, without changing candidate generation or acceptance.
