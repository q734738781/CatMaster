# Native xTB argv fragments

These fragments are deliberately incomplete. Each item ultimately placed in `argv` is one literal token and shell expansion is not used, but the fragments do not choose a GFN family, charge, spin, solvent, operation, optimization level, or output location. Assemble only the pieces required by the current task.

## Coordinate and asset fragment

```text
"argv": ["COORDINATE_FILE"],
"asset_mappings": [
  {"source_path": "WORKSPACE_COORDINATE", "stage_path": "COORDINATE_FILE"}
]
```

The staged filename used in `argv` must match `stage_path`.

## Operation-token fragments

Append the requested operation fragment when the native coordinate-only single point is not sufficient:

```text
["--opt", "OPT_LEVEL"]
["--hess"]
["--md"]
```

Treat them as alternatives unless the intended native workflow deliberately and validly combines operations; they are not a bundle to copy.

## Method, state, and solvation fragments

```text
["--gfn", "GFN_LEVEL"]
["--chrg", "CHARGE", "--uhf", "UNPAIRED_ELECTRONS"]
["--alpb", "SOLVENT"]
```

Choose each value for the actual molecular state and comparison protocol. Omit a fragment when it is not needed rather than replacing its placeholder with a familiar example value.

## Detailed-input fragment

```text
"argv": ["COORDINATE_FILE", "--input", "DETAIL_FILE"],
"asset_mappings": [
  {"source_path": "WORKSPACE_COORDINATE", "stage_path": "COORDINATE_FILE"},
  {"source_path": "WORKSPACE_DETAIL_FILE", "stage_path": "DETAIL_FILE"}
]
```

Use the staged detailed-input file for complete native constraints, fixes, metadynamics, or other `$...` sections. Add method, state, solvation, and operation tokens separately only when the task requires them.
