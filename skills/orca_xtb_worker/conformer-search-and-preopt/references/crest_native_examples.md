# Native CREST argv fragments

These fragments are deliberately incomplete. The prepare and execute surfaces preserve the final ordered `argv` exactly, but this reference does not choose a GFN family, molecular state, solvent, search mode, RMSD definition, or output location. Assemble only the task-relevant pieces.

## Coordinate-first fragment

```text
"argv": ["COORDINATE_FILE"],
"asset_mappings": [
  {"source_path": "WORKSPACE_COORDINATE", "stage_path": "COORDINATE_FILE"}
]
```

Append the selected native method option and any scientifically required state, solvent, or search controls; none are injected by the tool.

## State and solvation fragments

```text
["--chrg", "CHARGE", "--uhf", "UNPAIRED_ELECTRONS"]
["--alpb", "SOLVENT"]
```

The placeholders are not suggested values. Keep the chosen state and model consistent across conformers being ranked together.

## TOML-first fragment

```text
"argv": ["CREST_TOML"],
"asset_mappings": [
  {"source_path": "WORKSPACE_TOML", "stage_path": "CREST_TOML"},
  {"source_path": "WORKSPACE_COORDINATE", "stage_path": "COORDINATE_FILE"}
]
```

The TOML file must refer to the staged filenames it needs. Map each additional referenced asset explicitly.

## Constraint-file fragment

```text
["--cinp", "CONSTRAINT_FILE"]
```

Stage `CONSTRAINT_FILE` and append this pair to the coordinate-first argv. `--cinp` does not imply `--subrmsd`; add the latter only when the intended ensemble-comparison definition requires it. Native backend and scratch options remain available and are passed as authored.
