# mlff_relax

Use flat, uniquely named structure files directly under `input/`:

```text
stage/
  input/
    case_a.vasp
    case_b.extxyz
```

One stage may contain one or many compatible structures. Files within a stage run sequentially. Do not use subdirectories or recursive project-tree discovery. POSCAR/VASP Selective Dynamics constraints are inherited. An extxyz file must use ASE's standard `move_mask` (`L:1` for whole atoms or `L:3` for Cartesian components); false entries are fixed, and SP/relax returns the structure as `sp.extxyz` or `opt.extxyz` with that mask verified. Use POSCAR/VASP for scaled-coordinate `FixScaled` constraints. For SP force diagnosis, `task_config.document_extxyz=true` writes `sp.extxyz` beside `sp.vasp` or `sp.xyz`, using the same basename and retaining calculator energy, forces, and stress. The primary structure remains authoritative for any constraint type that extxyz cannot encode. Put MACE model artifacts under optional `models/` and refer to them through `backend_config.checkpoint_artifact`. Registered MACE `omol-0` charge and multiplicity-style spin belong in `backend_config.defaults` with per-input exceptions in `backend_config.items`; the same nested item pattern applies to UMA task/charge/spin. Key every item by the exact filename relative to `input/`; do not create a second metadata file.

For short, similarly sized relaxations, roughly 30-50 structures per stage is an empirical starting point. Use smaller groups for large or heterogeneous structures and substantially larger groups for cheap SP screening. The agent may create these stage copies directly; no generic automatic partitioner is required.
