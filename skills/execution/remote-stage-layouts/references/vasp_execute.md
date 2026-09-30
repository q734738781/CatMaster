# vasp_execute

Prepare one complete VASP calculation per stage:

```text
stage/
  INCAR
  POTCAR
  POSCAR
  KPOINTS
  optional required VASP inputs
```

For several calculations, use one complete VASP stage per selected directory. The default is first-level children; explicit `stage_paths` may select a subset or nested stages.
