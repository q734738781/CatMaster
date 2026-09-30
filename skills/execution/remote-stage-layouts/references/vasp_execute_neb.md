# vasp_execute_neb

Prepare the complete VASP NEB/dimer root locally before submission:

```text
stage/
  INCAR
  POTCAR
  KPOINTS
  00/POSCAR
  01/POSCAR
  ...
  NN/POSCAR
```

Use `vasp_neb_prepare` or an equivalent checked preparation path. Do not submit endpoint-only input and do not ask the remote boot script to interpolate. Reject atom-order/cell mismatches and inspect any `short_distance_count > 0` warning before submission.
