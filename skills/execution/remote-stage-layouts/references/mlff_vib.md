# mlff_vib

Place one or more accepted structures directly under `input/`:

```text
stage/
  input/
    minimum.vasp
    adsorbate.extxyz
```

Each structure is analyzed independently with the same backend/task configuration. Constraints in the structure define the exact normal-mode subspace: POSCAR/VASP may carry scaled-coordinate Selective Dynamics, while extxyz carries whole-atom or Cartesian-component `move_mask`. The runner does not optimize structures and does not assume they are transition states.

The output for each input is one `vibrations.npz` canonical bundle, one `frequencies.csv`, one multi-frame `modes.extxyz`, and `summary.json`. Use separate batch children when structures require different backend metadata or should run concurrently rather than sequentially.
