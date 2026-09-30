# lammps_execute_kokkos

```text
stage/
  in.lammps
  manifest.json
  every script-referenced data, restart, potential, table, molecule, or include file
```

Author the complete native script first, then call `lammps_prepare(input_path=..., asset_mappings=[...])`. It copies the script unchanged to `in.lammps`; the tool does not choose force fields, units, styles, fixes, ensembles, timesteps, or observables. Both tasks use this identical stage contract. Choose one available path compatible with the active LAMMPS styles.
