# cp2k_execute

```text
stage/
  job.inp
  manifest.json
  any explicitly referenced structure, restart, basis, potential, include, or PLUMED files
```

Author the complete native input first, then call `cp2k_prepare(input_path=..., asset_mappings=[...])`. It copies the input unchanged to `job.inp`; `manifest.json` only names `job.inp` and its explicitly mapped assets. All CP2K run types, electronic settings, dynamics, restarts, and print/property sections belong in the native input. Use one prepared CP2K stage per selected directory, with first-level children by default or explicit relative `stage_paths`.
