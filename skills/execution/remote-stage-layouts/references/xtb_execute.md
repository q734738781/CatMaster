# xtb_execute

Prepare xTB stages with `xtb_prepare`:

```text
stage/
  manifest.json
  every file named in input_assets
```

`manifest.json` contains only the exact native `argv` token list and explicit `input_assets`. Coordinate paths, GFN choice, charge, unpaired electrons, solvation, optimization, Hessian/MD modes, constraints, parameter files, and other supported xTB options remain native argv or referenced-file content. Do not pass scientific `template_overrides` to `xtb_execute`.

For xTB and CREST, each submission writes collected outputs under the authored stage's `attempts/<attempt-ref>/` directory. The canonical manifest and input assets remain the reusable authored stage.
