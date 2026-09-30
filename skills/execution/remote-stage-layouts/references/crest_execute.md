# crest_execute

```text
stage/
  manifest.json
  every file named in input_assets
```

The manifest stores the exact native `argv` token list and explicit `input_assets`;
native options and referenced-file contents hold the scientific settings.

Prepare CREST stages with `crest_prepare` using the same minimal `argv` plus `input_assets` contract. Native coordinate-first, TOML-first, constraint, NCI, entropy, and other documented invocations remain available. The task adds no scientific flags and accepts no scientific `template_overrides`.

For xTB and CREST, each submission writes collected outputs under the authored stage's `attempts/<attempt-ref>/` directory. The canonical manifest and input assets remain the reusable authored stage.
