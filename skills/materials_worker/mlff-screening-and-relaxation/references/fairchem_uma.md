# FairChem UMA backend reference

Use `backend="fairchem_uma"`. The model and device are stage-wide. Use only exact official model names returned by `enabled_models`: `uma-s-1p2`, `uma-s-1p1`, or `uma-m-1p1`. Physical domain data live under `backend_config.defaults` and `backend_config.items`; item keys are exact paths relative to `input/`.

`inference_settings` accepts `default` or `turbo`. Use `turbo` only for repeated inference with fixed atomic composition, such as one MD lineage; keep `default` for general mixed-structure screening.

Use `omat` for generic inorganic periodic materials, `oc20`/`oc22`/`oc25` for matching catalyst datasets, `odac` for direct-air-capture MOFs, `omc` for molecular crystals, and `omol` for molecules/polymers. `uma-s-1p2` supports all seven; the 1.1 S and M models support only `oc20`, `omat`, `omol`, `odac`, and `omc`. `auto` is not a FairChem task and is rejected.

For `omol`, charge and spin are applied to `atoms.info`. FairChem examples use multiplicity-style values such as spin 1 for singlets and spin 3 for triplets; spin must be positive. Non-`omol` tasks require both values to be zero.
