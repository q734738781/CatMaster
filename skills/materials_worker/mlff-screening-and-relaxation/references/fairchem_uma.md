# FairChem UMA backend reference

Use `backend="fairchem_uma"`. The enabled model alias and device are stage-wide. Physical domain data live under `backend_config.defaults` and `backend_config.items`; item keys are exact paths relative to `input/`.

`inference_settings` accepts `default` or `turbo`. Use `turbo` only for repeated inference with fixed atomic composition, such as one MD lineage; keep `default` for general mixed-structure screening.

Use `omat` for generic inorganic periodic materials, `oc20`/`oc22`/`oc25` for matching catalyst datasets, `odac` for direct-air-capture MOFs, `omc` for molecular crystals, and `omol` for molecules/polymers. `auto` only distinguishes periodic from nonperiodic inputs; it does not infer catalyst semantics.

For `omol`, charge and spin are applied to `atoms.info`. FairChem examples use multiplicity-style values such as spin 1 for singlets and spin 3 for triplets. Non-`omol` tasks require both values to be zero.
