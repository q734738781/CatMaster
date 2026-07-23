# MACE backend reference

Use `backend="mace"`. Query `get_remote_task_spec` with the intended model before choosing a head. The deployment profile exposes a strict model/head map:

- `model="mh-1"` uses `mace_mp()` and accepts only the heads returned for that checkpoint. `omat_pbe` is the deployment default; `omol` is a head of MH-1, not a separate model.
- `model="omol-0"` uses the standalone `mace_omol()` loader, fixes `head="omol"`, and requires charge plus multiplicity-style spin metadata. Put shared values in `backend_config.defaults`, for example `{"charge": 0, "spin": 3}` for neutral triplet O2, and use `backend_config.items` only for per-input exceptions.

`backend_config` also owns the mutually exclusive staged `checkpoint_artifact`, `dispersion`, `default_dtype`, `enable_cueq`, `compile_mode`, and `device`. Put staged checkpoints under the stage's `models/` directory and pass a stage-relative path; the submission validator records the file SHA-256 and size.

`enable_cueq` is allowed only for relaxation and MD in the current managed path. `compile_mode` is allowed only for MD. Keep both disabled unless the target GPU stack has a relevant benchmark or the user explicitly requests an acceleration experiment.

MACE model/head/dispersion/precision must be held fixed across structures used in one quantitative ranking. Invalid registered model/head pairs are preflight errors. Standalone `omol-0` does not use the `mace_mp` dispersion wrapper.
