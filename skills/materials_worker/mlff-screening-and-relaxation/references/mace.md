# MACE backend reference

Use `backend="mace"`. `backend_config` owns `model` or the mutually exclusive staged `checkpoint_artifact`, plus `head`, `dispersion`, `default_dtype`, `enable_cueq`, `compile_mode`, and `device`. Put staged checkpoints under the stage's `models/` directory and pass a stage-relative path; the submission validator records the file SHA-256 and size.

`enable_cueq` is allowed only for relaxation and MD in the current managed path. `compile_mode` is allowed only for MD. Keep both disabled unless the target GPU stack has a relevant benchmark or the user explicitly requests an acceleration experiment.

MACE model/head/dispersion/precision must be held fixed across structures used in one quantitative ranking.
