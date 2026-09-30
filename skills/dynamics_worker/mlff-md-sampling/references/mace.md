# MACE MD reference

MACE backend controls include model/checkpoint, head, dispersion, `default_dtype`, `enable_cueq`, `compile_mode`, and device. A staged checkpoint replaces the registered model. Put it under the stage's `models/` directory and pass its stage-relative path.

Leave cuEquivariance and compilation at the registered defaults for ordinary scientific sampling. Change them only for an explicit acceleration objective or after a concrete performance problem makes that choice relevant; any timing analysis then belongs to that performance question rather than MD acceptance.

For segmented runs, inspect `velocity_source`, `rng_source`, and `integrator_state_source`. Exact continuation claims require compatible stored state, not only matching coordinates.
