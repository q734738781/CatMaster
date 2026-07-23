# MACE MD reference

MACE backend controls include model/checkpoint, head, dispersion, `default_dtype`, `enable_cueq`, `compile_mode`, and device. A staged checkpoint replaces the registered model. Put it under the stage's `models/` directory and pass its stage-relative path; the submission validator records its SHA-256 and size.

Use cuEquivariance or compilation only after a benchmark on a comparable model, atom count, GPU class, and runtime stack. Do not add a probe run to every high-throughput submission. Record startup overhead and steady-state timing from the production run instead.

For segmented runs, inspect `velocity_source`, `rng_source`, and `integrator_state_source`. Exact continuation claims require compatible stored state, not only matching coordinates.
