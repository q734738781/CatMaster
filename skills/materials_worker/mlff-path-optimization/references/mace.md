# MACE path reference

The MACE NEB backend uses the same model/checkpoint, head, dispersion, precision, and device semantics as MLFF relaxation. Keep `default_dtype=float64` for pathway geometry unless the run is explicitly exploratory.

The managed runner uses fixed-image plain ASE NEB and FIRE. Legacy AutoNEB settings are not exposed through `mlff_neb`.
