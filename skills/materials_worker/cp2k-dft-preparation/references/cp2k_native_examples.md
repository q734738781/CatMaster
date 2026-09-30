# Native CP2K task and algorithm fragments

These fragments are deliberately incomplete. They show native section placement for one decision at a time; they do not select an XC functional, basis/potential family, grid cutoff, convergence threshold, spin state, periodic cell, k-point density, structure, or complete runnable input. Compose the full input for the actual system and consult [cp2k_numerical_settings.md](cp2k_numerical_settings.md) only when numerical choices need to be made.

## Energy and force task

```text
&GLOBAL
  RUN_TYPE ENERGY_FORCE
&END GLOBAL
```

Use another documented `RUN_TYPE` when the requested observable is different.

## Geometry-optimization task

```text
&GLOBAL
  RUN_TYPE GEO_OPT
&END GLOBAL

&MOTION
  &GEO_OPT
  &END GEO_OPT
&END MOTION
```

Optimization algorithm and stopping criteria belong in the actual method setup; they are intentionally absent here.

## Open-shell declarations

Insert these inside the existing `&DFT` section only when the physical state is open shell:

```text
UKS TRUE
MULTIPLICITY MULTIPLICITY_VALUE
```

Charge, multiplicity, and occupations must describe the intended state; the placeholder is not a suggested multiplicity.

## Periodicity and Poisson fragment

```text
&POISSON
  PERIODIC PERIODICITY
  PSOLVER POISSON_SOLVER
&END POISSON
```

Keep the Poisson choice consistent with the actual `&CELL` periodicity. Select the documented solver and cell dimensions separately for the system.

## Gamma-point OT fragment

```text
&SCF
  &OT
  &END OT
&END SCF
```

This fragment expresses an already chosen OT path; it does not make OT a universal SCF default.

## K-point diagonalization fragment

```text
&KPOINTS
  SCHEME MONKHORST-PACK NX NY NZ
&END KPOINTS

&SCF
  &DIAGONALIZATION
  &END DIAGONALIZATION
  &MIXING
  &END MIXING
&END SCF
```

Do not combine OT with k-point sampling. Select the k-point grid, smearing if any, and mixing controls from the actual electronic system rather than from this syntax fragment.
