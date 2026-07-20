# VASPKIT vs ASE Thermochemistry Fallback

> Validation note for maintainers and advanced auditors. User setup and feature
> guidance is in [Deployment, operations, and security](user-guide/10-deployment-operations.en.md).

This note records how closely the current ASE fallback matches the reference `vaspkit` outputs staged under:

- `reference_scripts/vaspkit_501_502/501_Z5_PX`
- `reference_scripts/vaspkit_501_502/502_PX`

## Scope

- `501`: adsorbate thermochemistry
  - `vaspkit` reference: task `501`
  - ASE fallback: `HarmonicThermo`
  - extra compatibility rule: frequencies below `50 cm^-1` are floored to `50 cm^-1`
- `502`: gas thermochemistry
  - `vaspkit` reference: task `502`
  - ASE fallback: `IdealGasThermo`
  - geometry and rotational symmetry are inferred from the final structure

## ASCII Summary

```text
501  Adsorbate thermo
scale: 1 block ~= 2e-4 eV (for S: 1 block ~= 5e-8 eV/K)

E_ZPE    +0.00164991 eV   +0.03974%   |########
Delta_U  +0.00002779 eV   +0.00055%   |
Delta_H  +0.00002779 eV   +0.00055%   |
Delta_G  +0.00003467 eV   +0.00112%   |
S        -0.00000040 eV/K -0.01266%   |
T*S      -0.00000588 eV   -0.00030%   |

502  Gas thermo
scale: 1 block ~= 2e-5 eV (for S: 1 block ~= 1e-7 eV/K)

E_ZPE    +0.00010279 eV   +0.00250%   |#####
Delta_U  +0.00012329 eV   +0.00249%   |######
Delta_H  +0.00012426 eV   +0.00248%   |######
Delta_G  +0.00004352 eV   +0.00248%   |##
S        +0.00000057 eV/K +0.01089%   |
T*S      +0.00008074 eV   +0.00248%   |####
```

## Detailed Table

```text
501
metric    VASPKIT        ASE            ASE-VASPKIT
E_ZPE     4.151605000    4.153254908    +0.001649908 eV
Delta_U   5.081731000    5.081758785    +0.000027785 eV
Delta_H   5.081731000    5.081758785    +0.000027785 eV
Delta_G   3.099589000    3.099623667    +0.000034667 eV
S         0.003182000    0.003181597    -0.000000403 eV/K
T*S       1.982141000    1.982135119    -0.000005881 eV

502
metric    VASPKIT        ASE            ASE-VASPKIT
E_ZPE     4.105209000    4.105311787    +0.000102787 eV
Delta_U   4.959939000    4.960062289    +0.000123289 eV
Delta_H   5.013624000    5.013748257    +0.000124257 eV
Delta_G   1.756929000    1.756972519    +0.000043519 eV
S         0.005227000    0.005227569    +0.000000569 eV/K
T*S       3.256695000    3.256775738    +0.000080738 eV
```

## Interpretation

- `502` is the better match.
  - The ASE `IdealGasThermo` fallback stays within about `1.3e-4 eV` for the main energy corrections.
- `501` is still usable, but its `E_ZPE` mismatch is noticeably larger.
  - The main gap is about `1.65e-3 eV` in `E_ZPE`.
  - The rest of the thermodynamic terms remain close, within about `3.5e-5 eV`.
- The dominant source of `501` mismatch is the low-frequency treatment.
  - We explicitly reproduce the VASPKIT-style `50 cm^-1` floor.
  - Small residual differences likely come from implementation details inside VASPKIT rather than the high-level thermodynamic model.

## Practical Guidance

- Prefer `vaspkit` when the environment has it.
- Accept ASE fallback for bounded experiment-lane work when `vaspkit` is unavailable.
- When comparing fine free-energy rankings with sub-meV sensitivity, treat fallback-derived `501` values as approximate and keep the backend label in the result summary.
