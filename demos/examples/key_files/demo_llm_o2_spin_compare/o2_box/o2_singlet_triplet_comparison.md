# O₂ in a vacuum box: singlet vs triplet (VASP relaxation)

This report compares gas-phase O₂ relaxations in a large cubic vacuum box, with the spin state explicitly enforced.

## Run metadata

- **Cell / vacuum box**: cubic, **a = b = c = 21.15093 Å**
- **k-point mesh**: **Γ-only (1×1×1)** (`k_product=1`; gas preset forces 1×1×1)
- **Workflow / presets**: `relax_prepare(calc_type=gas)` (MPRelaxSet-derived)
  - Gas-specific settings include **ISYM=0**, **ISMEAR=0**, **SIGMA=0.01**, **ISIF=2**, **LREAL=False**

### Spin enforcement (key INCAR settings)

- **Singlet** (`o2_box/singlet/INCAR`): `ISPIN=2`, `NUPDOWN=0`, `MAGMOM = 2*0.0`
- **Triplet** (`o2_box/triplet/INCAR`): `ISPIN=2`, `NUPDOWN=2`, `MAGMOM = 2*1.0`

## Results (relaxed)

Energies are the final electronic energies **E0** from the completed relaxations.

| Spin state | Total energy E0 (eV) | Energy / atom (eV/atom) | d(O–O) (Å) |
|---|---:|---:|---:|
| Singlet | -9.503295 | -4.751648 | 1.234044 |
| Triplet | -9.884646 | -4.942323 | 1.233254 |

### Energy difference

- **ΔE = E(singlet) − E(triplet) = +0.381351 eV**
  - The **triplet is lower** in energy by **0.381351 eV** for this setup.

## File locations

- Singlet run directory: `o2_box/singlet/`
- Triplet run directory: `o2_box/triplet/`
- Key outputs used: `OUTCAR`, `CONTCAR` (both runs converged: *"reached required accuracy - stopping structural energy minimisation"*)
