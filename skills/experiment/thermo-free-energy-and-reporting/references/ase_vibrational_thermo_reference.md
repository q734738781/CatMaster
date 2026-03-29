# ASE Vibrational Thermochemistry Reference

This reference embeds the current ASE thermochemistry implementation pattern used in CatMaster, so the agent can reuse the convention locally without reading source files outside the workspace.

## Principle

- `mace_analyze_frequencies` gives vibrational frequencies and modes, not thermodynamic state functions.
- For adsorbate or fixed-slab vibrational thermochemistry, use ASE `HarmonicThermo`.
- For gas-phase molecules, use ASE `IdealGasThermo`.
- Keep the same convention across all compared entries.

## Key constants

```python
_VASP_ATM_IN_PA = 101325.0
_CM1_TO_EV = 1.0 / 8065.54429
_ADSORBATE_FREQ_FLOOR_EV = 50.0 * _CM1_TO_EV
```

## Adsorbate / slab implementation pattern

```python
from ase.thermochemistry import HarmonicThermo

floored_vib_energies = [max(value, _ADSORBATE_FREQ_FLOOR_EV) for value in vib_energies]

thermo = HarmonicThermo(
    vib_energies=floored_vib_energies,
    potentialenergy=0.0,
    ignore_imag_modes=False,
)
zpe = float(thermo.get_ZPE_correction())
internal_energy = float(thermo.get_internal_energy(temperature_k, verbose=False))
entropy = float(thermo.get_entropy(temperature_k, verbose=False))
free_energy = float(thermo.get_helmholtz_energy(temperature_k, verbose=False))
```

Use this when only adsorbate vibrational degrees of freedom are active. Frequencies below `50 cm^-1` are floored before building `HarmonicThermo`.

## Gas-phase implementation pattern

```python
from ase import units
from ase.thermochemistry import IdealGasThermo

pressure_pa = float(pressure_atm) * _VASP_ATM_IN_PA
spin_quantum_number = max(0.0, (float(spin_multiplicity) - 1.0) / 2.0)

thermo = IdealGasThermo(
    vib_energies=vib_energies,
    potentialenergy=0.0,
    atoms=atoms,
    geometry=geometry,
    symmetrynumber=symmetry_number,
    spin=spin_quantum_number,
    ignore_imag_modes=False,
)
zpe = float(thermo.get_ZPE_correction())
enthalpy = float(thermo.get_enthalpy(temperature_k, verbose=False))
entropy = float(thermo.get_entropy(temperature_k, pressure_pa, verbose=False))
gibbs_energy = float(thermo.get_gibbs_energy(temperature_k, pressure_pa, verbose=False))
internal_energy = enthalpy - float(units.kB) * float(temperature_k)
```

Use this for isolated gas molecules with ideal-gas translational and rotational contributions included.

## Geometry / symmetry helpers for gas molecules

```python
def infer_gas_geometry(atoms):
    if len(atoms) <= 1:
        return "monatomic"
    inertias = sorted(float(value) for value in atoms.get_moments_of_inertia())
    largest = max(inertias[-1], 1.0)
    if inertias[0] / largest < 1.0e-3 and inertias[1] / largest > 1.0e-3:
        return "linear"
    return "nonlinear"
```

```python
def infer_symmetry_number(atoms):
    try:
        from pymatgen.io.ase import AseAtomsAdaptor
        from pymatgen.symmetry.analyzer import PointGroupAnalyzer

        molecule = AseAtomsAdaptor.get_molecule(atoms)
        analyzer = PointGroupAnalyzer(molecule)
        symmetry_number = int(analyzer.get_rotational_symmetry_number())
        point_group = str(analyzer.sch_symbol)
        if symmetry_number >= 1:
            return symmetry_number, point_group
    except Exception:
        pass
    return 1, None
```

## Using MACE frequencies

If MACE exported frequencies in `cm^-1`, convert them to `eV` before building ASE thermochemistry:

```python
vib_energies = [float(freq_cm1) * _CM1_TO_EV for freq_cm1 in frequencies_cm1]
```

Then:
- adsorbate/slab: feed them into `HarmonicThermo`
- gas molecule: feed them into `IdealGasThermo`

## Scope

- This is a local reference convention for post-processing.
- It does not imply that MACE frequencies are quantitatively interchangeable with converged DFT frequencies.
- Report whether the thermochemistry came from VASPKIT-native output or the ASE-style fallback convention.
