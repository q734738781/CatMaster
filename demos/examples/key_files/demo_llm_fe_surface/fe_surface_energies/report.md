# Fe (bcc) surface energies: (100), (110), (111)

Surface energies computed from relaxed *symmetric* slabs using:

\[ \gamma = \frac{E_\mathrm{slab} - N\,E_\mathrm{bulk/atom}}{2A} \]

- \(E_\mathrm{slab}\): final relaxed slab total energy (eV)
- \(N\): number of atoms in slab
- \(E_\mathrm{bulk/atom}\): bulk reference energy per atom (eV/atom)
- \(A\): in-plane surface area from slab lattice (Å²)
- factor **2**: two identical surfaces in symmetric slab
- unit conversion: \(1\,\mathrm{eV/\AA^2}=16.02176565\,\mathrm{J/m^2}\)

## Bulk reference
- Source: `vasp_inputs/fe_bcc_lattice_relax/vasprun.xml`
- Relaxed bcc Fe (Im-3m)
- Final bulk energy per atom:
  - **E_bulk_atom = -8.269541525 eV/atom**

## Results
| Surface | N atoms | A (Å²) | E_slab (eV) | γ (J/m²) |
|---|---:|---:|---:|---:|
| (100) | 12 | 8.0155689243 | -96.66824158 | **2.56475653** |
| (110) | 8 | 5.6678631549 | -64.39907216 | **2.48368821** |
| (111) | 20 | 13.8833726285 | -160.60660457 | **2.76055941** |

### Ranking (lowest → highest γ)
1. **(110)**: 2.48368821 J/m²
2. **(100)**: 2.56475653 J/m²
3. **(111)**: 2.76055941 J/m²

## Actual run parameters used
Slabs were generated as **symmetric** slabs with target **slab_thickness = 15 Å** and **vacuum_thickness = 15 Å** (build settings). Below are the *actual* parameters in the VASP runs plus the *measured* slab/vacuum thickness from the final relaxed structures.

Common VASP settings (all surfaces):
- ENCUT = 520 eV
- PREC = Accurate
- ISPIN = 2 (spin-polarized)
- MAGMOM = 5.0 μB per Fe (uniform initial moments)
- ISMEAR = 1, SIGMA = 0.2 eV
- EDIFF = 1e-6 eV
- EDIFFG = -0.02 eV/Å
- NSW = 500, IBRION = 2
- LREAL = False
- LASPH = True
- ISYM = 0
- k-point generation: **k_product = 40**, Γ-centered; slabs enforce **kz = 1**

### (100)
- Output: `vasp_outputs/fe_bcc_slab_relax_batch/100/vasprun.xml`
- KPOINTS mesh: **15×15×1** (k_product estimates: a≈42.47, b≈42.47)
- Final thickness (from relaxed structure):
  - slab ≈ **15.587 Å**
  - vacuum ≈ **18.387 Å**

### (110)
- Output: `vasp_outputs/fe_bcc_slab_relax_batch/110/vasprun.xml`
- KPOINTS mesh: **17×17×1** (k_product estimates: a≈41.68, b≈41.68)
- Final thickness (from relaxed structure):
  - slab ≈ **14.012 Å**
  - vacuum ≈ **18.019 Å**

### (111)
- Output: `vasp_outputs/fe_bcc_slab_relax_batch/111/vasprun.xml`
- KPOINTS mesh: **11×11×1** (k_product estimates: a≈44.04, b≈44.04)
- Final thickness (from relaxed structure):
  - slab ≈ **29.636 Å**
  - vacuum ≈ **26.987 Å**
