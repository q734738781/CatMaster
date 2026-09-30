# Final Report

## User Query
Compute surface energies of some common surfaces (100,110,111) on BCC Fe surface:Download thestructure from Materials Project (Fe bcc phase) and use it as the initial structure.Use with at least 15A slab thickness and 15A vacuum thickness and construct symmetric slab model for surface energy calculations.Perform VASP calculation to get the results. Report the surface energies of different surfaces and the relevant surface energy ranking. Report your proposed calculation parameters in the plan tasks for review.

## Final Answer
## Scientific report: BCC Fe surface energies for (100), (110), (111) using VASP

### Objective
Compute surface energies of common low-index surfaces of BCC Fe—(100), (110), and (111)—starting from a Materials Project BCC Fe structure, using symmetric slabs with **≥15 Å slab thickness** and **≥15 Å vacuum**, and performing VASP relaxations for bulk and slabs.

---

## Plan / proposed calculation parameters (for review)
1. **Bulk reference (BCC Fe)**
   - Source structure: Materials Project **mp-13** (`fe_bcc_mp/POSCAR`, selection record: `fe_bcc_mp_search.csv`)
   - Relaxation type: full lattice relaxation (cell + ions)
   - Core parameters:
     - `ENCUT = 520 eV`, `PREC = Accurate`
     - Spin-polarized: `ISPIN = 2`, initial `MAGMOM = 5.0 μB/Fe` (uniform)
     - Smearing: `ISMEAR = 1`, `SIGMA = 0.2 eV`
     - Convergence: `EDIFF = 1e-6`, `EDIFFG = -0.02 eV/Å`
     - Ionic relaxation: `IBRION = 2`, `NSW = 500`
     - `LREAL = False`, `LASPH = True`, `LWAVE = False`, `LCHARG = False`
   - k-mesh strategy: Gamma-centered mesh via `k_product = 40` → **15×15×15** for bulk

2. **Surface slabs**
   - Build **symmetric** slabs for (100), (110), (111)
   - Geometry: **15 Å slab thickness + 15 Å vacuum thickness**
   - Slab relaxations (ions only; in-plane cell fixed by slab construction)
   - Core parameters: same as bulk plus `ISYM = 0` (robust for slabs)
   - k-mesh strategy: Gamma-centered with `k_product = 40`, with **kz = 1** enforced

3. **Surface energy evaluation**
   - Use relaxed bulk energy per atom \(E_{\mathrm{bulk}}\) and relaxed slab total energy \(E_{\mathrm{slab}}\):
     \[
     \gamma = \frac{E_{\mathrm{slab}} - N E_{\mathrm{bulk}}}{2A}
     \]
     where \(N\) is number of atoms in slab, \(A\) is surface area of one face, factor 2 accounts for two equivalent surfaces (symmetric slab).
   - Report \(\gamma\) in **J/m²**.

---

## Executed workflow and key outputs

### 1) Bulk BCC Fe relaxation (reference state)
- Inputs/outputs: `vasp_inputs/fe_bcc_lattice_relax/`
- Convergence: electronic + ionic converged (from `vasp_inputs/fe_bcc_lattice_relax/vasprun.xml`)
- Ionic steps: **3**
- Final total energy (2-atom cell): **E_bulk_cell = -16.53908305 eV**
- Energy per atom: **E_bulk_atom = -8.269541525 eV/atom**
- Final relaxed lattice parameter: **a = 2.83117801 Å**
- Relaxed structure saved to: `fe_bcc_relaxed/POSCAR` (bulk energy summary: `fe_bcc_relaxed/bulk_energy.json`)

### 2) Symmetric slab construction (15 Å slab / 15 Å vacuum)
- Slab POSCARs:
  - (100): `fe_slabs/100/POSCAR`
  - (110): `fe_slabs/110/POSCAR`
  - (111): `fe_slabs/111/POSCAR`
- Slab metadata used in surface-energy evaluation:
  - (100): surface area **A = 8.0155689282 Å²**, **N = 12**
  - (110): surface area **A = 5.6678631442 Å²**, **N = 8**
  - (111): surface area **A = 13.8833726352 Å²**, **N = 20**

### 3) Slab relaxations (VASP)
- Inputs: `vasp_inputs/fe_bcc_slab_relax/{100,110,111}/`
- Outputs: `vasp_outputs/fe_bcc_slab_relax_batch/{100,110,111}/`
- Convergence: all slabs electronic + ionic converged (from each `vasprun.xml`)
- Final slab energies and relaxation length:
  - (100): **E_slab = -96.66824158 eV**, ionic steps **11**
  - (110): **E_slab = -64.39907216 eV**, ionic steps **5**
  - (111): **E_slab = -160.60660457 eV**, ionic steps **10**

---

## Surface energies (final results)
Using \(E_{\mathrm{bulk,atom}} = -8.269541525\) eV/atom and the symmetric-slab formula above:

| Surface | N (atoms) | A (Å²) | E_slab (eV) | γ (J/m²) |
|---|---:|---:|---:|---:|
| (100) | 12 | 8.0155689243 | -96.66824158 | **2.56475653** |
| (110) | 8 | 5.6678631549 | -64.39907216 | **2.48368821** |
| (111) | 20 | 13.8833726285 | -160.60660457 | **2.76055941** |

(Complete markdown summary also saved in `fe_surface_energies/report.md`.)

---

## Surface-energy ranking (lowest → highest)
\[
\gamma(110)=2.48368821 < \gamma(100)=2.56475653 < \gamma(111)=2.76055941 \;\; \text{J/m}^2
\]

---

## Notes / reproducibility pointers
- Bulk reference run: `vasp_inputs/fe_bcc_lattice_relax/`
- Slab input sets: `vasp_inputs/fe_bcc_slab_relax/100`, `vasp_inputs/fe_bcc_slab_relax/110`, `vasp_inputs/fe_bcc_slab_relax/111`
- Slab relaxed outputs (energies, convergence, final structures): `vasp_outputs/fe_bcc_slab_relax_batch/*/vasprun.xml`, `OUTCAR`, `CONTCAR`
- Final surface-energy report: `fe_surface_energies/report.md`
