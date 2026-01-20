# Final Report

## User Query
Compute surface energies of bcc Fe (100),(110),(111) facet of conventional cell and determine CO adsorption energy with best adsorption site on the most stable surfaceDownload the proper structure from Materials Project and use it as the initial structure. Get the relaxed structure under our calculation creteria.User enough thickness for slab construction (12A for slab and 15A for vacuum), and fix center part of slabs for surface for surface energy calculations.Then generate all possible adsorption structures on the most stable surface and calculate the adsorption energy for each site using the 2x2 supercell slab to avoid the pbc adsorbate co-interaction effect. Remember to fix the bottom atoms for adsorption calculation.Perform VASP calculation to get the results. Summarize the results in a markdown file, report the surface energies and the best site for CO adsorption geometry fileWrite down your proposed calculation parameters in the plan tasks for review.

## Final Answer
## Report: bcc Fe surface energies ((100),(110),(111)) and CO adsorption on the most stable surface

All calculations and deliverables are organized under `fe_surfaces/`. The final human-readable markdown report is at:

- `fe_surfaces/results/summary.md`

Machine/tabular outputs:

- Surface energies table: `fe_surfaces/results/surface_energies.tsv`
- Most-stable facet pointer: `fe_surfaces/results/most_stable_surface.json`
- CO adsorption energies table (all sites): `fe_surfaces/results/co_adsorption_energies.tsv`
- Best relaxed CO@slab geometry: `fe_surfaces/results/best_CO_on_Fe_110_ontop_0.vasp`

---

## 1) Initial structure (Materials Project) and bulk relaxation

- Material: bcc Fe, Im-3m (229), Materials Project ID **mp-13**
- Downloaded POSCAR: `fe_surfaces/00_mp/mp-13.vasp`
- Conventional standardized cell (2 atoms): `fe_surfaces/01_bulk/Fe_conv.vasp`
  - Initial lattice parameter: **a = b = c = 2.863035498949916 Å** (cubic)

Bulk VASP relaxation:
- Inputs: `fe_surfaces/01_bulk/vasp_in/Fe_conv/`
- Outputs (relaxed bulk used as reference): `fe_surfaces/01_bulk/vasp_out/Fe_conv/` (see `CONTCAR`, `vasprun.xml`)

Key bulk INCAR settings (from generated input):
- `ENCUT=520`, `ISPIN=2`, `ISIF=3`, `EDIFF=1e-6`, `EDIFFG=-0.02`, `NSW=500`, `IBRION=2`
- Smearing: `ISMEAR=1`, `SIGMA=0.2`
- Dispersion: D3 via `IVDW=11`
- POTCAR: `Fe_pv` (PAW_PBE Fe_pv (02Aug2007))
- KPOINTS: Γ-centered **17×17×17** (k_product = 45)

---

## 2) Slab construction and surface-energy workflow

Slabs were constructed with:
- **slab thickness = 12 Å**
- **vacuum thickness = 15 Å**
- orthogonalized cells, symmetric slabs (`get_symmetry_slab=True`, `orthogonal=True`)

Raw slabs (one symmetric termination each, t0):
- Fe(100): `fe_surfaces/02_slabs_raw/Fe_100/t0/POSCAR`
  - natoms = **10**, surface area A = **7.869936050638836 Å²**
- Fe(110): `fe_surfaces/02_slabs_raw/Fe_110/t0/POSCAR`
  - natoms = **7**, surface area A = **5.564885148911197 Å²**
- Fe(111): `fe_surfaces/02_slabs_raw/Fe_111/t0/POSCAR`
  - natoms = **16**, surface area A = **13.631129092024414 Å²**

Center constraints for surface energies (to mimic bulk-like interior):
- A **4 Å mid-slab slice** (z_mid ± 2 Å) was fixed for each slab:
  - Fe(100): 2 frozen / 8 relaxed atoms → `fe_surfaces/03_slabs_fixed/Fe_100_centerfixed.vasp`
  - Fe(110): 3 frozen / 4 relaxed atoms → `fe_surfaces/03_slabs_fixed/Fe_110_centerfixed.vasp`
  - Fe(111): 4 frozen / 12 relaxed atoms → `fe_surfaces/03_slabs_fixed/Fe_111_centerfixed.vasp`

Slab relaxations:
- Inputs: `fe_surfaces/04_surface_vasp_in/Fe_*_centerfixed/`
- Outputs: `fe_surfaces/05_surface_vasp_out/Fe_*_centerfixed/` (see `vasprun.xml`, `CONTCAR`)

Surface energies were computed as:
\[
\gamma = \frac{E_{\mathrm{slab}} - N\cdot E_{\mathrm{bulk/atom}}}{2A}
\]
(using the relaxed bulk reference from `fe_surfaces/01_bulk/vasp_out/Fe_conv/vasprun.xml` and each relaxed slab’s `vasprun.xml`).

### Surface energy results and most stable facet
The computed surface energies and underlying energies/areas are tabulated here:
- `fe_surfaces/results/surface_energies.tsv`

**Most stable facet identified:** **Fe(110)**  
(see `fe_surfaces/results/most_stable_surface.json`)

---

## 3) CO adsorption study on the most stable surface (Fe(110))

### Clean adsorption slab (2×2 supercell to reduce CO–CO PBC interactions)
Built from the **relaxed bulk** structure (`fe_surfaces/01_bulk/vasp_out/Fe_conv/CONTCAR`) with:
- Fe(110), **2×2** in-plane supercell
- slab thickness = **12 Å**, vacuum = **15 Å**
- **bottom 3 atomic layers fixed** (adsorption calculations)

Clean slab geometry:
- `fe_surfaces/06_adsorption/slab_clean/Fe_110_2x2_bottomfixed.vasp`

Clean slab VASP run:
- Inputs: `fe_surfaces/06_adsorption/vasp_in/slab_clean/`
- Outputs: `fe_surfaces/06_adsorption/vasp_out/slab_clean/`

### CO(g) reference
- Structure: `fe_surfaces/06_adsorption/co_gas/CO.vasp` (also `CO.xyz`)
- VASP inputs/outputs:
  - Inputs: `fe_surfaces/06_adsorption/vasp_in/CO_gas/`
  - Outputs: `fe_surfaces/06_adsorption/vasp_out/CO_gas/`

### Enumerated adsorption sites on Fe(110) 2×2
Generated adsorption structures (mode=`all`, initial placement distance = 2.0 Å):
- `fe_surfaces/06_adsorption/slab_ads_structures/ontop_0.vasp`
- `fe_surfaces/06_adsorption/slab_ads_structures/bridge_0.vasp`
- `fe_surfaces/06_adsorption/slab_ads_structures/bridge_1.vasp`
- `fe_surfaces/06_adsorption/slab_ads_structures/hollow_0.vasp`
- Index: `fe_surfaces/06_adsorption/slab_ads_structures/batch_structures.json`

Each site was relaxed with VASP:
- Inputs: `fe_surfaces/06_adsorption/vasp_in/{ontop_0,bridge_0,bridge_1,hollow_0}/`
- Outputs: `fe_surfaces/06_adsorption/vasp_out/{ontop_0,bridge_0,bridge_1,hollow_0}/`

Adsorption energies computed as:
\[
E_{\mathrm{ads}} = E_{\mathrm{slab+CO}} - E_{\mathrm{slab(clean)}} - E_{\mathrm{CO(g)}}
\]

### CO adsorption results (energies and ranking)
All computed adsorption energies (including the individual total energies used) are tabulated here:
- `fe_surfaces/results/co_adsorption_energies.tsv`

**Best (most exothermic) adsorption site:** **ontop_0 on Fe(110)**  
Best relaxed adsorption geometry (CONTCAR copied to results):
- `fe_surfaces/results/best_CO_on_Fe_110_ontop_0.vasp`

---

## 4) Where to find the key relaxed structures
- Relaxed bulk Fe (reference): `fe_surfaces/01_bulk/vasp_out/Fe_conv/CONTCAR`
- Relaxed slabs for surface energies:
  - `fe_surfaces/05_surface_vasp_out/Fe_100_centerfixed/CONTCAR`
  - `fe_surfaces/05_surface_vasp_out/Fe_110_centerfixed/CONTCAR`
  - `fe_surfaces/05_surface_vasp_out/Fe_111_centerfixed/CONTCAR`
- Relaxed clean adsorption slab:
  - `fe_surfaces/06_adsorption/vasp_out/slab_clean/CONTCAR`
- Best relaxed CO adsorption structure:
  - `fe_surfaces/results/best_CO_on_Fe_110_ontop_0.vasp`

---

For the exact numerical surface energies and adsorption energies, please use:
- `fe_surfaces/results/surface_energies.tsv`
- `fe_surfaces/results/co_adsorption_energies.tsv`
- and the consolidated write-up `fe_surfaces/results/summary.md`
