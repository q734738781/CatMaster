# Fe surfaces + CO adsorption summary

## 1) Starting structure (Materials Project) + bulk relaxation settings

- **Materials Project ID:** **mp-13** (bcc Fe, space group **Im\-3m (229)**)
- **Downloaded structure:** `fe_surfaces/00_mp/mp-13.vasp`
- **Bulk reference relaxation output used for energetics:** `fe_surfaces/01_bulk/vasp_out/Fe_conv/vasprun.xml`

**Bulk relaxation input recipe** (constraint: `calc_type='lattice'`, `k_product=45`, D3 on, DFT+U off):

- `relax_prepare(calc_type='lattice', k_product=45, use_d3=True, use_dft_plus_u=False)`
- Key INCAR settings (bulk):
  - `ENCUT = 520`
  - `ISPIN = 2`, `MAGMOM = 2*2.2` (Fe)
  - `ISIF = 3`, `IBRION = 2`, `NSW = 500`
  - `EDIFF = 1e-6`, `EDIFFG = -0.02`
  - `ISMEAR = 1`, `SIGMA = 0.2`
  - `ISYM = 0`
  - D3 dispersion: `IVDW = 11`
  - `LASPH = True`, `LWAVE = False`, `LCHARG = False`
- KPOINTS: Γ-centered **17×17×17** (from `k_product=45` on the ~2.86 Å cubic cell)
- POTCAR: **PAW_PBE Fe_pv (02Aug2007)**

Bulk reference energy extracted from the relaxed bulk calculation:

- \(E_\mathrm{bulk}\) = \(-17.16603492\) eV for \(N=2\) atoms
- \(E_\mathrm{bulk}^{\mathrm{(per\ atom)}}\) = **\(-8.58301746\) eV/atom**

---

## 2) Surface energies for Fe(100)/(110)/(111)

Surface energies computed from relaxed symmetric slabs using:

\[
\gamma = \frac{E_\mathrm{slab} - N E_\mathrm{bulk}^{\mathrm{(per\ atom)}}}{2A}
\]

- Slab builds: `slab_thickness=12 Å`, `vacuum_thickness=15 Å`, `get_symmetry_slab=True`, `orthogonal=True`
- Slab relaxations: `relax_prepare(calc_type='slab', k_product=35, use_d3=True)` with `ISIF=2` and **kz=1**

| Facet | Surface energy γ (eV/Å²) | Surface energy γ (J/m²) |
|---:|---:|---:|
| (100) | 0.20098831 | 3.2202 |
| (110) | 0.19401061 | 3.1084 |
| (111) | 0.20237527 | 3.2424 |

(Relaxed slab outputs: `fe_surfaces/05_surface_vasp_out/Fe_100_centerfixed/vasprun.xml`, `.../Fe_110_centerfixed/vasprun.xml`, `.../Fe_111_centerfixed/vasprun.xml`.)

---

## 3) Most stable surface

- **Most stable facet (lowest γ): Fe(110)**
- \(\gamma_\mathrm{min}\) = **0.19401061 eV/Å² = 3.1084 J/m²**

---

## 4) CO adsorption energies on Fe(110) (2×2 slab)

**Most-stable facet used:** Fe(110)

**Adsorption model details**

- Slab construction (constraint): built from relaxed bulk `fe_surfaces/01_bulk/vasp_out/Fe_conv/CONTCAR` with
  - `slab_thickness=12 Å`, `vacuum_thickness=15 Å`, `orthogonal=True`, `supercell=[2,2,1]`
- Clean slab constraints: bottom **3** layers fixed (`freeze_layers=3`, `layer_tol=0.2 Å`)
- Site enumeration: `mode='all'`, distance = **2.0 Å**
- Adsorption-stage relaxations: `k_product=25`, D3 (`IVDW=11`), dipole correction (`LDIPOL=True`, `IDIPOL=3`, `DIPOL=[0.5,0.5,0.5]`)

**Adsorption energy definition**

\[
E_\mathrm{ads} = E_{\mathrm{slab+CO}} - E_{\mathrm{slab}} - E_{\mathrm{CO(g)}}
\]

(So **more negative = stronger binding**.)

Reference energies (final electronic energies from `vasprun.xml`):

- Clean slab (Fe(110) 2×2): `fe_surfaces/06_adsorption/vasp_out/slab_clean/vasprun.xml` → \(E_{\mathrm{slab}}\)= \(-231.70830540\) eV
- CO(g): `fe_surfaces/06_adsorption/vasp_out/CO_gas/vasprun.xml` → \(E_{\mathrm{CO}}\)= \(-14.79176072\) eV

### Adsorption energies by site (Fe(110) 2×2)

| Site label | E(slab+CO) (eV) | E_ads (eV) |
|---|---:|---:|
| bridge_0 | -248.67204252 | -2.171976 |
| bridge_1 | -248.57588503 | -2.075819 |
| hollow_0 | -248.67265221 | -2.172586 |
| ontop_0 | -248.67326367 | **-2.173198** |

---

## 5) Best adsorption site + best relaxed geometry path

- **Best site:** `ontop_0`
- **Best adsorption energy:** **E_ads = -2.173198 eV**
- **Best relaxed geometry (CONTCAR):** `fe_surfaces/06_adsorption/vasp_out/ontop_0/CONTCAR`
