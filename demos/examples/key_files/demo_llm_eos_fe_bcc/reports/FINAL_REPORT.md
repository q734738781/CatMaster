# Final Report

## User Query
I need you to compute the equation of state diagram for BCC FeDownload the structure from Materials Project (mp-13) and use it as the initial structure.Scan from 8 A^3 to 16 A^3 per atom with 0.2 A^3 intervalPrepare VASP inputs from scratch. Use LREAL=False, disable D3 for accurate bulk energy calculations.Plot your results in a pyplot plot pdf, with x=volume and y=energy per atom, and try to use Birch-Murnaghan equation of state to fit the data, report the fitted parameters.

## Final Answer
## Equation of State (EOS) for BCC Fe (Materials Project mp-13)

### Workflow summary
- **Initial structure:** BCC Fe from Materials Project **mp-13**, downloaded as POSCAR to `eos_fe/mp13_struct/POSCAR` (2 atoms/cell).
- **Reference volume (mp-13):**  
  - Total cell volume: **23.468223 Å³**  
  - Per-atom volume: **11.734111 Å³/atom**
- **Volume scan:** isotropic scaling to target **8.0–16.0 Å³/atom** in **0.2 Å³/atom** steps (**41** structures), written to `eos_fe/scan_structures/V_<vol>/POSCAR`.
- **VASP inputs:** generated from scratch for each volume under `eos_fe/vasp_inputs/V_<vol>/` with:
  - **LREAL = .FALSE.**
  - **D3 disabled** (no dispersion correction)
- **Execution:** VASP ran successfully for all 41 points; outputs are in `eos_fe/vasp_outputs/V_<vol>/`.
- **Post-processing:** energies parsed and consolidated into `eos_fe/eos_data.csv`.

### Key numerical results

#### Minimum sampled energy point (from the discrete scan)
- **Lowest-energy volume point:** **11.4 Å³/atom** (`V_11.4`)
- **Total energy (2-atom cell):** **-16.5375663200 eV**
- **Energy per atom:** **-8.2687831600 eV/atom**

#### Birch–Murnaghan EOS fit (fit to all 41 points; per-atom form)
Fit performed on `eos_fe/eos_data.csv`, parameters saved to `eos_fe/bm_fit.json` and `eos_fe/bm_fit.txt`:

- **Equilibrium volume, V₀:** **11.413776 Å³/atom**
- **Equilibrium energy, E₀:** **-8.267484265 eV/atom**
- **Bulk modulus, B₀:** **1.098076 eV/Å³** = **175.931 GPa**
- **Pressure derivative, B′ (B1):** **5.087611**

### Outputs delivered
- **EOS dataset (volume vs energy/atom):** `eos_fe/eos_data.csv`
- **Birch–Murnaghan fit report:** `eos_fe/bm_fit.txt`
- **Birch–Murnaghan fit parameters (JSON):** `eos_fe/bm_fit.json`
- **Matplotlib EOS plot (PDF; energy/atom vs volume/atom + BM fit overlay):**  
  `eos_fe/fe_bcc_eos.pdf`
