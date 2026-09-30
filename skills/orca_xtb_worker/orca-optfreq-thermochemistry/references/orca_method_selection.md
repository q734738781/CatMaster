# ORCA electronic-method selection

Use this as a task-dependent starting policy, not a universal ORCA default. Preserve a method explicitly required by the user, an established project comparison, or authoritative task-specific evidence. Otherwise define the physical observable and electronic-structure regime before choosing a method.

## Functional token spelling

`orca_prepare` preserves native tokens rather than translating functional names.
For the selected method token, use `"LibXC(SCAN)"` for SCAN through ORCA's LibXC
interface, or `"r2SCAN"` for the native r²SCAN functional. These alternatives do
not select a basis, request optimization, or authorize changing the chosen method.
Only the simple-input aliases documented by ORCA support the `LibXC(...)` spelling;
other LibXC functionals use native `%method` syntax. Consult
[ORCA's LibXC input documentation](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/DensityFunctionalTheory.html#simple-input-of-libxc-functionals)
when the chosen functional has an unfamiliar input spelling.

## Routine multi-level starting point

- For routine main-group ground-state geometries and requested harmonic-frequency or thermochemistry work, start from `r2SCAN-3c`. It combines r2SCAN, a tailored triple-zeta basis, D4, and gCP as one composite method and is designed for robust geometries, thermochemistry, and noncovalent interactions at moderate cost.
- For main-group relative energies, reaction barriers, conformational energies, and noncovalent energies, use `WB97M-V/def2-TZVPP` as a strong routine final-DFT candidate, often as a single point on an appropriate lower-cost geometry. The original functional was tested across broad thermochemical and noncovalent data, and later general benchmarks support it as a strong hybrid meta-GGA.
- Keep `WB97X-V` for continuity with an established data set, a property/domain benchmark that favors it, or a justified cost or robustness tradeoff. It remains a strong range-separated hybrid rather than a deprecated fallback.
- Do not select B3LYP as the unprescribed routine starting point, with `def2-SVP` or otherwise, merely because it is familiar or cheap. It remains valid when required for comparison, continuity, or a property-specific benchmark.

This layering is a candidate workflow, not an instruction to add stages. A single-point request remains a single point; geometry optimization, frequencies, and thermochemistry appear only when the requested observable or supplied model requires them.

## Match the method to the property

- **Kohn-Sham orbital energies or gap:** report the selected functional and basis and label the result as a Kohn-Sham eigenvalue difference. Do not present it as an optical excitation or fundamental gap. Select a range-separated or otherwise validated functional for the actual system/property rather than inheriting the routine thermochemistry method without thought.
- **Fundamental gap, ionization, or electron attachment:** prefer a property-appropriate energy-difference treatment such as delta-SCF or a validated electron-removal/addition method. Add diffuse functions for anions and diffuse states.
- **Optical or charge-transfer excitation:** use a validated excited-state method and adequate diffuse basis where needed; TDDFT, STEOM/EOM, and multireference routes answer different regimes.
- **NMR, EPR, and other response properties:** follow property-specific benchmarks, basis requirements, relativistic treatment, solvent model, and referencing. A good energy functional is not automatically a good spectroscopy method.
- **Transition-metal spin energetics, bond breaking, or near degeneracy:** inspect the electronic state and relevant diagnostics. Do not treat either `WB97M-V` or `r2SCAN-3c` as a universal answer, and do not apply single-reference coupled cluster when the reference is qualitatively unsuitable.

## Escalating beyond routine DFT

Escalate only when the decision needs more accuracy than the routine DFT layer can support.

- A modern double hybrid can improve many well-behaved organic energy differences, but it is not a safe default for small-gap or multireference systems.
- `DLPNO-CCSD(T)` with a suitable triple-zeta, CBS, or F12 protocol can provide a high-level single-reference check at larger molecular sizes. Canonical CCSD(T) may be preferable for small benchmark systems. Choose the reference, basis/extrapolation, PNO treatment, and open-shell formulation from a property-appropriate protocol rather than automatically tightening every calculation.
- Use multireference methods only when the physical problem calls for them and the active space and state treatment can be justified.

Higher level means better matched evidence for the target observable, not simply more expensive keywords.

## ORCA composition

- Use native keywords `R2SCAN-3C`, `WB97M-V`, or `WB97X-V` as appropriate.
- `r2SCAN-3c` already owns its tailored basis and corrections. Do not append another basis, D4, or gCP token.
- `WB97M-V` and `WB97X-V` already contain VV10 nonlocal correlation. Do not add `D3`, `D4`, or another dispersion token; ORCA disallows that combination.
- `def2-TZVPP` is a strong routine final single-point basis. `def2-QZVPP`, systematic extrapolation, or another property-specific family may be justified for high-accuracy work. Add diffuse functions for anions, Rydberg states, charge transfer, and other diffuse densities.
- Start ordinary calculations from ORCA's documented normal SCF and grid settings. Tighten or increase them only for a method/property requirement or observed sensitivity.
- Check derivative support for the selected method in the current ORCA manual. Use `Freq` when an analytic Hessian is supported and `NumFreq` otherwise; do not silently change the electronic method merely to obtain frequencies.

## Primary and official references

- [ORCA 6.1 general recommendations: match method and property](https://www.faccts.de/docs/orca/6.1/manual/contents/quickstartguide/recommendations.html)
- [ORCA 6.1 composite methods and `r2SCAN-3c`](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/3cmethods.html)
- [Best-Practice DFT Protocols for Basic Molecular Computational Chemistry](https://doi.org/10.1002/anie.202205735)
- [Original `r2SCAN-3c` development](https://doi.org/10.1063/5.0040021)
- [Original `WB97M-V` development](https://doi.org/10.1063/1.4952647)
- [Original `WB97X-V` development](https://doi.org/10.1063/1.4868117)
- [GSCDB137 general benchmark](https://doi.org/10.1021/acs.jctc.5c01380)
- [ORCA 6.1 coupled-cluster and DLPNO methods](https://www.faccts.de/docs/orca/6.1/manual/contents/modelchemistries/mdci.html)
