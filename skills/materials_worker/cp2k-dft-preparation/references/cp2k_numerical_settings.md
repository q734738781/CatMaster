# Choosing CP2K numerical settings

The values below are practical starting candidates, not fixed standards, lower bounds, or a bundled "production" preset. CP2K grid convergence depends on the elements, Gaussian basis and pseudopotential, GPW versus GAPW treatment, cell, and requested observable. Use an already validated setting for the same method family when one exists; otherwise select the least expensive candidate that resolves the current scientific decision.

## Ordinary starting candidates

| Control | Practical starting point | Change it when |
| --- | --- | --- |
| `CUTOFF` / `REL_CUTOFF` | For a common GPW calculation with GTH pseudopotentials and MOLOPT basis sets, `400–500 Ry` and `50–60 Ry` are reasonable first candidates. They are not guaranteed values. | A hard pseudopotential, sharp basis functions, GAPW, force/stress-sensitive work, or an actual grid test shows that the target observable is not stable. Values of `600 Ry` or more can be justified in those cases, but are not the default response to the word "accurate". |
| `EPS_SCF` | `1e-6` is a useful routine starting point for ordinary energies, forces, geometry optimization, and Born–Oppenheimer MD. `1e-5` can be adequate for a preliminary or diagnostic stage. | Test `1e-7` for numerical derivatives, frequencies, very small energy differences, or visible force/energy noise. Do not jump to `1e-8` without evidence that the requested result changes at `1e-6` or `1e-7`. |
| `EPS_DEFAULT` | Keep the CP2K default `1e-10` for ordinary Quickstep work. | Change it only when a method-specific source or observed integral/grid truncation requires it. It is independent of `EPS_SCF`; setting both to smaller values is not a generic accuracy upgrade. |
| `MAX_SCF` | The CP2K default `50` is normal capacity. `50–100` is usually enough when a convergent calculation merely needs more room. | If many steps repeatedly reach the cap, diagnose the electronic state, guess, OT versus diagonalization, mixing, smearing, and grid quality. Raising the cap to `200` does not itself improve accuracy. Add an outer loop only for an actual algorithmic need. |
| `GEO_OPT` criteria | CP2K's ordinary `MAX_FORCE 4.5e-4` and `RMS_FORCE 3.0e-4` Ha/Bohr, with `MAX_ITER 200`, are a defensible routine starting point. | A preliminary cleanup may be looser. Tighten only when a downstream frequency, barrier, stress, or other target is demonstrably sensitive; difficult flexible systems may need more iterations without needing tighter force criteria. |

The official defaults (`CUTOFF 280 Ry`, `REL_CUTOFF 40 Ry`, `EPS_SCF 1e-5`) and tutorial outcomes are documented behavior or examples, not transferable validation. Conversely, the common `600/60 Ry` pattern is only a conservative candidate and must not become an automatic minimum.

## Minimal convergence workflow

1. Define the observable and useful resolution before changing parameters: an energy difference, force, optimized geometry, stress, frequency, or MD stability metric. Do not use the number of printed digits in total energy as the target.
2. Reuse a converged setting only when the element set, basis/potential family, method, and observable are sufficiently comparable.
3. For a new decision-sensitive GPW setup, hold `REL_CUTOFF` near `60 Ry`, test increasing `CUTOFF`, then hold the selected `CUTOFF` and test `REL_CUTOFF`. Stop at the lowest pair for which the target observable is stable enough for the stated claim. The official bulk-Si scan found `250/60 Ry` for that example only.
4. Test SCF tolerance separately from the grid. For geometry or MD, inspect force or trajectory stability; for cell optimization inspect stress; for rankings inspect the relevant energy differences. Do not tighten every control together.
5. A one-off demonstration with an established compatible setup does not require a fresh exhaustive sweep. State the adopted setting and its basis; add a targeted higher-setting comparison only when the result is decision-critical or shows sensitivity.

## Failure interpretation

- Reaching `MAX_SCF` means the requested `EPS_SCF` was not met. It is not evidence that the threshold must be tightened.
- If OT struggles, first check whether OT is appropriate for the periodicity and occupations. K-point calculations require diagonalization; metals commonly need suitable orbitals, smearing, and mixing.
- A larger `CUTOFF` with an unchanged, inadequate `REL_CUTOFF` need not improve the result monotonically because Gaussian products can move among grid levels.
- Grid, SCF, geometry, and iteration controls solve different problems. Change the control implicated by the output rather than applying a package of stricter values.

## Primary references

- [CP2K: converging `CUTOFF` and `REL_CUTOFF`](https://manual.cp2k.org/trunk/methods/dft/cutoff.html)
- [CP2K: MGRID keyword meanings and defaults](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/MGRID.html)
- [CP2K: SCF keyword meanings and defaults](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/SCF.html)
- [CP2K: Quickstep `EPS_DEFAULT`](https://manual.cp2k.org/trunk/CP2K_INPUT/FORCE_EVAL/DFT/QS.html)
- [CP2K: SCF convergence diagnosis](https://manual.cp2k.org/trunk/methods/dft/convergence.html)
- [CP2K: geometry and cell optimization](https://manual.cp2k.org/trunk/methods/optimization/geometry_and_cell_opt.html)
