# Choosing LAMMPS minimization criteria

There is no transferable LAMMPS tuple for `etol ftol maxiter maxeval`. The numerical meaning depends on `units`, the chosen force norm, atom count, potential smoothness, long-range solver accuracy, constraints, and what the relaxed structure will be used for. The bands below are CatMaster starting candidates, not LAMMPS defaults or fixed scientific standards.

## Select the criterion before the number

1. Read the active `units`. In `metal`, force is eV/Å; in `real`, it is kcal mol⁻¹ Å⁻¹; in `lj`, it is reduced force. Never copy the same numeric `ftol` across unit styles.
2. Match the norm to the claim. The default `min_modify norm two` checks the Euclidean norm of the entire `3N` force vector and becomes increasingly stringent as the system grows. With a minimizer that supports selectable norms (`cg`, `sd`, `quickmin`, `fire`, and the documented spin variants), use `min_modify norm max` if acceptance is stated as a maximum force on any atom; use `norm inf` only when the intended criterion is maximum Cartesian component.
3. Remember that any stopping condition can end the run. For a force-targeted geometry relaxation, `etol 0.0` avoids an unrelated relative-energy test stopping before the force target. Keep a nonzero `etol` only when relative energy change is itself the intended criterion, and still inspect the final forces.
4. Treat `maxiter` and `maxeval` as capacity. A small system may need only hundreds of iterations; a complex relaxation may reasonably start with about `1000/10000`. Increase them only after a cap-limited run is still making scientifically useful progress.

## Task-dependent starting bands

These bands assume `min_modify norm max`. They express maximum per-atom force-vector targets, not the default global 2-norm.

| Potential and task | Candidate force target | Interpretation |
| --- | --- | --- |
| Smooth analytic empirical potential in `metal`; ordinary static minimum | Start near `1e-6 eV/Å` | Appropriate when locating the potential's mathematical minimum is inexpensive. Test `1e-7–1e-8 eV/Å` only for a sensitive derivative, elastic, phonon, or tiny energy-difference result that actually changes at `1e-6`. |
| Smooth analytic empirical potential; overlap cleanup or pre-MD settling | About `1e-4–1e-3 eV/Å` | This is a preparation stage, not a final zero-temperature structure. Tighten only if residual forces disturb the intended MD initialization. |
| ML potential in `metal`; screening or routine relaxation | About `0.02–0.05 eV/Å` for screening, and `0.005–0.02 eV/Å` for a refined surrogate-PES geometry | Pick the band from the downstream decision and the model's validated domain. A smaller residual force solves the learned PES more closely but does not make the model itself more accurate. |
| Molecular mechanics in `real`; ordinary small-molecule refinement | About `0.01–0.1 kcal mol⁻¹ Å⁻¹` | Use the looser end for preparation and the tighter end when geometry sensitivity warrants it. Do not import a `metal`-unit value unchanged. |
| Discontinuous/tabulated/cutoff-sensitive potential or approximate long-range solver | No generic lower bound | First establish the numerical floor from the pair style, table resolution, neighbor/cutoff behavior, and Kspace tolerance. A requested `ftol` below that floor will not create a better minimum. |

The large difference between the analytic-potential and ML-potential bands is intentional. Numerical minimization error and model error are different quantities. An analytic potential can often be minimized very tightly at low cost; an ML potential used for screening is usually judged by whether tighter relaxation changes the scientific selection, not by forcing its residual force to `1e-8 eV/Å`.

## Recommended input pattern

For a force-targeted relaxation whose acceptance criterion is maximum force per atom:

```text
min_style cg
min_modify norm max
thermo_style custom step pe fnorm fmax
minimize 0.0 FTOL MAXITER MAXEVAL
```

Replace the uppercase fields from the actual unit style, potential class, system, and task. `thermo` keyword `fmax` is the largest Cartesian force component, whereas `min_modify norm max` uses the largest per-atom force-vector magnitude. If the exact latter value is needed in output, expose it explicitly:

```text
variable atom_fmag atom sqrt(fx*fx+fy*fy+fz*fz)
compute atom_fmax all reduce max v_atom_fmag
thermo_style custom step pe fnorm fmax c_atom_fmax
```

## Minimal sensitivity check

- Do not run a tolerance ladder for every routine relaxation. Use a compatible validated project setting when available.
- For a decision-sensitive new setup, repeat only the final stage with the relevant force target tightened by roughly one order of magnitude. Accept the cheaper setting when the geometry, energy difference, stress, barrier, or other claimed observable remains stable at the required resolution.
- For an analytic toy problem, derive a useful coordinate or energy tolerance from the requested comparison. Floating-point agreement is not the default target.
- If the run stops by energy change, line-search failure, iteration/evaluation limit, or a FIRE-specific stop rather than the intended force criterion, report that state instead of calling it force-converged.

## Numerical floors and false precision

LAMMPS documents that cutoff discontinuities, splined many-body potentials, and approximate PPPM/Ewald forces can prevent very tight minimization. Align the minimizer tolerance with the potential and long-range-solver accuracy before lowering it. If forces plateau, inspect the stopping reason and potential smoothness rather than repeatedly shrinking `ftol` or enlarging the caps.

## Primary references

- [LAMMPS `minimize`: stopping semantics, monitoring, and numerical floors](https://docs.lammps.org/minimize.html)
- [LAMMPS `min_modify`: `two`, `max`, and `inf` force norms](https://docs.lammps.org/min_modify.html)
- [LAMMPS `units`: force units for each unit style](https://docs.lammps.org/units.html)
- [LAMMPS `thermo_style`: `fnorm` and `fmax`](https://docs.lammps.org/thermo_style.html)
