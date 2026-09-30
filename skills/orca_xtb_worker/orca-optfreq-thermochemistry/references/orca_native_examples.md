# Native ORCA task-keyword fragments

These fragments are deliberately not complete `orca_prepare` calls. Select the electronic method, basis, dispersion, solvation, charge, and multiplicity independently from the scientific objective, then add only the fragment for the operation the user requested. See [orca_method_selection.md](orca_method_selection.md) for method selection.

## Ordinary single point

Add no task keyword. A complete `simple_keywords` list still needs the independently selected electronic method and basis.

## Analytic gradient

```text
simple_keywords += ["EnGrad"]
```

## Geometry optimization

```text
simple_keywords += ["Opt"]
```

`Opt` requests ordinary geometry optimization. Select a tighter geometry criterion separately only when the scientific target or observed behavior requires it.

## Vibrational frequencies

Use one frequency operation according to the selected method's derivative support:

```text
simple_keywords += ["Freq"]
```

or

```text
simple_keywords += ["NumFreq"]
```

SCF convergence controls are a separate method/property decision; they are not part of the task-keyword example.

## NMR shielding

```text
simple_keywords += ["NMR"]
```

Choose the NMR method, basis, solvent treatment, and reference strategy through the NMR workflow rather than from this keyword fragment.

## TDDFT roots

Add a complete native `%tddft ... end` block containing only the requested state/root controls. Choose the ground-state functional, basis, and solvation separately.

Relaxed scans, transition-state refinement, NEB, and IRC use their dedicated skills so their task-specific blocks are not loaded into ordinary single-point, optimization, or frequency work.
