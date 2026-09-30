# 6. Modeling and computation capabilities

[Previous](05-agents-and-modules.en.md) | [Contents](README.en.md) | [Next](07-literature-writing-review.en.md)

Chapter 5 described workers. This chapter follows complete research tasks and shows how CatMaster can turn partial inputs into structures, calculation stages, results, and analyses. These are capability stories, not fixed pipelines. The agent adapts to the system, existing files, enabled remote tasks, and intermediate checks.

## Starting from a formula or database record

You may not have a calculation-ready POSCAR. Given a formula, material name, or search criteria, Experiment can query Materials Project, compare phase, structure, stability, and identifier, then download one or more records. It preserves database originals separately from standardized, expanded, or calculation-ready derivatives.

Selecting among polymorphs is a scientific decision. State the phase or space group if it is known, or ask the agent to compare candidates and wait. After selection, Materials can prepare consistent bulk relaxation and static stages with recorded k-point density, pseudopotential, functional, spin, and convergence settings.

```text
I want to study oxygen vacancies at SrTiO3 surfaces, but I do not yet have a structure.
Use Experiment to find candidate bulk records from a reliable database. Compare phase, space group,
stability, and provenance. Do not silently choose one and continue. Save the original candidates and explain
which is a sensible starting point for cubic or low-temperature work. Wait for my choice before creating the
bulk calculation reference.
```

## From bulk to comparable slabs and adsorbates

Surface modeling is a connected capability, not a single slab button. The agent first checks whether the bulk reference is suitable. It then generates terminations for the requested Miller index and applies thickness, vacuum, and lateral-size requirements. Each slab is audited for orientation, top and bottom relation, stoichiometry, polarity, repeated layers, coordination, and fixed atoms, with rendered views when useful.

For adsorption, the agent reads or builds an adsorbate, identifies an anchor and relevant conformations, places it at deduplicated surface sites, and records site, orientation, and initial distance. Starting structures are not labeled stable states. Geometry checks come first, followed by optional MLFF single-point or relaxation screening and DFT preparation for the small candidate set that survives.

You can stop at any review boundary. On a first pass, stop after candidates and audit. Approve screening or DFT only after the structure model is accepted.

```text
Build Pt(111) adsorption candidates for O, OH, and OOH from structures/pt_bulk_relaxed.vasp
for a later ORR free-energy comparison. Ask Materials to combine slab, termination, adsorbate,
site-screening, and visual-inspection skills.

Create a surface with at least 15 angstrom of vacuum and a reasonable coverage, explain the fixed-layer policy,
and preserve constraints. For every intermediate, generate chemically sensible deduplicated sites and orientations,
then inspect periodic contacts and collisions. Save a candidate table, views, and generation settings. You may propose
an MLFF screen, but do not submit anything until I approve the slab, coverage, and candidates.
```

A local cleanup request can be much narrower:

```text
Inspect structures/stepped_ceo2.vasp only for oxygen atoms near the highest surface with coordination one.
Report the neighbor definition, cutoff, atom indices, and heights before changing the structure. Do not choose
atoms from the rendering alone. After I approve the target atoms, create a separate edited structure, preserve
all Selective Dynamics, and compare short contacts, coordination, and stoichiometry before and after.
```

## How far defect and dopant screening can go

CatMaster can enumerate symmetry-distinct sites, generate vacancy, substitution, and explicit interstitial candidates, and place each into a consistent calculation layout. A first screen may compare geometric stability, relative total energy, and local coordination under one bulk reference.

A formal defect-formation-energy study needs chemical-potential bounds, charge states, Fermi level, finite-size and charge corrections, and consistent references. The tools support candidate construction and standardized preparation, while the agent must state the additional physics rather than presenting a set of neutral-supercell energies as complete defect thermodynamics.

```text
Build surface and subsurface oxygen-vacancy candidates in a 3x3x1 anatase TiO2 supercell.
Enumerate symmetry-distinct oxygen sites and explain the grouping before generating neutral defects.
Inspect local coordination, overlaps, and Selective Dynamics, then prepare consistent VASP relaxation stages.

Label this as first-pass structural screening. In a separate section, explain what chemical potentials,
charge states, and corrections would be required for formal formation energies. Do not invent those conditions
or submit calculations in this turn.
```

## From endpoints to pathways, barriers, and free energies

Path calculations fail easily when endpoints or atom mapping are wrong. CatMaster first verifies that endpoints have the same composition, ordering, constraints, and periodic interpretation. It can remap mobile atoms while leaving frozen atoms untouched, estimate image count, create interpolation, and inspect periodic jumps, collisions, and discontinuous rearrangements.

An accepted path may pass through low-cost MLFF NEB, VASP plain NEB, CI-NEB, or dimer refinement. After execution, the agent can extract relative energies and barriers, inspect endpoint reference, forces, and profile shape, and recommend frequency or IRC evidence for a true transition state.

Free-energy work combines electronic energies with ZPE, thermal enthalpy, entropy, temperature, and standard state under one reference. Materials can process gas and adsorbate frequency corrections while retaining each contribution and assumption. It must not subtract values from incompatible methods or reference states.

```text
Compare the surface pathway between structures/co2_initial.vasp and structures/co2_final.vasp.
Audit endpoints, constraints, and atom mapping before creating a continuous, reviewable path.
Recommend an image count from displacement and chemistry and explain whether the current MLFF backend
is suitable for a preliminary optimization.

Do not prepare formal VASP NEB before I approve the path. The final plan should distinguish preliminary
screening, plain NEB, CI-NEB, transition-state validation, and free-energy correction, including what each stage
can and cannot prove.
```

## Band, DOS, phonon, elastic, and thermochemical work

These properties depend on accepted upstream structures and consistent settings. For band and DOS, the agent retains an SCF reference and records k-path provenance, Fermi level, broadening, spin, and projection definitions. `generate_kpath` and `vasp_band_prepare` build the path and input, but interpretation remains system-specific.

Finite-displacement phonons start from a consistent supercell and displacement set. Imaginary modes must be separated into numerical noise, acoustic behavior, inadequate supercell effects, and real instability. Finite-strain elasticity requires controlled strain amplitude, relaxation strategy, symmetry, and fit range. CatMaster can build the stages and collect outputs without reducing reliability to a single fitted number.

Thermochemistry tools compute gas or adsorbate corrections from VASP frequency outputs. The agent separates electronic energy, ZPE, enthalpy, entropy, and free energy and retains temperature, pressure, standard state, and constrained-degree assumptions.

```text
Prepare DOS and band work from calculations/bulk_relax/CONTCAR.
Verify that relaxation converged and that the structure remains a valid bulk reference. Decide which files
and settings must remain consistent across SCF, DOS, and band stages. Record the k-path source.

Generate inputs and an analysis plan without submitting. Explain how Fermi level, spin, projections,
and band gap will be defined, and which conclusions may require a higher-level method.
```

## From an initial structure to an analyzable trajectory

Dynamics normally has preparation, execution, health assessment, and property analysis. Preparation fixes ensemble, temperature, pressure, timestep, duration, thermostat or barostat, velocity source, seed, boundaries, and output interval. A restart also needs continuity of coordinates, velocities, integrator state, and random state. Copying the last frame and assigning new velocities begins a new segment instead of preserving that continuity.

CatMaster can prepare CP2K AIMD, LAMMPS, and MLFF MD. After remote execution, Dynamics checks logs, temperature, energy, volume, force anomalies, structural failure, and frame count. Only then does it select MSD, RDF, coordination, residence time, cluster connectivity, or diffusion analysis.

```text
Continue calculations/aimd_800K_part1/ to a total of 100 ps with CP2K AIMD.
Audit the latest valid restart, velocities, temperature, random state, and last completed step first.
Do not overwrite part1 and do not assign new velocities while calling the run continuous.

Build a separate continuation stage and document how segments will join and which restart files must survive.
Prepare and audit now, then wait for my approval before submitting cp2k_execute.
```

```text
Analyze Pd-cluster evolution in trajectories/pd_tio2_1073K.traj.
Perform a trajectory health audit first. Under periodic boundaries, compute the Pd-Pd contact graph and
connected components over time and also track nearest Pd-support contacts. Justify the cutoff and include
a sensitivity check. Do not infer sintering from a final snapshot alone. Save time series, representative frames,
and a reusable analysis script.
```

## From reference calculations to an evaluated MACE model

Training a potential is not a matter of feeding OUTCAR files into a command. CatMaster can extract structures, energies, forces, and stresses, reject unconverged or incomplete labels, normalize units and references, and create a manifest-backed extxyz dataset with fixed splits.

ML can prepare and run MACE training or fine-tuning, preserve configuration, logs, and checkpoints, then evaluate a held-out test set. Analysis should expose systematic error by element, configuration class, or high-energy region rather than reporting one aggregate number. Active learning can use diversity and committee disagreement to propose reference calculations without automatically authorizing their cost.

```text
Design one MACE active-learning round from calculations/vasp_labels/ and ml/candidate_pool/.
Audit labels, duplicates, element and local-environment coverage, and leakage in the current splits.
Then compare the pool with the training set and use diversity plus available model disagreement to propose
a candidate batch.

Save candidates, selection rationale, the environments they add, and a reference-VASP labeling plan.
Do not submit labels or training until I approve the batch size and cost.
```

## From SMILES to conformers, ORCA properties, and reaction paths

Molecular work can start from SMILES, XYZ, or existing ORCA output. The agent can build 3D structures, generate and deduplicate conformers, run CREST or xTB search and preoptimization, and select candidates for ORCA optimization, frequencies, thermochemistry, TDDFT, or NMR.

A transition-state workflow may use a relaxed coordinate scan and select a TS-side guess near the maximum, or use explicit reactant and product structures for NEB-TS. OptTS is checked for exactly the intended imaginary mode, and IRC tests endpoint connectivity. Flexible-molecule NMR requires conformer free energies and Boltzmann aggregation rather than one lowest-energy conformer.

```text
Plan a 298.15 K NMR workflow for structures/molecule.sdf with total charge 0, multiplicity 1,
and chloroform solvent.

Ask ORCA/xTB to build a sensible conformer ensemble, clean and deduplicate it with CREST/xTB,
then recommend conformers and methods for ORCA NMR. Retain structures, relative energies, and exclusion reasons.
Explain how frequency free energies, Boltzmann weights, and chemical-shift aggregation will connect.
Prepare candidates and stages, but do not submit remote tasks.
```

## Generated, executed, and answered are different states

An input or structure proves that preparation occurred. A normal process exit proves that execution ended. Only convergence, geometry, logs, physical plausibility, uncertainty, and comparison references determine whether the result answers the scientific question.

Ask the agent to point to files and state what the present evidence can and cannot support. Chapter 8 explains how prepared stages enter registered remote tasks, how receipts preserve execution identity, and how failures are recovered without duplicate jobs.
