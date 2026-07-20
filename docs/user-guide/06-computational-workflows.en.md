# 6. Computational workflows

[Previous](05-agents-and-modules.en.md) | [Contents](README.en.md) | [Next](07-literature-writing-review.en.md)

Computational modules share one pattern: define the scientific question, inspect
the input, prepare a stage, run pre-submission QC, query the current remote task
schema, obtain approval, submit, collect results, and perform numerical and
physical acceptance checks. A generated input or a scheduler completion state is
not a scientific conclusion.

## 6.1 General task contract

An Experiment request should state:

| Item | What to provide |
|---|---|
| Scientific objective | What is being compared, optimized, screened, or measured |
| Input | Workspace-relative path, structure version, and provenance |
| Physical settings | Charge, spin, periodicity, constraints, temperature, pressure, and fields |
| Method | Software, functional, basis, potentials, MLFF backend, and accuracy |
| Resource limits | Whether remote submission is allowed, CPU/GPU, time, and budget |
| Output | Stage, structures, tables, plots, logs, and report paths |
| Stop point | Choices that require user confirmation |

It is often safer to request preparation and review without submission, inspect
the files, and approve the remote job in a separate turn.

## 6.2 Structure preparation and audit

Structure work includes format conversion, standardization, supercells,
substitution, defects, adsorbates, surfaces, and path images. Check at least:

- Elements, atom count, lattice, periodicity, and coordinate units.
- Fractional versus Cartesian coordinate meaning.
- `Selective Dynamics`, fixed layers, constraints, and atom order.
- Minimum distances and overlaps under periodic boundaries.
- Molecular charge, spin multiplicity, and bond-breaking risk.
- Atom mapping before and after modification.

If `Selective Dynamics` must be preserved, treat it as a hard invariant across
conversion, sorting, cutting, and export. Do not lose it in an intermediate
format and reconstruct it by guesswork.

A completed structure task should include:

```text
structure file
change summary
key geometry or coordination audit
constraint-preservation check
input provenance and generation script
```

## 6.3 Bulk, surfaces, and adsorption

### Bulk

Typical tasks include primitive or conventional cells, supercells, bulk
relaxation, defects, and doping. Use consistent element order, k-point density,
energy reference, and constraints when comparing structures.

### Surfaces and terminations

Specify Miller index, layer count or minimum thickness, vacuum, symmetry,
fixed-layer rule, allowed stoichiometry changes, and polarity treatment. After
cutting, inspect:

- Slab normal and vacuum.
- Intended top and bottom surfaces.
- Surface coordination and dangling atoms.
- Duplicates, short bonds, and isolated fragments.
- Fixed and mobile layers.

Requests such as "highest atoms" or "coordination number one" should become
auditable geometry or graph filters with atom indexes and thresholds in the
report. Coordination is a screening heuristic, not a unique measure of chemical
stability.

### Adsorption

Define adsorbate conformation, coverage, site set, orientation, initial distance,
and whether reconstruction is allowed. Use consistent names and directories for
candidates, then check collisions, periodic distances, and chemical plausibility
before submission.

## 6.4 VASP and CP2K

The materials worker prepares VASP or CP2K stages and analyzes common outputs.

Before VASP submission, check:

- `INCAR`, `POSCAR`, `POTCAR`, and `KPOINTS` are present.
- Element order matches the concatenated `POTCAR` order.
- `ENCUT`, k-points, and electronic and ionic convergence have a rationale.
- Spin, magnetic moments, dipole correction, DFT+U, dispersion, and solvation
  match the system.
- Parallel settings are coordinated by the remote boot script and resource, not
  arbitrarily hard-coded in input.

Before CP2K submission, check:

- The main input is normally `job.inp`.
- `manifest.json` and referenced basis, potential, coordinate, or restart files
  are complete.
- `RUN_TYPE`, periodicity, CELL, POISSON, SCF, XC, and motion settings agree.
- AIMD restart, velocity, and random-state provenance are recorded.

Do not accept exit code alone. Inspect SCF or geometry convergence, warnings,
final structure, energy history, constraints, and expected output files.

## 6.5 Paths, NEB, and transition states

A path calculation needs reliable endpoints and atom mapping. Before submission:

- Ensure endpoint elements and order match.
- Keep constraints and periodicity consistent.
- Remove collisions and unphysical cell crossings from interpolated images.
- Use continuous image numbering and the task's expected endpoint layout.
- Define climbing image, spring, convergence, and later frequency validation.

NEB convergence describes the optimized discrete path. Important saddle points
still need geometry, force, and where appropriate frequency or IRC validation.

## 6.6 Phonons, elastic properties, bands, DOS, and thermodynamics

These analyses are sensitive to upstream calculations and units:

- Phonons: check supercell, displacement, force convergence, imaginary modes,
  and acoustic branches.
- Elasticity: check strain amplitude, relaxation policy, symmetry, and fit range.
- Bands and DOS: record k-path source, Fermi level, smearing, spin, and projection
  definitions.
- Thermodynamics: distinguish electronic energy, ZPE, thermal enthalpy, entropy,
  and free-energy reference, temperature, and standard state.

VASPKIT and ASE fallback are calculation routes. State which route, input, and
assumptions produced the reported value.

## 6.7 Dynamics

The dynamics worker covers CP2K AIMD, LAMMPS, and MLFF MD. A request should give:

- Initial or restart structure.
- Ensemble, temperature, pressure, timestep, and total steps.
- Thermostat or barostat and time constants.
- Velocity source, seed, and restart-continuity requirements.
- Output interval, trajectory format, and planned analysis.

Before submission, inspect units, masses, periodic boundaries, potential files,
neighbor settings, and time scales. After completion, first check temperature and
energy drift, volume, extreme force, overlaps, bond breaking, escape, frame count,
and restart usability. Only then calculate MSD, diffusion, RDF, coordination, or
residence times.

## 6.8 MLFF inference

Managed MLFF backends include MACE, FairChem UMA, MatterSim, and ORB-v3. Only
MACE is enabled in the template by default. An administrator must install,
enable, and bind the others to resources.

Task types:

```text
mlff_sp
mlff_relax
mlff_md
mlff_neb
```

Do not guess backend parameters in a prompt. Ask the agent to query
`get_avail_remote_task`, then use `get_remote_task_spec` for the full backend and
operation schema. Deployed models, default dtype, device, and supported
operations can change.

An MLFF result is a model prediction, not DFT truth. Check uncertainty or use an
independent method for out-of-domain elements, unusual coordination, high-energy
paths, charged systems, strong magnetism, or reactive bond breaking.

## 6.9 ML data and MACE training

The ML worker handles curation, active learning, MACE training, and evaluation.
Record before training:

- Data source, license, deduplication, and train/validation/test split.
- Energy, force, and stress units and reference-energy handling.
- Element coverage, configuration distribution, outliers, and leakage checks.
- Random seed, model version, hyperparameters, and hardware.

Save checkpoint, configuration, logs, learning curves, and independent test
metrics. Training error alone does not establish fitness for the target system.

## 6.10 Molecules, xTB, CREST, and ORCA

The ORCA/xTB worker handles nonperiodic molecules and clusters. State:

- Structure or SMILES, protonation, and tautomer state.
- Total charge and spin multiplicity.
- Conformer search scope and energy window.
- Solvent model, temperature, method, and basis.
- Whether frequency, thermochemistry, TS, IRC, TDDFT, or NMR is required.

xTB and CREST are useful for prescreening and preoptimization, not automatic
replacements for higher-level methods. After ORCA optimization, check termination
and gradients. For frequencies, confirm no imaginary mode at an intended minimum
or exactly the expected mode at a transition state, and inspect its displacement.

## 6.11 Submission and acceptance checklists

Before submission:

1. The input directory follows the registered task's canonical layout.
2. Every referenced file is inside the stage.
3. Method, units, constraints, charge, and spin are recorded.
4. The current task spec was queried instead of using stale parameter names.
5. In Review mode, inspect task, work directory, overrides, CPU/GPU, and cleanup.

After completion:

1. Read `status.json`, `stdout.log`, and `stderr.log`.
2. Check the receipt's `submission_hash`, task count, and state counts.
3. Confirm results were merged back into the original stage.
4. Inspect program-level convergence and warnings.
5. Perform domain QC on the final structure, trajectory, or model.
6. Write conclusions, limitations, and failed cases to a separate report.

See [Remote machines and execution](08-remote-execution.en.md) for layouts,
resource cards, and recovery.
