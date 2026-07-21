# 5. Experiment agent and its four computation workers

[Previous](04-webui.en.md) | [Contents](README.en.md) | [Next](06-computational-workflows.en.md)

Experiment is CatMaster's computation entry. It interprets the scientific objective, inspects available inputs and results, chooses the appropriate worker, and evaluates whether returned work is enough for the current question. Materials, Dynamics, ML, and ORCA/xTB workers perform the domain operations.

This separation lets a user begin with science rather than a tool chain. "Compare Pd adsorption near several oxygen vacancies" may involve defect structures, adsorption sites, candidate batches, geometry checks, and fast potential screening. Experiment can keep the main work with Materials, then hand accepted structures to Dynamics if the objective changes to high-temperature migration. The workers share a workspace but run sequentially, so each step can inspect the artifacts left by the previous one.

## How Experiment chooses a worker

| Primary question or deliverable | Worker that normally owns it |
|---|---|
| Crystals, surfaces, defects, adsorption, VASP/CP2K, MLFF inference, NEB, or solid-state properties | Materials |
| AIMD, LAMMPS, MLFF MD, restart continuity, trajectory health, or diffusion | Dynamics |
| Training data, MACE training or evaluation, active-learning selection | ML |
| Molecules, conformers, xTB, CREST, ORCA, TS, IRC, TDDFT, or NMR | ORCA/xTB |

The boundary follows the research objective, not only the program name. Materials can retain an MLFF relaxation within an adsorption screen. Dynamics is the better owner when MLFF MD and trajectory interpretation are central. MACE model training and benchmarking remain with ML.

All four workers also receive common project tools. `write_todos` maintains the current plan; `ls`, `glob`, `grep`, and `read_file` inspect files; `write_file` and `edit_file` save artifacts; and `execute` runs bounded local scripts or commands. `export_builtin_tool_source` can place the implementation of a registered built-in tool in the workspace for inspection. These common actions support the domain tools below and do not replace them. In Review mode, `write_file` and `edit_file` pause for approval.

## Materials worker: from crystals to surfaces, paths, and properties

Materials is the broadest computation worker. It can start from Materials Project or a workspace structure, establish a reliable bulk reference, and continue through surfaces, defects, adsorption, pathways, and properties. Deterministic modeling tools perform repeatable transformations, while domain skills guide candidate design, constraint preservation, quality checks, and calculation preparation.

### Materials discovery and bulk references

When no trusted structure exists, Materials can search by composition, elements, stability, or other Materials Project criteria and download selected records as POSCAR, CIF, or pymatgen JSON. The database file remains an original. Standardized, expanded, or calculation-ready structures are saved separately with provenance.

A consistent bulk reference supports surface energy, defect, band, and adsorption comparisons. The worker can prepare bulk relaxation and static stages, record symmetry-inequivalent sites, and use the accepted structure for later work.

```text
Use Experiment to establish a bulk reference for rutile TiO2 before surface modeling.
Ask Materials to inspect the workspace for a trustworthy structure and, if none exists, search Materials Project.
Compare candidate phase, space group, stability, and source before selecting one. Preserve the downloaded original,
the standardized structure, and a provenance note.

Prepare consistent VASP bulk-relax and static inputs, but do not submit. Record every setting that will affect
later surface-energy comparisons in notes/tio2_bulk_reference.md.
```

### Slabs, terminations, fixed layers, and surface inspection

Given a bulk structure and Miller index, `build_slab` generates all recognized terminations and can apply the same lateral expansion to each one. Surface skills help the worker choose thickness, vacuum, symmetry, top and bottom treatment, polarity checks, and a fixed-layer policy. Atoms can be fixed by bottom-layer count, height range, or explicit indices, while inherited Selective Dynamics remains attached to the structure.

Generating POSCAR files is only the beginning. Materials can inspect periodic short contacts, fragments, surface coordination, dangling atoms, and stoichiometry, then produce standardized structure views for human review. A request such as "the highest oxygen with coordination one" is converted into an auditable geometric or neighbor criterion with reported thresholds and atom indices.

```text
Read structures/relaxed_ceo2.vasp and build the CeO2(111) slab set needed for single-Pd adsorption.
Ask Materials to combine slab construction, termination screening, and visual inspection skills.

Compare every reasonable termination. Use at least 15 angstrom of vacuum and a lateral cell large enough
to avoid obvious Pd image interactions. Preserve Selective Dynamics. If a new fixed-layer policy is preferable,
show the options first. Audit top and bottom surfaces, stoichiometry, coordination, CN=1 atoms, short contacts,
and isolated fragments. Save views and a report. Do not prepare POTCAR or submit remote work in this turn.
```

### Adsorbates, adsorption sites, and candidate screening

The worker can build an adsorbate from SMILES or an existing structure, standardize its geometry, enumerate deduplicated top, bridge, hollow, and other representative sites, and place the adsorbate while inheriting slab constraints. `generate_batch_adsorption_structures` creates a candidate set from the site ledger.

Adsorption skills require site provenance, anchor atom, initial height, orientation, coverage, and consistent naming. Candidates are checked for collisions, periodic contacts, and implausible bonding. Large sets can pass through geometry filters and MLFF single-point or relaxation screening before a smaller collection enters DFT. An MLFF rank remains screening evidence, not a DFT adsorption energy.

```text
Build initial CO adsorption structures on structures/ceo2_111_selected.vasp.
Ask Materials to find symmetry-distinct sites and generate chemically sensible C-down and, where justified,
tilted orientations. Do not create redundant or colliding structures to increase the count.

Preserve slab constraints and record site type, anchor, starting distance, orientation, and provenance for every
candidate. Save structure views and a candidate ledger. You may recommend an MLFF screen, but wait for approval
before any remote execution.
```

### Defects, dopants, and site enumeration

Materials can enumerate symmetry-inequivalent sites and create vacancies, substitutions, or explicit-coordinate interstitials. `create_vacancy` and `substitute_species` accept either a selected site or representative sites from each symmetry group. `insert_interstitial_at_coords` handles defined interstitial positions.

The defect skill separates first-pass structural screening from a complete defect-formation-energy study. The latter also requires chemical potentials, charge states, Fermi level, finite-size treatment, and consistent references. The agent can plan that work without mislabeling a few neutral supercell energies as full defect thermodynamics.

### VASP, CP2K, and electronic-structure inputs

`vasp_prepare` creates canonical relax, static, frequency, DOS, or MD inputs. `vasp_band_prepare` builds a dedicated band directory with an explicit k-path source. `cp2k_prepare` covers single point, fixed-cell optimization, cell optimization, frequency, DOS-style, and related stages. Domain skills guide pseudopotential order, k points, functional, dispersion, spin, DFT+U, convergence, and constraints.

After execution, Materials can continue with band and DOS analysis, finite-displacement phonons, finite-strain elasticity, gas or adsorbate thermochemical corrections, and selected VASP MD diffusion analysis. The report records whether VASPKIT, ASE, a dedicated parser, or a project script produced the result.

### NEB, dimer, and reaction paths

Path work starts with reliable endpoints. Materials checks composition, atom order, constraints, and periodic mapping. It can remap mobile atoms while leaving frozen atoms untouched, estimate image count, and generate an interpolation. `vasp_neb_prepare` creates a VASP NEB tree. `vasp_dimer_prepare` and mode tools can derive a dimer direction from neighboring NEB images or MACE frequencies.

Skills cover plain NEB, CI-NEB, frequency or dimer refinement, barrier extraction, and path quality control. The agent checks for periodic jumps, collisions, and discontinuous rearrangements, and it does not treat an optimized discrete path as a frequency-validated transition state.

```text
Use structures/initial.vasp and structures/final.vasp to build a VASP NEB path.
Ask Materials to validate the endpoints, atom mapping, and Selective Dynamics first. Remap mobile atoms if needed,
then recommend an image count from displacement and chemistry.

Generate and visualize the interpolation and check for cell jumps, collisions, and discontinuities.
Only after endpoint and path QC should a NEB stage be created. Do not submit. Explain the recommended sequence
from plain NEB to CI-NEB and transition-state validation.
```

### MLFF screening and relaxation

Materials can query enabled MACE, FairChem UMA, MatterSim, or ORB-v3 backends, then use `mlff_sp`, `mlff_relax`, or `mlff_neb` for single-point screening, batch relaxation, or fixed-image path optimization. The worker reads the current task schema and considers element coverage, structural regime, accuracy needs, and cost.

MLFF is useful for identifying clearly unstable surface or adsorption candidates and reducing later DFT volume. Out-of-domain elements, unusual coordination, charged systems, strong magnetism, and bond-breaking pathways require caution and independent validation.

<details>
<summary>Current Materials tools and skills</summary>

Materials and structure tools: `mp_search_materials`, `mp_download_structure`, `supercell`, `enumerate_unique_sites`, `build_slab`, `fix_atoms_by_layers`, `fix_atoms_by_height`, `fix_atoms_by_indices`, `create_vacancy`, `substitute_species`, `insert_interstitial_at_coords`, `identify_structure_fragments`, and `render_vesta_views`.

Adsorption and path tools: `create_molecule_from_smiles`, `enumerate_adsorption_sites`, `place_adsorbate`, `generate_batch_adsorption_structures`, `estimate_neb_image_count`, `remap_neb_endpoint_atoms`, `make_neb_geometry`, `vasp_neb_prepare`, `vasp_dimer_prepare`, `make_dimer_mode_from_neb`, `make_dimer_mode_from_mace`, and `analyze_vasp_neb_results`.

Preparation and property tools: `vasp_prepare`, `vasp_band_prepare`, `cp2k_prepare`, `generate_kpath`, `generate_phonon_displacements`, `generate_strained_structures`, `mace_analyze_frequencies`, `analyze_trajectory`, `vaspkit_adsorbate_thermo_correction`, and `vaspkit_gas_thermo_correction`.

Visualization and implementation-inspection tools: `generate_nanobanana_figure` can draft a concept image that requires human scientific review, while `export_builtin_tool_source` exports registered tool source. Quantitative structures still use structure rendering or data plotting.

Execution tools: `get_avail_remote_task`, `get_remote_task_spec`, `get_avail_resources`, `remote_submission`, and `remote_submission_batch`.

Current domain skills are `materials-discovery-and-bulk-selection`, `bulk-relax-and-reference`, `slab-construction-and-surface-modeling`, `surface-and-termination-screening`, `adsorbate-and-intermediate-generation`, `adsorption-site-screening`, `adsorption-screening`, `defect-and-dopant-screening`, `vasp-input-preparation`, `vasp-batch-execution`, `cp2k-dft-preparation`, `cp2k-electronic-properties`, `cp2k-vibrational-analysis`, `cp2k-pathway-calculations`, `mlff-screening-and-relaxation`, `mlff-path-optimization`, `neb-prepare`, `neb-calculation`, `neb-analysis`, `band-and-dos-analysis`, `phonon-displacement-workflow`, `elastic-property-workup`, `md-diffusion-analysis`, `thermo-free-energy-and-reporting`, `structure-visual-inspection`, and `literature-grounding`.

</details>

## Dynamics worker: atomistic dynamics and trajectories

Dynamics focuses on how a system evolves with time. It prepares CP2K AIMD, LAMMPS, and managed MLFF MD, continues restarts, and assesses whether a trajectory is fit for analysis. It does not fit a diffusion coefficient before checking temperature, energy, volume, timestep, sampling length, abnormal forces, short contacts, broken structures, and trajectory continuity.

### CP2K AIMD

`cp2k_aimd_prepare` creates a fresh or continuation stage. Skills retain ensemble, temperature, pressure, timestep, thermostat or barostat, velocity source, random state, and restart lineage. `cp2k_output_summary` extracts general run health, while goal-specific properties use trajectory tools or a saved project script.

### LAMMPS

The worker can validate force-field files and element mapping before preparing minimization, NVE, NVT, NPT, annealing, or restart stages. LAMMPS skills examine units, atom style, masses, boundaries, neighbor settings, potential applicability, and restart integrity before treating a launching script as a valid physical setup.

### MLFF MD and trajectory analysis

When a backend is enabled, Dynamics can run `mlff_md` with restart-safe staging and continuity records. Model and operation parameters come from the current catalog.

`md_trajectory_summary` and `analyze_trajectory` can produce time series, MSD, diffusion fits, RDF, and related artifacts. The agent chooses mobile species, equilibration window, fit interval, and dimensionality according to the scientific target and records those choices.

```text
Ask Dynamics to determine whether calculations/mlff_md_1073K/ is suitable for Pd migration analysis.
Read its inputs, logs, restart, and trajectory without rerunning anything.

Check temperature, total energy, time continuity, frame count, Pd-cluster connectivity, short contacts,
atom escape, and restart provenance. Only after the health audit passes should you select equilibration and
mobile-atom windows for MSD, RDF, and diffusion fitting. Save methods, units, and uncertainty to
analysis/md_quality_and_diffusion.md.
```

<details>
<summary>Current Dynamics tools and skills</summary>

Tools include `cp2k_aimd_prepare`, `cp2k_output_summary`, `lammps_forcefield_validate`, `lammps_prepare`, `lammps_log_summary`, `md_trajectory_summary`, `analyze_trajectory`, `export_builtin_tool_source`, and the remote catalog and submission tools.

Current domain skills are `cp2k-aimd-preparation`, `cp2k-aimd-restart`, `cp2k-run-analysis`, `lammps-preparation`, `lammps-minimization`, `lammps-md-execution`, `lammps-restart`, `mlff-md-sampling`, and `trajectory-analysis`. Shared `remote-stage-layouts` and `dpdispatcher-remote-receipts` skills cover stage contracts and receipt-driven recovery.

</details>

## ML worker: datasets, training, and active learning

ML owns the data and model lifecycle for machine-learning potentials. It can extract energy, force, and stress labels from VASP result trees, create fixed train/validation/test splits, prepare remote MACE training or evaluation stages, and analyze held-out error.

Dataset work checks element coverage, units, reference energies, duplicate structures, outliers, missing labels, and leakage. Training retains configuration, seed, model origin, checkpoints, logs, and test results. `calculate_al_candidates` can rank a pool by diversity and optional committee disagreement, while the user retains control over expensive reference labeling.

```text
Ask ML to build a MACE fine-tuning dataset from calculations/reference_vasp/.
Audit which runs actually converged and contain usable energy, force, and stress labels before including them.
Normalize units and labels, check duplicates and mixed calculation settings, fix a random seed, and write a
train/validation/test manifest. Do not start training before the audit passes.

Save the dataset under ml/datasets/pd_ceo2_v1/ and document provenance, exclusions, element and configuration
coverage, leakage risk, and intended domain.
```

<details>
<summary>Current ML tools and skills</summary>

Tools are `build_dataset_from_runs`, `calculate_al_candidates`, `export_builtin_tool_source`, and the remote catalog, resource, and submission tools. Skills are `mace-dataset-curation`, `mace-finetuning-and-benchmark`, and `active-learning-relabel-loop`.

Managed remote tasks include `mace_train` and `mace_eval`. Training requires a configured GPU resource and MACE environment and is never silently run on the control plane.

</details>

## ORCA/xTB worker: molecules and quantum chemistry

ORCA/xTB handles nonperiodic molecules, complexes, and finite clusters. It can build 3D structures from SMILES, enumerate and deduplicate conformers, run CREST or xTB searches and preoptimization, then prepare selected conformers for ORCA optimization, frequencies, thermochemistry, TDDFT, or NMR.

For reaction paths, it can prepare a relaxed scan, take a TS-side guess near the scan maximum, and run OptTS. With explicit reactant and product structures it can prepare NEB-TS and then IRC. Flexible-molecule NMR can connect conformer generation, xTB cleanup, ORCA NMR, and evidence needed for later Boltzmann aggregation.

The worker requires a clear total charge and spin multiplicity and keeps multiplicity distinct from the number of unpaired electrons. Conformer ranking records method, solvent, and energy window. Frequency analysis distinguishes minima from transition states instead of treating geometry optimization alone as proof.

```text
Ask ORCA/xTB to build a conformer set for ORCA thermochemistry from the supplied molecular SMILES.
Use total charge 0, multiplicity 1, and acetonitrile solvent.

Choose a sensible conformer-generation, CREST/xTB screening, and deduplication strategy. Retain provenance and
relative energies. Before preparing ORCA opt+freq, report candidate count, energy window, duplicate checks,
and expected cost. Create reviewable ORCA stages, but do not submit remote work.
```

<details>
<summary>Current ORCA/xTB tools and skills</summary>

Molecule and conformer tools: `create_molecule_from_smiles`, `enumerate_molecular_conformers`, `filter_conformer_ensemble`, `extract_optimized_molecules`, and `identify_structure_fragments`.

ORCA preparation and analysis tools: `orca_prepare`, `orca_scan_prepare`, `orca_optts_prepare`, `orca_nebts_prepare`, `orca_irc_prepare`, `analyze_orca_results`, and `analyze_xtb_results`, plus remote catalog and submission tools. `export_builtin_tool_source` supports implementation inspection.

Skills include `conformer-search-and-preopt`, `xtb-screen-and-prune`, `mlff-molecular-screening`, `orca-optfreq-thermochemistry`, `scan-to-ts`, `nebts-and-irc`, and `nmr-ensemble-workup`.

</details>

## Continuing across workers

One objective may cross worker boundaries, but each handoff should leave clear artifacts. Materials may build adsorption candidates and reduce them with MLFF before Dynamics studies high-temperature stability. Dynamics may identify unusual configurations for ML active learning. ORCA/xTB gas-phase thermochemistry can join Materials surface-frequency corrections in a free-energy analysis.

Experiment schedules those steps sequentially. Users do not need to specify every delegation, but should state the main objective, allowed approximations, and stopping point. The next chapter follows complete modeling stories and shows what evidence each stage should retain.
