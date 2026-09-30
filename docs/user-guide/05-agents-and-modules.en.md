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

All four workers also receive common project tools. `write_todos` maintains the complete current plan; `ls`, `glob`, `grep`, and `read_file` inspect files; `write_file`, `edit_file`, and `delete` manage workspace artifacts; and `execute` runs bounded local scripts or commands. Each agent is given the absolute project `files/` directory as its workspace boundary. `execute` already starts there, so commands should use workspace-relative paths and must not inspect host files outside that directory. `write_file` replaces an existing file in full, so the agent reads an existing target first and uses `edit_file` for a local change. `delete` can recursively remove an explicit file or directory, but cannot delete the workspace root or the routed memory root. Codex OAuth workers can additionally use a Codex-compatible freeform `apply_patch` tool for precise multi-file patches. `export_builtin_tool_source` exports a builtin module and recursively follows its static CatMaster imports in one call, including parent package initializers. Each file keeps its original content and package path; read or search the returned source tree directly. Set `include_dependencies=false` for one file, or use `module_name` to export another known module. Identical existing files are reused; differing files require another output directory or explicit `overwrite=true`. The export is a reading reference: dynamic imports, non-Python assets and third-party libraries are not collected. These common actions support the domain tools below and do not replace them. Authorized local file edits and specific-path deletion run without Review approval; Review still pauses real remote submissions.

## Materials worker: from crystals to surfaces, paths, and properties

Materials is the broadest computation worker. It can start from Materials Project or a workspace structure, establish a reliable bulk reference, and continue through surfaces, defects, adsorption, pathways, and properties. Deterministic modeling tools perform repeatable transformations, while domain skills guide candidate design, constraint preservation, quality checks, and calculation preparation.

### Materials discovery and bulk references

When no trusted structure exists, Materials can search by composition, elements, stability, or other Materials Project criteria and download selected records as POSCAR, CIF, or pymatgen JSON. The search tool's visible default is 50 rows, not a maximum: an agent may request any positive total, and the CSV contains that requested number subject to the provider result count. Its result states whether the limit was explicitly selected or came from the visible default and labels a bounded subset without claiming an exhaustive database search. The database file remains an original. Standardized, expanded, or calculation-ready structures are saved separately with provenance.

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

The worker can build an adsorbate from SMILES or an existing structure, standardize its geometry, enumerate deduplicated top, bridge, hollow, and other representative sites, and place the adsorbate while inheriting slab constraints. `generate_batch_adsorption_structures` creates a candidate set from the site ledger. Before applying its bounded structure count, it writes a deterministic `all_sites.json` manifest. A later `site_offset` call continues from those stable site IDs after verifying the slab identity and generation settings, so an explicit subset remains bounded without rediscovering or skipping sites.

Adsorption skills require site provenance, anchor atom, initial height, orientation, coverage, and consistent naming. Candidates are checked for collisions, periodic contacts, and implausible bonding. Large sets can pass through geometry filters and MLFF single-point or relaxation screening before a smaller collection enters DFT. An MLFF rank remains screening evidence, not a DFT adsorption energy.

When independently generated structures are combined, site placement is a chemical seed rather than a finished geometry. Experiment, Materials, Dynamics, and ORCA/xTB share `constraint-guided-atomic-assembly` and `atomic-structure-validation-and-recovery`. The worker that owns the structure defines the rigid fragments, anchor distances, orientations, and allowed regions. It then samples translations and rotations, removes unintended contacts, and applies PBC-aware absolute and covalent-radius-normalized distance checks. For important models, it also inspects several views for interpenetrating layers, buried or reversed fragments, and periodic-image collisions. Physical single points and constrained pre-relaxations start only after the structure passes this geometry check. When a compatible MLFF is available, an unchanged-geometry SP can then record atom-resolved forces and help locate a remaining local problem.

A clear literature figure can also guide reconstruction through `literature-figure-guided-structure-reconstruction`. The structure-owning worker may give one exact image, its caption, and the modeling target to a general-purpose visual branch. The returned morphology brief separates visible features, statements from the caption, interpretation, and information that the figure does not contain. The worker then maps details such as step occupancy, termination, cyclic topology, adsorbate contacts, distribution, or substitution into named selections and operations in the construction script. This visual description can prevent a wrong model, but it is not an acceptance gate.

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

Materials can query enabled MACE, FairChem UMA, MatterSim, or ORB-v3 backends, then use `mlff_sp`, `mlff_relax`, `mlff_neb`, `mlff_vib`, or `mlff_ts` for single-point screening, batch relaxation, fixed-image path optimization, general normal-mode analysis, or local transition-state refinement. The worker reads the current task schema and considers element coverage, structural regime, accuracy needs, and cost.

MLFF is useful for identifying clearly unstable surface or adsorption candidates and reducing later DFT volume. Out-of-domain elements, unusual coordination, charged systems, strong magnetism, and bond-breaking pathways require caution and independent validation.

If optimization, SCF, energy, or force evaluation fails immediately after assembly, the agent inspects the submitted starting geometry before changing numerical settings. An overlap, interpenetrating layer, wrong anchor, wrong periodic image, or fragment expelled in the first steps sends the workflow back to the last chemically valid host and fragment files. Optimizer, electronic, potential, or model-domain diagnosis comes later, once the starting structure is coherent.

`mlff_ts` starts from one TS-like geometry and uses constrained RS-pRFO. It is not an open-ended saddle search. Optimizer convergence and first-order-saddle validation are reported separately; validation requires exactly one significant imaginary mode.

`mlff_vib` analyzes accepted minima, transition states, adsorbates, molecules, or constrained material structures without changing the geometry. Structure constraints define the exact mode subspace. The compact output contains one canonical `vibrations.npz`, one frequency table, and one multi-frame mode file rather than ASE displacement-cache JSON files.

<details>
<summary>Current Materials tools and skills</summary>

Materials and structure tools: `mp_search_materials`, `mp_download_structure`, `supercell`, `enumerate_unique_sites`, `build_slab`, `fix_atoms_by_layers`, `fix_atoms_by_height`, `fix_atoms_by_indices`, `create_vacancy`, `substitute_species`, `insert_interstitial_at_coords`, `identify_structure_fragments`, and `render_vesta_views`.

Adsorption and path tools: `create_molecule_from_smiles`, `enumerate_adsorption_sites`, `place_adsorbate`, `generate_batch_adsorption_structures`, `estimate_neb_image_count`, `remap_neb_endpoint_atoms`, `make_neb_geometry`, `vasp_neb_prepare`, `vasp_dimer_prepare`, `make_dimer_mode_from_neb`, `make_dimer_mode_from_mace`, and `analyze_vasp_neb_results`.

Preparation and property tools: `vasp_prepare`, `vasp_band_prepare`, `cp2k_prepare`, `cp2k_output_summary`, `generate_kpath`, `generate_phonon_displacements`, `generate_strained_structures`, `analyze_trajectory`, `vaspkit_adsorbate_thermo_correction`, and `vaspkit_gas_thermo_correction`.

Visualization and implementation-inspection tools: `generate_figure` can draft a concept image that requires human scientific review, while `export_builtin_tool_source` exports registered tool source. Quantitative structures still use structure rendering or data plotting.

Execution tools: `get_avail_remote_task`, `get_remote_task_spec`, `get_avail_resources`, `remote_submission`, and `remote_submission_batch`.

Current domain skills include `materials-discovery-and-bulk-selection`, `bulk-relax-and-reference`, `slab-construction-and-surface-modeling`, `surface-and-termination-screening`, `adsorbate-and-intermediate-generation`, `adsorption-site-screening`, `adsorption-screening`, `defect-and-dopant-screening`, `vasp-input-preparation`, `vasp-batch-execution`, `cp2k-dft-preparation`, `cp2k-electronic-properties`, `cp2k-vibrational-analysis`, `cp2k-pathway-calculations`, `mlff-screening-and-relaxation`, `mlff-path-optimization`, `mlff-vibrational-analysis`, `mlff-transition-state-refinement`, `neb-prepare`, `neb-calculation`, `neb-analysis`, `band-and-dos-analysis`, `phonon-displacement-workflow`, `elastic-property-workup`, `md-diffusion-analysis`, `thermo-free-energy-and-reporting`, `structure-visual-inspection`, and `literature-grounding`.

The shared atomistic skills are `constraint-guided-atomic-assembly`, `atomic-structure-validation-and-recovery`, and `literature-figure-guided-structure-reconstruction`. The first two include reference scripts for rigid assembly and overlap checking; a worker copies them into workspace `scripts/` before running or adapting them. The third produces an optional morphology brief and an image-to-script mapping. None adds tool permissions.

</details>

## Dynamics worker: atomistic dynamics and trajectories

Dynamics focuses on how a system evolves with time. It prepares CP2K AIMD, LAMMPS, and managed MLFF MD, continues restarts, and assesses whether a trajectory is fit for analysis. It does not fit a diffusion coefficient before checking temperature, energy, volume, timestep, sampling length, abnormal forces, short contacts, broken structures, and trajectory continuity.

Dynamics applies the shared geometry check to its starting structure. It does not use minimization or MD to repair collisions. If an input slab, adsorbate, defect, interface, or pathway model is already invalid, Dynamics returns the specific contact and geometry evidence to Materials for reconstruction.

### CP2K AIMD

The worker authors a complete native CP2K AIMD input and passes it to `cp2k_prepare`, which copies it unchanged to `job.inp` together with explicitly mapped restart, structure, wavefunction, or PLUMED files. Skills keep ensemble, timestep, thermostat/barostat, output strides, and restart lineage visible. `cp2k_output_summary` separates process, SCF, and task states; goal-specific properties use trajectory tools or a focused project script.

### LAMMPS

The worker authors a complete native LAMMPS script and stages it unchanged with `lammps_prepare`, along with every referenced data, restart, potential, table, or include file. The script retains full LAMMPS command freedom for minimization, NVE, NVT, NPT, annealing, and restart work. LAMMPS skills keep units, atom style, masses, boundaries, force field, timestep, fixes, and output semantics explicit.

### MLFF MD and trajectory analysis

When a backend is enabled, Dynamics can run `mlff_md` with restart-safe staging and continuity records. Model and operation parameters come from the current catalog.

`md_trajectory_summary` is a lightweight trajectory inventory tool. For an explicitly selected LAMMPS dump, XYZ, or ASE `.traj`, it reports the format, frame and atom counts, exports the final frame, and inventories existing RDF/MSD tables, CP2K `.ener` files, and restart files in the same directory; it does not infer diffusion or recalculate RDF. `analyze_trajectory` produces the quantitative time series, MSD, diffusion-fit, and RDF artifacts. For that analysis, the agent supplies the physical stored-frame interval, fit window, and mobile species; periodic generic XYZ/ASE files also require an explicit wrapped/unwrapped coordinate interpretation. Native LAMMPS dump columns and XDATCAR carry their own wrapping semantics.

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

Tools include `cp2k_prepare`, `cp2k_output_summary`, `lammps_prepare`, `lammps_log_summary`, `md_trajectory_summary`, `analyze_trajectory`, `export_builtin_tool_source`, and the remote catalog and submission tools.

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

ORCA/xTB also uses the shared rigid-assembly skill when a molecular encounter complex, finite cluster, or TS guess combines independent fragments. It keeps accepted reactant interiors fixed while imposing forming and breaking contacts, then moves to a local semi-rigid xTB/ORCA preoptimization. OptTS is not expected to recover the intended topology from a severely overlapped coordinate merge.

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

Quantum-chemistry preparation and analysis tools: `xtb_prepare`, `crest_prepare`, `orca_prepare`, `orca_nebts_prepare`, `analyze_xtb_results`, and `analyze_orca_results`, plus remote catalog and submission tools. `export_builtin_tool_source` supports implementation inspection.

ORCA preparation accepts a complete ordered native `simple_keywords` list, complete optional `input_blocks`, and explicit charge/multiplicity. Generic scan, OptTS, IRC, property, and ordinary calculations use `orca_prepare`; only mapped two-endpoint NEB staging has a specialized preparer. xTB and CREST preparation accept exact native argv token lists plus explicit file mappings, so documented engine options remain reachable without scientific execution-time overrides.

Skills include `conformer-search-and-preopt`, `xtb-screen-and-prune`, `mlff-molecular-screening`, `orca-optfreq-thermochemistry`, `scan-to-ts`, `nebts-and-irc`, and `nmr-ensemble-workup`.

</details>

## Continuing across workers

One objective may cross worker boundaries, but each handoff should leave clear artifacts. Materials may build adsorption candidates and reduce them with MLFF before Dynamics studies high-temperature stability. Dynamics may identify unusual configurations for ML active learning. ORCA/xTB gas-phase thermochemistry can join Materials surface-frequency corrections in a free-energy analysis.

Experiment schedules those steps sequentially. Users do not need to specify every delegation, but should state the main objective, allowed approximations, and stopping point. The next chapter follows complete modeling stories and shows what evidence each stage should retain.
