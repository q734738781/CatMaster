# Features And Daily Workflows

This chapter explains what each WebUI task lane is for and how to organize project spaces and prompts.

## 1. Basic Concepts

### Project Space

A project space is a normal directory that stores:

- Uploaded structures, manuscripts, data, and calculation outputs.
- Agent-created scripts, input files, reports, and figures.
- Run history, logs, and intermediate state.

Use one project space for one research project or one coherent problem.

### Run

One submission is one run. If a task is interrupted or needs more work, select a historical run in the WebUI and use `resume_selected_run`.

### Files

The WebUI file view can browse project-space files, upload files, preview text, and download a workspace zip.

## 2. Task Lanes

### Experiment

Use this for bounded computational tasks:

- Build slabs, adsorbate structures, and molecules.
- Prepare VASP, CP2K, LAMMPS, ORCA, and xTB inputs.
- Run MACE relax, single point, MD, NEB, training, or evaluation tasks, plus UMA single-point or relaxation screening tasks.
- Analyze existing outputs such as `OUTCAR`, `vasprun.xml`, CP2K output, or LAMMPS logs.
- Submit prepared remote stages.

Example:

```text
Read the Ni slab and CO molecule in the current project. Generate top, bridge, and hollow CO adsorption structures, then prepare VASP static inputs for each structure.
```

### Research

Use this for open research questions:

- Propose catalyst candidates for a target reaction.
- Combine literature and project results into a screening plan.
- Break a large question into literature, structure preparation, calculation, and result-synthesis steps.

Example:

```text
For CO2 hydrogenation to methanol, design a first-round Cu-based catalyst screening plan. Start from literature-supported candidates, then propose executable calculations.
```

### Literature Review

Use this for literature synthesis and public-source checking:

- Representative paper lists.
- DOI, year, journal, author, and metadata checks.
- Organization by catalyst structure, activity metric, and evidence type.

Example:

```text
Review representative work from the last five years on single-atom Ni catalysts for electrochemical CO2-to-CO conversion. Organize by catalyst structure, activity metrics, key evidence, and DOI.
```

### Writing

Use this for evidence-grounded writing inside the project space:

- Draft Results and Discussion.
- Revise abstracts, introductions, and discussions.
- Organize TeX drafts.
- Convert calculation results into reports or response drafts.

Example:

```text
Based on the calculation results and figures in this project, draft an ACS-style Results and Discussion section in TeX.
```

### Peer Review

Use this to review an existing PDF manuscript:

- Check whether conclusions are supported by data.
- Check whether methods are reproducible.
- Produce reviewer-style comments and revision suggestions.

Example:

```text
Review files/manuscript.pdf. Focus on whether the catalytic mechanism evidence is sufficient, the computational methods are reproducible, and the conclusions are supported by data.
```

## 3. Prompt Pattern

A good prompt usually includes:

- Goal: what final artifact you want.
- Inputs: which project-space files are available.
- Constraints: method, model, functional, structure scope, queue, or resource limits.
- Output format: report, CSV, JSON, VASP input directory, TeX, etc.
- Whether remote submission is allowed or preparation only.

Example:

```text
Use files/slab.vasp and files/CO.xyz to generate initial CO adsorption structures on Fe(110) for ontop, bridge, and hollow sites. Write outputs to adsorption_structures/ and create summary.csv with site name, initial height, and file path. Do not submit remote calculations yet.
```

Remote example:

```text
Treat each child directory under vasp_inputs/ as one VASP stage. Use remote_submission_batch with task_name=vasp_execute. Before submission, check that every child directory contains INCAR, KPOINTS, POSCAR, and POTCAR.
```

## 4. External Programs

Install only what your tasks require:

- OVITO: structure rendering and exported structure views.
- LaTeX / `pdflatex`: TeX manuscript compilation.
- VASPKIT: adsorbate and gas-phase thermochemistry corrections.
- ORCA, xTB, CREST: quantum chemistry, semiempirical calculations, and conformer search.
- VASP, CP2K, LAMMPS: first-principles and molecular simulation workflows.
- MACE: machine-learning potential relax, single point, MD, NEB, training, and evaluation.
- FairChem UMA: machine-learning single-point or relaxation screening for materials, catalyst structures, molecules, and clusters; use a remote environment separate from MACE.

External programs can run locally or remotely. Remote executable paths are normally loaded through resource environment setup in [Remote setup](02-remote.en.md).

## 5. Daily Workflows

### Prepare Structures

1. Upload bulk structures, slabs, molecules, or existing calculation outputs.
2. Use the Experiment lane to generate structures or input files.
3. Inspect output directories and summary files.
4. Decide whether to submit remote calculations.

### Submit Calculations

1. Confirm the stage directory is complete.
2. Use `remote_submission` or `remote_submission_batch`.
3. Specify `task_name`, such as `vasp_execute`, `mace_relax_dir`, or `uma_sp_dir`.
4. Record `remote_context_id`, `submission_hash`, and `receipt_rel`.

### Analyze Results

1. Upload or download calculation results.
2. Ask the Experiment lane to extract energies, structures, convergence status, and key metrics.
3. Output CSV/JSON/Markdown reports.

### Write And Review

1. Put figures, tables, CSV files, and calculation summaries in the project space.
2. Use the Writing lane for drafts, results, discussions, or responses.
3. Use the Peer Review lane to check evidence and reproducibility.

## 6. Resume A Historical Run

If a task was not finished:

1. Select the historical run in the WebUI.
2. Set run mode to `resume_selected_run`.
3. Add a precise instruction, such as "Continue the previous VASP input preparation. First inspect existing files and do not redo structures that are already complete."

## 7. Next Steps

- WebUI startup problem: return to [Local setup](01-local.en.md).
- Cluster submission: read [Remote setup](02-remote.en.md).
- Project capability overview: return to [README.md](../../README.md).
