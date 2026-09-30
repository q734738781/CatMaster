# CatMaster user manual

English | [中文](README.zh.md)

CatMaster is an autonomous agent workbench for computational catalysis, materials modeling, literature research, and scientific writing. You do not need to learn a list of tool names before you can use it, nor do you need to turn a research project into dozens of commands. In a typical session, you give the appropriate agent a research objective, the material already available, and the scientific constraints that must hold. The agent selects relevant skills, calls tools that can inspect or create real project files, and leaves its work in the shared workspace.

Autonomy does not mean giving up scientific control. CatMaster can make many implementation decisions, such as choosing a structure check, organizing intermediate files, or deciding which result to read first. When a choice changes the scientific meaning, consumes remote compute, or may overwrite important work, you can require the agent to compare options, explain its reasoning, and wait for approval. This manual explains that working relationship and describes how far each capability can take a research task.

## Where to begin

First-time users should read these four chapters in order:

1. [Quick installation and first conversation](01-quickstart.en.md) starts the WebUI and verifies it with a task that does not submit a calculation.
2. [How CatMaster works](02-concepts.en.md) explains how agents, workers, skills, tools, remote tasks, and the workspace fit together.
3. [The five agents](03-llm-configuration.en.md) introduces Research, Experiment, Literature Review, Writing, and Peer Review, including how they hand work to one another.
4. [Working in the WebUI](04-webui.en.md) covers project organization, uploads, visible agent activity, file review, and steering a running task.

After that, follow the path that matches your work:

| What you are doing | Read next |
|---|---|
| Building surfaces, adsorbates, defects, or VASP/CP2K workflows | [Experiment and its four workers](05-agents-and-modules.en.md), then [modeling and computation capabilities](06-computational-workflows.en.md) |
| Running AIMD, LAMMPS, MLFF MD, or trajectory analysis | [Dynamics worker](05-agents-and-modules.en.md#dynamics-worker-atomistic-dynamics-and-trajectories) |
| Curating training data, training MACE, or selecting active-learning candidates | [ML worker](05-agents-and-modules.en.md#ml-worker-datasets-training-and-active-learning) |
| Running conformer, xTB, CREST, ORCA, TS, IRC, or NMR work | [ORCA/xTB worker](05-agents-and-modules.en.md#orcaxtb-worker-molecules-and-quantum-chemistry) |
| Finding and reading papers, building evidence tables, or managing references | [Literature, Writing, and Peer Review](07-literature-writing-review.en.md) |
| Drafting papers, polishing, making figures or slides, answering reviewers, or preparing patent drafts | [Writing agent](07-literature-writing-review.en.md#writing-agent-turning-evidence-into-deliverables) |
| Submitting prepared calculations to a cluster or GPU server | [Remote tasks](08-remote-execution.en.md) |
| Continuing a long project, managing outputs, or preserving project methods | [Project files and continuity](09-tools-skills-evolution.en.md) |
| Installing, configuring models, connecting servers, or operating a shared deployment | [Installation, model configuration, and deployment](10-deployment-operations.en.md) |
| Copying a reference prompt or diagnosing a problem | [Prompt library and troubleshooting](11-reference-troubleshooting.en.md) |

## Capability map

This table is only an entry point. The manual describes how agents combine tools and skills into coherent research work instead of presenting each item as an isolated button.

| Capability area | Work CatMaster can participate in | Typical deliverables |
|---|---|---|
| Materials discovery and structure modeling | Find bulk structures; build reference cells, supercells, surfaces, terminations, defects, dopants, adsorption sites, and reaction paths | POSCAR/CIF/XYZ files, candidate sets, site ledgers, structure audits, reproducible scripts |
| First-principles calculations and properties | Prepare and inspect VASP or CP2K inputs for relaxation, static, frequency, band, DOS, phonon, elastic, NEB, and thermochemical work | Calculation stages, method records, convergence checks, barriers, property tables, analysis reports |
| Dynamics | Prepare CP2K AIMD, LAMMPS, and MLFF MD; continue restarts; assess trajectory health; analyze MSD, RDF, diffusion, and structural evolution | Input and restart stages, trajectories, health reports, time series, diffusion and coordination analyses |
| Machine-learning potentials | Build datasets from VASP results, create fixed splits, train or fine-tune MACE, evaluate held-out errors, and rank active-learning candidates | extxyz datasets, manifests, training configs, checkpoints, benchmark and candidate reports |
| Molecular quantum chemistry | Build molecules from SMILES or structures, search conformers, run xTB/CREST/ORCA, and handle frequency, thermochemistry, TS, IRC, TDDFT, or NMR work | Conformer ensembles, xTB/ORCA stages, optimized structures, frequencies, thermochemistry, reaction paths |
| Literature and evidence | Discover papers, use a controlled browser for authorized full text, ingest local corpora, read deeply, deduplicate, verify metadata, and map claims to evidence | Search records, corpora, evidence tables, bilingual readers, BibTeX/RIS/ENW, reviews |
| Scientific writing and communication | Draft and restructure manuscripts, polish prose, add citations, create figures and slides, prepare data statements, answer reviewers, and draft patents | Markdown, LaTeX, DOCX, PPTX, figures, PDF, response letters, patent documents |
| Independent peer review | Ask several reviewer models to inspect one canonical PDF, then synthesize agreements, disagreements, and submission risks | Reviewer reports, editor synthesis, revision issue lists |
| Research coordination | Connect literature, computation, writing, and review within an open objective while retaining hypotheses, evidence gaps, artifacts, and unresolved questions | Research plans, staged deliverables, evidence synthesis, limitations, resumable project state |

## How to read tool and skill names

The main text describes research capabilities first. Each agent or worker section includes an expandable list when the exact implementation surface matters.

- A tool performs an action, such as generating slabs, enumerating adsorption sites, preparing VASP inputs, reading a PDF, analyzing a trajectory, or submitting a remote task.
- A skill is a domain method. It tells the agent when to use those actions, what to check, how to organize outputs, and which interpretations would go beyond the evidence.
- A worker is a domain executor with a defined set of tools and skills. Experiment delegates to Materials, Dynamics, ML, or ORCA/xTB workers.
- A remote task is an administrator-registered execution contract. It moves a valid local stage to a configured machine, runs the managed scientific program, and returns results plus a recovery receipt.

Users normally do not need to name a tool in the prompt. State the scientific objective, inputs, constraints that must survive, allowed computation, and intended artifacts. The agent chooses the implementation. Naming a tool or remote task is useful when reproducing a known workflow, checking a deployment contract, or requiring a specific method.

## Boundaries

CatMaster can organize work, process files, generate and inspect inputs, call registered tools, submit configured remote calculations, and retain evidence. It does not supply VASP or ORCA licenses, bypass institutional login or cluster permissions, or cross publisher paywalls. Agent judgments, generated structures, and numerical results still require domain review, especially for charge, spin, constraints, energy references, convergence, out-of-domain ML predictions, and reaction pathways.

This manual describes the current DeepAgent specialist runtime and WebUI v2. Last checked: 2026-07-20.
