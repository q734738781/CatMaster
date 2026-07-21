# 2. Agents, workers, skills, and tools

[Previous](01-quickstart.en.md) | [Contents](README.en.md) | [Next](03-llm-configuration.en.md)

CatMaster accepts a research objective and turns it into files, calculation stages, and evidence that can be inspected and reused. This chapter only describes how its capability units connect. Later chapters list the detailed roles, tools, skills, and reference prompts for each entry.

## Four execution units

| Unit | Purpose | Examples |
|---|---|---|
| Agent | Accepts an objective for one research stage and decides who should execute it | Research, Experiment, Literature Review, Writing, Peer Review |
| Worker or specialist | Owns a bounded domain task | Materials, Dynamics, ML, ORCA/xTB, or the literature, writing, and review specialists |
| Skill | Supplies domain methods, checks, and delivery standards | Surface construction, termination screening, trajectory analysis, paper reading, manuscript writing |
| Tool | Reads or creates real results | Parse structures, build slabs, prepare inputs, submit remote tasks, analyze files, compile documents |

Research can dispatch execution across the other entries. It can split an open question into stages, send work to Literature Review, Experiment, Writing, or Peer Review, and read the returned files and evidence before continuing, requesting corrections, or closing the investigation. The specialist or worker that owns the relevant tools and skills performs each domain action.

Experiment delegates computation to four workers. Materials handles crystals, surfaces, adsorption, defects, reaction paths, and properties. Dynamics handles AIMD, LAMMPS, MLFF MD, restart, and trajectories. ML handles datasets, MACE, and active learning. ORCA/xTB handles molecules, conformers, xTB, CREST, ORCA, TS, IRC, TDDFT, and NMR.

## How far a research task can progress

For an objective such as "Explain why isolated Pd remains stable on CeO2," Research can ask Literature Review to build a mechanism and characterization evidence table, then ask Experiment which structural or energetic uncertainties can be calculated. After results return, it can ask Writing to combine the literature and computational evidence or send a fixed manuscript to Peer Review for an independent assessment.

```text
Research objective
  -> Literature Review: search record, evidence table, reference library
  -> Experiment: structure candidates, calculation stages, remote results, analysis reports
  -> Writing: Markdown, LaTeX, DOCX, figures, or PPTX
  -> Peer Review: reviewer reports, editor synthesis, revision issue list
```

The work can stop at any explicit boundary: literature evidence only, candidate structures only, prepared but unsubmitted calculations, a wait for remote results, or analysis of existing output. State the objective, input paths, scientific constraints, remote-compute authority, and stopping point in the prompt.

## Capability depends on the entry and deployment

Each worker receives tools and skills that match its responsibility. Materials can build slabs and VASP inputs. Writing can compile manuscripts and make figures. Literature Review can search, read, ingest a corpus, and finalize citations. Research or Experiment delegates cross-domain work to the appropriate executor.

Remote execution also depends on the tasks, resources, machines, and MLFF backends registered by the deployment. An agent can query the current catalog and submit only enabled tasks that belong to its role. Input preparation and remote execution are separate capabilities; see [Chapter 8](08-remote-execution.en.md).

## Workspaces and threads

Every workspace contains two parts:

```text
workspace/
  files/
  metadata/
```

`files/` is the shared project area for users and agents. Uploaded structures, papers, and data live there with generated candidates, scripts, reports, figures, and remote results. Use paths relative to `files/` in prompts, such as `structures/slab.vasp` or `writing/results.md`.

`metadata/` stores thread checkpoints, observability records, artifact indices, and remote recovery data. Users normally do not edit it. A complete backup includes both `files/` and `metadata/`.

A thread retains one continuing research context. When resuming work, name the files to reread, the conditions to preserve, and any steps that must not be repeated:

```text
Continue the surface screening. Read notes/termination_review.md and
structures/ceo2_111_candidates/ and compare the current candidates with the last audit.
Do not regenerate structures whose hashes are unchanged. Resume from the unresolved
termination choice, and do not submit remote calculations yet.
```

## What users can inspect

Chat shows delegation, Progress, and tool cards. Files holds the actual deliverables. Monitor records execution, and remote receipts provide recoverable task identities. Review protects common file writes and remote submission, but some domain tools create their declared output files within one call. Prompts should still state output paths and stopping points, and users should inspect the resulting files.

These records support review but do not replace scientific judgment. Before submission, check the system, charge, spin, constraints, method, convergence settings, sampling conditions, energy references, and cost. The next chapter describes the five entry agents and gives reference prompts for each.
