# 5. Agents and modules

[Previous](04-webui.en.md) | [Contents](README.en.md) | [Next](06-computational-workflows.en.md)

The five entrypoints are user-visible work modes, not five prompt skins on one
agent. Each has distinct model roles, tools, skills, and delegation topology.
Choosing the wrong entrypoint may still produce an answer, but often adds
unnecessary planning, omits domain tools, or crosses responsibility boundaries.

## 5.1 Quick selection

| Goal | Entrypoint | Reason |
|---|---|---|
| One bounded structure, calculation, trajectory, or ML task | Experiment | Direct access to four computational workers |
| An open research problem spanning literature, computation, and writing | Research | Can delegate the other four specialist types by stage |
| Drafting or revising text from existing evidence | Writing | Writing workers, polishing, figures, and compilation |
| Formal review of one fixed PDF | Peer Review | Multiple reviewer reports and editor synthesis |
| Systematic search, full-text reading, evidence tables, and references | Literature Review | Search, controlled browser, local corpus, and citation finalization |

Use the narrowest entrypoint for a bounded task. Research is not a generic
"stronger mode"; it is for goals that truly need cross-module decisions.

## 5.2 Research

Research is a research coordinator that can delegate:

```text
Research
  -> Experiment Specialist
  -> Writing Specialist
  -> Peer Review Specialist
  -> LitReview Agent
```

Good uses:

- Build a plan and identify evidence gaps from an open scientific question.
- Review literature before deciding whether computation is needed.
- Integrate existing calculations and literature into a report or manuscript
  section.
- Maintain decisions, assumptions, artifacts, and next steps across stages.

Poor uses:

- One structure conversion with known input and output.
- Only finding papers or polishing one paragraph.
- Using a vague "research this" request to hide an undefined calculation.

Research is a decision and integration layer, not the owner of every
computational tool. Property lookups first use workspace evidence and
literature. When evidence is missing, it should state the gap and ask whether to
calculate, not launch DFT or ORCA without authorization.

Delegation is sequential inside a shared workspace. CatMaster delegates one
specialist or worker, waits for its result, and then decides the next step. This
avoids parallel agents rewriting the same files.

## 5.3 Experiment

Experiment coordinates computational research and quality control. Its
coordinator can inspect available remote tasks and search or download Materials
Project structures. Domain work is normally delegated to four workers.

### Materials worker

Main capabilities:

- Bulk cells, supercells, structure standardization.
- Surface cuts, terminations, steps, defects, dopants, and adsorption sites.
- VASP and CP2K input preparation, batch stages, NEB, and dimer paths.
- Phonon, elastic, band, DOS, thermodynamic, and k-path helpers.
- SP, relaxation, and path work with MACE, UMA, MatterSim, and ORB.
- Structure, coordination, constraint, trajectory, and output audits.

### Dynamics worker

Main capabilities:

- CP2K AIMD preparation, restart, and analysis.
- LAMMPS minimization, MD, restart, and potential layouts.
- MLFF MD preparation and execution.
- Trajectory health, temperature, energy, drift, diffusion, and structural
  evolution analysis.

### ML worker

Main capabilities:

- Training and validation dataset curation.
- Active-learning candidate management.
- MACE training, fine-tuning, evaluation, and benchmarking.
- Reusable lightweight project scripts when no dedicated tool covers the task.

### ORCA/xTB worker

Main capabilities:

- Molecule and conformer generation from SMILES or structures.
- xTB optimization, energy, solvation, and short MD.
- CREST conformer search.
- ORCA geometry optimization, frequency, thermochemistry, scans, TS, IRC,
  TDDFT, and NMR.
- Conformer ensembles and molecular MLFF prescreening.

Workers prepare and inspect stages, but registered scientific engines use
managed remote execution. If machine, resource, or task configuration is
missing, the correct behavior is to report it instead of silently running the
engine on the control plane.

## 5.4 Writing

Writing produces manuscripts from existing evidence:

```text
Writing Specialist
  -> writing_worker_agent
  -> writing_polisher_agent
```

`writing_worker_agent` handles one bounded section, integration task, figure,
TeX compilation, or Markdown PDF. `writing_polisher_agent` performs conservative
language edits and should not alter numbers, citations, evidence scope,
conclusion strength, or technical structure.

Good uses:

- Draft an abstract, methods, results, or discussion from notes, tables, and
  figures.
- Revise existing Markdown, LaTeX, or extracted Word text.
- Normalize terminology and repair paragraph logic or language.
- Build scientific figures, compile a PDF, and inspect the compiled layout.

Writing does not invent experimental results or launch calculations. If source
support is missing, route the problem to Literature Review or Research instead
of generating plausible-looking references.

## 5.5 Peer Review

Peer Review accepts one canonical PDF, produces an independent report for every
label in `peer_review_models`, and then synthesizes an editor decision. One model
label corresponds to one reviewer report.

Typical outputs:

- Complete reports from each reviewer.
- Major and minor concerns with verifiable locations.
- Agreement and disagreement across reviewers.
- An editor synthesis or decision memo.

Use it for pre-submission review, revision quality control, or independent model
cross-review. It is not the mode for directly rewriting the paper or answering
every comment. Pass the review artifact to Writing or Research for revisions.

Identify exactly which PDF is the canonical manuscript. Do not provide several
near-identical versions without priority.

## 5.6 Literature Review

Literature Review uses one LitReview DeepAgent that combines:

- Public web search.
- Controlled `agent-browser`, including a session the user has legitimately
  authorized.
- Local literature ingestion and query.
- DOI, metadata, evidence-table, and citation finalization tools.
- Literature-reading and writing-quality skills.

It is suitable for paper discovery, full-text reading, thematic reviews,
evidence matrices, method comparison, and reference organization. It does not
run calculations or produce a complete manuscript.

The controlled browser does not bypass paywalls, CAPTCHAs, OTPs, or security
warnings. The user completes sign-in. A discovered abstract is not equivalent to
full-text access. Reports should distinguish metadata, abstract, full text, and
user-supplied evidence.

## 5.7 General-purpose subagent

The DeepAgents runtime may show a `general-purpose` subagent. It isolates context
inside the same responsibility lane, inherits the parent's tools, and receives
no new authority. It cannot bypass worker, remote-task, or safety boundaries.
Users normally do not need to request it by name.

## 5.8 Tools and skills

A tool performs one action, such as reading a file, inspecting a structure, or
submitting a stage. A skill is a workflow and checklist. Having a skill in a
worker's context does not grant every similarly named tool. The runtime allowlist
and task audience remain authoritative.

At run time, built-in skills are staged into the project and combined with
workspace `self_develop_skills`. A project skill can override a built-in skill of
the same name, but it loads only from the next run.

## 5.9 Selection examples

```text
"Expand this POSCAR to 3x3x1 and preserve Selective Dynamics"
-> Experiment

"Compare literature and computational evidence for Pt single atoms at CeO2
sites, then propose the next calculations"
-> Research

"Draft Results from results.csv and Figure 2 without adding data or citations"
-> Writing

"Give three reviewer reports and an editor assessment for manuscript.pdf"
-> Peer Review

"Find anti-sintering Pd catalyst papers since 2021 and build a DOI-deduplicated
evidence table"
-> Literature Review
```

If a task changes character during execution, stop at a clear boundary, save the
current artifacts, and continue in a new thread with the suitable entrypoint.
The entrypoint cannot change during a run.
