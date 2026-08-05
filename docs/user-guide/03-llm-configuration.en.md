# 3. The five agents: from objectives to deliverables

[Previous](02-concepts.en.md) | [Contents](README.en.md) | [Next](04-webui.en.md)

The WebUI exposes five entries: Research, Experiment, Writing, Peer Review, and Literature Review. They are not five prompt styles placed on top of the same chatbot. Each entry has a different role, delegation structure, tool surface, and set of skills.

The useful question is not which entry is strongest. Ask what the main deliverable of this session should be. A broad entry can add needless coordination to a small task. A narrow entry may lack the worker required for a cross-stage objective.

## Research agent: coordinating open objectives

Research is for questions that have not yet been reduced to one bounded task. It keeps the original objective active, decides whether the current evidence gap belongs to literature, computation, writing, or review, and delegates one stage at a time to Literature Review, Experiment, Writing, or Peer Review. After a delegated stage returns, Research checks whether the evidence actually advances the objective before choosing the next step.

For example, "Explain why isolated Pd resists sintering on CeO2" is not a single calculation. Research may first ask Literature Review to map reported mechanisms and characterization evidence. It can then ask Experiment which structural or energetic uncertainty is genuinely calculable, and later ask Writing to integrate literature and computed evidence. A missing literature value does not silently authorize DFT. New computation needs a scientific reason and user approval.

Research is a good fit for building an evidence-aware plan, comparing literature with existing project data, maintaining hypotheses and evidence gaps across stages, and closing a requested stage with limitations instead of expanding forever. It is not the best entry for a one-step supercell conversion or a narrowly scoped literature search.

Research can create a workspace Research Graph when a study must continue across threads, retain competing explanations, or apply one result to several hypotheses. The graph is not chat history and does not belong to the thread that created it. It stores three concise scientific node types:

- A Hypothesis contains a falsifiable claim, rationale, and observable predictions.
- An Experiment contains an objective, plan summary, decision rule, and execution lane.
- A Result contains a concise observation or outcome. A Result produced by a graph Experiment keeps that relationship; a literature finding, collaborator result, or historical observation can enter directly without inventing a retrospective Experiment. Typed relationships state which hypotheses it supports, opposes, or does not distinguish.

A single result does not turn a hypothesis into a terminal "supported" or "rejected" fact. The interface derives a Result-relationship summary from all incoming `supports`, `opposes`, and `inconclusive` edges. An Experiment may have several Results, so a replicate adds another observation without overwriting an earlier one.

The graph and notes have different jobs. Full papers, long analyses, structures, figures, logs, reports, and remote receipts stay in Files, artifacts, runs, or notes. A graph node keeps the scientific statement and controlled references to those sources. A planning turn starts with a partial focus snippet that contains the focus neighborhood and the complete runnable frontier. The bound read-only `query_research_graph_sql` surface keeps every node, relationship, reference, and referenced owner record in the selected graph reachable without copying the whole graph into the prompt. When several active graphs exist, the user or thread binding must select one explicitly.

Each graph has an editable completion criterion. Users can create seed hypotheses, experiment proposals, and observations or Results from the project, collaborators, or literature. A scientific input can attach a DOI, URL, note, artifact, run, thread, or message at creation time. Users can also start a bound Research planning thread from a Result. The `hypothesis_proposer` compares that Result with existing predictions and Results before deciding whether the current Hypotheses are enough or a distinct, falsifiable Hypothesis is needed. It can query the graph, search the web and local corpus, and read or acquire selected sources. The shared `evidence_judge` independently identifies only the Result-to-Hypothesis effects that the evidence actually addresses; an empty judgment set is valid. Automatic Experiment and Literature Review paths perform this judgment before the atomic Result writeback.

A Result keeps observation, derived analysis, interpretation, scientific modality, applicable conditions, and provenance in ordinary scientific language. It does not receive a global strength grade. The `supports`, `opposes`, and `inconclusive` edges describe its relationship to one Hypothesis, not the strength of the Result. Planning stages temporary branches without materializing them. A separate evaluator gives every candidate Experiment an innovation score and a conservative score for the current graph revision only. The durable Experiment has no route score, and any graph mutation invalidates the preview scores. Manual mode shows both recommendations. Automatic mode uses the conservative recommendation and waits when the recommendation or evaluation is missing, invalid, stale, or deliberately empty. It still runs at most one real Experiment and stops when the completion criterion is satisfied.

Research Graph does not bypass normal ownership. Literature checks still belong to Literature Review. DFT and experimental work still go through Experiment, the appropriate worker, managed execution, and any required approval. One-off answers and simple linear tasks do not need a graph.

<details>
<summary>Current roles, tools, and skills available to Research</summary>

Research delegates scientific plan formation to `hypothesis_proposer`, candidate evaluation to `experiment_evaluator`, evidence interpretation to `evidence_judge`, and execution to `experiment_specialist`, `writing_specialist`, `peer_review_specialist`, or `litreview_agent`. It retains common workspace, task-planning, and project-memory capabilities but does not directly own VASP, slab, or remote-submission tools. The proposer, evaluator, and judge use the narrow `research_reasoning` skills and read-only graph and source capabilities; they do not receive graph mutation, file-writing, shell, patch, or scientific execution tools.

Research can list, create, query, and edit graphs; add Hypotheses, Experiments, Results, judgments, and sources; and mark a real blocker. `query_research_graph_sql` accepts only a read-only SQL statement; the host binds the workspace, graph, revision, and referenced owner rows from the trusted thread. Ordinary mutations require the graph ID and current revision and return the exact changed entity and latest revision. The internal planning actions derive their binding from the planning thread. `stage_research_plan` writes only a disposable preview, while evaluation, materialization, and launch are separate transitions. Persistent state lives in the workspace `metadata/workspace.sqlite`. Detailed execution records and resource use remain in the existing thread, receipt, and artifact stores.

Its research skills include `research-graph-control`, `nature-citation`, `nature-data`, `nature-experiment-log`, `nature-figure`, `nature-literature-pipeline`, `nature-paper-to-patent`, `researchwrite`, `nature-reader`, and `nature-ref-verifier`. Manuscripts and other author-facing publication writing move into Writing and its unified `nature-writing` skill. Calculation execution moves into Experiment and its worker skills.

</details>

Reference prompt:

```text
Use Research to study possible mechanisms that stabilize isolated Pd on CeO2 surfaces.

Inspect the existing literature/, structures/, and calculations/ material first. Separate claims that already
have evidence from hypotheses that remain unsupported. Decide when Literature Review, Experiment, or Writing
is needed, but advance one bounded stage at a time and inspect its artifacts before delegating again.

The deliverable for this turn is an evidence map and a recommendation for the next stage. Do not launch DFT
just because a value is missing from the literature. If a new calculation would distinguish specific hypotheses,
describe its required inputs, cost, and evidential value, then wait for my approval.
```

## Experiment agent: organizing modeling, computation, and validation

Experiment handles bounded computational research. It reads the scientific objective and current inputs, then delegates work to Materials, Dynamics, ML, or ORCA/xTB workers. The coordinator can search and download Materials Project structures and inspect the deployment's task catalog. Domain modeling, input preparation, analysis, and remote submission belong to the worker that owns the method.

Its autonomy appears in worker selection and in the way it adapts after intermediate results. An adsorption screen may begin with slab and site construction, use MLFF to remove clearly poor candidates, and prepare DFT only for the small set that survives. The user does not need to switch workers manually, but should state which approximations are allowed, whether remote computation is authorized, and which scientific choices require confirmation. Experiment briefs preserve those scientific boundaries while leaving tool order, compatible execution routing, input-level repairs, and bounded recovery to the worker. A failed specialist-selected worker or route is handled first by an equivalent revised delegation, not by asking the user; human input is reserved for changes to user-controlled science, authority, cost, time, or safety boundaries.

The four workers cover complementary domains:

- Materials handles discovery, bulk and surface structures, adsorption, defects, VASP and CP2K, managed MLFF inference, paths, electronic properties, phonons, elasticity, and thermochemistry.
- Dynamics handles CP2K AIMD, LAMMPS, MLFF MD, restart continuity, trajectory health, and diffusion-related analyses.
- ML handles training data, MACE training and evaluation, and active-learning candidate selection.
- ORCA/xTB handles molecules, conformers, xTB, CREST, ORCA, TS, IRC, TDDFT, and NMR.

Chapter 5 expands each worker and its current tools and skills. Chapter 6 follows complete modeling workflows rather than worker boundaries.

<details>
<summary>Tools owned directly by the Experiment coordinator</summary>

Experiment can use `mp_search_materials` and `mp_download_structure` for Materials Project discovery. It can call `get_avail_remote_task` to understand what the deployment exposes to workers. It does not bypass the worker layer to call `remote_submission` directly.

</details>

Reference prompt:

```text
Use Experiment to inspect structures/POSCAR and build a reviewable set of surface candidates for CO adsorption.

Identify the material, cell, and existing Selective Dynamics first, then choose the appropriate workers, skills,
and tools. Compare reasonable (111) terminations, create representative adsorption sites and CO starting
geometries, and retain provenance and structure checks at every stage.

Do not prepare every possible VASP job at the outset. Reduce candidates using geometry and coordination first,
and explain any chemical choices that still need my decision. You may create structures and reports in this turn,
but do not submit remote computation.
```

## Literature Review agent: building traceable evidence

Literature Review works from the evidence that is actually available. Search summaries and abstracts can support claims they explicitly make; title and bibliographic metadata establish discovery only. The agent preserves those boundaries, deduplicates records, synthesizes evidence, and finalizes citation metadata without making full-text acquisition a per-paper completion requirement.

It begins with search summaries and scholarly metadata. When a selected paper needs deeper reading, one high-level acquisition tool tries legal open-access repositories and indexes first, then one internal ScanSci/CloakBrowser pass on the DOI landing page when available. It verifies any resulting PDF and otherwise saves one static public-page snapshot. The agent reads the local artifact; it never controls browser state or page actions itself. Corpus ingestion remains optional for repeated question-focused retrieval.

This entry supports topic reviews, method comparisons, full-paper reading, bilingual readers, claim-evidence matrices, citation placement, and reference verification. It does not run materials calculations or write detailed method claims from evidence that does not contain those details. When partial evidence materially limits a conclusion, it explains that limitation in ordinary language rather than requiring a confidence field for every paper.

<details>
<summary>Current Literature Review tools and skills</summary>

Direct tools are `web_search`, `acquire_literature_source`, `ingest_literature_files`, `query_literature_corpus`, and `finalize_citations`. Search follows the model bound to the role: `codex_oauth` and OpenAI Responses models use hosted `web_search`, while other providers use CatMaster's function. That function uses Tavily when available and can degrade to scholarly-index discovery after a classified Tavily failure; the result names the actual backend. CatMaster binds only one search implementation to an agent. Source acquisition uses pinned ScanSci and CloakBrowser integrations internally without exposing raw browser operations.

Core skills include `nature-academic-search`, `nature-reader`, `nature-citation`, `nature-ref-verifier`, and `nature-literature-pipeline`. Source acquisition guidance is integrated into `nature-academic-search`; tools own the actual download and validation behavior.

</details>

Reference prompt:

```text
Use Literature Review to study anti-sintering strategies for Pd catalysts published since 2021, with emphasis
on isolated atoms on oxide supports and reversible redispersion. Design a broad search, save the strategy,
and deduplicate titles, DOIs, and versions.

Distinguish records that were only discovered from papers read at abstract, full-text, or supplementary level.
Form a bounded synthesis from abstracts first, and read source text only when a conclusion depends on exact
conditions, values, or figures. Build a table of material, conditions, evidence basis, conclusion, and limitation. Save the candidate table,
evidence table, and final reference library. Do not invent parameters that cannot be verified.
```

## Writing agent: turning evidence into manuscripts and figures

Writing is for work that already has source material. You can give it notes, result tables, figures, references, existing sections, or a venue template. It can draft, restructure, polish, lay out, and compile. The coordinator delegates substantive composition to a writing worker, quantitative and data-native figures to a plot worker, and conservative language revision to a polisher.

The coordinator and text workers apply an academic-launch policy by default. They identify the strongest evidence-supported publishable value, reconstruct the title-to-conclusion argument around it, keep comparisons bounded to claims the evidence can support, and use experiments and displays as argument components rather than a result inventory. Journal-facing prose excludes project chronology, unsolicited defensive self-assessment, and irrelevant hardware, launcher, software-build, platform, or performance detail. Necessary qualifications remain precise and limited to conditions that materially change the core claim.

Its scope is much wider than English editing. Current skills cover manuscript sections, proposals, data-availability statements, citations and reference verification, publication figures, presentations, reviewer responses, pre-submission review, Chinese patent drafts, ACS LaTeX, Markdown PDF, and venue templates. It can read bounded PDF and Office content, work with existing LaTeX, produce editable figures, and compile deliverables.

Writing must not invent results or add plausible references to fill a gap. Missing literature should go to Literature Review. Missing computation should be reported explicitly or coordinated through Research.

<details>
<summary>Current Writing roles, tools, and skills</summary>

The entry agent can use `generate_nanobanana_figure` and `review_pdf_manuscript`, and it delegates to `writing_worker_agent`, `plot_worker`, and `writing_polisher_agent`. The writing worker can also use `polish_academic_prose`, `compile_text`, and `render_markdown_pdf`. The plot worker uses supplied quantitative data directly, writes reproducible plotting code, targets a clean Origin-like publication style, and inspects the rendered preview for palette quality, clipping, collisions, and overlap between text and scientific signals.

Available skills include `publication-launch-writing`, the unified `nature-writing`, `nature-polishing`, `nature-citation`, `citation-management`, `nature-data`, `nature-figure`, `nature-reader`, `nature-response`, `nature-reviewer`, `nature-paper2ppt`, `nature-paper-to-patent`, `nature-ref-verifier`, `nature-academic-search`, `researchwrite`, `scientific-visualization`, `achemso-latex-manuscript`, `venue-templates`, `markdown-pdf-export`, and the `humanizer` quality skill. `nature-writing` covers Nature-family and general-journal structure, IMRAD, and study-design reporting standards. The plot worker has the narrower `publication-data-plotting` skill.

</details>

Reference prompt:

```text
Use Writing to draft two Results subsections on surface stability from notes/result_contract.md,
data/summary.csv, figures/, and writing/references.bib.

Read the evidence first and propose the argumentative order, then select the relevant writing skills.
Every number, uncertainty, material name, and citation must trace to the supplied files. Do not add missing data
or new references. Write connected prose rather than an outline. Save the draft to writing/results_surface_v1.md
and include a short evidence note that identifies decisions still requiring an author.
```

## Peer Review agent: independently assessing a fixed manuscript

Peer Review starts from one canonical manuscript PDF. It sends the same PDF to the models listed under `peer_review_models`, collects independent reports, and produces an editor-level synthesis of novelty, method, evidence, reporting quality, and submission risk.

This differs from asking Writing to improve a paragraph. Peer Review keeps a referee perspective and does not directly rewrite the manuscript. Raw reports remain available because the editor synthesis may compress or select among them. The user decides which comments to accept, partly accept, clarify, or reject before handing a revision plan and source files to Writing.

<details>
<summary>Current Peer Review tools and skills</summary>

The main tool is `peer_review_request`, which sends one local PDF to every configured reviewer model and collects raw reports. The entry delegates a bounded episode to `peer_review_worker_agent`. That worker can read writing and writing-quality skills for review criteria and report quality, but it has no computation-worker tools.

</details>

Reference prompt:

```text
Use Peer Review on writing/submission/manuscript.pdf. This is the only canonical manuscript for this round.
The Supplementary Information is writing/submission/si.pdf.

Review it as a catalysis and computational-materials paper. Assess novelty, computational methods, structural
models, controls, evidence-to-claim fit, figures, and reproducibility. Preserve every complete reviewer report,
then produce an editor synthesis that distinguishes consensus, disagreement, required revisions, and optional
improvements. Review only in this turn. Do not edit source files or write an author response.
```

## How the five agents hand work over

The entries share a workspace, but their responsibilities do not automatically merge. Research can delegate other specialists within an open objective. Direct Experiment, Writing, Peer Review, and Literature Review sessions remain focused on their own work.

When a task naturally moves to a new stage, ask the current agent to save complete handoff artifacts first. A Literature Review evidence table and reference library can feed a new Writing thread. Peer Review reports can feed a Writing revision thread. Experiment can leave a result contract, tables, and figures for Writing. This is easier to audit than repeatedly changing the role of one thread.

The next chapter explains how to select entries, inspect delegation, review files, and steer a running agent in the WebUI. Model providers and role routing now live in Chapter 10 so that a first-time user does not need to learn configuration before understanding what CatMaster can do.
