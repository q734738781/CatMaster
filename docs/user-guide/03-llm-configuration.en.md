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

A single result does not turn a hypothesis into a terminal "supported" or "rejected" fact. The interface derives its evidence state from all incoming result relationships. An Experiment may have several Results, so a replicate adds evidence without overwriting an earlier observation.

The graph and notes have different jobs. Full papers, long analyses, structures, figures, logs, reports, and remote receipts stay in Files, artifacts, runs, or notes. A graph node keeps the short scientific statement and controlled references to those sources. Every bound Research turn receives a relevant subgraph with strict node and character limits. It never receives every graph in the workspace. When several active graphs exist, the user or thread binding must select one explicitly.

Each graph has an editable completion criterion. Users can create seed hypotheses, experiment proposals, and observations or Results from the project, collaborators, or literature. A scientific input can attach a DOI, URL, note, artifact, run, thread, or message at creation time. Users can also start a bound Research planning thread from a Result. The `hypothesis_proposer` reads the relevant graph evidence and can directly search the web, controlled browser, and local literature corpus before proposing temporary branches with DOI or URL sources. It returns an ordinary-language scientific memo and uses one bound staging tool only when there is a useful preview to publish; graph and revision IDs are not relayed through agent prose. `evidence_judge` likewise returns a concise scientific assessment, and Research records only the hypothesis effects actually addressed by the Result. Planning compares temporary branches with the complete runnable Experiment frontier and explains one recommendation when the evidence distinguishes a useful next step. No numeric route score is stored. Unselected branches are not written into the durable graph; the WebUI renders temporary branches as translucent nodes. Manual mode materializes a route only after a click, while automatic mode materializes only the selected route. After every graph change, including a new Result, automatic mode plans first and then runs at most one real Experiment. It stops when the completion criterion is satisfied.

Research Graph does not bypass normal ownership. Literature checks still belong to Literature Review. DFT and experimental work still go through Experiment, the appropriate worker, managed execution, and any required approval. One-off answers and simple linear tasks do not need a graph.

<details>
<summary>Current roles, tools, and skills available to Research</summary>

Research delegates scientific plan formation to `hypothesis_proposer`, evidence interpretation to `evidence_judge`, and execution to `experiment_specialist`, `writing_specialist`, `peer_review_specialist`, or `litreview_agent`. It retains common workspace, task-planning, and project-memory capabilities but does not directly own VASP, slab, or remote-submission tools. The proposer has web, controlled-browser, and local literature retrieval but no scientific experiment execution; the judge has no search, proposal, or execution tools.

Research can list, create, inspect, and edit graphs; add Hypotheses, Experiments, Results, judgments, and sources; and mark a real blocker. Ordinary mutations require the inspected graph ID and current revision and do not accept generic metadata. The internal `stage_research_plan` action instead derives both from its trusted planning-thread binding. Persistent state lives in the workspace `metadata/workspace.sqlite`. The graph stores scientific nodes, relationships, and refs. Detailed execution records and resource use remain in the existing thread, receipt, and artifact stores.

Its research skills include `research-graph-control`, `nature-citation`, `nature-data`, `nature-experiment-log`, `nature-figure`, `nature-literature-pipeline`, `nature-paper-to-patent`, `researchwrite`, `nature-reader`, `nature-ref-verifier`, and `nature-writing`. Calculation execution moves into Experiment and its worker skills.

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

Its autonomy appears in worker selection and in the way it adapts after intermediate results. An adsorption screen may begin with slab and site construction, use MLFF to remove clearly poor candidates, and prepare DFT only for the small set that survives. The user does not need to switch workers manually, but should state which approximations are allowed, whether remote computation is authorized, and which scientific choices require confirmation.

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

Literature Review does more than rewrite search snippets. It discovers papers, obtains legitimately accessible text, distinguishes metadata from abstract and full-text evidence, deduplicates records, reads selected sources, builds evidence tables, and finalizes citation metadata after papers have been chosen.

It can begin with public web search and, when configured, open a controlled browser. That browser may reuse the user's authorized institutional session, but it does not bypass CAPTCHA, paywalls, or security warnings. Accessible PDF, HTML, XML/JATS, Markdown, and other readable text can be read directly; corpus ingestion is optional when repeated question-focused retrieval is useful. A mixed ingest keeps successful documents and reports unreadable files separately. Citation finalization uses a deterministic batch tool rather than asking the model to guess every bibliographic field.

This entry supports topic reviews, method comparisons, full-paper reading, bilingual readers, claim-evidence matrices, citation placement, reference verification, and full-text availability records. It does not run materials calculations, and it must not write detailed method claims from title or abstract evidence alone.

<details>
<summary>Current Literature Review tools and skills</summary>

Direct tools are `web_search`, `ingest_literature_files`, `query_literature_corpus`, and `finalize_citations`. Search follows the model bound to the role: `codex_oauth` and OpenAI Responses models use hosted `web_search`, while other providers use CatMaster's Tavily-backed function. CatMaster binds only one of those implementations to an agent. A working `agent-browser` installation adds a separate filtered browser surface for dynamic pages and user-authorized sessions.

Core skills include `nature-academic-search`, `nature-downloader`, `nature-reader`, `nature-citation`, `nature-ref-verifier`, and `nature-literature-pipeline`. They cover search strategy, legitimate full-text acquisition, figure-aware reading, claim-level citation support, metadata verification, and larger literature workflows.

</details>

Reference prompt:

```text
Use Literature Review to study anti-sintering strategies for Pd catalysts published since 2021, with emphasis
on isolated atoms on oxide supports and reversible redispersion. Design a broad search, save the strategy,
and deduplicate titles, DOIs, and versions.

Distinguish records that were only discovered from papers read at abstract, full-text, or supplementary level.
Read the sources that directly discuss stabilization mechanisms, migration, or sintering experiments. Build a table
of material, conditions, evidence type, conclusion, and limitation. Save the candidate table, unavailable list,
evidence table, and final reference library. Do not invent parameters that cannot be verified.
```

## Writing agent: turning evidence into manuscripts and figures

Writing is for work that already has source material. You can give it notes, result tables, figures, references, existing sections, or a venue template. It can draft, restructure, polish, lay out, and compile. The coordinator delegates substantive composition to a writing worker and conservative language revision to a polisher so that a prose pass does not casually change scientific structure or stance.

Its scope is much wider than English editing. Current skills cover manuscript sections, proposals, data-availability statements, citations and reference verification, publication figures, presentations, reviewer responses, pre-submission review, Chinese patent drafts, ACS LaTeX, Markdown PDF, and venue templates. It can read bounded PDF and Office content, work with existing LaTeX, produce editable figures, and compile deliverables.

Writing must not invent results or add plausible references to fill a gap. Missing literature should go to Literature Review. Missing computation should be reported explicitly or coordinated through Research.

<details>
<summary>Current Writing roles, tools, and skills</summary>

The entry agent can use `generate_nanobanana_figure` and `review_pdf_manuscript`, and it delegates to `writing_worker_agent` and `writing_polisher_agent`. The writing worker can also use `polish_academic_prose`, `compile_text`, and `render_markdown_pdf`, along with common file and lightweight scripting capabilities.

Available skills include `nature-writing`, `nature-polishing`, `nature-citation`, `citation-management`, `nature-data`, `nature-figure`, `nature-reader`, `nature-response`, `nature-reviewer`, `nature-paper2ppt`, `nature-paper-to-patent`, `nature-ref-verifier`, `nature-academic-search`, `researchwrite`, `scientific-writing`, `scientific-visualization`, `achemso-latex-manuscript`, `venue-templates`, `markdown-pdf-export`, and the `humanizer` quality skill.

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
