# 3. The five agents: from objectives to deliverables

[Previous](02-concepts.en.md) | [Contents](README.en.md) | [Next](04-webui.en.md)

The WebUI exposes five specialist roles—Research, Experiment, Writing, Peer Review, and Literature Review—and adds Persistent Research as a continuous Research entry. These are not prompt styles placed on top of the same chatbot. Each role has a distinct delegation structure, tool surface, and set of skills; Persistent Research reuses the Research role and scientific state.

The useful question is not which entry is strongest. Ask what the main deliverable of this session should be. A broad entry can add needless coordination to a small task. A narrow entry may lack the worker required for a cross-stage objective.

## Research Agent: coordinate a scientific objective

Research delegates literature, computational, writing or formal review work and
closes when the requested stage is delivered. Persistent Research shares the same
ResearchSpecialist builder, model role, tools and specialist hierarchy, continuing
within the authorized objective on the same native thread.

Persistent Research creates and binds an auto Graph when needed. The Graph belongs
to the workspace and preserves Hypotheses, Experiments, Results and sources across
threads; its current root owns execution coordination. DBOS delivers child
completion turns to that root. Graph recovery reconciles accepted work and forwards
relevant evidence edits; it does not launch another planner or candidate tournament.

Independent scientific questions can run as concurrent ResearchSpecialist branches.
Each owns a complete hypothesis–method–result investigation with isolated context,
while sharing the workspace Graph. Native Experiment, Literature Review and other
delegates return synchronously within a branch; the main session integrates their
findings and owns overall completion.

Background tasks use coarse expected verification cost: low for literature and small
data baselines, medium for MLFF exploration or substantial training, high for DFT
campaigns including their prerequisites. Deployment defaults are a total of 16 and
low/medium/high caps of 8/4/2, configurable under `persistent_research` in the LLM
configuration. Computational waits retain the task's slot. These caps do not limit
Slurm jobs or batches. A researcher can change a formed task's cost at a safe native
checkpoint, queue and automatically resume the same context. Admission does not
authorize new scientific work or require an estimate of compute hours.

Result details preserve `methods`, `summary` and `conclusion`: the actual method,
observations, and scoped interpretation with open questions. The Graph UI edits all
three. Missing fields in old records display `missing due to old record`, which is
missing knowledge rather than negative evidence.

The existing `hypothesis_proposer` performs independent scientific reasoning from
Results, earlier evidence and counterevidence to scoped interpretations, revised
hypotheses and useful next checks. It can incrementally edit scientific records and
read complete graph/source evidence. Actual execution belongs to the existing
specialists. Routine Results do not require another independent summary.

Before unfinished persistent research stops for scientific stagnation, the root
records the reason and the native closeout path triggers one independent review.
A concrete feasible authorized remedy normally receives one bounded validation.
Actual authorization/input/resource barriers, a moot or equivalent completed check,
or an achieved user goal can override that default. Without materially new evidence,
retain partial findings, open premises and resumption conditions. User pause and
real budget boundaries take priority; renaming an unchanged issue or changing agents
does not justify another review.

Literature research or experimental recommendations never authorize new DFT, MD or
laboratory execution. Actionable laboratory proposals use the `external` lane for
later Result entry. Deliver and stop when the requested stage is satisfied. Graph
edits, judgment revisions and late evidence do not reopen completed delivery.

Judgment edges carry `scope` and `rationale`. A new H or R can `revise` an older
same-kind claim through a `revises` link with action `replace`, `qualify` or `withdraw`;
both records remain available. Findings under different conditions can coexist,
and failure of one proxy does not automatically erase other observations. Key
stopping decisions and validation outcomes remain linked to H/E/R in
`research_decisions`.

The Graph is the scientific index; original papers, data, reports and long analyses
remain in Files, artifacts, runs or notes. Focus context is partial navigation.
`query_research_graph_sql` preserves standard SQL, JSON1, recursive relation queries
and pagination over all bound scientific records. Follow revisions and decisive
sources before reusing an older conclusion.

Research writes incremental H/E/R through the existing tools, changes scoped
interpretations with `set_research_result_judgment`, links revisions with
`revise_research_claim` and records unfinished-stage disposition with
`record_research_disposition`. The independent reasoner persists its stopping review
with `record_research_review`. Existing graph transactions handle write conflicts;
DBOS owns execution and queues; native LangGraph owns checkpoints.

Example:

```text
Review recent CO2 reduction research on Cu electrodes and recommend potential new experiments.
Check decisive sources and reactor conditions; distinguish findings, inference and hypotheses.
This stage authorizes literature synthesis and recommendations only, without calculations or lab work.
Save the report, sources and Research Graph, then stop after delivering the recommendations.
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

It begins with search summaries and scholarly metadata. When a selected paper needs deeper reading, one high-level acquisition tool tries legal open-access repositories and indexes first, uses a configured official Elsevier API route for matching DOIs, and then may make one internal ScanSci browser pass on the DOI landing page. It verifies any main PDF, can save Supplementary Information when explicitly requested for a DOI, and otherwise saves the readable static page as a local source artifact. The agent reads those local artifacts; it never controls browser state or page actions itself. Corpus ingestion remains optional for repeated question-focused retrieval.

This entry supports topic reviews, method comparisons, full-paper reading, bilingual readers, claim-evidence matrices, citation placement, and reference verification. It does not run materials calculations or write detailed method claims from evidence that does not contain those details. When partial evidence materially limits a conclusion, it explains that limitation in ordinary language rather than requiring a confidence field for every paper.

<details>
<summary>Current Literature Review tools and skills</summary>

Direct tools include `web_search`, `acquire_literature_source`, `batch_acquire_literature_sources`, `ingest_literature_files`, `query_literature_corpus`, and `finalize_citations`. Web search follows the model bound to the role: `codex_oauth` and OpenAI Responses models use hosted `web_search`, while other providers use CatMaster's function. That function uses Tavily when available and can degrade to scholarly-index discovery after a classified Tavily failure; the result names the actual backend. CatMaster binds only one `web_search` implementation to an agent. Source acquisition uses pinned ScanSci, the optional official Elsevier API, and browser-backend integrations internally without exposing raw browser operations. Batch acquisition accepts a direct list or workspace list file and has a hard limit of 50 inputs.

Literature Review and its workers also receive `search_openalex`, `search_semantic_scholar`, `get_openalex_record`, `get_semantic_scholar_record`, and `recommend_semantic_scholar` for native scholarly metadata queries. Citation export defaults to one BibTeX `.bib` file; another format requires an explicit choice.

The local `literature-evidence-use` skill covers scoped discovery, source reading, evidence attributes and citation finalization. Tools own acquisition and access handling; optional scripts preserve complete citation metadata.

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

The Codex OAuth template uses Astra medium for the writing coordinator, prose,
presentations, plotting and review. Research main roles use Astra high and compute
workers/helpers use Astra medium. The default literature worker uses GPT-6 Luna xhigh;
title generation uses GPT-6 Luna low. Effective effort comes from each profile's
`reasoning.effort`; already-built agents retain their model configuration.

Writing is for work that already has source material. You can give it notes, result tables, figures, references, existing sections, or a venue template. It can draft, restructure, polish, lay out, and compile. The coordinator delegates drafting, revision, explicit prose polishing, and final integration to one writing worker, and quantitative or data-native figures to a plot worker. The configured `section_writer` model therefore owns the complete author-facing prose pass.

Writing uses shared `scientific-communication` guidance to connect the user question, audience, findings, evidence and interpretation. Manuscript argument, venue and submission instructions are selected for the actual task. Phrasing examples are an optional reference, without compulsory keyword audits, paragraph counts or multiple drafts. The coordinator retains whole-document judgment while workers own coherent section groups and final integration. Compact briefs distinguish user constraints and scientific conditions from adjustable editorial suggestions.

Its scope is much wider than English editing. Current skills cover manuscript sections, proposals, data-availability statements, citations and reference verification, publication figures, presentations, reviewer responses, pre-submission review, Chinese patent drafts, ACS LaTeX, Markdown PDF, and venue templates. It can read bounded PDF and Office content, work with existing LaTeX, produce editable figures, and compile deliverables.

Writing must not invent results or add plausible references to fill a gap. Missing literature should go to Literature Review. Missing computation should be reported explicitly or coordinated through Research.

<details>
<summary>Current Writing roles, tools, and skills</summary>

The entry agent can use `generate_figure` and `review_pdf_manuscript`, and it delegates to `writing_worker_agent`, `plot_worker`, and `presentation_worker`. The writing worker uses normal workspace file capabilities together with `generate_figure`, `compile_text`, and `render_markdown_pdf`; there is no separate polisher agent or direct prose-overwrite tool. The plot worker uses supplied quantitative data directly, writes reproducible matplotlib code with explicitly configured Origin-like axes, ticks, typography, and line work rather than library defaults, uses the Nature/NPG palette for ordinary categorical plots, and inspects the rendered preview for clipping, collisions, and overlap between text and scientific signals.

Writing loads `publication-launch-writing`, `citation-management`, `scientific-visualization`, `achemso-latex-manuscript`, `venue-templates`, and `markdown-pdf-export`, together with shared `scientific-communication` guidance. Study-design reporting is an on-demand reference under `publication-launch-writing`. The plot worker uses `publication-data-plotting`, including the customer-validated Origin style, NPG palette and rendered-figure inspection requirements. The presentation worker receives the presentation, writing-quality and plotting skill roots: EasySlides supplies editable deck production, shared guidance covers scientific content and prose, and the plotting skill supports direct data figures. Its model role can be configured independently and falls back to `section_writer`; see [EasySlides](../easyslides.md).

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
