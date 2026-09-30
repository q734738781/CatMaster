# 7. Literature, Writing, and Peer Review agents

[Previous](06-computational-workflows.en.md) | [Contents](README.en.md) | [Next](08-remote-execution.en.md)

Literature Review, Writing, and Peer Review share a workspace but have different evidence responsibilities. Literature Review discovers, verifies and synthesizes findings from sources. Writing uses existing evidence to create manuscripts and other deliverables. Peer Review independently examines a fixed manuscript. Keeping those roles distinct prevents the writer from inventing evidence while composing and keeps reviewer comments traceable to real revisions.

## Literature Review agent: from discovery to an evidence corpus

Literature Review can answer a focused verification question or conduct a larger topic review. The research question determines scale. A precise fact may need only a few strong sources; a topic review develops important advances, quantitative comparisons and mechanisms. Users can specify the date range, material or reaction system and intended use; the agent chooses search and reading depth accordingly.

The agent keeps the actual question, deliverable, relevance boundary, unresolved issues, and any user stopping instruction intact instead of converting every request into a full-review checklist. After each useful evidence batch it asks whether another search can still change the requested answer, its boundary, or the next decision. It synthesizes as soon as the bounded answer is supported and stops opening new branches when the user narrows or stops the search. There is no fixed paper count, branch count, or formal completion state behind this judgment.

### Discovery is not close reading

Public web search discovers papers, project pages, and database records. Title, author, and DOI metadata generally establish discovery only, but a complete or substantive abstract in the results can support the claims it explicitly makes. The agent preserves that boundary without discarding useful abstract evidence or extending it into unreported methods and values.

Literature Review and its workers can directly search OpenAlex and Semantic Scholar, look up known records, and request Semantic Scholar recommendations from a seed paper. OpenAlex search exposes the provider cursor; Semantic Scholar relevance search exposes its offset. Both accept page sizes up to 100. Small defaults are page sizes, not instructions to drain an entire result set. Records retain every provider author and the complete supplied abstract. Selected-source acquisition returns readable local artifacts for further inspection.

When a key decision genuinely depends on methods, conditions, values, figures, or supplementary material absent from the abstract, a controlled browser is one escalation path for open-access or user-authorized institutional content. After one reasonable access attempt fails, the agent states the limitation and continues with other evidence instead of cycling through pages or downloads. CAPTCHA, QR code, OTP, license prompts, and security warnings remain human actions.

Within one run, selected-source acquisition canonicalizes DOI, arXiv, PMID, and normalized URL identities and shares completed or in-flight outcomes between the parent and workers. Repeating the same identifier through the same route returns the existing source handle and access state rather than downloading it again. A retry requires a materially different authorized route or a transient or expired prior outcome; different editions are not merged.

```text
Review in situ research on aggregation and redispersion of single-atom catalysts since 2018.
Explain the main findings, influential factors and mechanisms using representative experimental data,
and include references.
```

### Local corpora support repeated project questions

Existing PDFs, Markdown, DOCX, and tables can be placed under `literature/` and ingested into a local corpus. The agent can then query the same material from several research angles while preserving source records. Parsed text may not capture every figure, formula, or supplementary detail, so important conclusions should return to the source page or publisher HTML.

The agent can prepare a bilingual, figure-aware reading document when requested, preserving source anchors and the requested reading depth. Source file reading and corpus retrieval remain available without a prescribed reader template.

```text
Read literature/papers/pd_redispersion.pdf closely and build a Chinese-English reader.
Preserve section order and place each important figure or table near the discussion it supports.
Keep page or source anchors for every block. Explain the key characterization, experimental results
and mechanistic significance of Pd redispersion in detail.
```

### Preserve findings, interpretation and sources

A literature handoff preserves useful findings, synthesis and sources. Evidence tables can organize observations, methods and conditions when comparison helps answer the question; the text still explains what those differences mean. Access depth and source independence are recorded when they affect interpretation, without requiring every source to fill a uniform verification table or receive a paper-level reliability grade.

Literature Review hands off the question, findings and interpretation in compact prose with source handles for recovery. Downstream writers reuse completed verification and may read relevant sources to understand the evidence, explain it or use a figure faithfully. This does not require revalidating unchanged evidence or creating a parallel manifest.

The literature coordinator delegates scientific question groups and returns findings, data and synthesis. Focused verification addresses an explicit checking request or a concrete inconsistency. The evidence-attribute reference supports those questions; ordinary reviews develop the subject. Literature notes leave final wording, chapter order and caption rules to the writing task. Literature reporting guidance is maintained separately from computational execution reporting.

A literature candidate pool records each candidate as `selected`, `deferred`, or `excluded`, with a concrete reason. Selected papers affect the current synthesis or decision, deferred papers are relevant but not needed yet, and excluded papers are out of scope, duplicates, or unable to answer the question. Access depth and claim relationship remain separate attributes. Selection does not assign a composite paper score or use journal prestige as an evidence grade.

Deduplicate title, DOI, preprint, and journal versions early. When a bibliography is part of the deliverable, `finalize_citations` resolves author, journal, year, DOI, and export records in one batch. It defaults to one `.bib` file, without equivalent JSON or Markdown copies. Set `output_format` to `json` or `md` only when explicitly needed; unresolved identifiers remain in the tool response. Optional export scripts also default to BibTeX. `citation-management` supports auditing existing bibliography metadata and flagging volume-year, author-order, pagination, and DOI conflicts.

```text
Use the papers under literature/corpus/ to explain Pd stabilization sites, oxygen-vacancy effects,
migration under redox conditions and redispersion mechanisms on CeO2.
Compare key experimental conditions and results in a table, and include references.
```

### Specialized search and citation work

Search scope follows the user's question and any explicit venue or date restriction. Available search and acquisition providers depend on the deployment. Report the sources actually accessed; a skill description does not establish access to a database.

<details>
<summary>Sources of Literature Review capability</summary>

Direct tools are `web_search`, `acquire_literature_source`, `batch_acquire_literature_sources`, `ingest_literature_files`, `query_literature_corpus`, and `finalize_citations`. `web_search` is provider-routed: OpenAI/Codex roles use hosted search and other providers use the CatMaster implementation with its configured fallback. Selected-source acquisition is direct-first, can use the configured official Elsevier API for matching DOIs, may use one internal ScanSci browser pass on a DOI landing page, validates main PDFs, optionally saves SI, and caches a static page once when no PDF is available. The batch interface accepts a direct identifier list or a workspace `.txt`, `.csv`, or `.tsv` list. Routine batches should group 10-30 sources with one scientific purpose; more than 50 input rows are rejected without truncation.

Native metadata tools are `search_openalex`, `search_semantic_scholar`, `get_openalex_record`, `get_semantic_scholar_record`, and `recommend_semantic_scholar`. They complement web search and are available to both Literature Review and its workers.

The local `literature-evidence-use` skill supplies scope and evidence-use guidance. Native tools handle discovery, source acquisition, ingestion, retrieval and final citations. Optional metadata-conversion scripts are available on demand.

</details>

## Writing agent: turning evidence into deliverables

Writing can begin from Chinese notes, result tables, figures, code output, a bibliography, an existing LaTeX project, an older PDF, or reviewer comments. Its job is not simply to polish the material. It interprets the writing target and evidence boundary, then selects writing skills for argument design, drafting, revision, figures, layout, or compilation.

Manuscripts, reports and presentations use free-text briefs carrying the user question, audience, useful findings and meaning, authoritative source paths, explicit user requirements and deliverable. Unspecified writing choices belong to the author's judgment and disciplinary conventions; briefs add no writing prohibitions, hypothetical misreading checklists or universal caption fields. Scientific conditions stay with the evidence, and authors select those that affect the reader's interpretation or decisions. Production instructions guide execution while the prose develops findings and implications. Section tasks retain shared purpose and relevant neighboring context; the coordinator judges coverage, coherence and reader relevance while workers select and reshape evidence notes into the document. Bibliography, conversion, compilation and local corrections remain within the relevant writing task, reusing completed checks.

Research and Writing distinguish explanatory scientific reports, internal technical records for informed readers, and venue-facing manuscripts in their handoffs, including what the reader already knows. Scientific reports and presentations without a specified audience default to a research advisor or experimental collaborator unfamiliar with the project's computational theory and internal history. They explain the question, what the comparison represents and what the results mean. Explicit technical summaries can remain concise, using tables, lists and appropriate shorthand; manuscripts follow venue and manuscript guidance. A compact handoff or delivery message does not limit document depth, and representative examples do not replace required coverage.

Authors write as knowledgeable colleagues addressing attentive readers: select the findings worth developing, explain relationships in the evidence and make supported interpretations and judgments clear. Detail follows explanatory value, with shared knowledge and routine implications left implicit. Avoid competing explanations and qualifications.

Writing assesses the actual artifact and its overall sequence against that purpose, including what the reader learns and whether the author's judgment is supported. Reading an existing source to explain a result or use a figure is distinct from repeating scientific validation. Corrections may be local or reorganize a chapter when the user authorizes reconstruction; optional aesthetic alternatives do not require endless revision.

### Use Research Graph to locate original evidence

When the current Writing thread explicitly attaches a Research Graph, the agent reads the partial focus and uses a read-only query to locate the relevant Hypotheses, every directly supporting, opposing, or inconclusive Result, their producing Experiments, and Sources. It then opens only the note, artifact, run, thread message, or literature source needed for the section. Claim-critical values, conditions, mechanisms, and limitations still come from the original owner material. If the graph does not cover a claim, Writing searches the workspace locally and states the gap. This adds no manuscript schema and gives Writing no graph mutation capability.

### Manuscripts, reports, and proposals

`publication-launch-writing` organizes manuscripts around the strongest evidence-supported contribution. Study-design reporting guidance is read only when applicable; venue templates supply required formats. Writing can draft or restructure manuscripts, reports and proposals from supplied evidence without a fixed chapter recipe.

Manuscripts describe the scientific work in an authorial voice. Incidental
drafting-session narration stays out of the prose, while agents, prompts, tools,
workflows, files and runs remain appropriate when they are research objects,
relevant methods or data, or required disclosures.

The shared `scientific-communication` skill connects concrete materials or
molecular systems with the observations, performance comparisons and interpretation
needed by the reader. Reports use relevant structures, experimental images and
plots alongside their explanation. Literature syntheses organize the scientific
questions and evidence; progress reports retain consequential negative results
and changes of direction. Methods and numerical results explain the findings,
while execution records and QC diagnostics remain a separate document purpose.
Research can route substantial reports from completed calculations to Writing
without starting new experiments. Missing claim-critical visuals are prepared
from existing evidence and integrated before an illustrated deliverable is final.

Research assesses scientific plausibility and evidence-claim fit before closeout.
Conditions and unresolved questions appear where they affect the answer; routine
reports, slides and artifact handoffs do not require an appended self-assessment.
An explicit request for verification or diagnosis receives that discussion.

A useful request identifies reader, document type, current section, available evidence, values that must remain exact, and content that must not be invented. Final prose should be connected paragraphs rather than a pile of outline fragments.

```text
Use Writing to rebuild the Discussion in writing/discussion_old.md for a catalysis-computation paper.
Trusted evidence is in notes/claims.md, data/final_results.csv, figures/, and references.bib.

Identify where the draft merely repeats Results and where literature comparison or limitations are missing.
Select appropriate writing skills and rebuild the argument as connected prose. Every number and citation must
come from the supplied files. Do not invent a mechanism. Preserve limitations on model domain and unvalidated
dynamics. Save the new draft and a concise revision note.
```

### Polishing, translation, and factual preservation

The writing worker uses `scientific-communication` and its optional phrasing reference to improve language and paragraph continuity while preserving numerical values, units, references, conclusion strength and scientific meaning. Chinese drafts can be translated into publication English. Language-only revision does not authorize changing claims.

Chinese prose, headings and tables use established terms and explain the object,
finding and reason for a judgment without coined shorthand or opaque internal
status labels. Necessary identifiers and abbreviations are introduced with their
meaning and object; incidental tracking codes remain in source records. Useful
classifications follow the reader's question rather than a preset taxonomy,
keeping distinct judgments separate and terminology consistent.
The coordinator also checks these requirements in the actual deliverable.

Keep the original or a revision record for important manuscripts. State terms, symbols, or phrases that must not change, and ask the agent to flag edits that could alter scientific meaning.

```text
Polish writing/abstract_v3.md in English. Preserve every number, catalyst name, tense, citation,
and level of certainty. Do not add background or turn correlation into causation. The target is
Nature Communications, but avoid promotional abstract language.

Check the scientific logic before editing. Save abstract_v4.md and list any terminology or overstrong
claim that still requires author judgment. The abstract must remain connected prose.
```

### Citations, bibliography, and data statements

Writing can use citation skills to find support for a supplied passage and verify DOI, author, volume, issue, and pages. Citation work begins from a specific claim rather than appending several vaguely related papers to a paragraph. The agent maps sources to claims and marks evidence that is abstract-only or incomplete.

Writing can draft Data Availability and Code Availability statements from the actual data situation and venue requirements. It does not invent repository accessions or upload material without authorization.

```text
Audit the [CITATION NEEDED] markers in writing/introduction.md.
Extract the externally verifiable claim in each sentence, find sources that directly support it,
and state whether evidence comes from full text or abstract. Do not force citations onto ordinary transitions.

Save a claim, candidate source, claim relationship, access depth, and DOI table. Wait for confirmation before updating
references.bib or the manuscript.
```

### Scientific figures, schematics, and PDF

The plot worker uses `publication-data-plotting` for quantitative figures, with Origin-like styling, the validated NPG categorical palette and inspection of the rendered output. Scientific color semantics and explicit user requirements determine applicable exceptions. The user should identify the conclusion, data, units, comparisons, and output format. Each logical figure keeps one final format: high-resolution PNG when unspecified, or the format explicitly required by the venue or downstream interface. Plotting code, source data, and the final figure have different roles and may coexist; a conversion used only for QA stays under `/tmp/` and is not a second deliverable.

When a requested report or presentation needs conceptual explanation, an image-generation route can draft a graphical abstract or explanatory schematic within the task's authority. Generated imagery does not replace quantitative plots or validated atomic structures.

`markdown-pdf-export` renders existing Markdown to PDF, while `compile_text` checks and compiles LaTeX. ACS projects can use the local achemso skill, and other venues can use venue templates. A successful compile still requires visual review of fonts, equations, images, references, and pagination.

```text
Use Writing to create the main manuscript figure from data/activity.csv and data/stability.csv.
The figure should show the activity-stability tradeoff and identify three candidate catalysts.
Inspect columns, units, replicates, and uncertainty definitions before proposing the panel logic.

After plotting, audit dimensions, fonts, color, and labels. Save Python source, processed plotting data,
and one 600 dpi PNG main figure; do not also retain equivalent SVG, PDF, or TIFF copies. Do not remove unfavorable points for visual clarity.
```

### Slides and reviewer responses

Writing delegates decks to `presentation_worker`, which uses the EasySlides skill and preinstalled runtime with full native file and shell capabilities plus `generate_figure` for image assets. Titles, body text, ordinary tables and layout remain native editable PPTX objects, with data plots and complex illustrations embedded as individual assets. The agent chooses layout for the audience and evidence, subject to user requirements and rendered-artifact checks. It can reuse built-in or supplied templates; see [EasySlides installation and configuration](../easyslides.md).

Writing can organize reviewer correspondence into point-by-point replies and revision plans. Every claimed change must correspond to an actual edit or supporting evidence.

```text
Create a 20-minute Chinese group-meeting presentation from writing/submission/manuscript.pdf.
Understand the research question, evidence chain, and limitations before selecting figures.
Do not follow paper page order mechanically and do not add a divider slide for every minor section.

Deliver an editable PPTX with speaker notes, then check image sharpness, overflow, color, slide numbers,
and citations. End with conclusions supported by the paper and questions that remain unresolved.
```

<details>
<summary>Sources of Writing capability</summary>

Entry tools are the read-only `query_research_graph_sql`, `generate_figure`, and `review_pdf_manuscript`. The Writing coordinator uses the graph query for evidence navigation; the single writing worker handles a coherent draft, section, or integration scope through common file and scripting capabilities together with `generate_figure`, `compile_text`, and `render_markdown_pdf`. The [figure-generation interface](../figure_generation.md) supports model selection, reference images, and iterative editing. Reconciliation batches known material corrections back to the same worker.

Local skills cover manuscript argument, citation management, scientific visualization, ACS LaTeX, venue formats and Markdown PDF export. `scientific-communication` provides prose guidance and optional phrasing examples. Other document tasks use the same writing capabilities, with software availability and the user's source materials determining the available execution paths.

</details>

## Peer Review agent: several independent reports on one manuscript

Peer Review needs one canonical PDF because the PDF contains text, figures, tables, equations, and final layout. Keep LaTeX or Word source in the workspace for later revision, but identify one PDF as authoritative.

`peer_review_request` sends that PDF to every model in `peer_review_models`. Reviewers independently assess novelty, method reliability, evidence, reporting, and risk. An editor-level synthesis then separates consensus from disagreement. Agreement among models is not automatic proof. Verify each major criticism against the cited page, source data, and methods.

```text
Use Peer Review on writing/submission/manuscript_r2.pdf for Journal of Catalysis.
This is the only canonical PDF. The SI is writing/submission/si_r2.pdf.

Ask reviewers to assess model construction, DFT settings, adsorption and free-energy references,
NEB evidence, experimental controls, figures, and reproducibility. Major comments must point to pages,
figures, or paragraphs. Preserve all reports, then provide an editor synthesis. Do not edit the manuscript
or write an author response in this turn.
```

### Moving from review to revision

After review, create a decision table. Mark each comment accepted, partly accepted, requiring clarification, or rejected with evidence. Identify which data or analysis it needs and where the manuscript will change. Give that table, source manuscript, reviewer reports, and editor synthesis to Writing.

Writing can draft a response and revise source files, but every claim of change must correspond to an actual diff. Compile a new PDF and check layout and scientific consistency. A second Peer Review round should explicitly identify the new canonical PDF.

```text
Use Writing to address the reviews under writing/review_round1/. The source is writing/manuscript.tex,
and author decisions are in writing/review_round1/decisions.md.

Verify each reviewer comment, author decision, and available evidence before drafting a response and edit plan.
Only comments marked accepted or partly accepted may change the manuscript. Put requests for new computation
on a pending list rather than inventing results. Point every response to the actual modified location and retain
a before-and-after record.
```

## A practical handoff order

For a full manuscript project, Literature Review often establishes sources and an evidence table, Writing creates or revises the manuscript, and Peer Review examines the compiled canonical PDF. Review outputs return to Writing for revision and response. Research can coordinate new literature or computation if a genuine evidence gap remains.

This is not a mandatory pipeline. Existing evidence can go directly to Writing. Reading one paper does not require Research. A layout-only check does not need multiple reviewer models. Choose the narrowest entry that can complete the work so that autonomy is spent on the task rather than role switching.
