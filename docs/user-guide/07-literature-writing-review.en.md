# 7. Literature, Writing, and Peer Review agents

[Previous](06-computational-workflows.en.md) | [Contents](README.en.md) | [Next](08-remote-execution.en.md)

Literature Review, Writing, and Peer Review share a workspace but have different evidence responsibilities. Literature Review finds and verifies source material. Writing uses existing evidence to create manuscripts and other deliverables. Peer Review independently examines a fixed manuscript. Keeping those roles distinct prevents the writer from inventing evidence while composing and keeps reviewer comments traceable to real revisions.

## Literature Review agent: from discovery to an evidence corpus

Literature Review can answer a focused verification question or conduct a larger topic review. The research question determines scale. A precise fact may need only a few strong sources. A perspective-style review requires a broad candidate set, explicit search boundaries, and saved screening records. State the date range, material or reaction system, document types, exclusions, and intended use when they matter.

### Discovery is not close reading

Public web search discovers papers, project pages, and database records. Title, author, and DOI metadata generally establish discovery only, but a complete or substantive abstract in the results can support the claims it explicitly makes. The agent preserves that boundary without discarding useful abstract evidence or extending it into unreported methods and values.

When a key decision genuinely depends on methods, conditions, values, figures, or supplementary material absent from the abstract, a controlled browser is one escalation path for open-access or user-authorized institutional content. After one reasonable access attempt fails, the agent states the limitation and continues with other evidence instead of cycling through pages or downloads. CAPTCHA, QR code, OTP, license prompts, and security warnings remain human actions.

```text
Use Literature Review to find in situ evidence for aggregation and redispersion of single-atom catalysts
since 2018. Begin with broad discovery and save the complete candidate set. Prioritize papers that use
operando or in situ methods, directly discuss aggregation or redispersion, and are scientifically relevant.

Separate metadata, abstract, full-text, and SI evidence. Synthesize what abstracts explicitly report first;
read source text only when catalyst, atmosphere, temperature, method, or figure details are decisive and absent.
State the remaining limitation naturally.
```

### Local corpora support repeated project questions

Existing PDFs, Markdown, DOCX, and tables can be placed under `literature/` and ingested into a local corpus. The agent can then query the same material from several research angles while preserving source records. Parsed text may not capture every figure, formula, or supplementary detail, so important conclusions should return to the source page or publisher HTML.

The `nature-reader` skill can build a bilingual, figure-aware Markdown reader that preserves source anchors and paper order. It is designed for full reading rather than collapsing a paper into a one-page summary.

```text
Read literature/papers/pd_redispersion.pdf closely and build a Chinese-English reader.
Preserve section order and place each important figure or table near the discussion it supports.
Keep page or source anchors for every block. Separate methods, direct results, and author interpretation.
Do not return only a summary. Explain in detail the evidence relevant to whether Pd is truly atomically dispersed
and list alternative explanations the paper does not exclude.
```

### Evidence tables connect claims to sources

The most useful intermediate artifact in a review is often an evidence table rather than prose. Literature Review can record system, method, conditions, result, limitation, and evidence level for each paper, then map individual claims to supporting or contradicting sources. Writing can use that map without guessing which citation belongs to each sentence.

Deduplicate title, DOI, preprint, and journal versions early. After paper selection, `finalize_citations` resolves author, journal, year, DOI, and export records in one batch. `nature-ref-verifier` can audit an existing bibliography field by field and flag volume-year, author-order, pagination, and DOI conflicts.

```text
Turn the papers under literature/corpus/ on Pd/CeO2 into a claim-evidence table.
Cover Pd stabilization sites, oxygen-vacancy effects, migration under redox conditions, sintering temperature,
and redispersion evidence. For each claim, list support, counterexamples, or insufficient evidence and label
whether it comes from an abstract, full text, or SI.

Finalize DOI and metadata only after selection. Write evidence.csv, references.bib, and unavailable.md.
Do not count a preprint and its journal article as separate evidence.
```

### Specialized search and citation work

Literature skills also support claim-level citation searches restricted to Nature Portfolio, the Science family, and Cell Press; coordinated multi-source search and citation metrics; and scheduled literature pipelines. Access to Scopus, ScienceDirect, PubMed, or other services depends on configured MCPs and accounts. The agent must report the sources it actually used rather than treating a skill description as proof of access.

<details>
<summary>Sources of Literature Review capability</summary>

Direct tools are `web_search`, `ingest_literature_files`, `query_literature_corpus`, and `finalize_citations`, plus controlled browser tools when `agent-browser` is available. `web_search` is provider-routed: OpenAI/Codex roles use hosted search and other providers use the Tavily implementation.

Skills are `nature-academic-search`, `nature-downloader`, `nature-reader`, `nature-citation`, `nature-ref-verifier`, and `nature-literature-pipeline`. Tools perform search, ingestion, retrieval, and metadata finalization. Skills define scope, evidence levels, lawful acquisition boundaries, reading form, and delivery quality.

</details>

## Writing agent: turning evidence into deliverables

Writing can begin from Chinese notes, result tables, figures, code output, a bibliography, an existing LaTeX project, an older PDF, or reviewer comments. Its job is not simply to polish the material. It interprets the writing target and evidence boundary, then selects writing skills for argument design, drafting, revision, figures, layout, or compilation.

### Manuscripts, reports, and proposals

`nature-writing` and `scientific-writing` build manuscript arguments from existing claims, results, and figures. `researchwrite` is proposal-oriented and establishes the evidence and argument contract before drafting. Writing can handle abstracts, introductions, methods, results, discussions, conclusions, and larger restructuring.

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

`nature-polishing` and the writing polisher improve language, paragraph logic, and academic style while preserving numerical values, units, references, conclusion strength, and scientific structure. Chinese drafts can be translated to publication English. A language-only request must not turn a cautious conclusion into promotion.

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

`nature-data` prepares Data Availability, Code Availability, repository plans, dataset citations, and FAIR metadata checks. It can draft language from the actual data situation, but it cannot upload the data or invent an accession number.

```text
Audit the [CITATION NEEDED] markers in writing/introduction.md.
Extract the externally verifiable claim in each sentence, find sources that directly support it,
and state whether evidence comes from full text or abstract. Do not force citations onto ordinary transitions.

Save a claim, candidate source, support strength, and DOI table. Wait for confirmation before updating
references.bib or the manuscript.
```

### Scientific figures, schematics, and PDF

`nature-figure` and `scientific-visualization` create publication figures in Python or R with multipanel layouts, uncertainty, significance, accessible color, and journal dimensions. The user should identify the conclusion, data, units, comparisons, and output formats. The agent retains source data and plotting code so the figure can be reproduced.

When explicitly requested, an image-generation route can draft a graphical abstract or mechanism schematic. Generated imagery does not replace quantitative plots or validated atomic structures.

`markdown-pdf-export` renders existing Markdown to PDF, while `compile_text` checks and compiles LaTeX. ACS projects can use the local achemso skill, and other venues can use venue templates. A successful compile still requires visual review of fonts, equations, images, references, and pagination.

```text
Use Writing to create the main manuscript figure from data/activity.csv and data/stability.csv.
The figure should show the activity-stability tradeoff and identify three candidate catalysts.
Inspect columns, units, replicates, and uncertainty definitions before proposing the panel logic.

After plotting, audit dimensions, fonts, color, and labels. Save Python source, processed plotting data,
SVG, PDF, and 600 dpi TIFF. Do not remove unfavorable points for visual clarity.
```

### Slides, reviewer responses, and patent drafts

`nature-paper2ppt` turns a paper, PDF, or reading notes into a Chinese academic presentation with selected figures, slide content, speaker notes, and an overflow and image-quality review. It is intended for journal club, group meetings, defenses, and talks, not for copying manuscript paragraphs into a template.

`nature-response` organizes editor and reviewer correspondence into point-by-point responses, revision cover letters, and marked-manuscript plans. Each response should correspond to a real edit or evidence record.

`nature-paper-to-patent` extracts evidence-supported technical contributions from papers, reports, code, and figures and drafts Chinese claims, specification, abstract, and abstract figure. Formal patentability review and filing remain professional legal work.

```text
Create a 20-minute Chinese group-meeting presentation from writing/submission/manuscript.pdf.
Understand the research question, evidence chain, and limitations before selecting figures.
Do not follow paper page order mechanically and do not add a divider slide for every minor section.

Deliver an editable PPTX with speaker notes, then check image sharpness, overflow, color, slide numbers,
and citations. End with conclusions supported by the paper and questions that remain unresolved.
```

<details>
<summary>Sources of Writing capability</summary>

Entry tools are `generate_nanobanana_figure` and `review_pdf_manuscript`. The writing worker also uses `polish_academic_prose`, `compile_text`, and `render_markdown_pdf`, plus common file and scripting capabilities.

Skills cover manuscript and proposal writing, polishing, citations, data statements, figures, full-paper reading, reference verification, response letters, pre-submission review, PPT, patent drafts, ACS LaTeX, venue templates, and Markdown PDF. `citation-management` supplies a general reference workflow, while `humanizer` performs the final style audit. External actions still depend on installed software, APIs, and supplied source material.

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
