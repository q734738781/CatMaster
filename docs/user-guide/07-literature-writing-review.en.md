# 7. Literature, writing, and review

[Previous](06-computational-workflows.en.md) | [Contents](README.en.md) | [Next](08-remote-execution.en.md)

Literature, writing, and review share project files but carry different evidence
responsibilities. Literature Review discovers, reads, and finalizes references.
Writing produces a manuscript from supplied material. Peer Review examines one
fixed PDF from reviewer and editor perspectives. Keeping the three separate
reduces fabricated references, version confusion, and prose invented during
search.

## 7.1 Search-service configuration

Common environment variables are:

```bash
export TAVILY_API_KEY="<KEY>"
export SEMANTIC_SCHOLAR_API_KEY="<KEY>"
export OPENALEX_API_KEY="<KEY>"
export NCBI_API_KEY="<KEY>"
export CROSSREF_MAILTO="you@example.org"
```

Not all are required. If a service is unavailable, the agent uses currently
available search and local-corpus routes. An API key grants access but does not
guarantee full text or correct metadata.

## 7.2 Controlled browser

See [Quick start](01-quickstart.en.md) for installation. Optional session
settings are:

```bash
export CATMASTER_AGENT_BROWSER_PROFILE="$HOME/.config/catmaster/browser-profile"
export CATMASTER_AGENT_BROWSER_HEADED=true
```

Or try to connect to a running Chrome instance:

```bash
export CATMASTER_AGENT_BROWSER_AUTO_CONNECT=true
```

Initial institutional sign-in usually needs headed mode. Keep the profile
outside project spaces and private to the current user. Never put passwords,
cookies, OTPs, session exports, or browser state in `.env.local`, YAML, prompts,
or `files/`.

When a CAPTCHA, QR code, text verification, license confirmation, or browser
security warning appears, the agent stops for the user. CatMaster does not
bypass paywalls or access controls.

## 7.3 Literature Review workflow

An auditable review process is:

1. Define the question, date range, material or reaction system, document types,
   and exclusions.
2. Record databases, web entrypoints, and search strings.
3. Deduplicate DOI, title, and publication versions.
4. Distinguish discovery records, abstracts, full text, and supplementary data.
5. Extract methods, systems, comparison conditions, results, and limitations
   from key papers.
6. Map claims to evidence instead of using one paper for many unrelated claims.
7. Use citation finalization to check DOI, author, journal, year, and exportable
   records.
8. Save the search log, evidence table, unavailable list, and final review.

Suggested layout:

```text
literature/
  query.md
  corpus/
  metadata/
  evidence.csv
  unavailable.md
  references.bib
  review.md
```

## 7.4 Evidence levels

A report should mark where information came from:

| Level | What it can support |
|---|---|
| Search result or metadata | Existence, title, authors, venue, year, and discovery facts |
| Abstract | Main purpose and findings explicitly stated in the abstract |
| Full text | Methods, numbers, conditions, figures, and discussion |
| Supplementary Information | Detailed experiments, computational settings, extended data, and extra figures |
| User-supplied data | Claims consistent with its provenance, version, and completeness |

State when full text was not read. Do not infer a method from the title, fill
exact parameters from an abstract, or double-count a preprint and journal
version.

## 7.5 Local corpora

Upload existing PDF, Markdown, or tables under `literature/corpus/`, then ask
Literature Review to ingest and query them. A large corpus should have a manifest
with:

```text
source path
DOI or stable identifier
title
document version
access date
parse status
notes
```

Successful parsing does not guarantee complete extraction of figures, formulas,
or supplementary material. Verify central claims against PDF pages or publisher
HTML.

## 7.6 Writing input contract

A Writing request should state:

- Language, target journal, or audience.
- Document type and section for this turn.
- Evidence files, tables, figures, and reference library.
- Numbers, terminology, citations, and conclusion boundaries to preserve.
- Prohibited additions, such as new results, unverified references, or causal
  claims.
- Output format, filename, and whether a revision record is required.

Example:

```text
Use notes/result_contract.md, calculations/summary.csv, and
writing/references.bib to draft two Results subsections. Preserve every number
and error value, add no references, and list the evidence file for each
paragraph. Write writing/results_v1.md.
```

## 7.7 Drafting, polishing, and fact preservation

The writing worker can restructure arguments, draft sections, create figures,
and compile documents. The polisher performs conservative language edits. Final
checks should ask:

- Do numbers, units, symbols, and uncertainties match their sources?
- Does each citation support its nearby claim?
- Was correlation turned into causation?
- Is conclusion strength wider than the data?
- Were methods and results mixed?
- Were limitations, uncertainty, or failed cases removed?
- Do figure, table, supplement, and cross-reference numbers agree?

Automated polishing does not replace author review. Keep a prior version or diff
for important text.

## 7.8 Figures, Markdown PDF, and TeX

Writing can invoke figure and compilation tools. Supply the data, intended
conclusion, panel logic, units, color constraints, output format, and journal
dimensions. Keep source data and the generation script for scientific figures.

The Markdown PDF route normally requires Pandoc, Chrome or Chromium, Fontconfig,
and appropriate CJK fonts. LaTeX requires `pdflatex` and also `bibtex` when a
bibliography is present. After compilation, inspect the PDF for blank pages,
overflow, cropped figures, substituted fonts, formulas, and links.

See [Deployment, operations, and security](10-deployment-operations.en.md) for
the environment.

## 7.9 Peer Review workflow

Prepare one canonical PDF, for example:

```text
writing/submission/manuscript.pdf
```

State the journal or review standard, article type, whether Supplementary
Information is included, and any methods or reporting concerns to emphasize. Do
not leave several same-named PDFs in different directories without identifying
the primary version.

Peer Review generates one report per `peer_review_models` label, followed by an
editor synthesis. A useful output layout is:

```text
writing/review/
  reviewer_1.md
  reviewer_2.md
  reviewer_3.md
  editor_synthesis.md
  review_memo.md
```

Review output is diagnostic material, not a factual verdict. Check each concern
against manuscript pages, source data, and methods.

## 7.10 Move from review to revision

Do not ask Peer Review to rewrite the paper directly. First classify comments as
accepted, partly accepted, or rejected. Then provide Writing or Research with:

- The canonical manuscript source.
- Reviewer and editor artifacts.
- A decision and evidence for every comment.
- Sections allowed to change.
- Whether a response letter and marked manuscript are needed.

Compile a new PDF after revision and perform an independent check on that new
canonical version.

## 7.11 Delivery checklist

A complete literature and writing delivery normally includes:

- Search strings and dates.
- Deduplicated records and stable identifiers.
- Full-text availability and evidence level.
- Claim-evidence table.
- Reference library and unverified entries.
- Editable source, figure sources, and compiled PDF.
- Original reviewer reports, editor synthesis, and revision decisions.
- Final data, citation, layout, and fact-preservation checks.
