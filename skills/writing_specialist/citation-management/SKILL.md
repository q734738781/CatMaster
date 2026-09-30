---
name: citation-management
description: Resolve paper identifiers, verify citation metadata, and prepare or repair a BibTeX bibliography. Use for bibliographic work, not broad literature synthesis or figure creation.
metadata:
    skill-author: K-Dense Inc.
---

# Citation management

Start from the supplied identifiers, bibliography and available metadata. Use
only the operation needed; correcting a DOI does not require a new literature
search, and resolving metadata does not establish scientific support.

## Choose an operation

| Need | Resource |
|---|---|
| Convert DOI/PMID/arXiv identifiers | `scripts/doi_to_bibtex.py`, `scripts/extract_metadata.py`; [metadata extraction](references/metadata_extraction.md) |
| Repair or deduplicate a bibliography | `scripts/format_bibtex.py`; [BibTeX formatting](references/bibtex_formatting.md) |
| Verify bibliographic identity | `scripts/validate_citations.py`; [citation validation](references/citation_validation.md) |
| Find a missing paper in Scholar | `scripts/search_google_scholar.py`; [Scholar queries](references/google_scholar_search.md) |
| Find biomedical papers | `scripts/search_pubmed.py`; [PubMed queries](references/pubmed_search.md) |

Scripts are optional. Execute them directly at
`"$CATMASTER_SKILLS_ROOT/writing_specialist/citation-management/scripts/<name>.py"`;
use `--help` for arguments. For batch exports use the script's output option (or
redirect JSON stdout to a workspace file) and inspect only relevant records.
DOI conversion, metadata extraction, PubMed/Scholar search and format conversion
print per-record progress only with `--verbose`; exports remain complete.
Citation validation with `--report` returns counts and the report path by default;
add `--verbose` to also print individual diagnostics. Without a report file,
errors and duplicate records remain on the console.
Read only the reference needed for a difficult
query or unresolved field. Verify title, authors, year and identifier against the
source; retain unresolved gaps rather than inventing metadata. Reuse earlier
verification on unchanged records.

## Journal-facing bibliography hygiene

Export one `.bib` bibliography by default. Do not prepare equivalent JSON,
Markdown, RIS or ENW companions unless explicitly requested or required by the
actual downstream interface. Preserve pre-existing user files.

Citation identity is separate from scientific support. For claim-relative use,
read `/.deepagents/skills/litreview_agent/literature-evidence-use/references/evidence-attributes.md`.
Metadata-only candidates remain `claim_relation=unassessed` with
`access_depth=metadata`. Provider relevance, metadata-verification status and
extraction confidence are not scientific evidence grades. Selection uses
`selected`, `deferred` or `excluded` with a task-specific reason.

Optional local helpers preserve complete abstracts and author lists:
`scripts/format-converter.py` converts PubMed, Crossref and arXiv metadata
(summary and errors by default, `--verbose` for per-reference progress);
`scripts/academic_search.py` queries OpenAlex metadata; and
`scripts/citation_records.py` exports selected Crossref JSON as BibTeX by default;
metadata-only JSON/TSV, RIS, ENW and Zotero RDF are explicit alternatives.
Run a helper with `--help` for its input format.
Their Apache-2.0 license is retained in `LICENSE.nature-skills`.

When the target output is a manuscript rather than an internal note:
- Generate publication-style BibTeX only. Do not leave internal provenance notes such as "workspace snippet", "accessible note", or "used in benchmark notes" inside final reference entries.
- Avoid padding weakly resolved sources with explanatory `note` fields just to justify a citation.
- If metadata cannot be validated to publication standard, mark it as a gap for cleanup or omit it from the journal-facing bibliography.
- Prefer a smaller clean bibliography over a larger one with questionable placeholder entries.

## Tool scope and examples

For a final DOI set, `finalize_citations` exports one bibliography to output_path or its default notes/literature location. Existing files are protected by default; use overwrite=true only when replacing that specific bibliography is intended. This tool resolves DOI identifiers; the optional conversion scripts above cover other identifier formats.
