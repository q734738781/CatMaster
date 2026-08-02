# Active LitReview tools

CatMaster's active literature surface is intentionally small:

| Capability | Tool surface |
|---|---|
| Efficient discovery | provider-routed `web_search` |
| Selected-source acquisition and local caching | `acquire_literature_source` |
| Optional local full-text evidence | `ingest_literature_files`, `query_literature_corpus` |
| Final metadata and bibliography | `finalize_citations` |

The LitReview model does not receive raw OpenAlex, Semantic Scholar, PubMed,
Scopus, ScienceDirect, Web of Science, browser-control, or academic-search MCP
inventories. Scholarly indexes and repositories may be used internally by the
high-level search, acquisition, and citation tools.
