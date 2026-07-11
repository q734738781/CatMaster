# Active LitReview tools

CatMaster's active literature surface is intentionally small:

| Capability | Tool surface |
|---|---|
| Efficient discovery | `web_search` (Tavily) |
| Dynamic or authorized page access | filtered `agent_browser_*` MCP tools |
| Full-text evidence | `ingest_literature_files`, `query_literature_corpus` |
| Final metadata and bibliography | `finalize_citations` |

The LitReview model does not receive OpenAlex, Semantic Scholar, PubMed,
Scopus, ScienceDirect, Web of Science, or an academic-search MCP inventory.
OpenAlex is only an internal deterministic fallback used by citation
finalization when Crossref fails.
