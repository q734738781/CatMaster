# Nature Academic Search In CatMaster

This skill follows CatMaster's single LitReview DeepAgent architecture. The active runtime exposes:

- `web_search` for efficient discovery;
- `acquire_literature_source` for verified authorized PDFs through direct-first routes, one internal ScanSci/CloakBrowser DOI-page fallback, or one cached static page;
- local full-text corpus ingest/query tools;
- one deterministic final citation batch.

The previous standalone academic-search MCP server and direct Scopus, ScienceDirect, PubMed, OpenAlex, and Semantic Scholar tool inventory are not active CatMaster tools and have been removed from this skill.

For standalone citation-file conversion outside the agent tool path:

```bash
python scripts/format-converter.py --doi 10.1038/s41586-020-2649-2 --format bib
```

For active review work, persist large candidate tables under `notes/literature/`, acquire and read selected sources locally, ingest full text only when repeated retrieval is useful, and call `finalize_citations` once for the final DOI set.
