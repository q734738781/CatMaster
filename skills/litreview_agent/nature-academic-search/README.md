# Nature Academic Search In CatMaster

This skill follows CatMaster's single LitReview DeepAgent architecture. The active runtime exposes:

- `web_search` for efficient discovery;
- a filtered `agent-browser` MCP session for dynamic and user-authorized pages;
- local full-text corpus ingest/query tools;
- one deterministic final citation batch.

The previous standalone academic-search MCP server and direct Scopus, ScienceDirect, PubMed, OpenAlex, and Semantic Scholar tool inventory are not active CatMaster tools and have been removed from this skill.

For standalone citation-file conversion outside the agent tool path:

```bash
python scripts/format-converter.py --doi 10.1038/s41586-020-2649-2 --format bib
```

For active review work, persist large candidate tables under `notes/literature/`, ingest selected full text, retrieve compact evidence spans, and call `finalize_citations` once for the final DOI set.
