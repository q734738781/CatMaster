---
name: literature-push-template
description: Configure a recurring multi-source literature search, reasoned candidate selection, digest, and optional raw-note archive.
version: 1.1.0
license: MIT
---
# Literature Push Template

```yaml
research_profile:
  field: "CHANGE_ME"
  subtopics: []
search:
  candidate_pool_size: 30
  lookback_days: 7
  sources: [arxiv, openalex, crossref, semantic_scholar_optional]
delivery:
  target: "CHANGE_ME"
archive:
  enabled: false
  root: "CHANGE_ME"
  write_wiki: false
```

## Task prompt

```text
Search the configured sources for {FIELD} over {LOOKBACK_DAYS} days and retain
the query provenance. Deduplicate by DOI, arXiv ID, OpenAlex ID, then normalized
title. Assign each candidate selected, deferred, or excluded with one observed,
task-specific reason. Record access depth separately. Read the selected papers
to the depth needed for the digest; do not infer findings from metadata. Deliver
the selected digest plus compact deferred/excluded lists. If fewer papers are
useful, select fewer. Do not modify a curated wiki. If archival is enabled,
write only authorized raw notes to {ARCHIVE_PATH}.
```

Verify that candidates are deduplicated, every status has a reason, every
reported claim matches its access depth, stable links are present where
available, and no private IDs, keys, or unrelated paths entered the template.
