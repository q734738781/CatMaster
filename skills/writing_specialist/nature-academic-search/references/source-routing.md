# Academic source routing

Choose sources for the requested operation by observable capability rather than
an overall reliability class.

| Need | Routing attributes |
|---|---|
| Biomedical identity and indexing | domain coverage, PMID/MeSH fields, structured abstracts |
| DOI and publisher metadata | DOI coverage, deposited field completeness, current availability |
| Preprints | repository coverage, version identity, full-text access |
| Citation graph | citing/cited edge coverage, update date, quota and entitlement |
| Full text | legal access route, document identity, article/SI availability |

For every call, consider coverage for the query, structured fields required by
the task, necessary access depth, current quota or entitlement, and observed
availability. Use parallel complementary sources when their coverage differs.
If a source fails or lacks a needed field, name that limitation and continue with
another fitting source. Do not treat venue, citation count, or source brand as a
scientific evidence grade. Venue may affect routing only when the user explicitly
requests a venue or journal family.
