# Candidate selection policy

Assign exactly one task-local status to every deduplicated candidate:

| Status | Meaning |
|---|---|
| `selected` | Needed for the current question and worth reading or reporting now |
| `deferred` | Potentially relevant, but another source or missing access/coverage condition makes it unnecessary now |
| `excluded` | Duplicate, outside scope, wrong document type, or unable to support the access-dependent task |

Write one concise reason using observed attributes: query/topic coverage, method,
material or dataset match, time window, redundancy, access depth, and the user's
explicit venue preference when one exists. Preserve stable identifiers and the
available access depth. Do not turn these attributes into component values,
totals, grades, or paper-quality tiers. Citations and venue can help discovery;
they do not establish scientific correctness.
