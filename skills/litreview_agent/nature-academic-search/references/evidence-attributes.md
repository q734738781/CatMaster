# Evidence Attributes

Describe evidence by properties relevant to the current claim. Do not assign a
global strength tier to a paper, journal, database, or retrieval interface. The
same source can be directly useful for one claim and merely contextual for
another.

## Attribute axes

Use only the axes that change the scientific interpretation:

- **Scientific modality:** experiment or measurement, computation or model,
  curated dataset or benchmark, review or synthesis, or metadata-only record.
- **Epistemic stage:** reported observation, derived analysis, author
  interpretation, or the current review's synthesis. Keep these distinct.
- **Access depth:** metadata or title, abstract or substantive summary, full
  text, or Supporting Information/source data. This records what was inspected;
  it is not a reliability grade.
- **Claim relationship:** directly addresses the claim, corroborates a
  compatible but non-discriminating part, supplies context, conflicts with or
  limits the claim, or does not resolve it.
- **Condition fit:** matched, partially comparable, materially different, or
  unclear with respect to the system, method, operating conditions, units, and
  reference state needed by the claim.
- **Independence and provenance:** independent evidence, a shared dataset or
  sample lineage, a derivative analysis, or unclear provenance. Multiple papers
  do not provide independent corroboration when they reuse the same underlying
  evidence.

These are descriptive attributes, not ordered labels. Do not convert them into
a score, confidence percentage, or high/medium/low evidence grade.

A separate candidate-selection or LATS utility score may rank what to read,
deliver, or archive next. Keep its component scores, total, rationale, access
depth, and provisional status together. That operational score does not measure
scientific truth and must not replace these claim-level evidence attributes.

## Claim-relative use

- Use authoritative metadata for identity, DOI, title, venue, and publication
  date, but not as support for a scientific mechanism or quantitative result.
- Use an abstract only for a bounded statement it explicitly makes. Inspect the
  full text, figure, table, or SI when exact conditions, numbers, controls,
  methods, or competing explanations matter.
- For a mechanistic or causal claim, identify the observation that distinguishes
  the live alternatives. Agreement without discrimination is corroborating, not
  direct resolution.
- For a field trend or consensus claim, use reviews to map the field and use
  representative independent primary evidence to show where the synthesis is
  grounded.
- For disagreements, compare condition fit and provenance before calling the
  results contradictory. Different systems or reference states can delimit a
  claim without opposing it.

## Synthesis language

Calibrate verbs to the evidence relationship and unresolved alternatives:

- Use direct-observation language only when the cited source actually contains
  the relevant observation under applicable conditions.
- Use `supports` or `is consistent with` when the result is compatible but does
  not uniquely distinguish the explanation.
- Use `suggests` or `proposes` for interpretation or synthesis beyond the direct
  observation.
- State `unresolved` when access depth, condition mismatch, shared provenance,
  or conflicting observations prevent a defensible conclusion.

A claim-evidence table may expose these attributes as useful columns, compact
notes, or prose. Do not force every source into a fixed schema when an attribute
does not matter to the current decision.
