export const DEFAULT_ENTRYPOINT = "research";

export const DEFAULT_ENTRYPOINTS = [
  {
    id: "research",
    label: "Research",
    summary: "Research coordinator with delegation to experiment, writing, peer review, and literature specialists.",
  },
  {
    id: "experiment",
    label: "Experiment",
    summary: "Computation and managed-execution specialist entry for bounded calculations and file-producing workflows.",
  },
  {
    id: "writing",
    label: "Writing",
    summary: "Manuscript, report, response, and author-facing scientific writing specialist.",
  },
  {
    id: "peer_review",
    label: "Peer Review",
    summary: "Reviewer-style critique and manuscript risk assessment specialist.",
  },
  {
    id: "literature_review",
    label: "Literature Review",
    summary: "Focused literature synthesis with dedicated literature research support.",
  },
];

const ALIASES = {
  litreview: "literature_review",
  literature: "literature_review",
};

function labelFromId(value) {
  return String(value || "")
    .replace(/[_-]+/g, " ")
    .replace(/\b\w/g, (letter) => letter.toUpperCase())
    .trim();
}

function publicSummary(id, value) {
  const defaultRow = DEFAULT_ENTRYPOINTS.find((item) => item.id === id);
  if (defaultRow) return defaultRow.summary;
  return String(value || "")
    .replace(/\bdeep[- ]research\s+lane\b/gi, "dedicated research support")
    .replace(/\blane\b/gi, "workflow")
    .replace(/\bworker\b/gi, "specialist")
    .trim();
}

export function normalizedEntrypoints(rows) {
  const source = Array.isArray(rows) && rows.length ? rows : DEFAULT_ENTRYPOINTS;
  const seen = new Set();
  const out = [];
  for (const row of source) {
    const id = normalizeEntrypoint(row?.id || row?.value || "");
    if (!id || seen.has(id)) continue;
    seen.add(id);
    out.push({
      id,
      label: String(row?.label || labelFromId(id)).trim() || labelFromId(id),
      summary: publicSummary(id, row?.summary || row?.description || ""),
    });
  }
  return out.length ? out : DEFAULT_ENTRYPOINTS;
}

export function normalizeEntrypoint(value, entrypoints = DEFAULT_ENTRYPOINTS) {
  const raw = String(value || "").trim().toLowerCase().replace(/[-\s]+/g, "_");
  const candidate = ALIASES[raw] || raw || DEFAULT_ENTRYPOINT;
  const valid = new Set((Array.isArray(entrypoints) ? entrypoints : DEFAULT_ENTRYPOINTS).map((item) => String(item.id || "")));
  return valid.has(candidate) ? candidate : DEFAULT_ENTRYPOINT;
}

export function entrypointMeta(value, entrypoints = DEFAULT_ENTRYPOINTS) {
  const rows = normalizedEntrypoints(entrypoints);
  const id = normalizeEntrypoint(value, rows);
  return rows.find((item) => item.id === id) || rows[0] || DEFAULT_ENTRYPOINTS[0];
}
