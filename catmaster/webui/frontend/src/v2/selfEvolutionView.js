const STATUS_RANK = {
  approved: 0,
  conflict: 1,
  proposed: 2,
  promoted: 3,
  invalid: 4,
  rejected: 5,
  rolled_back: 6,
};

export function sortSelfEvolutionCandidates(candidates) {
  return [...(Array.isArray(candidates) ? candidates : [])].sort((left, right) => {
    const rankDiff = (STATUS_RANK[left?.status] ?? 99) - (STATUS_RANK[right?.status] ?? 99);
    if (rankDiff !== 0) return rankDiff;
    return String(right?.updated_at || right?.created_at || "").localeCompare(
      String(left?.updated_at || left?.created_at || ""),
    );
  });
}

export function selfEvolutionCandidateTitle(candidate) {
  if (candidate?.action === "memory") return "Workspace memory";
  const group = String(candidate?.group || "").trim();
  const name = String(candidate?.name || "").trim();
  return [group, name].filter(Boolean).join("/") || "Skill candidate";
}

export function selfEvolutionStatusCounts(payload) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  const counts = { approved: 0, promoted: 0, invalid: 0, rejected: 0, conflict: 0, proposed: 0, rolled_back: 0 };
  for (const candidate of candidates) {
    const status = String(candidate?.status || "");
    if (Object.hasOwn(counts, status)) counts[status] += 1;
  }
  return { ...counts, ...(payload?.status_counts || {}) };
}
