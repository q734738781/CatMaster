const STATUS_RANK = {
  reviewed: 0,
  approved: 1,
  conflict: 2,
  proposed: 3,
  promoted: 4,
  invalid: 5,
  rejected: 6,
  rolled_back: 7,
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
  const counts = {
    reviewed: 0,
    approved: 0,
    promoted: 0,
    invalid: 0,
    rejected: 0,
    conflict: 0,
    proposed: 0,
    rolled_back: 0,
  };
  for (const candidate of candidates) {
    const status = String(candidate?.status || "");
    if (Object.hasOwn(counts, status)) counts[status] += 1;
  }
  return { ...counts, ...(payload?.status_counts || {}) };
}

export function selfEvolutionLifecycleLabel(candidate) {
  const status = String(candidate?.status || "");
  if (status === "reviewed") return "Awaiting human decision";
  if (status === "approved") return "Awaiting human decision (legacy)";
  if (status === "promoted") return "Human promoted";
  if (status === "rejected") {
    const source = candidate?.human_review?.human_decision?.decision_source;
    return source === "human" ? "Human rejected" : "Reviewer rejected";
  }
  if (status === "conflict") return "Conflict / stale";
  if (status === "rolled_back") return "Rolled back";
  if (status === "invalid") return "Structurally invalid";
  return status || "Unknown";
}

export function selfEvolutionHumanReview(candidate) {
  const projected = candidate?.human_review;
  if (projected && typeof projected === "object") {
    return {
      structuredReviewAvailable: projected.structured_review_available === true,
      recommendation: String(projected.reviewer_recommendation || "unavailable"),
      summary: String(projected.summary || "No reviewer summary was provided."),
      changePoints: Array.isArray(projected.change_points) ? projected.change_points : [],
      scopeAssessment: String(projected.scope_assessment || ""),
      proportionality: projected.proportionality_assessment && typeof projected.proportionality_assessment === "object"
        ? projected.proportionality_assessment
        : { status: "unavailable", explanation: "" },
      concerns: Array.isArray(projected.concerns) ? projected.concerns.map(String) : [],
      humanChecks: Array.isArray(projected.human_checks) ? projected.human_checks.map(String) : [],
      rationale: String(projected.rationale || ""),
    };
  }
  const legacyReview = candidate?.review && typeof candidate.review === "object" ? candidate.review : {};
  return {
    structuredReviewAvailable: false,
    recommendation: String(legacyReview.recommendation || legacyReview.decision || "unavailable"),
    summary: "Structured human-review summary is unavailable for this legacy candidate.",
    changePoints: [],
    scopeAssessment: "",
    proportionality: {
      status: "unavailable",
      explanation: "No structured proportionality assessment was stored for this legacy candidate.",
    },
    concerns: [],
    humanChecks: [],
    rationale: String(legacyReview.rationale || ""),
  };
}

export function selfEvolutionPromotionConfirmation(candidate, workspaceName = "") {
  const review = selfEvolutionHumanReview(candidate);
  const concerns = review.concerns.length
    ? review.concerns.map((item) => `- ${item}`).join("\n")
    : "None recorded.";
  return [
    `Promote ${selfEvolutionCandidateTitle(candidate)} in workspace ${workspaceName || "(unknown)"}?`,
    "",
    `Summary: ${review.summary}`,
    `AI recommendation: ${review.recommendation}`,
    `Proportionality: ${review.proportionality?.status || "unavailable"}`,
    "Concerns:",
    concerns,
    "",
    "The exact reviewed bundle will become active on the next run.",
  ].join("\n");
}
