import { presentError } from "./presentation.js";

const STATUS_LABELS = {
  pending: "Candidate preparation pending",
  review: "Ready for human review",
  revision: "Revision requested",
  canary: "Canary active",
  stable: "Stable",
  rejected: "Rejected",
  inactive: "Inactive",
};

export const SELF_EVOLUTION_STATUS_VALUES = Object.freeze(Object.keys(STATUS_LABELS));

const ROUTE_LABELS = {
  discard: "No durable change",
  workspace_preference: "Workspace preference",
  amend_existing_skill: "Update an existing skill",
  new_skill: "Create a new skill",
  tool_schema_or_test: "Tool or product fix",
  note_reference: "Research note",
};

const RECOMMENDATION_LABELS = {
  approve: "Reviewer supports this revision",
  reject: "Reviewer recommends rejection",
  needs_revision: "Reviewer requests revision",
  unavailable: "No reviewer recommendation",
};

const PROPORTIONALITY_LABELS = {
  pass: "Proportionate",
  warning: "Needs careful scope review",
  fail: "Disproportionate",
  unavailable: "Not assessed",
};

const SIGNAL_LABELS = {
  workspace_preference: "Workspace preference",
  skill_revision: "Existing skill revision",
  skill_discovery: "New reusable method",
};

const OBSERVATION_STATUS_LABELS = {
  open: "Available for proposal",
  consolidated: "Included in a candidate revision",
};

const ACTION_DEFINITIONS = {
  "run-review": {
    label: "Run independent review",
    submitLabel: "Start independent review",
    description: "Ask the independent reviewer to inspect this exact revision, its complete episode evidence, scope, and proportionality. The recommendation remains advisory.",
  },
  "request-revision": {
    label: "Request revision",
    submitLabel: "Send revision request",
    description: "Keep the current revision immutable and ask for a new, bounded revision.",
  },
  "start-canary": {
    label: "Start canary",
    submitLabel: "Start this canary",
    description: "Activate this exact version only for the selected thread or run. This does not start another conversation or model call.",
  },
  "promote-stable": {
    label: "Promote stable",
    submitLabel: "Promote this exact version",
    description: "Make this exact version the workspace default for future runs.",
  },
  reject: {
    label: "Reject",
    submitLabel: "Reject this revision",
    description: "Reject this exact revision while preserving its evidence and audit history.",
  },
  quarantine: {
    label: "Quarantine",
    submitLabel: "Quarantine this version",
    description: "Stop presenting this active version while its behavior is investigated.",
  },
  retire: {
    label: "Retire",
    submitLabel: "Retire this version",
    description: "Remove this stable version from future use without deleting its history.",
  },
  rollback: {
    label: "Roll back",
    submitLabel: "Roll back this version",
    description: "Move the active pointer away from this version and restore the previous verified state.",
  },
};

export const SELF_EVOLUTION_ACTION_ORDER = [
  "run-review",
  "request-revision",
  "start-canary",
  "promote-stable",
  "reject",
  "quarantine",
  "retire",
  "rollback",
];

function record(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function rawText(value) {
  if (typeof value === "string") return value;
  if (typeof value === "number" || typeof value === "boolean") return String(value);
  return "";
}

export function redactSelfEvolutionText(value) {
  let text = rawText(value);
  if (!text) return "";
  const trimmed = text.trim();
  if ((trimmed.startsWith("{") && trimmed.endsWith("}")) || (trimmed.startsWith("[") && trimmed.endsWith("]"))) {
    try {
      const parsed = JSON.parse(trimmed);
      if (parsed && typeof parsed === "object") return "";
    } catch {
      // This is ordinary prose or a diff that happens to use braces.
    }
  }
  text = text.replace(
    /(^|[\s([{"'`])\/(?!\/)(?:[^/\s]+\/)+[^\s)\]}"'`<>:,;]+/g,
    "$1[internal path hidden]",
  );
  text = text.replace(
    /(^|[\s([{"'`])[A-Za-z]:\\(?:[^\\\s]+\\)+[^\s)\]}"'`<>:,;]+/g,
    "$1[internal path hidden]",
  );
  text = text.replace(/\bfile:\/\/[^\s)\]}"'`<>]+/gi, "[internal path hidden]");
  return text;
}

export function selfEvolutionSafeText(value, fallback = "") {
  return redactSelfEvolutionText(value).trim() || fallback;
}

function humanizeIdentifier(value, fallback = "") {
  const text = rawText(value).trim();
  if (!text) return fallback;
  return text
    .replace(/[_-]+/g, " ")
    .replace(/\s+/g, " ")
    .replace(/^\w/, (letter) => letter.toUpperCase());
}

function itemText(value) {
  if (typeof value === "string" || typeof value === "number") {
    return selfEvolutionSafeText(value);
  }
  const item = record(value);
  return selfEvolutionSafeText(
    item.summary
      || item.claim
      || item.description
      || item.evidence
      || item.excerpt
      || item.result_label
      || item.outcome_summary
      || item.reason
      || item.label
      || item.title,
  );
}

export function selfEvolutionTextItems(value) {
  if (Array.isArray(value)) return value.map(itemText).filter(Boolean);
  const text = itemText(value);
  return text ? [text] : [];
}

function timestampValue(value) {
  const parsed = Date.parse(String(value || ""));
  return Number.isFinite(parsed) ? parsed : 0;
}

export function sortSelfEvolutionCandidates(candidates) {
  return [...(Array.isArray(candidates) ? candidates : [])].sort((left, right) => {
    const timeDiff = timestampValue(right?.updated_at || right?.created_at)
      - timestampValue(left?.updated_at || left?.created_at);
    if (timeDiff !== 0) return timeDiff;
    return String(right?.version || right?.revision || "").localeCompare(
      String(left?.version || left?.revision || ""),
      undefined,
      { numeric: true },
    );
  });
}

export function sortSelfEvolutionObservations(observations) {
  return [...(Array.isArray(observations) ? observations : [])].sort(
    (left, right) => timestampValue(right?.created_at) - timestampValue(left?.created_at),
  );
}

export function mergeSelfEvolutionCandidates(current, incoming) {
  const rows = new Map();
  for (const candidate of [...(Array.isArray(current) ? current : []), ...(Array.isArray(incoming) ? incoming : [])]) {
    const key = `${String(candidate?.candidate_id || "")}:${String(candidate?.revision ?? candidate?.version ?? "")}`;
    if (key !== ":") rows.set(key, candidate);
  }
  return sortSelfEvolutionCandidates([...rows.values()]);
}

export function mergeSelfEvolutionObservations(current, incoming) {
  const rows = new Map();
  for (const observation of [...(Array.isArray(current) ? current : []), ...(Array.isArray(incoming) ? incoming : [])]) {
    const key = String(observation?.observation_id || "");
    if (key) rows.set(key, observation);
  }
  return sortSelfEvolutionObservations([...rows.values()]);
}

export function selfEvolutionStatusLabel(status) {
  const value = String(status || "");
  return STATUS_LABELS[value] || humanizeIdentifier(value, "Status unavailable");
}

export function selfEvolutionRouteLabel(route) {
  const value = String(route || "");
  return ROUTE_LABELS[value] || humanizeIdentifier(value, "Route not specified");
}

export function normalizeSelfEvolutionAction(action) {
  return String(action || "").trim().toLowerCase().replaceAll("_", "-");
}

export function selfEvolutionActionDefinition(action) {
  return ACTION_DEFINITIONS[normalizeSelfEvolutionAction(action)] || null;
}

export function selfEvolutionAllowedActions(candidate) {
  const supplied = Array.isArray(candidate?.allowed_actions) ? candidate.allowed_actions : [];
  const allowed = new Set(
    supplied
      .map(normalizeSelfEvolutionAction)
      .filter((action) => Object.hasOwn(ACTION_DEFINITIONS, action)),
  );
  return SELF_EVOLUTION_ACTION_ORDER.filter((action) => allowed.has(action));
}

export function selfEvolutionFilterCandidates(candidates, statusFilter) {
  const rows = Array.isArray(candidates) ? candidates : [];
  if (!statusFilter || statusFilter === "all") return rows;
  if (statusFilter === "needs-action") {
    return rows.filter((candidate) => selfEvolutionAllowedActions(candidate).length > 0);
  }
  return rows.filter((candidate) => String(candidate?.status || "") === statusFilter);
}

export function selfEvolutionCandidateTitle(candidate) {
  return selfEvolutionSafeText(candidate?.target_label, "Skill revision");
}

export function selfEvolutionCandidateVersion(candidate) {
  const supplied = selfEvolutionSafeText(candidate?.version);
  if (supplied) return supplied;
  const revision = Number(candidate?.revision);
  return Number.isFinite(revision) && revision > 0
    ? `r${String(Math.trunc(revision)).padStart(4, "0")}`
    : "Version unavailable";
}

export function selfEvolutionLifecycleLabel(candidate) {
  return selfEvolutionStatusLabel(candidate?.status);
}

export function selfEvolutionStatusCounts(payload) {
  const counts = {};
  for (const candidate of Array.isArray(payload?.candidates) ? payload.candidates : []) {
    const status = String(candidate?.status || "");
    if (status) counts[status] = (counts[status] || 0) + 1;
  }
  return counts;
}

export function selfEvolutionBehaviorChange(candidate) {
  const value = candidate?.behavior_change;
  if (typeof value === "string") {
    return { summary: selfEvolutionSafeText(value), before: "", after: "", impact: "" };
  }
  const change = record(value);
  return {
    summary: selfEvolutionSafeText(
      change.summary || change.expected_step_change || change.description,
      "No behavior-change summary was supplied.",
    ),
    before: selfEvolutionSafeText(change.before),
    after: selfEvolutionSafeText(change.after),
    impact: selfEvolutionSafeText(change.impact || change.expected_impact),
  };
}

export function selfEvolutionEvidenceItems(value) {
  const supplied = Array.isArray(value) ? value : (value ? [value] : []);
  return supplied.flatMap((entry) => {
    const item = record(entry);
    const simple = typeof entry === "string" ? entry : "";
    const primary = {
      title: selfEvolutionSafeText(item.title || item.label || item.source_label || item.signal_label || item.reason),
      summary: selfEvolutionSafeText(
        simple || item.summary || item.claim || item.evidence || item.excerpt || item.description,
      ),
      sourceLabel: selfEvolutionSafeText(item.source_label || item.source || item.status_label),
      sourceRef: selfEvolutionSafeText(item.source_ref),
      href: typeof (item.href || item.source_ref) === "string"
        && (/^https?:\/\//.test(item.href || item.source_ref) || String(item.href || item.source_ref).startsWith("/"))
        ? String(item.href || item.source_ref)
        : "",
    };
    const nested = (Array.isArray(item.evidence) ? item.evidence : []).map((entryRef) => {
      const ref = record(entryRef);
      const sourceRef = selfEvolutionSafeText(ref.source_ref);
      return {
        title: selfEvolutionSafeText(ref.reason, "Evidence excerpt"),
        summary: selfEvolutionSafeText(ref.excerpt),
        sourceLabel: "Traceable source",
        sourceRef,
        href: /^https?:\/\//.test(sourceRef) || sourceRef.startsWith("/") ? sourceRef : "",
      };
    });
    return [primary, ...nested].filter((row) => row.title || row.summary);
  });
}

export function selfEvolutionHumanReview(candidate) {
  const review = record(candidate?.review);
  const proportionality = record(review.proportionality);
  const recommendation = String(review.recommendation || "unavailable");
  const changePoints = (Array.isArray(review.change_points) ? review.change_points : []).map((value) => {
    const point = record(value);
    return {
      title: selfEvolutionSafeText(point.title, "Behavior change"),
      before: selfEvolutionSafeText(point.before),
      after: selfEvolutionSafeText(point.after),
      impact: selfEvolutionSafeText(point.impact),
      evidence: selfEvolutionSafeText(point.evidence),
      evidenceSource: selfEvolutionSafeText(point.evidence_source || point.source_label),
    };
  });
  const proportionalityStatus = String(proportionality.status || "unavailable");
  return {
    structuredReviewAvailable: typeof review.available === "boolean"
      ? review.available
      : Object.keys(review).length > 0,
    recommendation,
    recommendationLabel: RECOMMENDATION_LABELS[recommendation]
      || humanizeIdentifier(recommendation, "No reviewer recommendation"),
    summary: selfEvolutionSafeText(review.summary || review.rationale, "No reviewer summary was supplied."),
    changePoints,
    evidenceSufficiency: selfEvolutionSafeText(review.evidence_sufficiency),
    scopeAssessment: selfEvolutionSafeText(review.scope_assessment),
    proportionality: {
      status: proportionalityStatus,
      label: PROPORTIONALITY_LABELS[proportionalityStatus]
        || humanizeIdentifier(proportionalityStatus, "Not assessed"),
      explanation: selfEvolutionSafeText(proportionality.explanation),
    },
    counterexamples: selfEvolutionTextItems(review.counterexamples),
    concerns: selfEvolutionTextItems(review.concerns),
    humanChecks: selfEvolutionTextItems(review.human_checks),
    rationale: selfEvolutionSafeText(review.rationale),
  };
}

export function selfEvolutionObservationView(observation) {
  const signal = String(observation?.signal_kind || observation?.signal || "");
  const status = String(observation?.status || "");
  return {
    title: selfEvolutionSafeText(observation?.claim || observation?.title, "Learning observation"),
    signalLabel: selfEvolutionSafeText(observation?.signal_label)
      || SIGNAL_LABELS[signal]
      || humanizeIdentifier(signal, "Observed evidence"),
    statusLabel: selfEvolutionSafeText(observation?.status_label)
      || OBSERVATION_STATUS_LABELS[status]
      || humanizeIdentifier(status, "Available"),
    summary: selfEvolutionSafeText(observation?.summary || observation?.evidence_summary),
    outcome: selfEvolutionSafeText(observation?.outcome_summary || observation?.outcome),
    evidence: selfEvolutionEvidenceItems(observation?.evidence || observation?.evidence_refs),
    createdAt: String(observation?.created_at || ""),
  };
}

export function selfEvolutionPromotionConfirmation(candidate, workspaceName = "", scopeLabel = "") {
  const review = selfEvolutionHumanReview(candidate);
  const behavior = selfEvolutionBehaviorChange(candidate);
  const applicability = selfEvolutionTextItems(candidate?.applicability_boundary);
  const exclusions = selfEvolutionTextItems(candidate?.non_applicability);
  const concerns = review.concerns.length ? review.concerns : ["No reviewer concerns were recorded."];
  return [
    "Confirm the exact release",
    `Version: ${selfEvolutionCandidateVersion(candidate)}`,
    `Target: ${selfEvolutionCandidateTitle(candidate)}`,
    `Scope: ${scopeLabel || `Stable for future runs in ${workspaceName || "this workspace"}`}`,
    `Behavior change: ${behavior.summary}`,
    `Applies when: ${applicability.length ? applicability.join("; ") : "No applicability boundary was supplied."}`,
    `Must not apply when: ${exclusions.length ? exclusions.join("; ") : "No exclusion boundary was supplied."}`,
    `Reviewer concerns: ${concerns.join("; ")}`,
  ].join("\n");
}

function candidateRevisionForEndpoint(candidate) {
  const revision = Number(candidate?.revision);
  if (Number.isFinite(revision) && revision > 0) return String(Math.trunc(revision));
  return "";
}

export function selfEvolutionActionEndpoint(ctx, candidate, action) {
  const normalizedAction = normalizeSelfEvolutionAction(action);
  const revision = candidateRevisionForEndpoint(candidate);
  if (!ctx || !candidate?.candidate_id || !revision || !Object.hasOwn(ACTION_DEFINITIONS, normalizedAction)) return "";
  return [
    "/api/session",
    encodeURIComponent(String(ctx)),
    "self-evolution/candidates",
    encodeURIComponent(String(candidate.candidate_id)),
    "revisions",
    encodeURIComponent(revision),
    encodeURIComponent(normalizedAction),
  ].join("/");
}

export function selfEvolutionActionRequest(
  action,
  {
    actor = "human",
    rationale = "",
    guidance = "",
    scopeKind = "",
    scopeId = "",
  } = {},
) {
  const normalizedAction = normalizeSelfEvolutionAction(action);
  const body = {
    actor: String(actor || "human").trim() || "human",
    rationale: String(rationale || "").trim(),
  };
  if (normalizedAction === "request-revision") body.guidance = String(guidance || "").trim();
  if (normalizedAction === "start-canary") {
    body.scope_kind = String(scopeKind || "").trim();
    body.scope_id = String(scopeId || "").trim();
  }
  return body;
}

export function selfEvolutionDisplayError(error, fallback = "The request could not be completed. Refresh and try again.") {
  const status = Number(error?.status || 0);
  const text = String(error?.message || error || "").toLowerCase();
  if (status === 401 || status === 403 || text.includes("authenticated")) {
    return "Your session is no longer authorized for this human decision. Sign in again and retry.";
  }
  if (status === 409 || text.includes("stale") || text.includes("changed") || text.includes("revision")) {
    return "This revision changed after it was opened. Refresh the candidate before deciding.";
  }
  return presentError(error, fallback).message || fallback;
}
