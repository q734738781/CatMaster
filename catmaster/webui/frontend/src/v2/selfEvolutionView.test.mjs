import test from "node:test";
import assert from "node:assert/strict";

import {
  mergeSelfEvolutionCandidates,
  mergeSelfEvolutionObservations,
  normalizeSelfEvolutionAction,
  redactSelfEvolutionText,
  selfEvolutionActionDefinition,
  selfEvolutionActionEndpoint,
  selfEvolutionActionRequest,
  selfEvolutionAllowedActions,
  selfEvolutionBehaviorChange,
  selfEvolutionCandidateTitle,
  selfEvolutionCandidateVersion,
  selfEvolutionDisplayError,
  selfEvolutionEvidenceItems,
  selfEvolutionFilterCandidates,
  selfEvolutionHumanReview,
  selfEvolutionJobView,
  selfEvolutionLifecycleLabel,
  selfEvolutionObservationView,
  selfEvolutionPromotionConfirmation,
  selfEvolutionRouteLabel,
  sortSelfEvolutionCandidates,
} from "./selfEvolutionView.js";

test("candidate pages remain newest-first and merge immutable revisions", () => {
  const rows = mergeSelfEvolutionCandidates(
    [
      { candidate_id: "alpha", revision: 1, version: "r0001", updated_at: "2026-01-01" },
      { candidate_id: "beta", revision: 1, version: "r0001", updated_at: "2026-01-03" },
    ],
    [
      { candidate_id: "alpha", revision: 1, version: "r0001", updated_at: "2026-01-04", status: "reviewed" },
      { candidate_id: "alpha", revision: 2, version: "r0002", updated_at: "2026-01-05" },
    ],
  );

  assert.deepEqual(
    rows.map((row) => `${row.candidate_id}:${row.revision}`),
    ["alpha:2", "alpha:1", "beta:1"],
  );
  assert.equal(rows[1].status, "reviewed");
  assert.deepEqual(
    sortSelfEvolutionCandidates([
      { version: "r0001", updated_at: "2026-01-01" },
      { version: "r0002", updated_at: "2026-01-02" },
    ]).map((row) => row.version),
    ["r0002", "r0001"],
  );
});

test("observation pagination deduplicates by public observation key", () => {
  const rows = mergeSelfEvolutionObservations(
    [{ observation_id: "one", claim: "Old", created_at: "2026-01-01" }],
    [
      { observation_id: "two", claim: "New", created_at: "2026-01-03" },
      { observation_id: "one", claim: "Updated", created_at: "2026-01-01" },
    ],
  );
  assert.deepEqual(rows.map((row) => row.observation_id), ["two", "one"]);
  assert.equal(rows[1].claim, "Updated");
});

test("candidate lifecycle labels do not recreate a manual release queue", () => {
  assert.equal(selfEvolutionLifecycleLabel({ status: "review" }), "Reviewer approved");
  assert.equal(selfEvolutionLifecycleLabel({ status: "canary" }), "Legacy canary");
  assert.equal(selfEvolutionLifecycleLabel({ status: "stable" }), "Legacy stable");
});

test("candidate identity is human-readable and does not fall back to internal IDs", () => {
  assert.equal(
    selfEvolutionCandidateTitle({
      candidate_id: "candidate_94f1",
      target_label: "Dynamics worker · restart guidance",
    }),
    "Dynamics worker · restart guidance",
  );
  assert.equal(selfEvolutionCandidateTitle({ candidate_id: "candidate_94f1" }), "Skill revision");
  assert.equal(selfEvolutionCandidateVersion({ revision: 12 }), "r0012");
  assert.equal(selfEvolutionLifecycleLabel({ status: "revision" }), "Revision requested");
  assert.equal(selfEvolutionRouteLabel("amend_existing_skill"), "Update an existing skill");
});

test("allowed actions are supplied by the server and never inferred from status", () => {
  const candidate = {
    status: "review",
    allowed_actions: [
      "reject",
      "promote_stable",
      "unknown_action",
      "run_review",
      "request-revision",
    ],
  };
  assert.deepEqual(
    selfEvolutionAllowedActions(candidate),
    ["run-review", "request-revision", "promote-stable", "reject"],
  );
  assert.deepEqual(selfEvolutionAllowedActions({ status: "review" }), []);
  assert.equal(normalizeSelfEvolutionAction("start_canary"), "start-canary");
  assert.equal(selfEvolutionActionDefinition("promote_stable").label, "Promote stable");
  assert.equal(selfEvolutionActionDefinition("obsolete_action"), null);
  assert.deepEqual(
    selfEvolutionFilterCandidates([candidate, { status: "stable", allowed_actions: [] }], "needs-action"),
    [candidate],
  );
});

test("review projection exposes behavior, evidence, scope, concerns, and checks without JSON fallback", () => {
  const candidate = {
    behavior_change: {
      summary: "Stop adding transfer checks to ordinary handoffs.",
      before: "Every handoff gained a checksum.",
      after: "Integrity checks run only when the transfer contract requires them.",
      impact: "Removes an unnecessary tool call.",
    },
    review: {
      recommendation: "needs_revision",
      summary: "The direction is useful but the recovery boundary is incomplete.",
      evidence_sufficiency: "Two independent user corrections support the change.",
      scope_assessment: "Limit it to ordinary successful handoffs.",
      proportionality: { status: "warning", explanation: "Retain a lightweight recovery fallback." },
      counterexamples: ["A verified corrupted transfer still needs validation."],
      concerns: ["The recovery clause is ambiguous."],
      human_checks: ["Confirm explicit integrity contracts still activate validation."],
    },
  };

  const behavior = selfEvolutionBehaviorChange(candidate);
  const review = selfEvolutionHumanReview(candidate);

  assert.equal(behavior.after, "Integrity checks run only when the transfer contract requires them.");
  assert.equal(review.recommendationLabel, "Reviewer requests revision");
  assert.equal(review.proportionality.label, "Needs careful scope review");
  assert.deepEqual(review.concerns, ["The recovery clause is ambiguous."]);
  assert.deepEqual(review.humanChecks, ["Confirm explicit integrity contracts still activate validation."]);
});

test("evidence projection preserves semantic trace anchors", () => {
  const evidence = selfEvolutionEvidenceItems([{
    claim: "A repaired run succeeded after removing an unconditional checksum.",
    signal_label: "Existing skill revision",
    status_label: "Included in a candidate revision",
    evidence: [{
      source_ref: "run:run-7:event:31",
      reason: "First changed decision",
      excerpt: "The ordinary handoff continued without an extra integrity ritual.",
    }],
  }]);
  assert.equal(evidence.length, 2);
  assert.equal(evidence[1].sourceRef, "run:run-7:event:31");
  assert.equal("omittedCount" in evidence[1], false);
});

test("promotion confirmation repeats exact version, target, scope, boundaries, and concerns", () => {
  const text = selfEvolutionPromotionConfirmation(
    {
      version: "r0007",
      target_label: "Materials worker · transfer guidance",
      behavior_change: "Remove unconditional checksum checks.",
      applicability_boundary: ["Ordinary successful handoffs"],
      non_applicability: ["Explicit integrity contracts"],
      review: { concerns: ["Keep recovery-only validation."] },
    },
    "demo-space",
    "Stable for every future run in demo-space",
  );

  assert.match(text, /r0007/);
  assert.match(text, /Materials worker · transfer guidance/);
  assert.match(text, /Stable for every future run in demo-space/);
  assert.match(text, /Ordinary successful handoffs/);
  assert.match(text, /Explicit integrity contracts/);
  assert.match(text, /Keep recovery-only validation/);
});

test("action endpoint and request body follow the immutable revision contract", () => {
  const candidate = { candidate_id: "candidate/a", revision: 7 };
  assert.equal(
    selfEvolutionActionEndpoint("ctx 1", candidate, "promote_stable"),
    "/api/session/ctx%201/self-evolution/candidates/candidate%2Fa/revisions/7/promote-stable",
  );
  assert.equal(
    selfEvolutionActionEndpoint("ctx 1", { candidate_id: "candidate/a", version: "2.4.0" }, "promote-stable"),
    "",
  );
  assert.deepEqual(
    selfEvolutionActionRequest("request-revision", {
      actor: "human",
      rationale: "The boundary is too broad.",
      guidance: "Add the explicit recovery exception.",
    }),
    {
      actor: "human",
      rationale: "The boundary is too broad.",
      guidance: "Add the explicit recovery exception.",
    },
  );
  assert.deepEqual(
    selfEvolutionActionRequest("start-canary", {
      actor: "human",
      rationale: "Low-risk validation.",
      scopeKind: "thread",
      scopeId: "thread-7",
    }),
    {
      actor: "human",
      rationale: "Low-risk validation.",
      scope_kind: "thread",
      scope_id: "thread-7",
    },
  );
});

test("candidate action errors keep safe server guidance instead of collapsing to a generic failure", () => {
  const validationError = new Error("candidate validation did not pass");
  validationError.status = 400;
  assert.equal(
    selfEvolutionDisplayError(validationError),
    "candidate validation did not pass",
  );

  const staleError = new Error("candidate revision changed");
  staleError.status = 409;
  assert.equal(
    selfEvolutionDisplayError(staleError),
    "This revision changed after it was opened. Refresh the candidate before deciding.",
  );
});

test("observations remain readable and internal paths or JSON are not rendered", () => {
  const observation = selfEvolutionObservationView({
    signal_kind: "skill_revision",
    status: "open",
    claim: "Do not add checksum checks to ordinary handoffs.",
    evidence_summary: "Correction recorded in /home/user/project/thread.json",
    outcome_summary: '{"trace_id":"secret"}',
  });

  assert.equal(observation.signalLabel, "Existing skill revision");
  assert.equal(observation.statusLabel, "Available for proposal");
  assert.doesNotMatch(observation.summary, /\/home\/user/);
  assert.equal(observation.outcome, "");
  assert.equal(redactSelfEvolutionText("Read C:\\Users\\name\\trace.json"), "Read [internal path hidden]");
});

test("observation projection accepts public signal and evidence fields", () => {
  const observation = selfEvolutionObservationView({
    signal: "workspace_preference",
    signal_label: "Workspace preference",
    status: "consolidated",
    status_label: "Included in a candidate revision",
    target: "materials_worker/initial-owner",
    resolved_target: "materials_worker/actual-owner",
    claim: "The extra validation changed no decision.",
    evidence: [{
      reason: "User correction",
      excerpt: "Do not add this check.",
    }],
  });

  assert.equal(observation.signalLabel, "Workspace preference");
  assert.equal(observation.target, "materials_worker/initial-owner");
  assert.equal(observation.resolvedTarget, "materials_worker/actual-owner");
  assert.equal(observation.evidence[0].title, "User correction");
  assert.equal(observation.evidence[0].summary, "Do not add this check.");
  assert.equal("omittedCount" in observation.evidence[0], false);
});

test("job view accepts a text completion without presenting a failure or a candidate", () => {
  const text = "The conclusion needs discussion. " + "Detailed evidence. ".repeat(500);
  const view = selfEvolutionJobView({
    result_kind: "text_completed",
    result_label: "Completed with a text response",
    summary: text,
    outcome: { reflection_items: [], no_change: false },
  });
  assert.equal(view.resultKind, "text_completed");
  assert.equal(view.resultLabel, "Completed with a text response");
  assert.equal(view.summary, text.trim());
  assert.deepEqual(view.items, []);
});

test("job view keeps no-change and execution-lapse items visible", () => {
  const view = selfEvolutionJobView({
    result_kind: "completed_without_candidate",
    result_label: "Completed without a candidate",
    summary: "The run was checked and no skill update was warranted.",
    outcome: {
      reflection_items: [
        {
          item_ref: "no-change",
          kind: "no_change",
          title: "No durable change",
          status: "no_change",
          status_label: "No durable change supported",
          reason: "The run already followed the best available SOP.",
          retry_allowed: false,
        },
        {
          item_ref: "lapse",
          kind: "execution_lapse",
          status: "execution_lapse",
          status_label: "Covered by existing guidance",
          retry_allowed: false,
        },
      ],
    },
  });

  assert.equal(view.resultKind, "completed_without_candidate");
  assert.equal(view.items.length, 2);
  assert.equal(view.items[0].title, "No durable change");
  assert.equal(view.items[0].reason, "The run already followed the best available SOP.");
  assert.equal(view.items[1].title, "Execution lapse");
  assert.equal(view.items.some((item) => item.retryAllowed), false);
});

test("job view preserves an exact failed-item retry without hiding other results", () => {
  const view = selfEvolutionJobView({
    result_kind: "partial",
    result_label: "Completed with issues",
    summary: "One independent finding needs attention.",
    outcome: {
      reflection_items: [
        {
          item_ref: "completed",
          kind: "skill_revision",
          title: "Existing skill revision",
          status_label: "Candidate created",
          retry_allowed: false,
        },
        {
          item_ref: "failed",
          kind: "skill_discovery",
          title: "New reusable skill",
          status_label: "Needs attention",
          error: "Review failed.",
          retry_allowed: true,
        },
      ],
    },
  });

  assert.deepEqual(view.items.map((item) => item.itemRef), ["completed", "failed"]);
  assert.equal(view.items[1].retryAllowed, true);
  assert.equal(view.items[1].error, "Review failed.");
});

test("job view preserves historical reflect content and the resolved owner", () => {
  const evidence = [{
    source_ref: "run:run-reflect#event:7",
    reason: "Observed method choice",
    excerpt: "The scientific attribution remains uncertain.",
  }];
  const view = selfEvolutionJobView({
    result_kind: "deferred",
    result_label: "Waiting for more real evidence",
    summary: "Wait for another independent use.",
    outcome: {
      reflection_items: [{
        item_ref: "finding-one",
        kind: "skill_revision",
        title: "materials_worker/initial-owner",
        resolved_target: "materials_worker/actual-owner",
        status: "deferred",
        status_label: "Deferred for more real evidence",
        finding: {
          change: "Keep the successful scientific method invariant.",
          rationale: "One episode does not separate method choice from runtime noise.",
          evidence_refs: evidence,
        },
      }],
    },
  });

  assert.equal(view.items[0].resolvedTarget, "materials_worker/actual-owner");
  assert.equal(view.items[0].change, "Keep the successful scientific method invariant.");
  assert.equal(view.items[0].rationale, "One episode does not separate method choice from runtime noise.");
  assert.deepEqual(view.items[0].evidenceRefs, evidence);
});
