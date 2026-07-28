import test from "node:test";
import assert from "node:assert/strict";

import {
  selfEvolutionCandidateTitle,
  selfEvolutionHumanReview,
  selfEvolutionLifecycleLabel,
  selfEvolutionPromotionConfirmation,
  selfEvolutionStatusCounts,
  sortSelfEvolutionCandidates,
} from "./selfEvolutionView.js";

test("workspace evolution candidates put human review first", () => {
  const rows = sortSelfEvolutionCandidates([
    { candidate_id: "rejected", status: "rejected", updated_at: "2026-01-03" },
    { candidate_id: "approved-old", status: "approved", updated_at: "2026-01-01" },
    { candidate_id: "approved-new", status: "approved", updated_at: "2026-01-02" },
    { candidate_id: "conflict", status: "conflict", updated_at: "2026-01-04" },
  ]);

  assert.deepEqual(rows.map((row) => row.candidate_id), ["approved-new", "approved-old", "conflict", "rejected"]);
});

test("workspace evolution summaries and titles are stable", () => {
  assert.deepEqual(
    selfEvolutionStatusCounts({ candidates: [{ status: "approved" }, { status: "invalid" }] }),
    { reviewed: 0, approved: 1, promoted: 0, invalid: 1, rejected: 0, conflict: 0, proposed: 0, rolled_back: 0 },
  );
  assert.equal(selfEvolutionCandidateTitle({ action: "skill", group: "materials_worker", name: "demo" }), "materials_worker/demo");
  assert.equal(selfEvolutionCandidateTitle({ action: "memory" }), "Workspace memory");
});

test("workspace evolution card projection exposes readable review without raw JSON", () => {
  const candidate = {
    action: "skill",
    group: "dynamics_worker",
    name: "restart",
    status: "reviewed",
    human_review: {
      structured_review_available: true,
      reviewer_recommendation: "needs_revision",
      summary: "Limit identity validation to failed-tail restart.",
      change_points: [{ title: "Narrow retry check" }],
      proportionality_assessment: { status: "warning", explanation: "Keep it recovery-only." },
      concerns: ["Do not apply this to ordinary handoffs."],
      human_checks: ["Inspect the activation clause."],
    },
  };

  const review = selfEvolutionHumanReview(candidate);

  assert.equal(selfEvolutionLifecycleLabel(candidate), "Awaiting human decision");
  assert.equal(review.recommendation, "needs_revision");
  assert.equal(review.changePoints.length, 1);
  assert.equal(review.proportionality.status, "warning");
  assert.deepEqual(review.concerns, ["Do not apply this to ordinary handoffs."]);
});

test("legacy candidates stay readable without fabricated change points", () => {
  const review = selfEvolutionHumanReview({
    review: { decision: "approve", rationale: "Legacy rationale." },
  });

  assert.equal(review.structuredReviewAvailable, false);
  assert.equal(review.recommendation, "approve");
  assert.match(review.summary, /legacy candidate/i);
  assert.deepEqual(review.changePoints, []);
  assert.deepEqual(review.concerns, []);
});

test("promotion confirmation repeats target summary and concerns", () => {
  const text = selfEvolutionPromotionConfirmation(
    {
      action: "skill",
      group: "materials_worker",
      name: "handoff",
      human_review: {
        structured_review_available: true,
        reviewer_recommendation: "approve",
        summary: "Remove an unsupported checksum requirement.",
        proportionality_assessment: { status: "pass", explanation: "" },
        concerns: ["Confirm no recovery-only identity check was removed."],
      },
    },
    "demo-space",
  );

  assert.match(text, /materials_worker\/handoff/);
  assert.match(text, /Remove an unsupported checksum requirement/);
  assert.match(text, /Confirm no recovery-only identity check was removed/);
  assert.match(text, /demo-space/);
});
