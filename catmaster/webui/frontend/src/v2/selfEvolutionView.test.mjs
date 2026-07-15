import test from "node:test";
import assert from "node:assert/strict";

import {
  selfEvolutionCandidateTitle,
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
    { approved: 1, promoted: 0, invalid: 1, rejected: 0, conflict: 0, proposed: 0, rolled_back: 0 },
  );
  assert.equal(selfEvolutionCandidateTitle({ action: "skill", group: "materials_worker", name: "demo" }), "materials_worker/demo");
  assert.equal(selfEvolutionCandidateTitle({ action: "memory" }), "Workspace memory");
});
