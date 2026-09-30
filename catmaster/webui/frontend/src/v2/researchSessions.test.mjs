import assert from "node:assert/strict";
import test from "node:test";

import {
  buildResearchSessionRows,
  preferredWorkspaceThreadId,
  RELATED_RESEARCH_ACTIVITY_ID,
  researchSessionDefaultOpen,
  researchStateLabel,
} from "./researchSessions.js";

const threads = [
  {
    thread_id: "root-a",
    title: "ExampleResearch study with a long recoverable session title",
    thread_role: "research_root",
    parent_thread_id: "",
    updated_at: 10,
    research_activity: { state: "running" },
  },
  {
    thread_id: "execution-a2",
    title: "Run: complete ligand descriptor validation",
    thread_role: "research_execution",
    parent_thread_id: "root-a",
    updated_at: 30,
  },
  {
    thread_id: "execution-a1",
    title: "Run: baseline descriptor",
    thread_role: "research_execution",
    parent_thread_id: "root-a",
    updated_at: 20,
  },
  {
    thread_id: "task-a1",
    title: "Literature branch: benchmark evidence",
    thread_role: "agent_task",
    parent_thread_id: "root-a",
    updated_at: 25,
  },
  {
    thread_id: "planning-a",
    title: "Plan next step: internal",
    thread_role: "research_planning",
    parent_thread_id: "root-a",
    updated_at: 40,
  },
  {
    thread_id: "comparison-a",
    title: "Compare ready Experiments",
    thread_role: "research_comparison",
    parent_thread_id: "root-a",
    updated_at: 50,
  },
  {
    thread_id: "ordinary",
    title: "Direct user Experiment",
    thread_role: "primary",
    parent_thread_id: "",
    updated_at: 15,
  },
  {
    thread_id: "legacy-execution",
    title: "Recovered execution without an unambiguous owner",
    thread_role: "research_execution",
    parent_thread_id: "",
    updated_at: 12,
  },
];

test("research session rows hide diagnostics and nest only explicit execution children", () => {
  const rows = buildResearchSessionRows(threads);
  const root = rows.find((row) => row.root.thread_id === "root-a");
  const ordinary = rows.find((row) => row.root.thread_id === "ordinary");
  const related = rows.find(
    (row) => row.root.thread_id === RELATED_RESEARCH_ACTIVITY_ID,
  );

  assert.ok(root);
  assert.deepEqual(
    root.children.map((thread) => thread.thread_id),
    ["execution-a2", "execution-a1"],
  );
  assert.equal(root.isResearchSession, true);
  assert.ok(ordinary);
  assert.equal(ordinary.isResearchSession, false);
  assert.ok(related);
  assert.equal(related.isRelatedGroup, true);
  assert.deepEqual(
    related.children.map((thread) => thread.thread_id),
    ["legacy-execution"],
  );
  assert.equal(
    rows.some((row) => ["planning-a", "comparison-a"].includes(row.root.thread_id)),
    false,
  );
  assert.equal(
    rows.some((row) => row.root.thread_id === "task-a1"),
    false,
  );
});

test("child search keeps its session context and never exposes diagnostic titles", () => {
  const matches = buildResearchSessionRows(threads, "complete ligand");
  assert.equal(matches.length, 1);
  assert.equal(matches[0].root.thread_id, "root-a");
  assert.equal(matches[0].rootMatch, false);
  assert.deepEqual(
    matches[0].children.map((thread) => thread.thread_id),
    ["execution-a2"],
  );
  assert.equal(buildResearchSessionRows(threads, "Compare ready").length, 0);
  assert.equal(buildResearchSessionRows(threads, "Plan next step").length, 0);
  assert.equal(buildResearchSessionRows(threads, "benchmark evidence").length, 0);
});

test("workspace boot prefers a user thread over a newer background task", () => {
  assert.equal(
    preferredWorkspaceThreadId([
      { ...threads[3], updated_at: 100 },
      threads[0],
      threads[1],
    ]),
    "root-a",
  );
});

test("workspace refresh restores an explicitly opened background thread", () => {
  assert.equal(
    preferredWorkspaceThreadId(threads, "task-a1"),
    "task-a1",
  );
  assert.equal(
    preferredWorkspaceThreadId(threads, "missing-thread"),
    "root-a",
  );
});

test("active and action-required sessions default open while settled sessions do not", () => {
  assert.equal(researchSessionDefaultOpen({ research_activity: { state: "running" } }), true);
  assert.equal(researchSessionDefaultOpen({ research_activity: { state: "waiting" } }), true);
  assert.equal(researchSessionDefaultOpen({ research_activity: { state: "waiting_review" } }), true);
  assert.equal(researchSessionDefaultOpen({ research_activity: { state: "completed" } }), false);
  assert.equal(researchSessionDefaultOpen({ research_activity: { state: "paused" } }), false);
});

test("research states retain their distinct user-facing meanings", () => {
  assert.equal(researchStateLabel("waiting_continue"), "Waiting to continue");
  assert.equal(researchStateLabel("waiting"), "Waiting — research unfinished");
  assert.equal(researchStateLabel("waiting_review"), "Waiting for review");
  assert.equal(researchStateLabel("operationally_incomplete"), "Needs attention");
  assert.equal(researchStateLabel("comparing"), "Comparing experiments");
});
