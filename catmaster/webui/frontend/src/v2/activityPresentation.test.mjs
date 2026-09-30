import test from "node:test";
import assert from "node:assert/strict";

import {
  LONG_ACTIVITY_THRESHOLD,
  LONG_REASONING_TEXT_THRESHOLD,
  hasVisibleTurnPresentation,
  isLongActivityGroup,
  isSemanticProgressPart,
  latestTodoParts,
  organizeTurnParts,
  todoCardProjection,
  withCanonicalTodoParts,
} from "./activityPresentation.js";

test("keeps only the latest Todo state per source and lifts it out of activity", () => {
  const parts = [
    {
      id: "plan-old",
      type: "progress",
      title: "Materials plan",
      activity_group_title: "Materials",
      items: [{ label: "Inspect input", status: "pending" }],
    },
    { id: "answer", type: "text", text: "Result" },
    {
      id: "plan-new",
      type: "progress",
      title: "Materials plan",
      activity_group_title: "Materials",
      items: [{ label: "Inspect input", status: "completed" }],
    },
  ];

  assert.deepEqual(latestTodoParts(parts).map((part) => part.id), ["plan-new"]);
  const presentation = organizeTurnParts(parts);
  assert.deepEqual(presentation.planParts.map((part) => part.id), ["plan-new"]);
  assert.deepEqual(presentation.contentParts.map((part) => part.id), ["answer"]);
  assert.equal(presentation.activityGroups.length, 0);
});

test("lifts semantic phase updates above the collapsible activity trace", () => {
  const parts = [
    { id: "reasoning", type: "reasoning", status: "streaming", text: "Inspecting sources" },
    {
      id: "phase-one",
      type: "progress",
      status: "completed",
      summary: "Candidate screening has started.",
    },
    { id: "tool", type: "tool", status: "running", title: "Specialist task" },
    {
      id: "phase-two",
      type: "progress",
      status: "completed",
      summary: "The reproducibility comparison is starting.",
    },
  ];

  assert.equal(isSemanticProgressPart(parts[1]), true);
  const presentation = organizeTurnParts(parts);
  assert.deepEqual(
    presentation.semanticProgressParts.map((part) => part.id),
    ["phase-one", "phase-two"],
  );
  assert.deepEqual(
    presentation.activityGroups[0].parts.map((part) => part.id),
    ["reasoning", "tool"],
  );
  assert.equal(hasVisibleTurnPresentation(presentation), true);
});

test("canonical Todo push replaces paginated inline plan history, including an empty terminal state", () => {
  const parts = [
    { id: "answer", type: "text", text: "Result" },
    {
      id: "stale-plan",
      type: "progress",
      title: "Materials plan",
      items: [{ label: "Old", status: "pending" }],
    },
  ];
  const canonical = [
    {
      id: "current-plan",
      type: "progress",
      title: "Materials plan",
      items: [{ label: "Current", status: "completed" }],
    },
  ];

  assert.deepEqual(
    withCanonicalTodoParts(parts, canonical).map((part) => part.id),
    ["answer", "current-plan"],
  );
  assert.deepEqual(withCanonicalTodoParts(parts, []).map((part) => part.id), ["answer"]);
});

test("assigns one Todo card owner per user task and coalesces resume snapshots", () => {
  const plan = (id, status) => ({
    id,
    type: "progress",
    title: "Materials plan",
    activity_group_title: "Materials",
    items: [{ label: "Optimize O2", status }],
  });
  const messages = [
    { id: "user-a", role: "user", parts: [] },
    { id: "assistant-a1", role: "assistant", parts: [plan("plan-a1", "pending")] },
    { id: "assistant-a2", role: "assistant", parts: [plan("plan-a2", "in_progress")] },
    { id: "assistant-a3", role: "assistant", parts: [plan("plan-a3", "completed")] },
    { id: "user-b", role: "user", parts: [] },
    { id: "assistant-b1", role: "assistant", parts: [plan("plan-b1", "pending")] },
    { id: "assistant-b2", role: "assistant", parts: [plan("plan-b2", "in_progress")] },
  ];
  const current = [plan("plan-b-canonical", "completed")];

  const projection = todoCardProjection(messages, current);

  assert.deepEqual([...projection.ownerMessageIds], ["assistant-a3", "assistant-b2"]);
  assert.equal(projection.currentOwnerId, "assistant-b2");
  assert.deepEqual(
    projection.partsByMessageId.get("assistant-a3").map((part) => part.id),
    ["plan-a3"],
  );
  assert.deepEqual(
    projection.partsByMessageId.get("assistant-b2").map((part) => part.id),
    ["plan-b-canonical"],
  );
  assert.deepEqual(
    withCanonicalTodoParts(messages[1].parts, []).map((part) => part.id),
    [],
  );
});

test("treats an empty current Todo projection as authoritative", () => {
  const messages = [
    { id: "user", role: "user", parts: [] },
    {
      id: "assistant",
      role: "assistant",
      parts: [{
        id: "stale-plan",
        type: "progress",
        title: "Specialist plan",
        items: [{ label: "Scratch task", status: "pending" }],
      }],
    },
  ];

  const projection = todoCardProjection(messages, []);
  assert.equal(projection.currentOwnerId, "assistant");
  assert.deepEqual(projection.partsByMessageId.get("assistant"), []);
});

test("a Todo-only snapshot becomes an empty presentation after coalescing", () => {
  const staleParts = [{
    id: "stale-plan",
    type: "progress",
    title: "Materials plan",
    items: [{ label: "Optimize O2", status: "in_progress" }],
  }];

  assert.equal(
    hasVisibleTurnPresentation(organizeTurnParts(withCanonicalTodoParts(staleParts, []))),
    false,
  );
  assert.equal(
    hasVisibleTurnPresentation(organizeTurnParts(staleParts)),
    true,
  );
});

test("separates same-named subagents by lifecycle and exposes only the latest active item", () => {
  const parts = [
    {
      id: "a-progress",
      type: "progress",
      status: "completed",
      title: "Materials",
      activity_group_id: "run-a",
      activity_group_title: "Materials",
    },
    {
      id: "a-read",
      type: "tool",
      status: "completed",
      title: "Materials · Read file",
      activity_group_id: "run-a",
      activity_group_title: "Materials",
    },
    {
      id: "b-progress",
      type: "progress",
      status: "running",
      title: "Materials",
      activity_group_id: "run-b",
      activity_group_title: "Materials",
    },
    {
      id: "b-running",
      type: "tool",
      status: "running",
      title: "Materials · Relax structure",
      activity_group_id: "run-b",
      activity_group_title: "Materials",
    },
    {
      id: "b-finished-after",
      type: "tool",
      status: "completed",
      title: "Materials · Write note",
      activity_group_id: "run-b",
      activity_group_title: "Materials",
    },
  ];

  const groups = organizeTurnParts(parts).activityGroups;
  assert.equal(groups.length, 2);
  assert.deepEqual(groups.map((group) => group.id), ["run-a", "run-b"]);
  assert.equal(groups[0].status, "completed");
  assert.equal(groups[1].status, "running");
  assert.equal(groups[1].activePart.id, "b-running");
  assert.equal(LONG_ACTIVITY_THRESHOLD, 3);
});

test("collapses a single substantial reasoning trace without removing its text", () => {
  const longText = "Planning an independently checkable literature route. ".repeat(30);
  const group = {
    parts: [
      {
        id: "reasoning-long",
        type: "reasoning",
        status: "running",
        text: longText,
      },
    ],
  };

  assert.ok(longText.length > LONG_REASONING_TEXT_THRESHOLD);
  assert.equal(isLongActivityGroup(group), true);
  assert.equal(group.parts[0].text, longText);
  assert.equal(isLongActivityGroup({
    parts: [{ type: "reasoning", text: "Checking one source." }],
  }), false);
});

test("puts root reasoning and tools in one CatMaster activity group", () => {
  const presentation = organizeTurnParts([
    { id: "reasoning", type: "reasoning", status: "completed", title: "Progress" },
    { id: "tool", type: "tool", status: "completed", title: "Read file" },
    { id: "artifact", type: "artifact", status: "completed", title: "Report" },
  ]);

  assert.deepEqual(presentation.contentParts.map((part) => part.id), ["artifact"]);
  assert.equal(presentation.activityGroups.length, 1);
  assert.equal(presentation.activityGroups[0].title, "CatMaster");
  assert.deepEqual(presentation.activityGroups[0].parts.map((part) => part.id), ["reasoning", "tool"]);
});
