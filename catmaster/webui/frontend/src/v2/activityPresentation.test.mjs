import test from "node:test";
import assert from "node:assert/strict";

import {
  LONG_ACTIVITY_THRESHOLD,
  LONG_REASONING_TEXT_THRESHOLD,
  isLongActivityGroup,
  latestTodoParts,
  organizeTurnParts,
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
