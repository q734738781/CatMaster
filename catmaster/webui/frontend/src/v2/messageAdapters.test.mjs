import test from "node:test";
import assert from "node:assert/strict";

import { catMessageToAssistant } from "./messageAdapters.js";
import { entrypointMeta, normalizeEntrypoint, normalizedEntrypoints } from "./entrypoints.js";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection.js";
import { applyThreadEvent } from "./threadEventReducer.js";
import { artifactForSelection } from "./artifactSelection.js";
import { canonicalTodoPartsFromEvent, todoGroupsFromMessages } from "./todoPanel.js";
import { normalizeMathMarkdown } from "./markdown.js";

test("catMessageToAssistant preserves only projected presentation parts", () => {
  const converted = catMessageToAssistant({
    id: "msg_1",
    role: "assistant",
    status: "interrupted",
    created_at: 1,
    parts: [
      { id: "part_text", type: "text", text: "hello", fields: [], actions: [], items: [] },
      { id: "part_tool", type: "tool", title: "Write file", summary: "The operation completed.", fields: [{ label: "Path", value: "files/x" }], actions: [], items: [] },
      { id: "part_artifact", type: "artifact", artifact_id: "art_1", title: "table", renderer: "csv", path: "files/table.csv" },
      { id: "part_interrupt", type: "interrupt", status: "pending", title: "Review required", summary: "Choose an action.", actions: [], items: [] },
      { id: "part_sources", type: "citations", title: "Sources", items: [{ label: "OpenAI API", href: "https://developers.openai.com/api/docs/guides/tools-web-search" }] },
    ],
  });

  assert.equal(converted.status.type, "requires-action");
  assert.equal(converted.content[0].type, "data");
  assert.equal(converted.content[0].data.text, "hello");
  assert.equal(converted.content[1].data.type, "tool");
  assert.equal(converted.content[1].data.title, "Write file");
  assert.equal(converted.content[1].data.meta, undefined);
  assert.equal(converted.content[2].data.type, "artifact");
  assert.equal(converted.content[3].data.type, "interrupt");
  assert.equal(converted.content[4].data.type, "citations");
  assert.equal(converted.content[4].data.items[0].label, "OpenAI API");
});

test("normalizeMathMarkdown converts common LLM math delimiters", () => {
  const source = "inline \\(E=mc^2\\) and display:\\n\\[x^2 + y^2 = z^2\\]";
  const normalized = normalizeMathMarkdown(source);
  assert.match(normalized, /inline \$E=mc\^2\$/);
  assert.match(normalized, /\$\$\nx\^2 \+ y\^2 = z\^2\n\$\$/);
});

test("applyThreadEvent consumes projected activity updates without raw tool metadata", () => {
  const messages = [
    {
      id: "msg_1",
      role: "assistant",
      status: "streaming",
      parts: [{ id: "part_text", type: "text", text: "", status: "streaming" }],
    },
  ];

  const started = applyThreadEvent(messages, {
    event: "activity.updated",
    message_id: "msg_1",
    data: {
      part: {
        id: "part_tool_1",
        type: "tool",
        status: "running",
        title: "Research plan",
        summary: "Work is in progress.",
        fields: [],
        actions: [],
        items: [],
      },
    },
  });
  const completed = applyThreadEvent(started, {
    event: "activity.updated",
    message_id: "msg_1",
    data: {
      part: {
        id: "part_tool_1",
        type: "tool",
        status: "completed",
        title: "Research plan",
        summary: "The operation completed.",
        fields: [],
        actions: [],
        items: [],
      },
    },
  });

  assert.equal(started[0].parts[1].title, "Research plan");
  assert.equal(completed[0].parts[1].status, "completed");
  assert.equal(completed[0].parts[1].meta, undefined);
  assert.equal(catMessageToAssistant(completed[0]).content[1].data.summary, "The operation completed.");
});

test("todoGroupsFromMessages keeps latest write_todos per agent source", () => {
  const messages = [
    {
      id: "msg_1",
      role: "user",
      updated_at: 0,
      parts: [{ id: "part_user_1", type: "text", text: "run old plan" }],
    },
    {
      id: "msg_2",
      updated_at: 1,
      role: "assistant",
      parts: [
        {
          id: "part_tool_1",
          type: "progress",
          status: "completed",
          title: "Research specialist plan",
          items: [{ label: "Old plan", status: "completed" }],
        },
      ],
    },
    {
      id: "msg_3",
      role: "user",
      updated_at: 1.5,
      parts: [{ id: "part_user_2", type: "text", text: "run new plan" }],
    },
    {
      id: "msg_4",
      updated_at: 2,
      role: "assistant",
      parts: [
        {
          id: "part_tool_2",
          type: "progress",
          status: "running",
          title: "Research specialist plan",
          items: [{ label: "New plan", status: "in_progress" }],
        },
        {
          id: "part_tool_3",
          type: "progress",
          status: "completed",
          title: "Materials specialist plan",
          items: [{ label: "Prepare input", status: "pending" }],
        },
      ],
    },
  ];

  const groups = todoGroupsFromMessages(messages);

  assert.equal(groups.length, 2);
  assert.equal(groups[0].source, "Materials specialist");
  assert.deepEqual(groups[0].rows, [{ content: "Prepare input", status: "pending" }]);
  assert.equal(groups[1].source, "Research specialist");
  assert.deepEqual(groups[1].rows, [{ content: "New plan", status: "in_progress" }]);
});

test("todoGroupsFromMessages scopes todos to the latest user turn", () => {
  const completedPreviousTurn = [
    {
      id: "msg_user_1",
      role: "user",
      updated_at: 1,
      parts: [{ id: "part_user_1", type: "text", text: "first request" }],
    },
    {
      id: "msg_assistant_1",
      role: "assistant",
      updated_at: 2,
      status: "completed",
      parts: [
        {
          id: "part_old_todos",
          type: "progress",
          status: "completed",
          title: "Research specialist plan",
          items: [{ label: "Old completed task", status: "completed" }],
        },
      ],
    },
    {
      id: "msg_user_2",
      role: "user",
      updated_at: 3,
      parts: [{ id: "part_user_2", type: "text", text: "second request" }],
    },
  ];

  assert.deepEqual(todoGroupsFromMessages(completedPreviousTurn), []);

  const interruptedNewTurn = [
    ...completedPreviousTurn,
    {
      id: "msg_assistant_2",
      role: "assistant",
      updated_at: 4,
      status: "interrupted",
      parts: [
        {
          id: "part_new_todos",
          type: "progress",
          status: "completed",
          title: "Experiment specialist plan",
          items: [{ label: "New interrupted task", status: "in_progress" }],
        },
      ],
    },
  ];

  const groups = todoGroupsFromMessages(interruptedNewTurn);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].source, "Experiment specialist");
  assert.deepEqual(groups[0].rows, [{ content: "New interrupted task", status: "in_progress" }]);
});

test("message completion pushes the canonical terminal todo projection", () => {
  const completed = [{
    id: "part_final_plan",
    type: "progress",
    items: [{ label: "Write synthesis", status: "completed" }],
  }];
  assert.deepEqual(canonicalTodoPartsFromEvent({
    event: "message.completed",
    data: { todo_parts: completed },
  }), completed);
  assert.deepEqual(canonicalTodoPartsFromEvent({
    event: "message.completed",
    data: {},
  }), []);
  assert.equal(canonicalTodoPartsFromEvent({
    event: "activity.updated",
    data: { todo_parts: completed },
  }), null);
});

test("applyThreadEvent completes a message without requiring a repeated full snapshot", () => {
  const messages = [
    {
      id: "msg_1",
      role: "assistant",
      status: "streaming",
      parts: [{ id: "part_text", type: "text", text: "", status: "streaming" }],
    },
  ];

  const updated = applyThreadEvent(messages, {
    event: "message.completed",
    message_id: "msg_1",
    data: {
      status: "completed",
    },
  });

  assert.equal(updated[0].status, "completed");
  assert.equal(updated[0].parts[0].text, "");
});

test("applyThreadEvent streams reasoning deltas into audit parts", () => {
  const messages = [
    {
      id: "msg_1",
      role: "assistant",
      status: "streaming",
      parts: [{ id: "part_text", type: "text", text: "", status: "streaming" }],
    },
  ];

  const withPart = applyThreadEvent(messages, {
    event: "message.part.created",
    message_id: "msg_1",
    data: {
      part: {
        id: "part_reasoning",
        type: "reasoning",
        text: "",
        status: "streaming",
        meta: { source: "model" },
      },
    },
  });
  const updated = applyThreadEvent(withPart, {
    event: "reasoning.delta",
    message_id: "msg_1",
    data: {
      message_id: "msg_1",
      part_id: "part_reasoning",
      delta: "Plan: inspect files",
    },
  });

  assert.equal(updated[0].parts[1].type, "reasoning");
  assert.equal(updated[0].parts[1].text, "Plan: inspect files");
  assert.equal(updated[0].parts[1].status, "streaming");
});

test("inspector selection serializes artifact and file selections", () => {
  const artifactHash = selectionToHash({ type: "artifact", artifact_id: "art_123" });
  assert.equal(artifactHash, "#inspect=artifact&artifact_id=art_123");
  assert.deepEqual(selectionFromHash(artifactHash), { type: "artifact", artifact_id: "art_123" });

  const fileHash = selectionToHash({ type: "file", path: "files/report.md" });
  assert.equal(fileHash, "#inspect=file&path=files%2Freport.md");
  assert.deepEqual(selectionFromHash(fileHash), { type: "file", path: "files/report.md" });
});

test("artifact selection falls back when message meta is empty", () => {
  const listed = {
    artifact_id: "art_1",
    path: "files/structures/o2/o2_start.vasp",
    title: "o2_start.vasp",
    renderer: "structure",
  };

  assert.deepEqual(
    artifactForSelection({ type: "artifact", artifact_id: "art_1", artifact: {} }, [listed]),
    listed,
  );
  assert.deepEqual(
    artifactForSelection({ type: "artifact", artifact_id: "art_1", artifact: { title: "override.vasp" } }, [listed]),
    { ...listed, title: "override.vasp" },
  );
});

test("workspace tab hash state preserves monitor, research map, evolution, and files routes", () => {
  assert.equal(tabFromHash("#tab=files"), "files");
  assert.equal(tabFromHash("#tab=monitor&inspect=file&path=o2.xyz"), "monitor");
  assert.equal(tabFromHash("#tab=hypotheses"), "hypotheses");
  assert.equal(tabFromHash("#tab=evolution"), "evolution");
  assert.equal(tabFromHash("#tab=artifacts&inspect=artifact&artifact_id=art_1"), "chat");
  assert.equal(tabFromHash("#tab=unknown"), "chat");

  const fileHash = selectionToHash({ type: "file", path: "files/o2.xyz" }, "files");
  assert.equal(fileHash, "#tab=files&inspect=file&path=files%2Fo2.xyz");
  assert.deepEqual(selectionFromHash(fileHash), { type: "file", path: "files/o2.xyz" });

  assert.equal(selectionToHash(null, "monitor"), "#tab=monitor");
  assert.equal(selectionToHash(null, "hypotheses"), "#tab=hypotheses");
  assert.equal(selectionToHash(null, "evolution"), "#tab=evolution");
});

test("entrypoint helpers preserve specialist entries and aliases without internal labels", () => {
  const rows = normalizedEntrypoints([
    { id: "research", label: "Research" },
    { id: "experiment", label: "Experiment" },
    { id: "writing", label: "Writing" },
    { id: "peer_review", label: "Peer Review" },
    { id: "literature_review", label: "Literature Review" },
  ]);

  assert.deepEqual(rows.map((item) => item.id), ["research", "experiment", "writing", "peer_review", "literature_review"]);
  assert.equal(normalizeEntrypoint("literature", rows), "literature_review");
  assert.equal(normalizeEntrypoint("bad-entry", rows), "research");
  assert.equal(entrypointMeta("peer-review", rows).label, "Peer Review");
  assert.doesNotMatch(entrypointMeta("literature", rows).summary, /\blane\b/i);

  const backendRows = normalizedEntrypoints([
    { id: "research", summary: "Routes work to the coordinator worker lane." },
  ]);
  assert.equal(backendRows[0].label, "Research");
  assert.doesNotMatch(backendRows[0].summary, /\b(?:worker|lane)\b/i);
});
