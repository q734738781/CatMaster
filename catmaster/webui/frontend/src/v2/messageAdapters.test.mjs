import test from "node:test";
import assert from "node:assert/strict";

import { catMessageToAssistant } from "./messageAdapters.js";
import { entrypointMeta, normalizeEntrypoint, normalizedEntrypoints } from "./entrypoints.js";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection.js";
import { interruptActions, repeatInterruptDecision } from "./interruptPayload.js";
import { applyThreadEvent } from "./threadEventReducer.js";
import { artifactForSelection } from "./artifactSelection.js";
import { todoGroupsFromMessages } from "./todoPanel.js";

test("catMessageToAssistant converts text, tool, artifact, and interrupt parts", () => {
  const converted = catMessageToAssistant({
    id: "msg_1",
    role: "assistant",
    status: "interrupted",
    created_at: 1,
    parts: [
      { id: "part_text", type: "text", text: "hello" },
      { id: "part_tool", type: "tool-call", meta: { tool_call_id: "tc1", tool: "write_file", input: { path: "x" }, agent_name: "materials_worker" } },
      { id: "part_artifact", type: "artifact", artifact_id: "art_1", title: "table", renderer: "csv", path: "files/table.csv" },
      { id: "part_interrupt", type: "interrupt", status: "pending", text: "Review", meta: { title: "Review required" } },
    ],
  });

  assert.equal(converted.status.type, "requires-action");
  assert.equal(converted.content[0].type, "text");
  assert.equal(converted.content[1].type, "tool-call");
  assert.equal(converted.content[1].toolName, "write_file");
  assert.equal(converted.content[1].source, "materials_worker");
  assert.deepEqual(converted.content[1].args, { path: "x" });
  assert.equal(converted.content[2].type, "data");
  assert.equal(converted.content[2].data.type, "artifact");
  assert.equal(converted.content[3].type, "data");
  assert.equal(converted.content[3].data.type, "interrupt");
});

test("applyThreadEvent preserves tool call agent source metadata", () => {
  const messages = [
    {
      id: "msg_1",
      role: "assistant",
      status: "streaming",
      parts: [{ id: "part_text", type: "text", text: "", status: "streaming" }],
    },
  ];

  const started = applyThreadEvent(messages, {
    event: "tool_call.started",
    message_id: "msg_1",
    data: {
      part_id: "part_tool_1",
      tool_call_id: "tc1",
      tool: "write_todos",
      input: { todos: [] },
      agent_name: "experiment_specialist",
    },
  });
  const completed = applyThreadEvent(started, {
    event: "tool_call.completed",
    message_id: "msg_1",
    data: {
      part_id: "part_tool_1",
      tool_call_id: "tc1",
      tool: "write_todos",
      input: { todos: [] },
      output: "ok",
      subagent_source: "materials_worker",
    },
  });

  assert.equal(started[0].parts[1].meta.agent_name, "experiment_specialist");
  assert.equal(completed[0].parts[1].meta.agent_name, "experiment_specialist");
  assert.equal(completed[0].parts[1].meta.subagent_source, "materials_worker");
  assert.equal(catMessageToAssistant(completed[0]).content[1].source, "materials_worker");
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
          type: "tool-call",
          status: "completed",
          meta: {
            tool_call_id: "tc1",
            tool: "write_todos",
            agent_name: "research_specialist",
            input: { todos: [{ content: "Old plan", status: "completed" }] },
          },
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
          type: "tool-call",
          status: "running",
          meta: {
            tool_call_id: "tc2",
            tool: "write_todos",
            agent_name: "research_specialist",
            input: { todos: [{ content: "New plan", status: "in_progress" }] },
          },
        },
        {
          id: "part_tool_3",
          type: "tool-call",
          status: "completed",
          meta: {
            tool_call_id: "tc3",
            tool: "write_todos",
            subagent_source: "materials_worker",
            input: { todos: ["Prepare input"] },
          },
        },
      ],
    },
  ];

  const groups = todoGroupsFromMessages(messages);

  assert.equal(groups.length, 2);
  assert.equal(groups[0].source, "materials_worker");
  assert.deepEqual(groups[0].rows, [{ content: "Prepare input", status: "pending" }]);
  assert.equal(groups[1].source, "research_specialist");
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
          type: "tool-call",
          status: "completed",
          meta: {
            tool: "write_todos",
            agent_name: "research_specialist",
            input: { todos: [{ content: "Old completed task", status: "completed" }] },
          },
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
          type: "tool-call",
          status: "completed",
          meta: {
            tool: "write_todos",
            agent_name: "experiment_specialist",
            input: { todos: [{ content: "New interrupted task", status: "in_progress" }] },
          },
        },
      ],
    },
  ];

  const groups = todoGroupsFromMessages(interruptedNewTurn);
  assert.equal(groups.length, 1);
  assert.equal(groups[0].source, "experiment_specialist");
  assert.deepEqual(groups[0].rows, [{ content: "New interrupted task", status: "in_progress" }]);
});

test("applyThreadEvent replaces completed messages from SSE snapshots", () => {
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
      message: {
        id: "msg_1",
        role: "assistant",
        status: "completed",
        parts: [{ id: "part_text", type: "text", text: "Final answer.", status: "completed" }],
      },
    },
  });

  assert.equal(updated[0].status, "completed");
  assert.equal(updated[0].parts[0].text, "Final answer.");
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

test("workspace tab hash state preserves full-page monitor and files routes", () => {
  assert.equal(tabFromHash("#tab=files"), "files");
  assert.equal(tabFromHash("#tab=monitor&inspect=file&path=o2.xyz"), "monitor");
  assert.equal(tabFromHash("#tab=artifacts&inspect=artifact&artifact_id=art_1"), "chat");
  assert.equal(tabFromHash("#tab=unknown"), "chat");

  const fileHash = selectionToHash({ type: "file", path: "files/o2.xyz" }, "files");
  assert.equal(fileHash, "#tab=files&inspect=file&path=files%2Fo2.xyz");
  assert.deepEqual(selectionFromHash(fileHash), { type: "file", path: "files/o2.xyz" });

  assert.equal(selectionToHash(null, "monitor"), "#tab=monitor");
});

test("interrupt payload helpers repeat decisions for each pending action", () => {
  const part = {
    meta: {
      payload: {
        interrupts: [
          {
            value: {
              action_requests: [
                { name: "write_file", args: { path: "a" } },
                { name: "edit_file", args: { path: "b" } },
              ],
            },
          },
        ],
      },
    },
  };

  assert.deepEqual(interruptActions(part).map((item) => item.name), ["write_file", "edit_file"]);
  assert.deepEqual(repeatInterruptDecision(part, { type: "approve" }), [{ type: "approve" }, { type: "approve" }]);
});

test("entrypoint helpers preserve the specialist lanes and aliases", () => {
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
});
