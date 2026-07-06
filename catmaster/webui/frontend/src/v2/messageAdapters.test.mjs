import test from "node:test";
import assert from "node:assert/strict";

import { catMessageToAssistant } from "./messageAdapters.js";
import { entrypointMeta, normalizeEntrypoint, normalizedEntrypoints } from "./entrypoints.js";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection.js";
import { interruptActions, repeatInterruptDecision } from "./interruptPayload.js";
import { applyThreadEvent } from "./threadEventReducer.js";
import { artifactForSelection } from "./artifactSelection.js";

test("catMessageToAssistant converts text, tool, artifact, and interrupt parts", () => {
  const converted = catMessageToAssistant({
    id: "msg_1",
    role: "assistant",
    status: "interrupted",
    created_at: 1,
    parts: [
      { id: "part_text", type: "text", text: "hello" },
      { id: "part_tool", type: "tool-call", meta: { tool_call_id: "tc1", tool: "write_file", input: { path: "x" } } },
      { id: "part_artifact", type: "artifact", artifact_id: "art_1", title: "table", renderer: "csv", path: "files/table.csv" },
      { id: "part_interrupt", type: "interrupt", status: "pending", text: "Review", meta: { title: "Review required" } },
    ],
  });

  assert.equal(converted.status.type, "requires-action");
  assert.equal(converted.content[0].type, "text");
  assert.equal(converted.content[1].type, "tool-call");
  assert.equal(converted.content[1].toolName, "write_file");
  assert.deepEqual(converted.content[1].args, { path: "x" });
  assert.equal(converted.content[2].type, "data");
  assert.equal(converted.content[2].data.type, "artifact");
  assert.equal(converted.content[3].type, "data");
  assert.equal(converted.content[3].data.type, "interrupt");
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
