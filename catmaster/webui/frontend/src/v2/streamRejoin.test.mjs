import test from "node:test";
import assert from "node:assert/strict";
import { applyThreadEvent } from "./threadEventReducer.js";
import { insertById } from "./messageAdapters.js";
import { organizeTurnParts } from "./activityPresentation.js";

test("compaction progress survives reload and completes separately from answer text", () => {
  const progress = {id: "compact", type: "progress", title: "上下文压缩", status: "running", text: "正在压缩上下文…"};
  let rows = applyThreadEvent([], {event: "message.part.created", message_id: "m", data: {part: progress}});
  rows = JSON.parse(JSON.stringify(rows)); // reloaded REST representation
  rows = applyThreadEvent(rows, {event: "activity.updated", message_id: "m", data: {part: {...progress, status: "completed", text: "上下文已压缩，继续处理请求。"}}});
  rows = applyThreadEvent(rows, {event: "message.delta", message_id: "m", data: {part_id: "reply", delta: "报告已整理", text_offset: 0}});
  const presentation = organizeTurnParts(rows[0].parts);
  assert.equal(presentation.semanticProgressParts.length, 1);
  assert.equal(presentation.semanticProgressParts[0].status, "completed");
  assert.equal(presentation.semanticProgressParts[0].title, "上下文压缩");
  assert.deepEqual(presentation.contentParts.map((part) => part.text), ["报告已整理"]);
});

test("snapshot and positioned replay preserve digits and intentional repetition", () => {
  const full = "🧪 E=-4090.864 eV; yes yes";
  const fragments = ["🧪 E=-", "409", "0.864 eV; ", "yes ", "yes"];
  for (let boundary = 0; boundary <= full.length; boundary++) {
    let rows = [{id: "m", parts: [{id: "p", type: "text", text: full.slice(0, boundary)}]}];
    let offset = 0;
    for (const fragment of fragments) {
      const event = {event: "message.delta", message_id: "m", data: {part_id: "p", delta: fragment, text_offset: offset}};
      rows = applyThreadEvent(rows, event);
      rows = applyThreadEvent(rows, event); // transport replay is idempotent
      offset += fragment.length;
    }
    assert.equal(rows[0].parts[0].text, full);
  }
});

test("replayed create events cannot erase a newer REST snapshot", () => {
  const message = {id: "m", parts: [{id: "p", type: "text", text: "E=-409"}]};
  let rows = applyThreadEvent([message], {event: "message.created", data: {message: {id: "m", parts: []}}});
  rows = applyThreadEvent(rows, {event: "message.part.created", message_id: "m", data: {part: {id: "p", text: ""}}});
  assert.equal(rows[0].parts[0].text, "E=-409");
});

test("late historical deltas cannot modify a canonical completed answer", () => {
  const message = {id: "m", status: "completed", parts: [{id: "p", text: "Final result"}]};
  assert.deepEqual(applyThreadEvent([message], {event: "message.delta", message_id: "m", data: {part_id: "p", text_offset: 0, delta: "Earlier tool commentary that was longer"}}), [message]);
});

test("late submit acknowledgement cannot erase tokens received through SSE", () => {
  const rows = [{id: "m", parts: [{id: "p", text: "E=-409"}]}];
  assert.equal(insertById(rows, {id: "m", parts: []}), rows);
  assert.equal(insertById([], rows[0]).length, 1);
});

test("native child status updates independently of the completed parent reply", () => {
  const message = {id: "m", status: "completed", parts: [{id: "child", type: "progress", status: "pending"}]};
  let rows = applyThreadEvent([message], {event: "subagent.started", message_id: "m", data: {part: {id: "child", status: "running"}}});
  assert.equal(rows[0].parts[0].status, "running");
  rows = applyThreadEvent(rows, {event: "subagent.completed", message_id: "m", data: {part: {id: "child", status: "completed"}}});
  assert.equal(rows[0].parts[0].status, "completed");
});

test("async message snapshots complete tools after deltas regardless of browser clock", () => {
  let rows = [{id: 'worker', updated_at: 10, status: 'streaming', parts: [{id: 'text', type: 'text', text: ''}]}];
  rows = applyThreadEvent(rows, {event: 'message.delta', message_id: 'worker', data: {part_id: 'text', delta: 'Cu report', text_offset: 0}});
  rows = applyThreadEvent(rows, {event: 'message.updated', data: {message: {
    id: 'worker', updated_at: 11, status: 'completed', source: 'Writing Worker',
    parts: [{id: 'text', type: 'text', text: 'Cu report'}, {id: 'tool', type: 'tool', status: 'completed'}],
  }}});
  assert.equal(rows[0].status, 'completed');
  assert.equal(rows[0].source, 'Writing Worker');
  assert.equal(rows[0].parts[1].status, 'completed');
  rows = applyThreadEvent(rows, {event: 'message.updated', data: {message: {id: 'worker', updated_at: 9, parts: []}}});
  assert.equal(rows[0].parts.length, 2);
});
