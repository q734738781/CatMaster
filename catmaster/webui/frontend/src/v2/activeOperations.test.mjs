import assert from "node:assert/strict";
import test from "node:test";

import {
  formatElapsedSeconds,
  sortActiveOperations,
  updateActiveOperations,
} from "./activeOperations.js";

test("active operations keep the oldest known start first", () => {
  const sorted = sortActiveOperations([
    { id: "new", type: "tool", status: "running", started_at: 200 },
    { id: "unknown", type: "tool", status: "running", started_at: 0 },
    { id: "old", type: "tool", status: "running", started_at: 100 },
    { id: "done", type: "tool", status: "completed", started_at: 50 },
  ]);

  assert.deepEqual(sorted.map((part) => part.id), ["old", "new", "unknown"]);
});

test("terminal update removes the same operation without touching others", () => {
  const current = [
    { id: "one", type: "tool", status: "running", started_at: 100 },
    { id: "two", type: "tool", status: "running", started_at: 200 },
  ];

  const updated = updateActiveOperations(
    current,
    { id: "one", type: "tool", status: "completed", started_at: 100, duration_seconds: 25 },
  );

  assert.deepEqual(updated.map((part) => part.id), ["two"]);
});

test("elapsed formatting remains compact across long executions", () => {
  assert.equal(formatElapsedSeconds(8), "8s");
  assert.equal(formatElapsedSeconds(81), "1m 21s");
  assert.equal(formatElapsedSeconds(3723), "1h 02m 03s");
});

test('native subagents survive parent completion while compaction does not', async () => {
  const { retainActiveAsyncSubagents } = await import('./activeOperations.js');
  const child = { id: 'child', type: 'subagent', status: 'running' };
  const parts = [child, {id: 'compact', type: 'progress', status: 'running'}, {id: 'tool', type: 'tool', status: 'running'}];
  assert.deepEqual(retainActiveAsyncSubagents(parts), [child]);
  assert.deepEqual(updateActiveOperations([child], {...child, status: 'completed'}), []);
});
