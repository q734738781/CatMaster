import test from "node:test";
import assert from "node:assert/strict";
import { subscribeThreadStream } from "./threadStream.js";

const settle = () => new Promise((resolve) => setImmediate(resolve));

function fixture(loadSnapshot) {
  const connections = [];
  const events = [];
  const errors = [];
  const visibility = new EventTarget();
  visibility.hidden = false;
  class Source extends EventTarget {
    constructor(url) { super(); this.url = url; this.readyState = 0; connections.push(this); }
    close() { this.closed = true; }
    open() { this.readyState = 1; this.onopen?.(); }
    disconnect() { this.readyState = 0; this.onerror?.(); }
    emit(value) { this.dispatchEvent(new MessageEvent("activity.updated", { data: value })); }
  }
  const stop = subscribeThreadStream({ url: "/stream", eventNames: ["activity.updated"],
    loadSnapshot, onEvent: (event) => events.push(event.data), onError: (error) => errors.push(error),
    visibility, EventSourceClass: Source });
  function hide(hidden) { visibility.hidden = hidden; visibility.dispatchEvent(new Event("visibilitychange")); }
  return { connections, events, errors, stop, hide };
}

test("returning to the tab jumps to the latest snapshot without replaying old tools", async () => {
  let cursor = 10;
  let displayed = "read_sources";
  const f = fixture(async () => { displayed = cursor === 10 ? "read_sources" : "completed"; return cursor; });
  await settle();
  const old = f.connections[0];
  old.open();
  old.emit("live tool");
  f.hide(true);
  assert.equal(old.closed, true);
  cursor = 200;
  old.emit("buffered old tool");
  f.hide(false);
  await settle();
  assert.equal(displayed, "completed");
  assert.equal(f.connections[1].url, "/stream?last_seq=200");
  assert.deepEqual(f.events, ["live tool"]);
  f.stop();
});

test("network recovery reloads the snapshot before accepting replay events", async () => {
  let cursor = 1;
  const f = fixture(async () => cursor);
  await settle();
  const old = f.connections[0];
  old.open(); old.disconnect();
  cursor = 99;
  old.emit("queued tool");
  old.open(); old.emit("replayed tool");
  await settle();
  assert.equal(old.closed, true);
  assert.equal(f.connections[1].url, "/stream?last_seq=99");
  f.connections[1].open(); f.connections[1].emit("new tool");
  assert.deepEqual(f.events, ["new tool"]);
  f.stop();
});

test("a stale snapshot cannot reconnect after hiding or switching threads", async () => {
  const pending = [];
  const f = fixture((signal) => new Promise((resolve) => pending.push({ signal, resolve })));
  f.hide(true);
  assert.equal(pending[0].signal.aborted, true);
  f.hide(false);
  pending[0].resolve(1);
  await settle();
  assert.equal(f.connections.length, 0);
  pending[1].resolve(50);
  await settle();
  assert.equal(f.connections.length, 1);
  f.stop();
  assert.equal(f.connections[0].closed, true);
  f.hide(false);
  assert.equal(pending.length, 2);
});
