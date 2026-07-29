import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

const read = (name) => readFile(new URL(name, import.meta.url), "utf8");

test("read-only MatterViz hides mutation controls and bridge drops mutation callbacks", async () => {
  const [host, bridge] = await Promise.all([
    read("./MatterVizHost.svelte"),
    read("./MatterVizBridge.jsx"),
  ]);
  assert.match(host, /hidden:\s*\["measure-mode",\s*"controls"\]/);
  assert.match(host, /if\s*\(!read_only\)\s*bonds\s*=\s*next_bonds/);
  assert.match(host, /if\s*\(current\s*&&\s*!read_only\)/);
  assert.match(bridge, /!readOnlyRef\.current\s*&&\s*!projectionRef\.current\.isLod/);
  assert.match(bridge, /selection_gesture:\s*readOnly\s*\?\s*""/);
});

test("interstitial picker uses a unit-cell ray segment and exact keyboard coordinates", async () => {
  const host = await read("./MatterVizHost.svelte");
  assert.match(host, /ray_cell_segment/);
  assert.match(host, /Sightline depth/);
  assert.match(host, /Fractional interstitial coordinates/);
  assert.match(host, /Use centre sightline/);
  assert.doesNotMatch(host, /structure plane|plane_origin|centroid/i);
});

test("canvas, sliders, spinbuttons, and pointer overlays have keyboard-readable contracts", async () => {
  const host = await read("./MatterVizHost.svelte");
  assert.match(host, /input\[type="range"\].*input\[type="number"\]/s);
  assert.match(host, /Interactive three-dimensional structure canvas/);
  assert.match(host, /canvas\.getAttribute\("role"\)\s*!==\s*"img"/);
  assert.match(host, /aria-label="Interstitial depth along the selected unit-cell sightline"/);
  assert.match(host, /Keyboard: arrows move the rectangle/);
  assert.match(host, /onkeydown=\{handle_overlay_keydown\}/);
});

test("mobile workbench keeps the scientific summary in the rendered center pane", async () => {
  const [workbench, styles] = await Promise.all([
    read("./StructureWorkbench.jsx"),
    read("../../styles.css"),
  ]);
  assert.match(workbench, /narrow\s*\?\s*\(\s*<ScientificStructureSummary/s);
  for (const field of ["Formula", "Sites", "PBC", "Cell"]) {
    assert.match(workbench, new RegExp(`<dt>${field}</dt>`));
  }
  assert.match(styles, /\.v2-workbench-summary\.compact/);
  assert.match(styles, /@media \(max-width: 1100px\)[\s\S]*minmax\(0, 1fr\)/);
});
