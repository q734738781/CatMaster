import assert from "node:assert/strict";
import test from "node:test";

import {
  buildLargeStructureView,
  displayIndicesToBase,
  LOD_RENDER_TIERS,
} from "./largeStructureView.js";

function structureWith(count) {
  return {
    lattice: { matrix: [[10, 0, 0], [0, 10, 0], [0, 0, 10]] },
    properties: {
      bonds: [
        { site_idx_1: 0, site_idx_2: 1, order: 1 },
        { site_idx_1: count - 2, site_idx_2: count - 1, order: 1 },
      ],
    },
    sites: Array.from({ length: count }, (_, index) => ({
      species: [{ element: index === count - 1 ? "Pt" : "H", occu: 1 }],
      xyz: [index, 0, 0],
      abc: [index / count, 0, 0],
      properties: {},
    })),
  };
}

test("large structure view is bounded, deterministic, species-aware, and maps selection", () => {
  const structure = structureWith(50_000);
  const first = buildLargeStructureView(structure, structure.properties.bonds, [49_999], 4000);
  const second = buildLargeStructureView(structure, structure.properties.bonds, [49_999], 4000);

  assert.equal(first.isLod, true);
  assert.equal(first.visible, 4000);
  assert.equal(first.total, 50_000);
  assert.deepEqual(first.displayToBase, second.displayToBase);
  assert.ok(first.displayToBase.includes(49_999));
  assert.deepEqual(
    displayIndicesToBase(first.selection, first.displayToBase),
    [49_999],
  );
  assert.equal(first.structure.sites.length, 4000);
  assert.equal(structure.sites.length, 50_000);
});

test("small structure preserves the authoritative object and indices", () => {
  const structure = structureWith(1000);
  const projected = buildLargeStructureView(structure, [], [3, 9], 4000);
  assert.equal(projected.isLod, false);
  assert.equal(projected.structure, structure);
  assert.deepEqual(projected.selection, [3, 9]);
  assert.deepEqual(displayIndicesToBase([3, 9], null), [3, 9]);
});

test("default LOD tiers preserve selected neighborhoods and spatial density", () => {
  const structure = structureWith(50_000);
  const selected = 25_000;
  const projected = buildLargeStructureView(structure, [], [selected]);

  assert.equal(projected.visible, 5_000);
  assert.ok(projected.displayToBase.includes(selected));
  for (let offset = -12; offset <= 12; offset += 1) {
    assert.ok(
      projected.displayToBase.includes(selected + offset),
      `selected-site neighbor ${selected + offset} should remain visible`,
    );
  }
  assert.ok(projected.displayToBase.includes(49_999), "dilute Pt species remains visible");
  assert.deepEqual(
    LOD_RENDER_TIERS.map(({ budget }) => budget),
    [1_000, 2_500, 5_000, 6_000],
  );
});

test("browser-model projection benchmark is measured at 1k, 10k, and 50k sites", (context) => {
  const results = [];
  for (const count of [1_000, 10_000, 50_000]) {
    const structure = structureWith(count);
    const heapBefore = process.memoryUsage().heapUsed;
    const started = performance.now();
    const projected = buildLargeStructureView(structure, structure.properties.bonds, [Math.floor(count / 2)]);
    const elapsedMs = performance.now() - started;
    const heapDelta = Math.max(0, process.memoryUsage().heapUsed - heapBefore);
    results.push({ count, visible: projected.visible, elapsedMs, heapDelta });

    assert.ok(elapsedMs < 2_000, `${count.toLocaleString()}-site projection took ${elapsedMs.toFixed(1)} ms`);
    assert.ok(heapDelta < 192 * 1024 * 1024, `${count.toLocaleString()}-site projection retained too much heap`);
    assert.ok(projected.visible <= 5_000 || count === 1_000);
    assert.ok(projected.structure.sites.every(
      (site, displayIndex) => site === structure.sites[projected.displayToBase?.[displayIndex] ?? displayIndex],
    ), "LOD must reference, not clone, site records");
  }
  context.diagnostic(`measured V8 browser-model benchmark: ${JSON.stringify(
    results.map(({ count, visible, elapsedMs, heapDelta }) => ({
      sites: count,
      visible,
      projection_ms: Number(elapsedMs.toFixed(2)),
      retained_heap_mib: Number((heapDelta / 1024 / 1024).toFixed(2)),
    })),
  )}`);
});
