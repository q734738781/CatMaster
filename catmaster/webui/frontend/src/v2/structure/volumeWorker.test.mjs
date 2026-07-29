import assert from "node:assert/strict";
import { readFile } from "node:fs/promises";
import test from "node:test";

import {
  MAX_INTERACTIVE_GRID_POINTS,
  parseVolumeBuffer,
  readResponseBuffer,
} from "./volumeWorker.js";

const encoder = new TextEncoder();
const asBuffer = (text) => encoder.encode(text).buffer;

const cube = [
  "CUBE fixture",
  "bounded parser",
  "1 0 0 0",
  "-2 0.5 0 0",
  "-2 0 0.5 0",
  "-2 0 0 0.5",
  "6 0 0 0 0",
  "-1 0 1 2 3 4 5 6",
].join("\n");

const chgcar = [
  "CHGCAR fixture",
  "1.0",
  "1 0 0",
  "0 1 0",
  "0 0 1",
  "H",
  "1",
  "Direct",
  "0 0 0",
  "",
  "2 2 2",
  "0 1 2 3 4 5 6 7",
].join("\n");

const xsf = [
  "CRYSTAL",
  "PRIMVEC",
  "1 0 0",
  "0 1 0",
  "0 0 1",
  "PRIMCOORD",
  "1 1",
  "6 0.25 0.25 0.25",
  "BEGIN_BLOCK_DATAGRID_3D",
  "density",
  "BEGIN_DATAGRID_3D_density",
  "2 2 2",
  "0 0 0",
  "1 0 0",
  "0 1 0",
  "0 0 1",
  "-4 -3 -2 -1 1 2 3 4",
  "END_DATAGRID_3D",
  "END_BLOCK_DATAGRID_3D",
].join("\n");

test("CUBE is parsed directly from ArrayBuffer with structure and bounded grid", () => {
  const parsed = parseVolumeBuffer(asBuffer(cube), "density.cube");
  assert.equal(parsed.structure.sites[0].species[0].element, "C");
  assert.deepEqual(parsed.volumes[0].grid_dims, [2, 2, 2]);
  assert.deepEqual(parsed.volumes[0].source_grid_dims, [2, 2, 2]);
  assert.equal(parsed.volumes[0].grid[0][0][0], -1);
  assert.equal(parsed.volumes[0].grid[1][1][1], 6);
});

test("CHGCAR x-fastest density and normalization are parsed without a full source grid", () => {
  const parsed = parseVolumeBuffer(asBuffer(chgcar), "CHGCAR");
  const volume = parsed.volumes[0];
  assert.equal(volume.source, "CHGCAR");
  assert.equal(volume.grid[0][0][0], 0);
  assert.equal(volume.grid[1][0][0], 1);
  assert.equal(volume.grid[0][0][1], 4);
  assert.deepEqual(parsed.structure.lattice.pbc, [true, true, true]);
});

test("XSF structure and signed scalar field preserve source semantics", () => {
  const parsed = parseVolumeBuffer(asBuffer(xsf), "density.xsf");
  assert.equal(parsed.structure.sites[0].species[0].element, "C");
  assert.deepEqual(parsed.structure.sites[0].abc.map((value) => Number(value.toFixed(3))), [0.25, 0.25, 0.25]);
  assert.equal(parsed.volumes[0].label, "density");
  assert.equal(parsed.volumes[0].data_range.min, -4);
  assert.equal(parsed.volumes[0].data_range.max, 4);
});

test("parsed grids expose the complete slice and isosurface contract used by MatterViz", async () => {
  const volume = parseVolumeBuffer(asBuffer(xsf), "density.xsf").volumes[0];
  assert.deepEqual(Object.keys(volume).filter((key) => [
    "grid", "grid_dims", "lattice", "origin", "data_range", "periodic",
  ].includes(key)).sort(), [
    "data_range", "grid", "grid_dims", "lattice", "origin", "periodic",
  ]);
  assert.equal(volume.data_range.abs_max, 4);
  assert.ok(volume.grid.every((plane) => plane.every((row) => row instanceof Float32Array)));

  // MatterViz's package entry imports Svelte components and cannot be evaluated
  // by Node's test loader. Verify the compiled host wires both tested data
  // contracts; the Vite production build compiles these imports end-to-end.
  const host = await readFile(new URL("./MatterVizHost.svelte", import.meta.url), "utf8");
  assert.match(host, /sample_hkl_slice\(active,/);
  assert.match(host, /sample_plane_slice\(\s*active,/);
  assert.match(host, /<VolumeSlice/);
  assert.match(host, /bind:isosurface_settings/);
});

test("million-point CUBE is averaged while the interactive grid remains bounded", () => {
  const largeCube = [
    "CUBE large fixture",
    "bounded parser",
    "0 0 0 0",
    "-100 0.01 0 0",
    "-100 0 0.01 0",
    "-100 0 0 0.01",
    "1 ".repeat(1_000_000),
  ].join("\n");
  const parsed = parseVolumeBuffer(asBuffer(largeCube), "large.cube");
  const volume = parsed.volumes[0];
  assert.equal(volume.source_grid_dims.reduce((left, right) => left * right, 1), 1_000_000);
  assert.ok(volume.grid_dims.reduce((left, right) => left * right, 1) <= MAX_INTERACTIVE_GRID_POINTS);
  assert.ok(volume.downsample_factor > 1);
  assert.equal(volume.data_range.min, 1);
  assert.equal(volume.data_range.max, 1);
});

test("stream cancellation aborts the reader and does not return partial data", async () => {
  const controller = new AbortController();
  let cancelled = false;
  const stream = new ReadableStream({
    pull(streamController) {
      streamController.enqueue(new Uint8Array([1, 2, 3, 4]));
    },
    cancel() {
      cancelled = true;
    },
  });
  const response = new Response(stream, {
    headers: { "content-length": "64" },
  });
  await assert.rejects(
    readResponseBuffer(response, {
      signal: controller.signal,
      onProgress() {
        controller.abort();
      },
    }),
    (error) => error?.name === "AbortError",
  );
  assert.equal(cancelled, true);
});
