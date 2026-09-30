import test from "node:test";
import assert from "node:assert/strict";

import { inferPreviewKind, parseCompleteJson, yamlOutline } from "./structuredPreview.js";

test("structured file suffixes select formatted preview modes", () => {
  assert.equal(inferPreviewKind("text", "settings.yaml"), "yaml");
  assert.equal(inferPreviewKind("text", "settings.YML"), "yaml");
  assert.equal(inferPreviewKind("text", "result.json"), "json");
  assert.equal(inferPreviewKind("csv", "table.csv"), "csv");
});

test("JSON is parsed only when the source slice is complete", () => {
  assert.deepEqual(parseCompleteJson('{"atoms":[{"symbol":"Pd"}]}'), {
    ok: true,
    value: { atoms: [{ symbol: "Pd" }] },
  });
  assert.equal(parseCompleteJson('{"atoms":[', true).ok, false);
  assert.equal(parseCompleteJson("not json").ok, false);
});

test("YAML outline preserves every non-empty source line and nesting", () => {
  const rows = yamlOutline([
    "hypothesis:",
    "  statement: Pd resists sintering",
    "  evidence:",
    "    - microscopy",
    "    - spectroscopy",
    "# reviewer note",
  ].join("\n"));

  assert.equal(rows.length, 2);
  assert.equal(rows[0].label, "hypothesis:");
  assert.equal(rows[0].children[0].label, "statement: Pd resists sintering");
  assert.equal(rows[0].children[1].children.length, 2);
  assert.equal(rows[1].label, "# reviewer note");
  const flatten = (nodes) => nodes.flatMap((node) => [node, ...flatten(node.children)]);
  assert.equal(flatten(rows).length, 6);
});
