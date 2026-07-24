import assert from "node:assert/strict";
import test from "node:test";

import {
  compactNodeLabel,
  layoutResearchTechTree,
} from "./researchTechTree.js";

test("research tech tree keeps shared evidence as one node with several impact edges", () => {
  const graph = {
    nodes: [
      { id: "hypothesis:h1", kind: "hypothesis", label: "H1", status: "open" },
      { id: "hypothesis:h2", kind: "hypothesis", label: "H2", status: "open" },
      { id: "action:a1", kind: "action", label: "Search", status: "eligible" },
      { id: "evidence:a1", kind: "evidence", label: "Shared result", status: "judged" },
    ],
    edges: [
      { id: "e1", source: "evidence:a1", target: "hypothesis:h1", kind: "supports" },
      { id: "e2", source: "evidence:a1", target: "hypothesis:h2", kind: "opposes" },
    ],
  };

  const layout = layoutResearchTechTree(graph);

  assert.equal(layout.nodes.filter((node) => node.kind === "evidence").length, 1);
  assert.equal(layout.edges.length, 2);
  assert.ok(layout.edges.every((edge) => edge.path.startsWith("M ")));
  assert.deepEqual(layout.columns.map((column) => column.kind), ["hypothesis", "action", "evidence"]);
});

test("research tech tree drops dangling view edges and compacts long labels", () => {
  const layout = layoutResearchTechTree({
    nodes: [{ id: "hypothesis:h1", kind: "hypothesis", label: "H1", status: "open" }],
    edges: [{ id: "missing", source: "missing", target: "hypothesis:h1", kind: "targets" }],
  });

  assert.equal(layout.edges.length, 0);
  assert.equal(compactNodeLabel("1234567890", 6), "12345…");
});

test("research map preserves high-cost actions as ordinary eligible nodes", () => {
  const layout = layoutResearchTechTree({
    nodes: [
      {
        id: "action:expensive",
        kind: "action",
        label: "Decisive calculation",
        status: "eligible",
        cost: "high",
      },
    ],
    edges: [],
  });

  assert.equal(layout.nodes[0].status, "eligible");
  assert.equal(layout.nodes[0].cost, "high");
});
