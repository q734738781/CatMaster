import assert from "node:assert/strict";
import test from "node:test";

import {
  boundedResearchGraph,
  evidenceStateLabel,
  executionLaneLabel,
  experimentStateLabel,
  orchestrationModeLabel,
  relationLabel,
} from "./researchTechTree.js";

const graph = {
  nodes: [
    { node_id: "h1", kind: "hypothesis", title: "Mechanism A" },
    { node_id: "e1", kind: "experiment", title: "Discriminating run" },
    { node_id: "r1", kind: "result", title: "Supporting result" },
    { node_id: "h2", kind: "hypothesis", title: "Mechanism B" },
    { node_id: "e2", kind: "experiment", title: "Follow-up" },
    { node_id: "isolated", kind: "hypothesis", title: "Unrelated branch" },
  ],
  edges: [
    { source_node_id: "h1", target_node_id: "e1", relation: "tests" },
    { source_node_id: "e1", target_node_id: "r1", relation: "produces" },
    { source_node_id: "r1", target_node_id: "h1", relation: "supports" },
    { source_node_id: "r1", target_node_id: "h2", relation: "suggests" },
    { source_node_id: "h2", target_node_id: "e2", relation: "tests" },
  ],
};

test("bounded research graph preserves typed scientific cycles and shared result nodes", () => {
  const view = boundedResearchGraph(graph, { limit: 100 });

  assert.equal(view.nodes.filter((node) => node.kind === "result").length, 1);
  assert.equal(view.edges.length, 5);
  assert.equal(view.totalCount, 6);
  assert.equal(view.omittedCount, 0);
  assert.deepEqual(
    view.edges.map((edge) => edge.relation),
    ["tests", "produces", "supports", "suggests", "tests"],
  );
});

test("focus mode is a deterministic two-hop neighborhood with an explicit omission count", () => {
  const view = boundedResearchGraph(graph, {
    limit: 25,
    focusNodeId: "r1",
    hops: 2,
  });

  assert.deepEqual(
    view.nodes.map((node) => node.node_id),
    ["r1", "e1", "h1", "h2", "e2"],
  );
  assert.equal(view.edges.length, 5);
  assert.equal(view.omittedCount, 1);
});

test("density caps visible nodes and drops dangling edges without altering source data", () => {
  const view = boundedResearchGraph(graph, { limit: 5 });

  assert.equal(view.nodes.length, 5);
  assert.equal(view.omittedCount, 1);
  assert.ok(
    view.edges.every((edge) => (
      view.nodes.some((node) => node.node_id === edge.source_node_id)
      && view.nodes.some((node) => node.node_id === edge.target_node_id)
    )),
  );
  assert.equal(graph.nodes.length, 6);
});

test("search can discover a node outside the density cap without exposing raw graph data", () => {
  const manyNodes = {
    nodes: Array.from({ length: 130 }, (_, index) => ({
      node_id: `h${index + 1}`,
      kind: "hypothesis",
      title: index === 119 ? "Rare operando reconstruction mechanism" : `Branch ${index + 1}`,
      body: { claim: `Claim ${index + 1}` },
    })),
    edges: [],
  };
  const defaultView = boundedResearchGraph(manyNodes, { limit: 100 });
  const searchView = boundedResearchGraph(manyNodes, {
    limit: 100,
    query: "operando reconstruction",
  });

  assert.equal(defaultView.nodes.some((node) => node.node_id === "h120"), false);
  assert.deepEqual(searchView.nodes.map((node) => node.node_id), ["h120"]);
  assert.equal(searchView.matchingCount, 1);
  assert.equal(searchView.totalCount, 130);
  assert.equal(searchView.omittedCount, 0);
});

test("relations and evidence states have readable non-colour labels", () => {
  assert.equal(relationLabel("depends_on"), "depends on");
  assert.equal(relationLabel("unknown"), "related");
  assert.equal(
    evidenceStateLabel("conflicting_evidence"),
    "Conflicting supporting and opposing evidence",
  );
  assert.equal(experimentStateLabel("has_results"), "Results recorded");
  assert.equal(executionLaneLabel("literature_review"), "Literature review");
  assert.equal(orchestrationModeLabel("auto"), "Automatic");
});
