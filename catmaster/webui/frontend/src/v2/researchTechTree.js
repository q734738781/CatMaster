const RELATION_LABELS = {
  tests: "tests",
  produces: "produces",
  supports: "supports",
  opposes: "opposes",
  inconclusive: "inconclusive for",
  suggests: "suggests",
  depends_on: "depends on",
};

const EVIDENCE_LABELS = {
  conflicting_evidence: "Both supporting and opposing results recorded",
  supporting_evidence: "Supporting result recorded",
  opposing_evidence: "Opposing result recorded",
  not_distinguished: "Recorded result does not distinguish",
  no_results: "No linked result yet",
};

const EXPERIMENT_STATE_LABELS = {
  draft: "Draft proposal",
  ready: "Ready to run",
  running: "Running",
  has_results: "Results recorded",
  blocked: "Blocked",
};

const EXECUTION_LANE_LABELS = {
  experiment: "Experiment",
  research: "Research",
  literature_review: "Literature review",
};

const ORCHESTRATION_MODE_LABELS = {
  manual: "Manual",
  auto: "Automatic",
};

export function relationLabel(relation) {
  return RELATION_LABELS[String(relation || "")] || "related";
}

export function evidenceStateLabel(state) {
  return EVIDENCE_LABELS[String(state || "")] || "No result relation summary";
}

export function experimentStateLabel(state) {
  return EXPERIMENT_STATE_LABELS[String(state || "")] || "Unknown state";
}

export function executionLaneLabel(lane) {
  return EXECUTION_LANE_LABELS[String(lane || "")] || "Unassigned";
}

export function orchestrationModeLabel(mode) {
  return ORCHESTRATION_MODE_LABELS[String(mode || "")] || "Manual";
}

export function boundedResearchGraph(payload, {
  limit = 25,
  focusNodeId = "",
  hops = 2,
  query = "",
} = {}) {
  const sourceNodes = Array.isArray(payload?.nodes) ? payload.nodes : [];
  const sourceEdges = Array.isArray(payload?.edges) ? payload.edges : [];
  const cappedLimit = Math.max(1, Math.min(100, Number(limit) || 25));
  const adjacency = new Map();
  for (const edge of sourceEdges) {
    const source = String(edge?.source_node_id || "");
    const target = String(edge?.target_node_id || "");
    if (!source || !target) continue;
    if (!adjacency.has(source)) adjacency.set(source, []);
    if (!adjacency.has(target)) adjacency.set(target, []);
    adjacency.get(source).push(target);
    adjacency.get(target).push(source);
  }

  let candidates = sourceNodes;
  if (focusNodeId) {
    const distance = new Map([[focusNodeId, 0]]);
    const queue = [focusNodeId];
    while (queue.length) {
      const current = queue.shift();
      const depth = distance.get(current);
      if (depth >= Math.max(1, Math.min(2, Number(hops) || 2))) continue;
      for (const neighbor of adjacency.get(current) || []) {
        if (distance.has(neighbor)) continue;
        distance.set(neighbor, depth + 1);
        queue.push(neighbor);
      }
    }
    candidates = sourceNodes.filter((node) => distance.has(String(node.node_id || "")));
    candidates.sort((left, right) => {
      const leftDistance = distance.get(String(left.node_id || "")) ?? 99;
      const rightDistance = distance.get(String(right.node_id || "")) ?? 99;
      return leftDistance - rightDistance || String(left.node_id).localeCompare(String(right.node_id));
    });
  }
  const normalizedQuery = String(query || "").trim().toLocaleLowerCase();
  if (normalizedQuery) {
    const searchableText = (value) => {
      if (Array.isArray(value)) return value.map(searchableText).join(" ");
      if (value && typeof value === "object") {
        return Object.values(value).map(searchableText).join(" ");
      }
      return typeof value === "string" || typeof value === "number"
        ? String(value)
        : "";
    };
    candidates = candidates.filter((node) => (
      searchableText({
        kind: node.kind,
        title: node.title,
        state: node.state,
        evidence_state: node.evidence_state,
        body: node.body,
      }).toLocaleLowerCase().includes(normalizedQuery)
    ));
  }
  const nodes = candidates.slice(0, cappedLimit);
  const ids = new Set(nodes.map((node) => String(node.node_id || "")));
  const edges = sourceEdges.filter((edge) => (
    ids.has(String(edge.source_node_id || ""))
    && ids.has(String(edge.target_node_id || ""))
  ));
  return {
    nodes,
    edges,
    totalCount: sourceNodes.length,
    matchingCount: candidates.length,
    omittedCount: Math.max(
      0,
      (normalizedQuery ? candidates.length : sourceNodes.length) - nodes.length,
    ),
  };
}
