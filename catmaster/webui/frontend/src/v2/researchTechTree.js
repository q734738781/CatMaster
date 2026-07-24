const KIND_ORDER = ["hypothesis", "action", "evidence"];
const COLUMN_LABELS = {
  hypothesis: "Hypotheses",
  action: "Scientific checks",
  evidence: "Evidence judgments",
};
export function layoutResearchTechTree(graph) {
  const sourceNodes = Array.isArray(graph?.nodes) ? graph.nodes : [];
  const sourceEdges = Array.isArray(graph?.edges) ? graph.edges : [];
  const grouped = Object.fromEntries(KIND_ORDER.map((kind) => [kind, []]));
  for (const node of sourceNodes) {
    if (grouped[node?.kind]) grouped[node.kind].push(node);
  }
  for (const kind of KIND_ORDER) {
    grouped[kind].sort((left, right) => String(left.id).localeCompare(String(right.id)));
  }

  const nodeWidth = 238;
  const nodeHeight = 84;
  const columnGap = 90;
  const rowGap = 34;
  const left = 52;
  const top = 76;
  const maxRows = Math.max(1, ...KIND_ORDER.map((kind) => grouped[kind].length));
  const width = left * 2 + KIND_ORDER.length * nodeWidth + (KIND_ORDER.length - 1) * columnGap;
  const height = top + maxRows * (nodeHeight + rowGap) + 42;
  const nodes = [];

  KIND_ORDER.forEach((kind, columnIndex) => {
    const x = left + columnIndex * (nodeWidth + columnGap);
    const rows = grouped[kind];
    const occupied = rows.length * nodeHeight + Math.max(0, rows.length - 1) * rowGap;
    const available = maxRows * (nodeHeight + rowGap) - rowGap;
    const offset = Math.max(0, (available - occupied) / 2);
    rows.forEach((node, rowIndex) => {
      nodes.push({
        ...node,
        x,
        y: top + offset + rowIndex * (nodeHeight + rowGap),
        width: nodeWidth,
        height: nodeHeight,
      });
    });
  });

  const byId = new Map(nodes.map((node) => [node.id, node]));
  const edges = sourceEdges.flatMap((edge) => {
    const source = byId.get(edge?.source);
    const target = byId.get(edge?.target);
    if (!source || !target) return [];
    const sourceCenter = {
      x: source.x + source.width / 2,
      y: source.y + source.height / 2,
    };
    const targetCenter = {
      x: target.x + target.width / 2,
      y: target.y + target.height / 2,
    };
    let path;
    if (source.kind === target.kind) {
      const bend = Math.max(source.x + source.width, target.x + target.width) + 44;
      path = `M ${sourceCenter.x} ${sourceCenter.y} C ${bend} ${sourceCenter.y}, ${bend} ${targetCenter.y}, ${targetCenter.x} ${targetCenter.y}`;
    } else {
      const middle = (sourceCenter.x + targetCenter.x) / 2;
      path = `M ${sourceCenter.x} ${sourceCenter.y} C ${middle} ${sourceCenter.y}, ${middle} ${targetCenter.y}, ${targetCenter.x} ${targetCenter.y}`;
    }
    return [{ ...edge, path }];
  });

  return {
    width,
    height,
    nodes,
    edges,
    columns: KIND_ORDER.map((kind, columnIndex) => ({
      kind,
      label: COLUMN_LABELS[kind],
      x: left + columnIndex * (nodeWidth + columnGap),
      width: nodeWidth,
    })),
  };
}

export function compactNodeLabel(label, limit = 92) {
  const text = String(label || "").trim();
  if (text.length <= limit) return text;
  return `${text.slice(0, Math.max(0, limit - 1)).trimEnd()}…`;
}
