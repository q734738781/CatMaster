export function inferPreviewKind(kind, path) {
  const declared = String(kind || "").trim().toLowerCase();
  const sourcePath = String(path || "").trim().toLowerCase();
  if (sourcePath.endsWith(".yaml") || sourcePath.endsWith(".yml")) return "yaml";
  if (sourcePath.endsWith(".json") || sourcePath.endsWith(".jsonl")) return "json";
  return declared;
}

export function parseCompleteJson(text, truncated = false) {
  if (truncated) return { ok: false, value: null };
  try {
    return { ok: true, value: JSON.parse(String(text || "")) };
  } catch {
    return { ok: false, value: null };
  }
}

export function yamlOutline(text) {
  const root = [];
  const stack = [{ indent: -1, children: root }];
  String(text || "").split(/\r?\n/).forEach((source, index) => {
    if (!source.trim()) return;
    const indent = source.match(/^[ \t]*/)?.[0].replace(/\t/g, "  ").length || 0;
    const label = source.trim();
    const node = { id: `line-${index + 1}`, label, line: index + 1, children: [] };
    while (stack.length > 1 && indent <= stack[stack.length - 1].indent) stack.pop();
    stack[stack.length - 1].children.push(node);
    stack.push({ indent, children: node.children });
  });
  return root;
}
