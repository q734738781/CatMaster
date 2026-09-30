export function normalizeMathMarkdown(text) {
  const source = String(text || "");
  if (!source) return "";
  return source
    .replace(/\\\[((?:.|\n)*?)\\\]/g, (_match, expr) => `\n$$\n${String(expr || "").trim()}\n$$\n`)
    .replace(/\\\(((?:.|\n)*?)\\\)/g, (_match, expr) => `$${String(expr || "").trim()}$`);
}
