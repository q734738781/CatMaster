import path from "node:path";

export function normalizeTitle(title = "") {
  return String(title).replace(/\s+/g, " ").trim().toLowerCase();
}

export function exactTitleMatch(candidate, expected) {
  return normalizeTitle(candidate) === normalizeTitle(expected);
}

export function normalizeArxivId(value = "") {
  const text = String(value).trim();
  const match = text.match(/(?:arxiv\.org\/(?:abs|pdf)\/)?([a-z-]+\/\d{7}|\d{4}\.\d{4,5})(v\d+)?(?:\.pdf)?/i);
  if (!match) return "";
  return `${match[1]}${match[2] || ""}`;
}

export function arxivPdfUrl(id) {
  const normalized = normalizeArxivId(id);
  if (!normalized) return "";
  return `https://arxiv.org/pdf/${normalized}`;
}

function decodeXml(text = "") {
  return text
    .replace(/&lt;/g, "<")
    .replace(/&gt;/g, ">")
    .replace(/&amp;/g, "&")
    .replace(/&quot;/g, '"')
    .replace(/&#39;/g, "'");
}

export function parseArxivAtom(xml, expectedTitle) {
  const entries = String(xml).match(/<entry>[\s\S]*?<\/entry>/g) || [];
  for (const entry of entries) {
    const title = decodeXml((entry.match(/<title>([\s\S]*?)<\/title>/) || [])[1] || "").replace(/\s+/g, " ").trim();
    if (!exactTitleMatch(title, expectedTitle)) continue;
    const rawId = decodeXml((entry.match(/<id>([\s\S]*?)<\/id>/) || [])[1] || "");
    const id = normalizeArxivId(rawId);
    if (!id) continue;
    return { id, title, pdfUrl: arxivPdfUrl(id) };
  }
  return null;
}

export async function findArxivByTitle(title, { fetchImpl = fetch } = {}) {
  const url = new URL("https://export.arxiv.org/api/query");
  url.searchParams.set("search_query", `ti:"${title}"`);
  url.searchParams.set("start", "0");
  url.searchParams.set("max_results", "5");
  const response = await fetchImpl(url, { signal: AbortSignal.timeout(30000) });
  if (!response.ok) {
    throw new Error(`arXiv lookup failed: HTTP ${response.status}`);
  }
  return parseArxivAtom(await response.text(), title);
}

export function filenameForPdfUrl(url, title = "") {
  if (title) {
    const safeTitle = title
      .trim()
      .replace(/[\/:*?"<>|]+/g, "")
      .replace(/\s+/g, "_")
      .slice(0, 120);
    if (safeTitle) return `${safeTitle}.pdf`;
  }
  const base = path.basename(new URL(url).pathname) || "paper.pdf";
  return base.endsWith(".pdf") ? base : `${base}.pdf`;
}
