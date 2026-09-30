import { defaultUrlTransform } from "react-markdown";

function normalizedWorkspaceFilePath(value) {
  const raw = String(value || "").trim().replace(/^\/+/, "");
  const relative = raw.startsWith("files/") ? raw.slice("files/".length) : raw;
  if (!relative) return "";

  try {
    const parts = relative.split("/").map((part) => decodeURIComponent(part));
    if (parts.some((part) => !part || part === "." || part === ".." || part.includes("/") || part.includes("\\"))) {
      return "";
    }
    return `files/${parts.join("/")}`;
  } catch {
    return "";
  }
}

export function workspacePathFromSandboxHref(value) {
  const href = String(value || "").trim();
  if (!href.toLowerCase().startsWith("sandbox:")) return "";

  const raw = href.slice("sandbox:".length).split(/[?#]/, 1)[0].replace(/^\/+/, "");
  return normalizedWorkspaceFilePath(raw);
}

export function workspaceInlineImageUrl(value, threadId) {
  const path = workspacePathFromSandboxHref(value);
  const thread = String(threadId || "").trim();
  if (!path || !thread) return "";
  return `/api/threads/${encodeURIComponent(thread)}/files/image?path=${encodeURIComponent(path)}`;
}

export function workspacePathFromInlineImageUrl(value, threadId) {
  const source = String(value || "").trim();
  const thread = String(threadId || "").trim();
  if (!source.startsWith("/") || !thread) return "";

  try {
    const parsed = new URL(source, "http://catmaster.local");
    const expected = `/api/threads/${encodeURIComponent(thread)}/files/image`;
    if (parsed.origin !== "http://catmaster.local" || parsed.pathname !== expected) return "";
    return normalizedWorkspaceFilePath(parsed.searchParams.get("path"));
  } catch {
    return "";
  }
}

export function workspaceMarkdownUrlTransform(url, key, node, threadId = "") {
  const path = workspacePathFromSandboxHref(url);
  if (key === "href" && node?.tagName === "a" && path) {
    return url;
  }
  if (key === "src" && node?.tagName === "img" && path) {
    return workspaceInlineImageUrl(url, threadId);
  }
  return defaultUrlTransform(url);
}
