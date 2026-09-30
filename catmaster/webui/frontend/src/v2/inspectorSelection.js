export function selectionFromHash(hashValue) {
  const hash = String(hashValue || "").replace(/^#/, "");
  if (!hash) return null;
  const params = new URLSearchParams(hash);
  const type = params.get("inspect");
  if (type === "artifact") {
    const artifactId = params.get("artifact_id") || "";
    return artifactId ? { type: "artifact", artifact_id: artifactId } : null;
  }
  if (type === "file") {
    const path = params.get("path") || "";
    return path ? { type: "file", path } : null;
  }
  return null;
}

export function tabFromHash(hashValue) {
  const hash = String(hashValue || "").replace(/^#/, "");
  const tab = new URLSearchParams(hash).get("tab") || "chat";
  return ["chat", "monitor", "hypotheses", "evolution", "files"].includes(tab) ? tab : "chat";
}

export function selectionToHash(selection, activeTab = "chat") {
  const params = new URLSearchParams();
  if (activeTab && activeTab !== "chat") {
    params.set("tab", activeTab);
  }
  if (selection?.type === "artifact" && selection.artifact_id) {
    params.set("inspect", "artifact");
    params.set("artifact_id", selection.artifact_id);
    params.delete("path");
    return `#${params.toString()}`;
  }
  if (selection?.type === "file" && selection.path) {
    params.set("inspect", "file");
    params.set("path", selection.path);
    params.delete("artifact_id");
    return `#${params.toString()}`;
  }
  const serialized = params.toString();
  return serialized ? `#${serialized}` : "";
}
