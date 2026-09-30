const ACTIVE_STATUSES = new Set(["created", "streaming", "running", "queued", "pending", "started"]);

export function isActiveOperation(part) {
  const type = String(part?.type || "");
  const displayableType = type === "tool"
    || type === "subagent"
    || (type === "progress" && !(Array.isArray(part?.items) && part.items.length));
  return displayableType
    && ACTIVE_STATUSES.has(String(part?.status || "").toLowerCase());
}

export function sortActiveOperations(parts) {
  return [...(Array.isArray(parts) ? parts : [])]
    .filter(isActiveOperation)
    .sort((left, right) => {
      const leftStart = Number(left?.started_at || 0);
      const rightStart = Number(right?.started_at || 0);
      if (leftStart > 0 && rightStart > 0 && leftStart !== rightStart) return leftStart - rightStart;
      if (leftStart > 0 && rightStart <= 0) return -1;
      if (leftStart <= 0 && rightStart > 0) return 1;
      return String(left?.id || "").localeCompare(String(right?.id || ""));
    });
}

export function updateActiveOperations(current, part) {
  const partId = String(part?.id || "").trim();
  if (!partId || !["tool", "progress", "subagent"].includes(String(part?.type || ""))) {
    return sortActiveOperations(current);
  }
  const remaining = (Array.isArray(current) ? current : []).filter((item) => item?.id !== partId);
  return sortActiveOperations(isActiveOperation(part) ? [...remaining, part] : remaining);
}

export function retainActiveAsyncSubagents(parts) {
  return sortActiveOperations((Array.isArray(parts) ? parts : []).filter((part) => (
    part?.type === "subagent" || (part?.actions || []).some((action) => action?.id === "update_async_subagent")
  )));
}

export function formatElapsedSeconds(value) {
  const total = Math.max(0, Math.floor(Number(value) || 0));
  const hours = Math.floor(total / 3600);
  const minutes = Math.floor((total % 3600) / 60);
  const seconds = total % 60;
  if (hours) return `${hours}h ${String(minutes).padStart(2, "0")}m ${String(seconds).padStart(2, "0")}s`;
  if (minutes) return `${minutes}m ${String(seconds).padStart(2, "0")}s`;
  return `${seconds}s`;
}
