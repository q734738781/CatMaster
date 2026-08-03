const ACTIVE_STATUSES = new Set(["created", "streaming", "running", "queued", "pending", "started"]);
const FAILED_STATUSES = new Set(["failed", "error", "incomplete"]);
const TERMINAL_STATUSES = new Set(["completed", "complete", "done", "success", "resolved"]);
const ACTIVITY_TYPES = new Set(["reasoning", "progress", "tool"]);

export const LONG_ACTIVITY_THRESHOLD = 3;
export const LONG_REASONING_TEXT_THRESHOLD = 800;

export function isLongActivityGroup(group) {
  const parts = Array.isArray(group?.parts) ? group.parts : [];
  if (parts.length > LONG_ACTIVITY_THRESHOLD) return true;
  return parts.some((part) => (
    String(part?.type || "") === "reasoning"
    && (
      Boolean(part?.truncation?.truncated)
      || String(part?.text || "").trim().length > LONG_REASONING_TEXT_THRESHOLD
    )
  ));
}

export function isTodoPart(part) {
  return part?.type === "progress" && Array.isArray(part.items) && part.items.length > 0;
}

function planSource(part) {
  const title = String(part?.activity_group_title || part?.title || "Research plan").trim();
  return title.replace(/\s+plan$/i, "") || "CatMaster";
}

export function latestTodoParts(parts) {
  const latest = new Map();
  (Array.isArray(parts) ? parts : []).forEach((part, index) => {
    if (!isTodoPart(part)) return;
    const source = planSource(part);
    latest.set(source.toLowerCase(), { part, index, source });
  });
  return [...latest.values()]
    .sort((left, right) => left.index - right.index)
    .map(({ part, source }) => ({ ...part, plan_source: source }));
}

export function withCanonicalTodoParts(parts, canonicalTodoParts) {
  const content = (Array.isArray(parts) ? parts : []).filter((part) => !isTodoPart(part));
  const canonical = (Array.isArray(canonicalTodoParts) ? canonicalTodoParts : []).filter(isTodoPart);
  return [...content, ...canonical];
}

function groupIdentity(part) {
  const explicitId = String(part?.activity_group_id || "").trim();
  const explicitTitle = String(part?.activity_group_title || "").trim();
  if (explicitId) {
    return { id: explicitId, title: explicitTitle || "Specialist" };
  }
  if (explicitTitle) {
    return { id: `legacy:${explicitTitle.toLowerCase()}`, title: explicitTitle };
  }
  if (part?.type === "progress") {
    const title = String(part.title || "").trim();
    if (title && !["progress", "update", "execution update"].includes(title.toLowerCase())) {
      return { id: `legacy:${title.toLowerCase()}`, title };
    }
  }
  return { id: "activity:catmaster", title: "CatMaster" };
}

function groupState(parts) {
  const rows = Array.isArray(parts) ? parts : [];
  const activePart = [...rows].reverse().find((part) => ACTIVE_STATUSES.has(String(part?.status || "").toLowerCase()));
  const latestPart = activePart || rows.at(-1) || {};
  const statuses = rows.map((part) => String(part?.status || "").toLowerCase());
  const status = statuses.some((value) => FAILED_STATUSES.has(value))
    ? "failed"
    : statuses.some((value) => ACTIVE_STATUSES.has(value))
      ? "running"
      : statuses.length && statuses.every((value) => TERMINAL_STATUSES.has(value) || !value)
        ? "completed"
        : String(latestPart.status || "updated");
  return { activePart: latestPart, status };
}

export function organizeTurnParts(parts) {
  const rows = Array.isArray(parts) ? parts : [];
  const planParts = latestTodoParts(rows);
  const contentParts = [];
  const groups = new Map();

  rows.forEach((part, index) => {
    if (isTodoPart(part)) return;
    if (!ACTIVITY_TYPES.has(String(part?.type || ""))) {
      contentParts.push(part);
      return;
    }
    const identity = groupIdentity(part);
    const existing = groups.get(identity.id);
    if (existing) {
      existing.parts.push(part);
      return;
    }
    groups.set(identity.id, {
      id: identity.id,
      title: identity.title,
      firstIndex: index,
      parts: [part],
    });
  });

  const activityGroups = [...groups.values()]
    .sort((left, right) => left.firstIndex - right.firstIndex)
    .map((group) => ({ ...group, ...groupState(group.parts) }));
  return { planParts, contentParts, activityGroups };
}
