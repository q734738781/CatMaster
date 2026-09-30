const ACTIVE_STATUSES = new Set(["created", "streaming", "running", "queued", "pending", "started"]);
const FAILED_STATUSES = new Set(["failed", "error", "incomplete"]);
const TERMINAL_STATUSES = new Set(["completed", "complete", "done", "success", "resolved"]);
const ACTIVITY_TYPES = new Set(["reasoning", "progress", "tool"]);

export function asyncFollowupLabel(status) {
  return {
    pending: "补充指令 · 等待处理",
    running: "补充指令 · 正在处理",
    success: "补充指令 · 该轮已完成",
    error: "补充指令 · 该轮失败",
    interrupted: "补充指令 · 已中断",
  }[status] || "补充指令";
}

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

export function isSemanticProgressPart(part) {
  return part?.type === "progress" && !isTodoPart(part);
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

export function todoCardProjection(messages, currentTodoParts = []) {
  const rows = Array.isArray(messages) ? messages : [];
  const ownerMessageIds = new Set();
  const partsByMessageId = new Map();
  let assistantBlock = [];

  const finishBlock = () => {
    if (!assistantBlock.length) return;
    const owner = assistantBlock.at(-1);
    const ownerId = String(owner?.id || "").trim();
    if (ownerId) {
      ownerMessageIds.add(ownerId);
      partsByMessageId.set(
        ownerId,
        latestTodoParts(assistantBlock.flatMap((message) => (
          Array.isArray(message?.parts) ? message.parts : []
        ))),
      );
    }
    assistantBlock = [];
  };

  rows.forEach((message) => {
    const role = String(message?.role || "").toLowerCase();
    if (role === "user") {
      finishBlock();
      return;
    }
    if (role === "assistant") assistantBlock.push(message);
  });
  finishBlock();

  const latestUserIndex = rows.reduce((latest, message, index) => (
    String(message?.role || "").toLowerCase() === "user" ? index : latest
  ), -1);
  const currentOwnerId = [...rows.slice(latestUserIndex + 1)]
    .reverse()
    .find((message) => String(message?.role || "").toLowerCase() === "assistant")?.id || "";
  if (currentOwnerId) {
    ownerMessageIds.add(currentOwnerId);
    // The server computes this projection from the complete persisted current
    // turn, including Todo calls outside the inline message page. An empty
    // terminal projection is authoritative and deliberately removes stale
    // child-agent scratch plans.
    partsByMessageId.set(currentOwnerId, latestTodoParts(currentTodoParts));
  }

  return { currentOwnerId, ownerMessageIds, partsByMessageId };
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
  const semanticProgressParts = [];
  const contentParts = [];
  const groups = new Map();

  rows.forEach((part, index) => {
    if (isTodoPart(part)) return;
    if (isSemanticProgressPart(part)) {
      semanticProgressParts.push(part);
      return;
    }
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
  return { planParts, semanticProgressParts, contentParts, activityGroups };
}

export function hasVisibleTurnPresentation(presentation) {
  return Boolean(
    presentation?.planParts?.length
    || presentation?.semanticProgressParts?.length
    || presentation?.contentParts?.length
    || presentation?.activityGroups?.length,
  );
}
