function normalizeTodoRows(value) {
  const candidates = Array.isArray(value) ? value : [];
  return candidates
    .map((item) => {
      if (item && typeof item === "object") {
        const content = String(item.label || "").trim();
        const status = String(item.status || "pending").trim() || "pending";
        return content ? { content, status } : null;
      }
      const content = String(item || "").trim();
      return content ? { content, status: "pending" } : null;
    })
    .filter(Boolean);
}

export function todoGroupsFromMessages(messages) {
  const rows = Array.isArray(messages) ? messages : [];
  const latestUserIndex = rows.reduce((latest, message, index) => (
    message?.role === "user" ? index : latest
  ), -1);
  const scopedMessages = latestUserIndex >= 0 ? rows.slice(latestUserIndex + 1) : rows;
  const groups = new Map();
  for (const message of scopedMessages) {
    const parts = Array.isArray(message?.parts) ? message.parts : [];
    parts.forEach((part, index) => {
      if (part?.type !== "progress" || !Array.isArray(part.items) || !part.items.length) return;
      const todoRows = normalizeTodoRows(part.items);
      if (!todoRows.length) return;
      const source = String(part.title || "CatMaster").replace(/\s+plan$/i, "") || "CatMaster";
      groups.set(source, {
        source,
        rows: todoRows,
        status: String(part.status || "running"),
        toolCallId: String(part.id || ""),
        updatedAt: Number(message.updated_at || message.created_at || 0) || index,
      });
    });
  }
  return [...groups.values()].sort((left, right) => {
    const timeDelta = Number(right.updatedAt || 0) - Number(left.updatedAt || 0);
    if (timeDelta) return timeDelta;
    return String(left.source).localeCompare(String(right.source));
  });
}

export function todoGroupsFromParts(parts) {
  return (Array.isArray(parts) ? parts : [])
    .filter((part) => part?.type === "progress" && Array.isArray(part.items) && part.items.length)
    .map((part, index) => ({
      source: String(part.title || "CatMaster").replace(/\s+plan$/i, "") || "CatMaster",
      rows: normalizeTodoRows(part.items),
      status: String(part.status || "running"),
      toolCallId: String(part.id || ""),
      updatedAt: -index,
    }));
}

export function todoSummary(groups) {
  const rows = (Array.isArray(groups) ? groups : []).flatMap((group) => group.rows || []);
  return {
    total: rows.length,
    done: rows.filter((row) => ["done", "completed", "complete"].includes(String(row.status || "").toLowerCase())).length,
    active: rows.filter((row) => ["in_progress", "running", "active"].includes(String(row.status || "").toLowerCase())).length,
  };
}
