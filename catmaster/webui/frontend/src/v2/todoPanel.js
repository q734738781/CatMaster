function sourceFromMeta(meta = {}) {
  return String(meta.subagent_source || meta.agent_name || meta.source || "").trim() || "CatMaster";
}

function normalizeTodoRows(value) {
  const input = value && typeof value === "object" && !Array.isArray(value) ? value : {};
  const candidates = Array.isArray(value)
    ? value
    : (Array.isArray(input.todos) ? input.todos : Array.isArray(input.items) ? input.items : Array.isArray(input.tasks) ? input.tasks : []);
  return candidates
    .map((item) => {
      if (item && typeof item === "object") {
        const content = String(item.content || item.task || item.text || "").trim();
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
      if (part?.type !== "tool-call") return;
      const meta = part.meta || {};
      const tool = String(part.tool || meta.tool || "").trim();
      if (tool !== "write_todos") return;
      const rows = normalizeTodoRows(meta.input || part.input || {});
      if (!rows.length) return;
      const source = sourceFromMeta(meta);
      groups.set(source, {
        source,
        rows,
        status: String(part.status || "running"),
        toolCallId: String(meta.tool_call_id || part.tool_call_id || part.id || ""),
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

export function todoSummary(groups) {
  const rows = (Array.isArray(groups) ? groups : []).flatMap((group) => group.rows || []);
  return {
    total: rows.length,
    done: rows.filter((row) => ["done", "completed", "complete"].includes(String(row.status || "").toLowerCase())).length,
    active: rows.filter((row) => ["in_progress", "running", "active"].includes(String(row.status || "").toLowerCase())).length,
  };
}
