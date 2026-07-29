export function catPartText(part) {
  if (!part) return "";
  if (part.type === "text" || part.type === "reasoning" || part.type === "interrupt") {
    return String(part.text || "");
  }
  return "";
}

function assistantStatus(status) {
  const value = String(status || "").toLowerCase();
  if (value === "streaming" || value === "created") {
    return { type: "running" };
  }
  if (value === "failed") {
    return { type: "incomplete", reason: "error" };
  }
  if (value === "interrupted") {
    return { type: "requires-action", reason: "interrupt" };
  }
  return { type: "complete", reason: "unknown" };
}

function catPartToAssistant(part) {
  const type = String(part?.type || "");
  return {
    type: "data",
    name: `catmaster-${type || "part"}`,
    data: part || {
      type: "unknown",
      title: "This activity cannot be displayed yet",
      summary: "The record remains available to developer diagnostics.",
    },
  };
}

export function catMessageToAssistant(message) {
  const role = String(message?.role || "assistant");
  const content = Array.isArray(message?.parts)
    ? message.parts.map(catPartToAssistant)
    : [{ type: "text", text: String(message?.content || "") }];
  const base = {
    id: String(message?.id || ""),
    role,
    content,
    createdAt: new Date(Number(message?.created_at || Date.now() / 1000) * 1000),
    metadata: {
      custom: {
        catmaster: message,
      },
    },
  };
  if (role === "user") {
    base.attachments = [];
  }
  if (role === "assistant") {
    base.status = assistantStatus(message?.status);
  }
  return base;
}

export function catMessagesToAssistant(messages) {
  return (Array.isArray(messages) ? messages : []).map(catMessageToAssistant).filter((item) => item.id);
}

export function requestFromAssistantAppend(message) {
  if (!message) return { text: "", attachments: [] };
  const parts = Array.isArray(message.content) ? message.content : [];
  const text = parts
    .map((part) => {
      if (typeof part === "string") return part;
      if (part?.type === "text") return String(part.text || "");
      return "";
    })
    .join("")
    .trim();
  const attachments = [];
  const appendPart = (part, source = "content") => {
    if (!part || typeof part !== "object" || part.type === "text") return;
    attachments.push({
      source,
      type: String(part.type || "file"),
      filename: part.filename || part.name || "",
      mime_type: part.mimeType || part.contentType || "",
      size_bytes: part.sizeBytes || part.size_bytes || 0,
      data: part.data || part.image || part.audio?.data || "",
      text: part.text || "",
      name: part.name || "",
      raw: part,
    });
  };
  parts.forEach((part) => appendPart(part, "content"));
  const rawAttachments = Array.isArray(message.attachments) ? message.attachments : [];
  rawAttachments.forEach((attachment) => {
    const content = Array.isArray(attachment?.content) ? attachment.content : [];
    if (!content.length) {
      attachments.push({
        source: "attachment",
        type: String(attachment?.type || "file"),
        filename: attachment?.name || "",
        mime_type: attachment?.contentType || "",
        size_bytes: attachment?.file?.size || attachment?.sizeBytes || 0,
        data: "",
        text: "",
        name: attachment?.name || "",
      });
    }
    content.forEach((part) => appendPart({ ...part, name: attachment?.name || part?.name }, "attachment"));
  });
  return { text, attachments };
}

export function textFromAssistantAppend(message) {
  return requestFromAssistantAppend(message).text;
}

export function upsertById(rows, row) {
  if (!row?.id) return rows;
  const next = [...rows];
  const index = next.findIndex((item) => item.id === row.id);
  if (index >= 0) next[index] = row;
  else next.push(row);
  return next;
}
