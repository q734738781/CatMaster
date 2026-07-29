import { upsertById } from "./messageAdapters.js";

function updateMessagePart(message, partId, updater) {
  const parts = Array.isArray(message.parts) ? [...message.parts] : [];
  const index = parts.findIndex((part) => part.id === partId);
  if (index >= 0) {
    parts[index] = updater(parts[index]);
  }
  return { ...message, parts, updated_at: Date.now() / 1000 };
}

function upsertPart(message, part) {
  if (!part?.id) return message;
  const parts = Array.isArray(message.parts) ? [...message.parts] : [];
  const index = parts.findIndex((item) => item.id === part.id);
  if (index >= 0) parts[index] = { ...parts[index], ...part };
  else parts.push(part);
  return { ...message, parts, updated_at: Date.now() / 1000 };
}

function ensureMessage(messages, messageId) {
  if (!messageId || messages.some((message) => message.id === messageId)) return messages;
  return [
    ...messages,
    {
      id: messageId,
      role: "assistant",
      status: "streaming",
      created_at: Date.now() / 1000,
      updated_at: Date.now() / 1000,
      parts: [],
    },
  ];
}

export function applyThreadEvent(messages, payload) {
  const name = String(payload?.event || "");
  const data = payload?.data || {};
  const messageId = String(payload?.message_id || data.message_id || "");

  if (name === "message.created" && data.message) {
    return upsertById(messages, data.message);
  }
  if (name === "message.delta" || name === "reasoning.delta" || name === "subagent.delta") {
    const seeded = ensureMessage(messages, messageId);
    return seeded.map((message) => {
      if (message.id !== messageId) return message;
      const partId = String(data.part_id || "");
      const existing = (message.parts || []).some((part) => part.id === partId);
      const partType = name === "reasoning.delta" ? "reasoning" : name === "subagent.delta" ? "progress" : "text";
      const base = existing ? message : upsertPart(message, {
        id: partId,
        type: partType,
        status: "streaming",
        title: partType === "progress" ? "Specialist progress" : "",
        text: "",
        fields: [],
        actions: [],
        items: [],
      });
      return updateMessagePart(base, partId, (part) => ({
        ...part,
        text: `${String(part.text || "")}${String(data.delta || "")}`,
        status: "streaming",
      }));
    });
  }
  if ((name === "message.part.created" || name === "subagent.started") && data.part) {
    const seeded = ensureMessage(messages, messageId);
    return seeded.map((message) => (
      message.id === messageId ? upsertPart(message, data.part) : message
    ));
  }
  if (name === "activity.updated" && data.part) {
    const seeded = ensureMessage(messages, messageId);
    return seeded.map((message) => (
      message.id === messageId ? upsertPart(message, data.part) : message
    ));
  }
  if (name === "message.completed") {
    return messages.map((message) => (
      message.id === messageId ? { ...message, status: "completed" } : message
    ));
  }
  if (name === "run.failed") {
    const seeded = ensureMessage(messages, messageId);
    return seeded.map((message) => {
      if (message.id !== messageId) return message;
      return {
        ...upsertPart(message, {
          id: `part_error_${payload.seq || "run"}`,
          type: "error",
          status: "failed",
          title: data.title || "The task stopped before it completed",
          summary: data.summary || "CatMaster could not complete this run.",
          fields: data.fields || [],
          actions: data.actions || [],
          items: [],
          diagnostics_ref: data.diagnostics_ref || "",
        }),
        status: "failed",
      };
    });
  }
  if (name === "interrupt.resolved") {
    return messages.map((message) => {
      const parts = Array.isArray(message.parts) ? message.parts : [];
      const nextParts = parts.map((part) => (
        part.type === "interrupt" && part.status !== "resolved"
          ? { ...part, status: "resolved", actions: [] }
          : part
      ));
      return { ...message, parts: nextParts };
    });
  }
  return messages;
}
