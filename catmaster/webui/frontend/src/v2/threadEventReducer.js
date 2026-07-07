import { upsertById } from "./messageAdapters.js";

function updateMessagePart(message, partId, updater) {
  const parts = Array.isArray(message.parts) ? [...message.parts] : [];
  const index = parts.findIndex((part) => part.id === partId);
  if (index >= 0) {
    parts[index] = updater(parts[index]);
  }
  return { ...message, parts, updated_at: Date.now() / 1000 };
}

function appendPart(message, part) {
  const parts = Array.isArray(message.parts) ? [...message.parts] : [];
  if (!parts.some((item) => item.id === part.id)) {
    parts.push(part);
  }
  return { ...message, parts, updated_at: Date.now() / 1000 };
}

function upsertPart(message, part) {
  const parts = Array.isArray(message.parts) ? [...message.parts] : [];
  const index = parts.findIndex((item) => item.id === part.id);
  if (index >= 0) parts[index] = { ...parts[index], ...part };
  else parts.push(part);
  return { ...message, parts, updated_at: Date.now() / 1000 };
}

export function applyThreadEvent(messages, payload) {
  const name = String(payload?.event || "");
  const data = payload?.data || {};
  if (name === "message.created" && data.message) {
    return upsertById(messages, data.message);
  }
  if (name === "message.delta") {
    return messages.map((message) => {
      if (message.id !== data.message_id) return message;
      const partId = data.part_id;
      return updateMessagePart(message, partId, (part) => ({
        ...part,
        text: `${String(part.text || "")}${String(data.delta || "")}`,
        status: "streaming",
      }));
    });
  }
  if (name === "message.part.created" && data.part) {
    return messages.map((message) => (
      message.id === payload.message_id ? upsertPart(message, data.part) : message
    ));
  }
  if (name === "reasoning.delta" || name === "subagent.delta") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      return updateMessagePart(message, data.part_id, (part) => ({
        ...part,
        text: `${String(part.text || "")}${String(data.delta || "")}`,
        status: name === "subagent.delta" ? "running" : "streaming",
      }));
    });
  }
  if (name === "subagent.started" && data.part) {
    return messages.map((message) => (
      message.id === payload.message_id ? upsertPart(message, data.part) : message
    ));
  }
  if (name === "message.completed" || name === "message.failed") {
    if (data.message) {
      return upsertById(messages, data.message);
    }
    return messages.map((message) => (
      message.id === payload.message_id
        ? {
            ...updateMessagePart(message, data.part_id, (part) => ({
              ...part,
              text: data.text !== undefined ? String(data.text || "") : part.text,
              status: name === "message.completed" ? "completed" : part.status,
            })),
            status: name === "message.completed" ? "completed" : "failed",
            structured_sidecar: data.structured_sidecar || message.structured_sidecar || {},
          }
        : message
    ));
  }
  if (name === "tool_call.started") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      return appendPart(message, {
        id: data.part_id,
        type: "tool-call",
        status: "running",
        text: "",
        meta: {
          tool_call_id: data.tool_call_id,
          tool: data.tool,
          input: data.input || {},
          agent_name: data.agent_name || "",
          subagent_source: data.subagent_source || "",
          stream_namespace: data.stream_namespace,
        },
      });
    });
  }
  if (name === "tool_call.delta") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      return updateMessagePart(message, data.part_id, (part) => {
        const meta = part.meta || {};
        return {
          ...part,
          status: "running",
          meta: {
            ...meta,
            tool_call_id: data.tool_call_id || meta.tool_call_id || "",
            tool: data.tool || meta.tool || "",
            input: data.input !== undefined ? data.input : meta.input || {},
            delta: data.delta || meta.delta,
            agent_name: data.agent_name || meta.agent_name || "",
            subagent_source: data.subagent_source || meta.subagent_source || "",
            stream_namespace: data.stream_namespace !== undefined ? data.stream_namespace : meta.stream_namespace,
          },
        };
      });
    });
  }
  if (name === "tool_call.completed" || name === "tool_call.failed") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      if (!data.part_id) {
        return appendPart(message, {
          id: `part_tool_done_${data.tool_call_id || payload.seq || Date.now()}`,
          type: "tool-call",
          status: name === "tool_call.completed" ? "completed" : "failed",
          text: typeof data.output === "string" ? data.output : "",
          meta: {
            tool_call_id: data.tool_call_id || "",
            tool: data.tool || data.tool_call_id || "tool",
            input: data.input || {},
            output: data.output,
            agent_name: data.agent_name || "",
            subagent_source: data.subagent_source || "",
            stream_namespace: data.stream_namespace,
          },
        });
      }
      return updateMessagePart(message, data.part_id, (part) => ({
        ...part,
        status: name === "tool_call.completed" ? "completed" : "failed",
        text: typeof data.output === "string" ? data.output : part.text,
        meta: {
          ...(part.meta || {}),
          tool_call_id: data.tool_call_id || part.meta?.tool_call_id || "",
          tool: data.tool || part.meta?.tool || "",
          input: data.input !== undefined ? data.input : part.meta?.input || {},
          output: data.output,
          agent_name: data.agent_name || part.meta?.agent_name || "",
          subagent_source: data.subagent_source || part.meta?.subagent_source || "",
          stream_namespace: data.stream_namespace !== undefined ? data.stream_namespace : part.meta?.stream_namespace,
        },
      }));
    });
  }
  if (name === "artifact.created") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      return appendPart(message, {
        id: `part_${data.artifact_id}`,
        type: "artifact",
        status: "completed",
        text: "",
        artifact_id: data.artifact_id,
        renderer: data.renderer,
        title: data.title,
        summary: data.summary,
        path: data.path,
        meta: data,
      });
    });
  }
  if (name === "interrupt.created") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      return appendPart({ ...message, status: "interrupted" }, {
        id: data.part_id,
        type: "interrupt",
        status: "pending",
        text: data.body || "Review required.",
        meta: data,
      });
    });
  }
  if (name === "interrupt.resolved") {
    const resolved = Array.isArray(data.resolved_parts) ? data.resolved_parts : [];
    const partIds = new Set(resolved.map((item) => item.part_id).filter(Boolean));
    return messages.map((message) => {
      const parts = Array.isArray(message.parts) ? message.parts : [];
      let changed = false;
      const nextParts = parts.map((part) => {
        if (part.type !== "interrupt") return part;
        if (partIds.size && !partIds.has(part.id)) return part;
        changed = true;
        return {
          ...part,
          status: "resolved",
          meta: {
            ...(part.meta || {}),
            status: "resolved",
            resolution: { decisions: data.decisions || [] },
          },
        };
      });
      return changed ? { ...message, parts: nextParts } : message;
    });
  }
  if (name === "task_receipt.updated") {
    return messages.map((message) => {
      if (message.id !== payload.message_id) return message;
      const receipt = data.receipt || data;
      const part = {
        id: data.part_id || `part_receipt_${receipt.remote_context_id || receipt.submission_hash || Date.now()}`,
        type: "receipt",
        status: payload.status || receipt.status || "updated",
        text: receipt.receipt_rel || receipt.remote_context_id || "Remote task receipt",
        meta: receipt,
      };
      return appendPart(message, part);
    });
  }
  return messages;
}
