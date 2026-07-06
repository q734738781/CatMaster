import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  CompositeAttachmentAdapter,
  SimpleImageAttachmentAdapter,
  SimpleTextAttachmentAdapter,
  useExternalStoreRuntime,
} from "@assistant-ui/react";

import { catMessagesToAssistant, requestFromAssistantAppend, upsertById } from "./messageAdapters.js";
import { applyThreadEvent } from "./threadEventReducer.js";

export async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  const text = await response.text();
  if (!response.ok) {
    let message = text || `Request failed: ${response.status}`;
    try {
      const payload = JSON.parse(text || "{}");
      message = String(payload.detail || payload.message || message);
    } catch {
      // Keep raw text.
    }
    throw new Error(message);
  }
  return text ? JSON.parse(text) : {};
}

function threadIsRunning(thread) {
  return ["running", "stopping"].includes(String(thread?.status || "").toLowerCase());
}

export function useCatMasterThreadRuntime({ thread, onThreadUpdate, onSelectArtifact }) {
  const [messages, setMessages] = useState([]);
  const [artifacts, setArtifacts] = useState([]);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef(null);

  const refreshMessages = useCallback(async () => {
    if (!thread?.thread_id) return;
    const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/messages`);
    setMessages(Array.isArray(payload.messages) ? payload.messages : []);
  }, [thread?.thread_id]);

  const refreshArtifacts = useCallback(async () => {
    if (!thread?.thread_id) return;
    const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/artifacts`);
    setArtifacts(Array.isArray(payload.artifacts) ? payload.artifacts : []);
  }, [thread?.thread_id]);

  useEffect(() => {
    let cancelled = false;
    async function load() {
      if (!thread?.thread_id) {
        setMessages([]);
        setArtifacts([]);
        setEvents([]);
        return;
      }
      setLoading(true);
      setError("");
      try {
        const [messagePayload, artifactPayload] = await Promise.all([
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/messages`),
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/artifacts`),
        ]);
        if (!cancelled) {
          setMessages(Array.isArray(messagePayload.messages) ? messagePayload.messages : []);
          setArtifacts(Array.isArray(artifactPayload.artifacts) ? artifactPayload.artifacts : []);
        }
      } catch (err) {
        if (!cancelled) setError(err.message || String(err));
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [thread?.thread_id]);

  useEffect(() => {
    if (!thread?.thread_id) return undefined;
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const source = new EventSource(`/api/threads/${encodeURIComponent(thread.thread_id)}/stream`);
    eventSourceRef.current = source;
    const handleEvent = (event) => {
      try {
        const payload = JSON.parse(event.data || "{}");
        setEvents((prev) => [...prev.slice(-299), payload]);
        setMessages((prev) => applyThreadEvent(prev, payload));
        if (payload.event === "artifact.created" && payload.data?.artifact_id) {
          setArtifacts((prev) => {
            if (prev.some((item) => item.artifact_id === payload.data.artifact_id)) return prev;
            return [...prev, payload.data];
          });
        }
        if (payload.event === "thread.status" && thread?.thread_id) {
          onThreadUpdate?.({
            thread_id: thread.thread_id,
            status: payload.status || payload.data?.status || thread.status,
          });
        }
      } catch (err) {
        setError(err.message || String(err));
      }
    };
    [
      "thread.created",
      "thread.updated",
      "thread.status",
      "message.created",
      "message.delta",
      "message.part.created",
      "message.part.delta",
      "reasoning.delta",
      "message.completed",
      "message.failed",
      "tool_call.started",
      "tool_call.delta",
      "tool_call.completed",
      "tool_call.failed",
      "artifact.created",
      "artifact.updated",
      "interrupt.created",
      "interrupt.updated",
      "interrupt.resolved",
      "usage.updated",
      "task_receipt.updated",
      "subagent.started",
      "subagent.delta",
      "subagent.completed",
      "trace.event",
      "error",
    ].forEach((name) => source.addEventListener(name, handleEvent));
    source.onerror = () => {
      // Browser will reconnect automatically; keep the last visible state.
    };
    return () => {
      source.close();
      if (eventSourceRef.current === source) eventSourceRef.current = null;
    };
  }, [thread?.thread_id, onThreadUpdate]);

  const submitText = useCallback(async (text, attachments = []) => {
    const body = String(text || "").trim();
    const attachmentRows = Array.isArray(attachments) ? attachments : [];
    if (!thread?.thread_id || (!body && !attachmentRows.length)) return;
    setError("");
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/submit`, {
        method: "POST",
        body: JSON.stringify({
          text: body,
          entrypoint: thread.entrypoint || "research",
          permission_mode: thread?.meta?.permission_mode || "auto",
          attachments: attachmentRows,
        }),
      });
      if (payload.message) setMessages((prev) => upsertById(prev, payload.message));
      if (payload.assistant_message) setMessages((prev) => upsertById(prev, payload.assistant_message));
      if (payload.thread) onThreadUpdate?.(payload.thread);
    } catch (err) {
      setError(err.message || String(err));
    }
  }, [thread, onThreadUpdate]);

  const stop = useCallback(async (emergency = false) => {
    if (!thread?.thread_id) return;
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/stop`, {
        method: "POST",
        body: JSON.stringify({ emergency }),
      });
      if (payload.thread) onThreadUpdate?.(payload.thread);
    } catch (err) {
      setError(err.message || String(err));
    }
  }, [thread, onThreadUpdate]);

  const resume = useCallback(async (decision) => {
    if (!thread?.thread_id) return;
    const decisions = Array.isArray(decision) ? decision : [decision];
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/resume`, {
        method: "POST",
        body: JSON.stringify({ decisions }),
      });
      if (payload.assistant_message) setMessages((prev) => upsertById(prev, payload.assistant_message));
      if (payload.thread) onThreadUpdate?.(payload.thread);
    } catch (err) {
      setError(err.message || String(err));
    }
  }, [thread, onThreadUpdate]);

  const assistantMessages = useMemo(() => catMessagesToAssistant(messages), [messages]);
  const attachmentAdapter = useMemo(() => new CompositeAttachmentAdapter([
    new SimpleImageAttachmentAdapter(),
    new SimpleTextAttachmentAdapter(),
  ]), []);
  const runtime = useExternalStoreRuntime({
    messages: assistantMessages,
    isRunning: threadIsRunning(thread),
    onNew: async (message) => {
      const request = requestFromAssistantAppend(message);
      await submitText(request.text, request.attachments);
    },
    onCancel: async () => {
      await stop(false);
    },
    adapters: {
      attachments: attachmentAdapter,
    },
  });

  return {
    runtime,
    messages,
    artifacts,
    events,
    loading,
    error,
    isRunning: threadIsRunning(thread),
    submitText,
    stop,
    resume,
    refreshMessages,
    refreshArtifacts,
    onSelectArtifact,
  };
}
