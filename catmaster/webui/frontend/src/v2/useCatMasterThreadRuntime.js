import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { useExternalStoreRuntime } from "@assistant-ui/react";

import { CatMasterAttachmentAdapter } from "./catmasterAttachmentAdapter.js";
import { catMessagesToAssistant, requestFromAssistantAppend, upsertById } from "./messageAdapters.js";
import { applyThreadEvent } from "./threadEventReducer.js";
import { makeApiError } from "./presentation.js";
import { canonicalTodoPartsFromEvent } from "./todoPanel.js";

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
    throw makeApiError(response.status, text, response.headers.get("content-type") || "");
  }
  if (!text) return {};
  try {
    return JSON.parse(text);
  } catch {
    const error = new Error("CatMaster received an unreadable server response. Refresh the workspace and try again.");
    error.status = response.status;
    error.details = {};
    error.technicalDetails = `HTTP ${response.status}\nThe server returned non-JSON content where application data was expected.`;
    throw error;
  }
}

function threadIsRunning(thread) {
  return ["running", "stopping"].includes(String(thread?.status || "").toLowerCase());
}

export function useCatMasterThreadRuntime({ thread, onThreadUpdate, onSelectArtifact }) {
  const [messages, setMessages] = useState([]);
  const [messagePage, setMessagePage] = useState({});
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [artifacts, setArtifacts] = useState([]);
  const [todoParts, setTodoParts] = useState([]);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const eventSourceRef = useRef(null);

  const refreshMessages = useCallback(async () => {
    if (!thread?.thread_id) return;
    const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/messages?limit=50`);
    setMessages(Array.isArray(payload.messages) ? payload.messages : []);
    setMessagePage(payload.page || {});
    setTodoParts(Array.isArray(payload.todo_parts) ? payload.todo_parts : []);
  }, [thread?.thread_id]);

  const loadOlderMessages = useCallback(async () => {
    const cursor = String(messagePage?.next_cursor || "");
    if (!thread?.thread_id || !cursor || loadingOlder) return;
    setLoadingOlder(true);
    setError("");
    try {
      const payload = await apiFetch(
        `/api/threads/${encodeURIComponent(thread.thread_id)}/messages?limit=50&before=${encodeURIComponent(cursor)}`,
      );
      const older = Array.isArray(payload.messages) ? payload.messages : [];
      setMessages((current) => {
        const existing = new Set(current.map((message) => message.id));
        return [...older.filter((message) => !existing.has(message.id)), ...current];
      });
      setMessagePage(payload.page || {});
    } catch (err) {
      setError(err);
    } finally {
      setLoadingOlder(false);
    }
  }, [thread?.thread_id, messagePage?.next_cursor, loadingOlder]);

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
        setMessagePage({});
        setArtifacts([]);
        setTodoParts([]);
        setEvents([]);
        return;
      }
      setLoading(true);
      setError("");
      try {
        const [messagePayload, artifactPayload] = await Promise.all([
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/messages?limit=50`),
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/artifacts`),
        ]);
        if (!cancelled) {
          setMessages(Array.isArray(messagePayload.messages) ? messagePayload.messages : []);
          setMessagePage(messagePayload.page || {});
          setTodoParts(Array.isArray(messagePayload.todo_parts) ? messagePayload.todo_parts : []);
          setArtifacts(Array.isArray(artifactPayload.artifacts) ? artifactPayload.artifacts : []);
        }
      } catch (err) {
        if (!cancelled) setError(err);
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
        const canonicalTodoParts = canonicalTodoPartsFromEvent(payload);
        if (canonicalTodoParts !== null) {
          setTodoParts(canonicalTodoParts);
        }
        const todoPart = payload.event === "activity.updated" && payload.data?.part?.type === "progress"
          ? payload.data.part
          : null;
        if (todoPart?.items?.length) {
          setTodoParts((prev) => {
            const key = String(todoPart.title || "Research plan").toLowerCase();
            return [
              todoPart,
              ...prev.filter((item) => String(item.title || "Research plan").toLowerCase() !== key),
            ];
          });
        }
        const artifactPart = payload.event === "activity.updated" && payload.data?.part?.type === "artifact"
          ? payload.data.part
          : null;
        if (artifactPart?.artifact_id) {
          setArtifacts((prev) => {
            if (prev.some((item) => item.artifact_id === artifactPart.artifact_id)) return prev;
            return [...prev, artifactPart];
          });
        }
        if (payload.event === "thread.status" && thread?.thread_id) {
          onThreadUpdate?.({
            thread_id: thread.thread_id,
            status: payload.status || payload.data?.status || thread.status,
          });
        }
      } catch (err) {
        setError(err);
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
      "activity.updated",
      "run.failed",
      "tool_call.started",
      "tool_call.delta",
      "tool_call.completed",
      "tool_call.failed",
      "artifact.created",
      "artifact.updated",
      "multimodal.prepared",
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

  const submitText = useCallback(async (text, attachments = [], submitOptions = {}) => {
    const body = String(text || "").trim();
    const attachmentRows = Array.isArray(attachments) ? attachments : [];
    if (!thread?.thread_id || (!body && !attachmentRows.length)) return;
    setError("");
    setTodoParts([]);
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/submit`, {
        method: "POST",
        body: JSON.stringify({
          text: body,
          entrypoint: submitOptions.entrypoint || thread.entrypoint || "research",
          permission_mode: submitOptions.permission_mode || thread?.permission_mode || "auto",
          attachments: attachmentRows,
        }),
      });
      if (payload.message) setMessages((prev) => upsertById(prev, payload.message));
      if (payload.assistant_message) setMessages((prev) => upsertById(prev, payload.assistant_message));
      if (payload.thread) onThreadUpdate?.(payload.thread);
      return payload;
    } catch (err) {
      setError(err);
      throw err;
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
      setError(err);
    }
  }, [thread, onThreadUpdate]);

  const resume = useCallback(async (review) => {
    if (!thread?.thread_id) return;
    const actions = Array.isArray(review) ? review : [review];
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/resume`, {
        method: "POST",
        body: JSON.stringify({ actions }),
      });
      if (payload.assistant_message) setMessages((prev) => upsertById(prev, payload.assistant_message));
      if (payload.thread) onThreadUpdate?.(payload.thread);
    } catch (err) {
      setError(err);
    }
  }, [thread, onThreadUpdate]);

  const continueFromCheckpoint = useCallback(async (messageId) => {
    if (!thread?.thread_id || !messageId) return;
    setError("");
    try {
      const payload = await apiFetch(
        `/api/threads/${encodeURIComponent(thread.thread_id)}/continue-from-checkpoint`,
        {
          method: "POST",
          body: JSON.stringify({ message_id: messageId }),
        },
      );
      if (payload.assistant_message) {
        setMessages((prev) => upsertById(prev, payload.assistant_message));
      }
      if (payload.thread) onThreadUpdate?.(payload.thread);
      return payload;
    } catch (err) {
      setError(err);
      throw err;
    }
  }, [thread, onThreadUpdate]);

  const assistantMessages = useMemo(() => catMessagesToAssistant(messages), [messages]);
  const attachmentAdapter = useMemo(() => new CatMasterAttachmentAdapter(), []);
  const runtime = useExternalStoreRuntime({
    messages: assistantMessages,
    isRunning: threadIsRunning(thread),
    onNew: async (message) => {
      const request = requestFromAssistantAppend(message);
      await submitText(request.text, request.attachments);
    },
    onCancel: async () => {
      await stop(true);
    },
    adapters: {
      attachments: attachmentAdapter,
    },
  });

  return {
    runtime,
    messages,
    todoParts,
    artifacts,
    events,
    messagePage,
    loading,
    loadingOlder,
    error,
    isRunning: threadIsRunning(thread),
    submitText,
    stop,
    resume,
    continueFromCheckpoint,
    refreshMessages,
    loadOlderMessages,
    refreshArtifacts,
    onSelectArtifact,
  };
}
