import { useCallback, useEffect, useMemo, useState } from "react";
import { useExternalStoreRuntime } from "@assistant-ui/react";

import { CatMasterAttachmentAdapter } from "./catmasterAttachmentAdapter.js";
import { catMessagesToAssistant, insertById, requestFromAssistantAppend, upsertById } from "./messageAdapters.js";
import { applyThreadEvent } from "./threadEventReducer.js";
import { makeApiError } from "./presentation.js";
import { canonicalTodoPartsFromEvent } from "./todoPanel.js";
import { retainActiveAsyncSubagents, updateActiveOperations } from "./activeOperations.js";
import { subscribeThreadStream } from "./threadStream.js";

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

export function useCatMasterThreadRuntime({ thread, onThreadUpdate, onSelectArtifact, readOnly = false, transformMessages }) {
  const [messages, setMessages] = useState([]);
  const [messagePage, setMessagePage] = useState({});
  const [loadingOlder, setLoadingOlder] = useState(false);
  const [artifacts, setArtifacts] = useState([]);
  const [todoParts, setTodoParts] = useState([]);
  const [activeParts, setActiveParts] = useState([]);
  const [events, setEvents] = useState([]);
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(false);
  const [refreshGeneration, setRefreshGeneration] = useState(0);

  const refreshMessages = useCallback(async () => {
    setRefreshGeneration((value) => value + 1);
  }, []);

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
    let lastSeq = 0;
    async function loadSnapshot(signal) {
      if (!thread?.thread_id) {
        setMessages([]);
        setMessagePage({});
        setArtifacts([]);
        setTodoParts([]);
        setActiveParts([]);
        setEvents([]);
        return;
      }
      setLoading(true);
      setError("");
      try {
        // Refresh native lifecycle before taking the UI snapshot, so a run
        // completed while this tab was hidden does not remain visibly active.
        const threadPayload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}`, { signal });
        const [messagePayload, artifactPayload] = await Promise.all([
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/messages?limit=50`, { signal }),
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/artifacts`, { signal }),
        ]);
        if (!cancelled && !signal.aborted) {
          setMessages(Array.isArray(messagePayload.messages) ? messagePayload.messages : []);
          setMessagePage(messagePayload.page || {});
          setTodoParts(Array.isArray(messagePayload.todo_parts) ? messagePayload.todo_parts : []);
          setActiveParts(Array.isArray(messagePayload.active_parts) ? messagePayload.active_parts : []);
          setArtifacts(Array.isArray(artifactPayload.artifacts) ? artifactPayload.artifacts : []);
          lastSeq = Number(messagePayload.stream_cursor || 0);
          if (threadPayload.thread) onThreadUpdate?.(threadPayload.thread);
        }
        return lastSeq;
      } finally {
        if (!cancelled && !signal.aborted) setLoading(false);
      }
    }
    const handleEvent = (event) => {
      try {
        if (cancelled || !event.data) return;
        const payload = JSON.parse(event.data || "{}");
        if (Number(payload.seq) <= lastSeq) return;
        lastSeq = Number(payload.seq);
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
        if (payload.event === "activity.updated" && payload.data?.part) {
          setActiveParts((prev) => updateActiveOperations(prev, payload.data.part));
        }
        if (["subagent.started", "subagent.completed"].includes(payload.event) && payload.data?.part) {
          setActiveParts((prev) => updateActiveOperations(prev, payload.data.part));
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
          const nextStatus = String(payload.status || payload.data?.status || "").toLowerCase();
          if (["idle", "completed", "error", "failed", "interrupted", "stopped"].includes(nextStatus)) {
            setActiveParts((current) => retainActiveAsyncSubagents(current));
          }
          onThreadUpdate?.({
            thread_id: thread.thread_id,
            status: payload.status || payload.data?.status || thread.status,
            ...(Object.hasOwn(payload.data || {}, "run_id") ? { active_run_id: payload.data.run_id } : {}),
            ...(Number.isFinite(Number(payload.data?.pending_run_count))
              ? { pending_run_count: Number(payload.data.pending_run_count) }
              : {}),
          });
        }
        if (payload.event === "thread.updated" && payload.data?.thread) {
          onThreadUpdate?.(payload.data.thread);
        }
        if (["message.completed", "message.failed", "run.failed"].includes(payload.event)) {
          setActiveParts((current) => retainActiveAsyncSubagents(current));
        }
        if (payload.event === "message.completed") {
          const outputs = (payload.data?.message?.parts || []).filter((part) => part.type === "artifact" && part.artifact_id);
          setArtifacts((prev) => outputs.reduce((rows, part) => rows.some((row) => row.artifact_id === part.artifact_id) ? rows : [...rows, part], prev));
          // Artifacts can be beyond the first page of a long activity trace.
          apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/artifacts`)
            .then((result) => { if (!cancelled) setArtifacts(result.artifacts || []); })
            .catch((err) => { if (!cancelled) setError(err); });
        }
      } catch (err) {
        setError(err);
      }
    };
    const eventNames = [
      "thread.created",
      "thread.updated",
      "thread.status",
      "message.created",
      "message.updated",
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
    ];
    if (!thread?.thread_id) {
      loadSnapshot(new AbortController().signal);
      return () => { cancelled = true; };
    }
    const unsubscribe = subscribeThreadStream({
      url: `/api/threads/${encodeURIComponent(thread.thread_id)}/stream`,
      eventNames, loadSnapshot, onEvent: handleEvent, onError: setError,
    });
    return () => {
      cancelled = true;
      unsubscribe();
    };
  }, [thread?.thread_id, onThreadUpdate, refreshGeneration]);

  const submitText = useCallback(async (text, attachments = [], submitOptions = {}) => {
    const body = String(text || "").trim();
    const attachmentRows = Array.isArray(attachments) ? attachments : [];
    if (!thread?.thread_id || (!body && !attachmentRows.length)) return;
    setError("");
    setTodoParts([]);
    setActiveParts((current) => retainActiveAsyncSubagents(current));
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/submit`, {
        method: "POST",
        body: JSON.stringify({
          text: body,
          entrypoint: submitOptions.entrypoint || thread.entrypoint || "research",
          permission_mode: submitOptions.permission_mode || thread?.permission_mode || "auto",
          attachments: attachmentRows,
          strategy: submitOptions.strategy || "enqueue",
        }),
      });
      if (payload.message) setMessages((prev) => upsertById(prev, payload.message));
      if (payload.assistant_message) setMessages((prev) => insertById(prev, payload.assistant_message));
      if (payload.thread) onThreadUpdate?.(payload.thread);
      return payload;
    } catch (err) {
      setError(err);
      throw err;
    }
  }, [thread, onThreadUpdate]);

  const stop = useCallback(async (action = "interrupt", runId = "") => {
    if (!thread?.thread_id) return;
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/stop`, {
        method: "POST",
        body: JSON.stringify({
          run_id: runId || thread.active_run_id || "",
          action,
        }),
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
      if (payload.assistant_message) setMessages((prev) => insertById(prev, payload.assistant_message));
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
        setMessages((prev) => insertById(prev, payload.assistant_message));
      }
      if (payload.thread) onThreadUpdate?.(payload.thread);
      return payload;
    } catch (err) {
      setError(err);
      throw err;
    }
  }, [thread, onThreadUpdate]);

  const updateAsyncSubagent = useCallback(async (action, message) => {
    const endpoint = String(action?.endpoint || "");
    const prefix = `/api/threads/${encodeURIComponent(thread?.thread_id || "")}/async-subagents/`;
    if (!thread?.thread_id || !endpoint.startsWith(prefix)) return;
    setError("");
    try {
      const payload = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ message: String(message || "").trim() }),
      });
      await refreshMessages();
      return payload;
    } catch (err) {
      setError(err);
      throw err;
    }
  }, [thread?.thread_id, refreshMessages]);

  const stopAsyncSubagent = useCallback(async (action) => {
    const endpoint = String(action?.endpoint || "");
    const prefix = `/api/threads/${encodeURIComponent(thread?.thread_id || "")}/async-subagents/`;
    if (!thread?.thread_id || !endpoint.startsWith(prefix)) return;
    setError("");
    try {
      const payload = await apiFetch(endpoint, {
        method: "POST",
        body: JSON.stringify({ action: "interrupt" }),
      });
      await refreshMessages();
      return payload;
    } catch (err) {
      setError(err);
      throw err;
    }
  }, [thread?.thread_id, refreshMessages]);

  const displayedMessages = useMemo(() => transformMessages ? transformMessages(messages) : messages, [messages, transformMessages]);
  const assistantMessages = useMemo(() => catMessagesToAssistant(displayedMessages), [displayedMessages]);
  const attachmentAdapter = useMemo(() => new CatMasterAttachmentAdapter(), []);
  const runtime = useExternalStoreRuntime({
    messages: assistantMessages,
    isRunning: threadIsRunning(thread),
    isDisabled: readOnly,
    onNew: async (message) => {
      if (readOnly) return;
      const request = requestFromAssistantAppend(message);
      await submitText(request.text, request.attachments, { strategy: "enqueue" });
    },
    onCancel: async () => {
      if (readOnly) return;
      await stop("interrupt");
    },
    adapters: {
      attachments: attachmentAdapter,
    },
  });

  return {
    runtime,
    messages,
    todoParts,
    activeParts,
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
    updateAsyncSubagent,
    stopAsyncSubagent,
    refreshMessages,
    loadOlderMessages,
    refreshArtifacts,
    onSelectArtifact,
  };
}
