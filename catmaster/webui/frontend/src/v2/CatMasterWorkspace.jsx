import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { LogIn, LogOut, Menu, PanelRight, RefreshCw, ShieldAlert, ShieldCheck, UserPlus, Workflow, X } from "lucide-react";

import WorkspaceRail from "./components/WorkspaceRail";
import ThreadMessages from "./components/ThreadMessages";
import ThreadComposer from "./components/ThreadComposer";
import ActiveOperationsPanel from "./components/ActiveOperationsPanel";
import ArtifactPreviewModal from "./components/ArtifactPreviewModal";
import PlanPanel from "./components/PlanPanel";
import { FilesPanel, MonitorPanel, SelfEvolutionPanel } from "./components/WorkspacePanels";
import { apiFetch, useCatMasterThreadRuntime } from "./useCatMasterThreadRuntime";
import { DEFAULT_ENTRYPOINT, entrypointMeta, newThreadEntrypoint, normalizedEntrypoints, normalizeEntrypoint } from "./entrypoints";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection";
import { artifactForSelection } from "./artifactSelection.js";
import { todoGroupsFromParts } from "./todoPanel.js";
import { displayValue, presentError, userFacingFileTitle } from "./presentation.js";
import {
  ACTIVE_RESEARCH_STATES,
  preferredWorkspaceThreadId,
  RESEARCH_GRAPH_ACTIVITY_EVENTS,
  researchStateLabel,
} from "./researchSessions.js";

const ResearchTechTreePanel = lazy(
  () => import("./components/ResearchTechTreePanel"),
);

const RESEARCH_CHILD_ACTIVITY_EVENTS = [
  "thread.updated",
  "thread.status",
  "activity.updated",
  "message.completed",
  "message.failed",
  "tool_call.started",
  "tool_call.completed",
  "tool_call.failed",
  "interrupt.created",
  "interrupt.resolved",
  "error",
];

const NATIVE_ROOT_ACTIVITY_EVENTS = [
  "thread.updated",
  "thread.status",
  "activity.updated",
  "subagent.started",
  "subagent.completed",
  "message.completed",
  "message.failed",
  "error",
];

function ErrorNotice({ error }) {
  const presented = presentError(error);
  if (!presented.message) return null;
  return (
    <div className="v2-error" role="alert">
      <span>{presented.message}</span>
      {presented.technicalDetails ? (
        <details className="v2-error-details">
          <summary>Technical details</summary>
          <pre>{presented.technicalDetails}</pre>
        </details>
      ) : null}
    </div>
  );
}

function threadStatusLabel(value) {
  const status = String(value || "idle").toLowerCase();
  return {
    idle: "Turn ready",
    created: "Turn ready",
    queued: "Queued",
    pending: "Waiting",
    running: "Turn active",
    stopping: "Stopping",
    interrupted: "Waiting for review",
    completed: "Completed",
    failed: "Needs attention",
  }[status] || displayValue(status.replace(/[_-]+/g, " "), "Ready");
}

function WorkspaceEmptyState({ hasWorkspaces, onCreate }) {
  const [name, setName] = useState("");
  const [busy, setBusy] = useState(false);
  async function submit(event) {
    event.preventDefault();
    if (!name.trim() || busy) return;
    setBusy(true);
    try { await onCreate(name.trim()); }
    finally { setBusy(false); }
  }
  return (
    <div className="v2-workspace-empty">
      <form onSubmit={submit}>
        <h2>{hasWorkspaces ? "Choose a workspace" : "Create your first workspace"}</h2>
        <p>{hasWorkspaces
          ? "Select a workspace from the sidebar, or create a new one below."
          : "Give your project a name to start a conversation and keep its files together."}</p>
        <label htmlFor="new-workspace-name">Workspace name</label>
        <input id="new-workspace-name" value={name} onChange={(event) => setName(event.target.value)} placeholder="e.g. CO2 copper catalysts" disabled={busy} required />
        <button type="submit" className="v2-primary-btn" disabled={busy || !name.trim()}>{busy ? "Creating…" : "Create workspace"}</button>
      </form>
    </div>
  );
}

function AuthPanel({ onReady, registrationEnabled = true }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [captcha, setCaptcha] = useState(null);
  const [captchaAnswer, setCaptchaAnswer] = useState("");
  const [error, setError] = useState("");
  const isRegister = registrationEnabled && mode === "register";

  useEffect(() => {
    if (!isRegister) return;
    apiFetch("/api/auth/captcha").then(setCaptcha).catch(setError);
  }, [isRegister]);

  useEffect(() => {
    if (!registrationEnabled && mode !== "login") {
      setMode("login");
      setCaptcha(null);
      setCaptchaAnswer("");
      setError("");
    }
  }, [mode, registrationEnabled]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    try {
      const action = isRegister ? "register" : "login";
      const body = isRegister
        ? { username, password, captcha_id: captcha?.captcha_id || "", captcha_answer: captchaAnswer }
        : { username, password };
      await apiFetch(`/api/auth/${action}`, { method: "POST", body: JSON.stringify(body) });
      onReady();
    } catch (err) {
      setError(err);
    }
  }

  return (
    <main className="v2-auth">
      <form className="v2-auth-card" onSubmit={submit}>
        <h1>CatMaster</h1>
        <input aria-label="Username" value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Username" autoComplete="username" />
        <input aria-label="Password" value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" type="password" autoComplete={isRegister ? "new-password" : "current-password"} />
        {isRegister && captcha?.question ? (
          <label className="v2-captcha">
            <span>{captcha.question}</span>
            <input aria-label="Captcha answer" value={captchaAnswer} onChange={(event) => setCaptchaAnswer(event.target.value)} placeholder="Answer" />
          </label>
        ) : null}
        <ErrorNotice error={error} />
        <button type="submit" className="v2-primary-btn">
          {isRegister ? <UserPlus size={15} /> : <LogIn size={15} />}
          {isRegister ? "Register" : "Log in"}
        </button>
        {registrationEnabled ? (
          <button type="button" className="v2-link-btn" onClick={() => setMode(isRegister ? "login" : "register")}>
            {isRegister ? "Use existing account" : "Create account"}
          </button>
        ) : null}
      </form>
    </main>
  );
}

function PermissionModeToggle({ mode, disabled, onChange }) {
  const normalized = mode === "auto" ? "auto" : "hitl";
  return (
    <div className="v2-permission-toggle" aria-label="Permission mode">
      <button
        type="button"
        className={normalized === "hitl" ? "active" : ""}
        aria-pressed={normalized === "hitl"}
        disabled={disabled}
        onClick={() => onChange("hitl")}
        title="Review protected tool calls before they run"
      >
        <ShieldAlert size={15} />
        Review
      </button>
      <button
        type="button"
        className={normalized === "auto" ? "active" : ""}
        aria-pressed={normalized === "auto"}
        disabled={disabled}
        onClick={() => onChange("auto")}
        title="Automatically approve protected tool calls"
      >
        <ShieldCheck size={15} />
        Auto
      </button>
    </div>
  );
}

function EntryPointPicker({ value, entrypoints, disabled, onChange }) {
  const rows = normalizedEntrypoints(entrypoints);
  const selected = entrypointMeta(value, rows);
  return (
    <label className="v2-entrypoint-picker" title={selected.summary}>
      <span>
        <Workflow size={14} />
        Agent
      </span>
      <select
        value={selected.id}
        disabled={disabled}
        onChange={(event) => onChange(event.target.value)}
        aria-label="Thread entry point"
      >
        {rows.map((item) => (
          <option key={item.id} value={item.id}>{item.label}</option>
        ))}
      </select>
    </label>
  );
}

function writeSelectionHash(selection, activeTab) {
  if (typeof window === "undefined") return;
  const current = String(window.location.hash || "");
  const nextHash = selectionToHash(selection, activeTab);
  if (current === nextHash) return;
  const nextUrl = `${window.location.pathname}${window.location.search}${nextHash}`;
  window.history.replaceState(null, "", nextUrl);
}

function activeThreadStorageKey(workspaceName) {
  return `catmaster:v2:active-thread:${String(workspaceName || "").trim()}`;
}

function rememberedWorkspaceThreadId(workspaceName) {
  if (typeof window === "undefined" || !String(workspaceName || "").trim()) return "";
  try {
    return window.sessionStorage.getItem(activeThreadStorageKey(workspaceName)) || "";
  } catch {
    return "";
  }
}

function rememberWorkspaceThreadId(workspaceName, threadId) {
  if (
    typeof window === "undefined"
    || !String(workspaceName || "").trim()
    || !String(threadId || "").trim()
  ) return;
  try {
    window.sessionStorage.setItem(activeThreadStorageKey(workspaceName), String(threadId));
  } catch {
    // Browsers may disable session storage; selection still works for this render.
  }
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function ColumnResizeHandle({
  className = "",
  label,
  value,
  min,
  max,
  expandToward = "left",
  onResize,
  onResizeValue,
}) {
  const [dragging, setDragging] = useState(false);

  function startResize(event) {
    event.preventDefault();
    setDragging(true);
    document.body.classList.add("v2-resizing-columns");
    const move = (moveEvent) => onResize(moveEvent);
    const stop = () => {
      setDragging(false);
      document.body.classList.remove("v2-resizing-columns");
      window.removeEventListener("pointermove", move);
      window.removeEventListener("pointerup", stop);
      window.removeEventListener("pointercancel", stop);
    };
    window.addEventListener("pointermove", move);
    window.addEventListener("pointerup", stop);
    window.addEventListener("pointercancel", stop);
  }

  function resizeWithKeyboard(event) {
    let next = null;
    if (event.key === "ArrowLeft") next = value + (expandToward === "left" ? 16 : -16);
    if (event.key === "ArrowRight") next = value + (expandToward === "right" ? 16 : -16);
    if (event.key === "Home") next = min;
    if (event.key === "End") next = max;
    if (next === null) return;
    event.preventDefault();
    onResizeValue(clampNumber(next, min, max));
  }

  return (
    <div
      className={`v2-resize-handle ${className} ${dragging ? "dragging" : ""}`}
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      aria-valuemin={min}
      aria-valuemax={max}
      aria-valuenow={Math.round(value)}
      aria-valuetext={`${Math.round(value)} pixels`}
      tabIndex={0}
      onPointerDown={startResize}
      onKeyDown={resizeWithKeyboard}
    />
  );
}

export default function CatMasterWorkspace({ boot }) {
  const [auth, setAuth] = useState(null);
  const [bootstrap, setBootstrap] = useState(null);
  const [threads, setThreads] = useState([]);
  const [activeThreadId, setActiveThreadId] = useState("");
  const [selection, setSelection] = useState(() => (typeof window === "undefined" ? null : selectionFromHash(window.location.hash)));
  const [activeTab, setActiveTab] = useState(() => (typeof window === "undefined" ? "chat" : tabFromHash(window.location.hash)));
  const [previewItem, setPreviewItem] = useState(null);
  const [previewBackItem, setPreviewBackItem] = useState(null);
  const [previewOpen, setPreviewOpen] = useState(false);
  const [railDrawerOpen, setRailDrawerOpen] = useState(false);
  const [planDrawerOpen, setPlanDrawerOpen] = useState(false);
  const [contextCollapsed, setContextCollapsed] = useState(false);
  const [railWidth, setRailWidth] = useState(() => {
    if (typeof window === "undefined") return 260;
    const saved = Number(window.localStorage.getItem("catmaster:v2:rail-width"));
    return Number.isFinite(saved) && saved > 0 ? clampNumber(saved, 250, 520) : 260;
  });
  const [planWidth, setPlanWidth] = useState(() => {
    if (typeof window === "undefined") return 300;
    const saved = Number(window.localStorage.getItem("catmaster:v2:plan-width"));
    return Number.isFinite(saved) && saved > 0 ? saved : 300;
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [researchActionBusy, setResearchActionBusy] = useState(false);
  const [selfEvolutionPayload, setSelfEvolutionPayload] = useState(null);
  const [selfEvolutionLoading, setSelfEvolutionLoading] = useState(false);
  const [selfEvolutionError, setSelfEvolutionError] = useState("");
  const selfEvolutionRequestKey = useRef(0);
  const railDrawerButtonRef = useRef(null);
  const planDrawerButtonRef = useRef(null);
  const drawerCloseButtonRef = useRef(null);

  const requestedProjectSpace = useMemo(() => {
    if (boot?.project_space) return String(boot.project_space);
    if (typeof window === "undefined") return "";
    return new URLSearchParams(window.location.search).get("project_space") || "";
  }, [boot?.project_space]);
  const workspaceName = bootstrap?.workspace_name || "";
  const selfEvolutionEnabled = auth?.self_evolution_available === true;
  const entrypoints = normalizedEntrypoints(bootstrap?.entrypoints);
  const activeThread = useMemo(
    () => threads.find((thread) => thread.thread_id === activeThreadId) || threads[0] || null,
    [threads, activeThreadId],
  );
  const activateThread = useCallback((threadId, targetWorkspace = workspaceName) => {
    const nextThreadId = String(threadId || "").trim();
    if (!nextThreadId) return;
    rememberWorkspaceThreadId(targetWorkspace, nextThreadId);
    setActiveThreadId(nextThreadId);
  }, [workspaceName]);
  const taskContextThread = useMemo(() => {
    if (!activeThread?.parent_thread_id) {
      return activeThread;
    }
    const parent = threads.find(
      (thread) => thread.thread_id === activeThread.parent_thread_id,
    );
    if (activeThread.thread_role === "agent_task") return parent || activeThread;
    if (
      activeThread.thread_role === "research_execution"
      && parent?.thread_role === "research_root"
    ) {
      return parent;
    }
    return activeThread;
  }, [activeThread, threads]);

  const updateThread = useCallback((thread) => {
    if (!thread?.thread_id) return;
    setThreads((prev) => {
      const next = [...prev];
      const index = next.findIndex((item) => item.thread_id === thread.thread_id);
      if (index >= 0) {
        const existing = next[index];
        const incomingActivity = thread?.research_activity;
        next[index] = {
          ...existing,
          ...thread,
          ...(
            existing?.thread_role === "research_root"
            && Number(incomingActivity?.updated_at || 0) <= 0
            && Number(existing?.research_activity?.updated_at || 0) > 0
              ? { research_activity: existing.research_activity }
              : {}
          ),
        };
      }
      else next.unshift(thread);
      return next;
    });
  }, []);

  const refreshThreads = useCallback(async () => {
    if (!workspaceName) return [];
    const payload = await apiFetch(
      `/api/workspaces/${encodeURIComponent(workspaceName)}/threads`,
    );
    const next = Array.isArray(payload.threads) ? payload.threads : [];
    setThreads(next);
    setActiveThreadId((current) => (
      current && next.some((thread) => thread.thread_id === current)
        ? current
        : preferredWorkspaceThreadId(next, rememberedWorkspaceThreadId(workspaceName))
    ));
    return next;
  }, [workspaceName]);

  const runtimeState = useCatMasterThreadRuntime({
    thread: activeThread,
    onThreadUpdate: updateThread,
    onSelectArtifact: (nextSelection) => handleSelection(nextSelection),
  });
  const researchActivity = activeThread?.thread_role === "research_root"
    ? (activeThread.research_activity || {})
    : null;
  const taskContextResearchActivity = taskContextThread?.thread_role === "research_root"
    ? (taskContextThread.research_activity || {})
    : null;
  const activeOperationParts = useMemo(() => {
    const rows = [
      ...(Array.isArray(runtimeState.activeParts) ? runtimeState.activeParts : []),
      ...(Array.isArray(researchActivity?.active_parts) ? researchActivity.active_parts : []),
    ];
    const seen = new Set();
    return rows.filter((part) => {
      const key = String(part?.id || "");
      if (!key || seen.has(key)) return false;
      seen.add(key);
      return true;
    });
  }, [researchActivity?.active_parts, runtimeState.activeParts]);
  const activeAsyncSubagents = useMemo(
    () => activeOperationParts.filter((part) => String(part?.type || "") === "subagent"),
    [activeOperationParts],
  );
  const researchGraphStreamIds = useMemo(() => [
    ...new Set(
      threads
        .filter((thread) => (
          thread.thread_role === "research_root"
          && (
            ACTIVE_RESEARCH_STATES.has(
              String(thread.research_activity?.state || "idle"),
            )
            || (
              String(thread.research_activity?.state || "idle") === "idle"
              && !thread.research_activity?.automation_paused
            )
          )
        ))
        .map((thread) => String(thread.active_research_graph_id || ""))
        .filter(Boolean),
    ),
  ].sort(), [threads]);
  const researchChildStreamIds = useMemo(() => [
    ...new Set(
      threads
        .filter((thread) => (
          thread.thread_role === "research_root"
          && ACTIVE_RESEARCH_STATES.has(
            String(thread.research_activity?.state || "idle"),
          )
        ))
        .map((thread) => String(thread.research_activity?.active_child_thread_id || ""))
        .filter(Boolean),
    ),
  ].sort(), [threads]);
  const taskContextRootStreamIds = useMemo(() => (
    taskContextThread?.thread_id
    && taskContextThread.thread_id !== activeThread?.thread_id
      ? [taskContextThread.thread_id]
      : []
  ), [activeThread?.thread_id, taskContextThread?.thread_id]);
  const todoGroups = useMemo(
    () => todoGroupsFromParts(runtimeState.todoParts),
    [runtimeState.todoParts],
  );

  useEffect(() => {
    function closeDrawers(event) {
      if (event.key !== "Escape") return;
      if (document.body.classList.contains("v2-preview-modal-open")) return;
      if (planDrawerOpen) {
        setPlanDrawerOpen(false);
        planDrawerButtonRef.current?.focus();
      } else if (railDrawerOpen) {
        setRailDrawerOpen(false);
        railDrawerButtonRef.current?.focus();
      }
    }
    function closeAtDesktop() {
      if (window.innerWidth >= 1200) {
        setRailDrawerOpen(false);
        setPlanDrawerOpen(false);
      }
    }
    window.addEventListener("keydown", closeDrawers);
    window.addEventListener("resize", closeAtDesktop);
    return () => {
      window.removeEventListener("keydown", closeDrawers);
      window.removeEventListener("resize", closeAtDesktop);
    };
  }, [planDrawerOpen, railDrawerOpen]);

  useEffect(() => {
    if (!railDrawerOpen && !planDrawerOpen) return undefined;
    const frame = window.requestAnimationFrame(() => drawerCloseButtonRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [planDrawerOpen, railDrawerOpen]);

  const checkAuthAndBootstrap = useCallback(async (projectSpace = "") => {
    setLoading(true);
    setError("");
    setSelfEvolutionPayload(null);
    setSelfEvolutionError("");
    try {
      const authStatus = await apiFetch("/api/auth/status");
      setAuth(authStatus);
      if (authStatus.auth_enabled && !authStatus.authenticated) {
        setLoading(false);
        return;
      }
      const params = projectSpace ? `?project_space=${encodeURIComponent(projectSpace)}` : "";
      const bootPayload = await apiFetch(`/api/bootstrap${params}`);
      setBootstrap(bootPayload);
      const ws = bootPayload.workspace_name || "";
      if (!ws) {
        setThreads([]);
        setActiveThreadId("");
        setSelection(null);
        setPreviewOpen(false);
        return;
      }
      const threadPayload = await apiFetch(`/api/workspaces/${encodeURIComponent(ws)}/threads`);
      let nextThreads = Array.isArray(threadPayload.threads) ? threadPayload.threads : [];
      if (!nextThreads.length) {
        const defaultEntrypoint = normalizeEntrypoint(bootPayload.default_entrypoint || DEFAULT_ENTRYPOINT, bootPayload.entrypoints);
        const created = await apiFetch(`/api/workspaces/${encodeURIComponent(ws)}/threads`, {
          method: "POST",
          body: JSON.stringify({ entrypoint: defaultEntrypoint }),
        });
        nextThreads = created.thread ? [created.thread] : [];
      }
      setThreads(nextThreads);
      setActiveThreadId((current) => (
        current && nextThreads.some((thread) => thread.thread_id === current)
          ? current
          : preferredWorkspaceThreadId(nextThreads, rememberedWorkspaceThreadId(ws))
      ));
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuthAndBootstrap(requestedProjectSpace);
  }, [checkAuthAndBootstrap, requestedProjectSpace]);

  useEffect(() => {
    if (!workspaceName || (
      !researchGraphStreamIds.length
      && !researchChildStreamIds.length
      && !taskContextRootStreamIds.length
    )) {
      return undefined;
    }
    const sources = [];
    let refreshTimer = null;
    const scheduleRefresh = () => {
      window.clearTimeout(refreshTimer);
      refreshTimer = window.setTimeout(() => {
        // EventSource reconnects on its own. A transient network break must not
        // replace the last durable research state with a task-failure banner.
        refreshThreads().catch(() => {});
      }, 100);
    };
    researchGraphStreamIds.forEach((graphId) => {
      const source = new EventSource(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graphId)}/stream`,
      );
      RESEARCH_GRAPH_ACTIVITY_EVENTS.forEach((eventName) => (
        source.addEventListener(eventName, scheduleRefresh)
      ));
      sources.push([source, RESEARCH_GRAPH_ACTIVITY_EVENTS]);
    });
    researchChildStreamIds.forEach((threadId) => {
      const source = new EventSource(
        `/api/threads/${encodeURIComponent(threadId)}/stream`,
      );
      RESEARCH_CHILD_ACTIVITY_EVENTS.forEach((eventName) => (
        source.addEventListener(eventName, scheduleRefresh)
      ));
      sources.push([source, RESEARCH_CHILD_ACTIVITY_EVENTS]);
    });
    taskContextRootStreamIds.forEach((threadId) => {
      const source = new EventSource(
        `/api/threads/${encodeURIComponent(threadId)}/stream`,
      );
      NATIVE_ROOT_ACTIVITY_EVENTS.forEach((eventName) => (
        source.addEventListener(eventName, scheduleRefresh)
      ));
      sources.push([source, NATIVE_ROOT_ACTIVITY_EVENTS]);
    });
    return () => {
      window.clearTimeout(refreshTimer);
      sources.forEach(([source, eventNames]) => {
        eventNames.forEach((eventName) => source.removeEventListener(eventName, scheduleRefresh));
        source.close();
      });
    };
  }, [
    refreshThreads,
    researchChildStreamIds.join("|"),
    researchGraphStreamIds.join("|"),
    taskContextRootStreamIds.join("|"),
    workspaceName,
  ]);

  const refreshSelfEvolution = useCallback(async () => {
    if (!selfEvolutionEnabled || !bootstrap?.ctx || !workspaceName) {
      selfEvolutionRequestKey.current += 1;
      setSelfEvolutionPayload(null);
      setSelfEvolutionError("");
      setSelfEvolutionLoading(false);
      return;
    }
    const requestKey = selfEvolutionRequestKey.current + 1;
    selfEvolutionRequestKey.current = requestKey;
    setSelfEvolutionLoading(true);
    setSelfEvolutionError("");
    try {
      const payload = await apiFetch(`/api/session/${encodeURIComponent(bootstrap.ctx)}/self-evolution/candidates?project_space=${encodeURIComponent(workspaceName)}`);
      if (selfEvolutionRequestKey.current === requestKey) setSelfEvolutionPayload(payload);
    } catch (err) {
      if (selfEvolutionRequestKey.current === requestKey) setSelfEvolutionError(err.message || String(err));
    } finally {
      if (selfEvolutionRequestKey.current === requestKey) setSelfEvolutionLoading(false);
    }
  }, [bootstrap?.ctx, selfEvolutionEnabled, workspaceName]);

  useEffect(() => {
    refreshSelfEvolution();
  }, [refreshSelfEvolution]);

  useEffect(() => {
    if (auth && !selfEvolutionEnabled && activeTab === "evolution") setActiveTab("chat");
  }, [auth, activeTab, selfEvolutionEnabled]);

  useEffect(() => {
    const restoreSelection = () => {
      const next = selectionFromHash(window.location.hash);
      setSelection(next || null);
      setActiveTab(tabFromHash(window.location.hash));
    };
    restoreSelection();
    window.addEventListener("hashchange", restoreSelection);
    return () => window.removeEventListener("hashchange", restoreSelection);
  }, []);

  useEffect(() => {
    writeSelectionHash(selection, activeTab);
  }, [selection?.type, selection?.artifact_id, selection?.path, activeTab]);

  async function createThread() {
    if (!workspaceName) return;
    const entrypoint = newThreadEntrypoint(
      activeThread?.entrypoint,
      bootstrap?.default_entrypoint || DEFAULT_ENTRYPOINT,
      entrypoints,
    );
    try {
      const payload = await apiFetch(`/api/workspaces/${encodeURIComponent(workspaceName)}/threads`, {
        method: "POST",
        body: JSON.stringify({ entrypoint }),
      });
      if (payload.thread) {
        setThreads((prev) => [payload.thread, ...prev]);
        activateThread(payload.thread.thread_id);
        setActiveTab("chat");
        setRailDrawerOpen(false);
        handleSelection(null);
      }
    } catch (err) {
      setError(err);
    }
  }

  async function openThread(threadId) {
    if (!threadId) return;
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(threadId)}`);
      if (payload.thread) updateThread(payload.thread);
      activateThread(threadId);
      setActiveTab("chat");
    } catch (err) {
      setError(err);
    }
  }

  async function renameThread(threadId, title) {
    const nextTitle = String(title || "").trim();
    if (!threadId || !nextTitle) throw new Error("Thread name cannot be empty.");
    const payload = await apiFetch(`/api/threads/${encodeURIComponent(threadId)}`, {
      method: "PATCH",
      body: JSON.stringify({ title: nextTitle }),
    });
    if (!payload.thread) throw new Error("The thread was renamed, but the updated thread was not returned.");
    updateThread(payload.thread);
    return payload.thread;
  }

  async function createWorkspace(requestedName) {
    if (!bootstrap?.ctx) return;
    const name = typeof requestedName === "string" ? requestedName : window.prompt("New workspace name");
    if (!name?.trim()) return;
    try {
      const payload = await apiFetch(`/api/session/${encodeURIComponent(bootstrap.ctx)}/workspace/create`, {
        method: "POST",
        body: JSON.stringify({ workspace: name.trim() }),
      });
      if (payload.ok === false) throw new Error(payload.status_message || "Workspace create failed.");
      await checkAuthAndBootstrap(name.trim());
    } catch (err) {
      setError(err);
    }
  }

  async function deleteWorkspace(defaultName = "") {
    if (!bootstrap?.ctx) return;
    const name = window.prompt("Workspace name to delete. The active workspace cannot be deleted.", defaultName || "");
    if (!name?.trim()) return;
    if (name.trim() === workspaceName) {
      setError("Switch to another workspace before deleting the active workspace.");
      return;
    }
    const confirmed = window.prompt(`Type ${name.trim()} to confirm deletion`);
    if (confirmed !== name.trim()) return;
    try {
      await apiFetch(`/api/session/${encodeURIComponent(bootstrap.ctx)}/workspace/delete`, {
        method: "DELETE",
        body: JSON.stringify({ workspace: name.trim(), confirm_name: confirmed, active_workspace: workspaceName }),
      });
      await checkAuthAndBootstrap("");
    } catch (err) {
      setError(err);
    }
  }

  async function logout() {
    await apiFetch("/api/auth/logout", { method: "POST", body: "{}" });
    setAuth(null);
    setBootstrap(null);
    setThreads([]);
    setActiveThreadId("");
    checkAuthAndBootstrap("");
  }

  function previewItemFromSelection(nextSelection) {
    if (nextSelection?.type === "agent" && nextSelection.part?.detail_ref) {
      return { id: `agent:${nextSelection.part.detail_ref}`, type: "agent", title: nextSelection.part.title,
        part: nextSelection.part, steer: Boolean(nextSelection.steer), showInstructions: Boolean(nextSelection.showInstructions) };
    }
    if (nextSelection?.type === "file" && nextSelection.path) {
      return {
        id: `file:${nextSelection.path}`,
        type: "file",
        path: nextSelection.path,
        title: nextSelection.node?.name || nextSelection.preview?.name || nextSelection.path,
        preview: nextSelection.preview || null,
      };
    }
    if (nextSelection?.type === "artifact" && (nextSelection.artifact_id || nextSelection.path)) {
      const artifact = artifactForSelection(nextSelection, runtimeState.artifacts) || nextSelection.artifact || null;
      const artifactId = nextSelection.artifact_id || artifact?.artifact_id || "";
      const artifactPath = artifact?.path || nextSelection.path || "";
      return {
        id: `artifact:${artifactId || artifactPath}`,
        type: "artifact",
        artifact_id: artifactId,
        path: artifactPath,
        title: userFacingFileTitle(artifact?.title, artifactPath, "Artifact"),
        artifact,
      };
    }
    if (nextSelection?.type === "activity" && nextSelection.part) {
      const part = nextSelection.part;
      const stableKey = part.id || `${part.type || "activity"}:${part.title || "details"}:${part.path || ""}`;
      return {
        id: `activity:${stableKey}`,
        type: "activity",
        title: displayValue(part.title, "Activity details"),
        part,
      };
    }
    return null;
  }

  function openPreview(nextSelection) {
    const item = previewItemFromSelection(nextSelection);
    if (!item) return;
    if (previewItem?.type === "agent" && item.type !== "agent") setPreviewBackItem(previewItem);
    if (item.type === "agent") setPreviewBackItem(null);
    setPreviewItem((current) => (
      current?.id === item.id
        ? { ...current, ...item, preview: item.preview || current.preview || null }
        : item
    ));
    setPreviewOpen(true);
  }

  function handleSelection(nextSelection) {
    if (["file", "artifact", "activity", "agent"].includes(nextSelection?.type)) {
      openPreview(nextSelection);
    } else if (!nextSelection) {
      setPreviewOpen(false);
      setPreviewItem(null);
    }
    setSelection(nextSelection || null);
  }

  function closePreview() {
    setPreviewBackItem(null);
    setPreviewOpen(false);
    setPreviewItem(null);
    setSelection(null);
  }

  useEffect(() => {
    if (["file", "artifact", "activity", "agent"].includes(selection?.type)) {
      openPreview(selection);
    } else if (!selection) {
      setPreviewOpen(false);
      setPreviewItem(null);
    }
  }, [selection?.type, selection?.path, selection?.artifact_id, selection?.part?.id, runtimeState.artifacts.length]);

  function selectFile(node) {
    if (!node) return;
    handleSelection({ type: "file", path: node.path || "", node });
    setActiveTab("chat");
  }

  const hasInterrupt = runtimeState.messages.some((message) => (
    Array.isArray(message.parts) && message.parts.some((part) => part.type === "interrupt" && part.status !== "resolved")
  ));
  const permissionMode = activeThread?.permission_mode === "hitl" ? "hitl" : "auto";
  const selectedEntrypoint = normalizeEntrypoint(activeThread?.entrypoint || bootstrap?.default_entrypoint || DEFAULT_ENTRYPOINT, entrypoints);
  const planVisible = Boolean(workspaceName) && activeTab === "chat";
  const pendingEvolutionCount = Number(selfEvolutionPayload?.attention_count || 0);

  const planMaxWidth = Math.max(300, Math.min(760, (typeof window === "undefined" ? 1366 : window.innerWidth) - 620));

  function setPlanWidthPersisted(value) {
    const next = clampNumber(value, 280, planMaxWidth);
    setPlanWidth(next);
    window.localStorage.setItem("catmaster:v2:plan-width", String(Math.round(next)));
  }

  function resizePlan(event) {
    setPlanWidthPersisted(window.innerWidth - event.clientX);
  }

  function setRailWidthPersisted(value) {
    const next = clampNumber(value, 250, 520);
    setRailWidth(next);
    window.localStorage.setItem("catmaster:v2:rail-width", String(Math.round(next)));
  }

  function resizeRail(event) {
    setRailWidthPersisted(event.clientX);
  }

  async function updateEntrypoint(nextEntrypoint) {
    if (!activeThread?.thread_id) return;
    const normalized = normalizeEntrypoint(nextEntrypoint, entrypoints);
    if (normalized === selectedEntrypoint) return;
    setError("");
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(activeThread.thread_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ entrypoint: normalized }),
      });
      if (payload.thread) updateThread(payload.thread);
    } catch (err) {
      setError(err);
    }
  }

  async function updatePermissionMode(nextMode) {
    if (!activeThread?.thread_id || nextMode === permissionMode) return;
    setError("");
    try {
      const payload = await apiFetch(`/api/threads/${encodeURIComponent(activeThread.thread_id)}`, {
        method: "PATCH",
        body: JSON.stringify({ permission_mode: nextMode }),
      });
      if (payload.thread) updateThread(payload.thread);
    } catch (err) {
      setError(err);
    }
  }

  function openResearchSessionGraph(root = activeThread) {
    if (root?.thread_id) activateThread(root.thread_id);
    setActiveTab("hypotheses");
    setRailDrawerOpen(false);
  }

  async function toggleResearchPause() {
    const threadId = taskContextThread?.thread_id;
    if (!threadId || researchActionBusy) return;
    setResearchActionBusy(true);
    setError("");
    try {
      const paused = taskContextResearchActivity?.automation_paused === true;
      await apiFetch(
        `/api/threads/${encodeURIComponent(threadId)}/${paused ? "submit" : "stop"}`,
        {
          method: "POST",
          body: JSON.stringify(paused ? {
            text: "Continue the existing research objective within the authorization already given. Reuse ongoing work and completed findings, and carry out the next useful actions until the requested deliverable is complete.",
            entrypoint: "persistent_research",
            strategy: "enqueue",
          } : { action: "interrupt", run_id: "" }),
        },
      );
      await refreshThreads();
    } catch (err) {
      setError(err);
    } finally {
      setResearchActionBusy(false);
    }
  }

  if (auth?.auth_enabled && !auth?.authenticated) {
    return (
      <AuthPanel
        onReady={() => checkAuthAndBootstrap(requestedProjectSpace)}
        registrationEnabled={auth?.registration_enabled === true}
      />
    );
  }

  return (
    <AssistantRuntimeProvider runtime={runtimeState.runtime}>
      <main
        className={`v2-shell v2-workspace tab-${activeTab} ${planVisible ? "has-plan" : ""} ${contextCollapsed ? "context-collapsed" : ""} ${railDrawerOpen ? "rail-drawer-open" : ""} ${planDrawerOpen ? "plan-drawer-open" : ""}`}
        style={{
          "--v2-rail-width": `${railWidth}px`,
          ...(planVisible ? { "--v2-plan-width": `${planWidth}px` } : {}),
        }}
      >
        <WorkspaceRail
          ctx={bootstrap?.ctx || ""}
          workspaceName={workspaceName}
          workspaceChoices={bootstrap?.workspaces || []}
          threads={threads}
          activeThreadId={activeThread?.thread_id || ""}
          activeView={activeTab}
          selfEvolutionEnabled={selfEvolutionEnabled}
          pendingEvolutionCount={pendingEvolutionCount}
          onViewChange={(view) => {
            setActiveTab(view);
            setRailDrawerOpen(false);
            if (view === "evolution") refreshSelfEvolution();
          }}
          onWorkspaceChange={(name) => checkAuthAndBootstrap(name)}
          onCreateWorkspace={createWorkspace}
          onDeleteWorkspace={deleteWorkspace}
          onCreateThread={createThread}
          onRenameThread={renameThread}
          onSelectThread={(threadId) => {
            activateThread(threadId);
            setActiveTab("chat");
            handleSelection(null);
            setRailDrawerOpen(false);
          }}
          onOpenResearchDecisions={(root) => openResearchSessionGraph(root)}
          onSelectFile={(item) => {
            selectFile(item);
            setRailDrawerOpen(false);
          }}
        />
        <ColumnResizeHandle
          className="v2-resize-handle-rail"
          label="Resize workspace navigation"
          value={railWidth}
          min={250}
          max={520}
          expandToward="right"
          onResize={resizeRail}
          onResizeValue={setRailWidthPersisted}
        />
        <section className="v2-center">
          <header className="v2-topbar">
            <div className="v2-drawer-triggers">
              <button
                ref={railDrawerButtonRef}
                type="button"
                className="v2-icon-btn"
                aria-label="Open workspace navigation"
                aria-expanded={railDrawerOpen}
                onClick={() => setRailDrawerOpen(true)}
              >
                <Menu size={17} />
              </button>
              {planVisible ? (
                <button
                  ref={planDrawerButtonRef}
                  type="button"
                  className="v2-icon-btn"
                  aria-label="Open task context"
                  aria-expanded={planDrawerOpen}
                  aria-controls="task-context-panel"
                  onClick={() => setPlanDrawerOpen(true)}
                >
                  <PanelRight size={17} />
                  {activeAsyncSubagents.length + Number(taskContextThread?.pending_run_count || 0) > 0 ? (
                    <span className="v2-context-badge active">
                      {activeAsyncSubagents.length + Number(taskContextThread?.pending_run_count || 0)}
                    </span>
                  ) : null}
                </button>
              ) : null}
            </div>
            <div className="v2-page-heading">
              <div className="v2-eyebrow">{workspaceName || "Workspace"} <span>/</span> {activeTab === "chat" ? "Conversation" : "Workspace"}</div>
              <h1 title={activeThread?.title}>{!workspaceName ? "Your workspaces" : activeTab === "chat" ? (activeThread?.title || "New conversation") : ({ monitor: "Monitor", hypotheses: "Research Graph", evolution: "Skill Evolution", files: "Files" }[activeTab])}</h1>
            </div>
            {workspaceName && activeTab === "chat" ? (
              <div className="v2-thread-status-strip">
                <span className={`v2-turn-status status-${activeThread?.status || "idle"}`} title={`Conversation status: ${threadStatusLabel(activeThread?.status)}`}>
                  {threadStatusLabel(activeThread?.status)}
                </span>
                {researchActivity ? (
                  <span title="Research Session background status">
                    Research: {researchStateLabel(researchActivity.state)}
                  </span>
                ) : null}
                {Number(activeThread?.pending_run_count || 0) > 0 ? (
                  <span title="Queued turns">{activeThread.pending_run_count} queued</span>
                ) : null}
              </div>
            ) : null}
            <div className="v2-topbar-actions">
              <button type="button" className="v2-icon-btn" aria-label="Refresh workspace" title="Refresh workspace" onClick={() => checkAuthAndBootstrap(workspaceName)}>
                <RefreshCw size={15} />
              </button>
              {planVisible ? (
                <button type="button" className="v2-icon-btn v2-context-toggle" aria-label={contextCollapsed ? "Show task context" : "Hide task context"} title={contextCollapsed ? "Show task context" : "Hide task context"} aria-expanded={!contextCollapsed} aria-controls="task-context-panel" onClick={() => setContextCollapsed((value) => !value)}>
                  <PanelRight size={17} />
                </button>
              ) : null}
              {auth?.auth_enabled ? (
                <button type="button" className="v2-ghost-btn" onClick={logout}>
                  <LogOut size={15} />
                  Logout
                </button>
              ) : null}
            </div>
          </header>
          <ErrorNotice error={error} />
          {!workspaceName ? (
            loading ? <div className="v2-empty">Loading workspaces…</div> :
            <WorkspaceEmptyState hasWorkspaces={Boolean(bootstrap?.workspaces?.length)} onCreate={createWorkspace} />
          ) : <>
          {activeTab === "chat" ? (
            <>
              <div className="v2-thread-scroll">
                <ThreadMessages
                  threadId={activeThread?.thread_id || ""}
                  messages={runtimeState.messages}
                  loading={loading || runtimeState.loading}
                  error={runtimeState.error}
                  onSelect={handleSelection}
                  onResume={runtimeState.resume}
                  onContinueFromCheckpoint={runtimeState.continueFromCheckpoint}
                  hasMore={Boolean(runtimeState.messagePage?.truncated)}
                  onLoadOlder={runtimeState.loadOlderMessages}
                  loadingOlder={runtimeState.loadingOlder}
                  todoParts={runtimeState.todoParts}
                />
              </div>
              <div className="v2-chat-controls">
                <ActiveOperationsPanel
                  parts={activeOperationParts.filter((part) => !(part.actions || []).some((action) => action.id === "update_async_subagent"))}
                  isRunning={runtimeState.isRunning}
                  pendingRuns={activeThread?.pending_run_count || 0}
                  onSelect={handleSelection}
                />
                <ThreadComposer
                  thread={activeThread}
                  isRunning={runtimeState.isRunning}
                  hasInterrupt={hasInterrupt}
                  onSubmit={runtimeState.submitText}
                  onStop={runtimeState.stop}
                  controls={(
                    <>
                      <EntryPointPicker value={selectedEntrypoint} entrypoints={entrypoints} disabled={!activeThread?.thread_id || runtimeState.isRunning} onChange={updateEntrypoint} />
                      <PermissionModeToggle mode={permissionMode} disabled={!activeThread?.thread_id || runtimeState.isRunning} onChange={updatePermissionMode} />
                    </>
                  )}
                />
              </div>
            </>
          ) : null}
          {activeTab === "monitor" ? <MonitorPanel ctx={bootstrap?.ctx || ""} workspaceName={workspaceName} thread={activeThread} entrypoint={selectedEntrypoint} events={runtimeState.events} /> : null}
          {activeTab === "hypotheses" ? (
            <Suspense fallback={<div className="v2-empty">Loading Research Graph workspace…</div>}>
              <ResearchTechTreePanel
                workspaceName={workspaceName}
                thread={activeThread}
                onOpenThread={openThread}
                onThreadUpdate={updateThread}
                onOpenReference={handleSelection}
              />
            </Suspense>
          ) : null}
          {activeTab === "evolution" && selfEvolutionEnabled ? (
            <SelfEvolutionPanel
              ctx={bootstrap?.ctx || ""}
              workspaceName={workspaceName}
              payload={selfEvolutionPayload}
              loading={selfEvolutionLoading}
              error={selfEvolutionError}
              onRefresh={refreshSelfEvolution}
            />
          ) : null}
          {activeTab === "files" ? (
            <FilesPanel
              ctx={bootstrap?.ctx || ""}
              workspaceName={workspaceName}
              selectedFilePath={selection?.type === "file" ? selection.path : ""}
              onSelectFile={(item) => handleSelection({ type: "file", path: item.path || "", preview: item.preview })}
            />
          ) : null}
          </>}
        </section>
        {planVisible ? (
          <ColumnResizeHandle
            className="v2-resize-handle-plan"
            label="Resize task context"
            value={planWidth}
            min={280}
            max={planMaxWidth}
            onResize={resizePlan}
            onResizeValue={setPlanWidthPersisted}
          />
        ) : null}
        {planVisible ? (
          <PlanPanel
            groups={todoGroups}
            artifacts={runtimeState.artifacts}
            onSelectArtifact={handleSelection}
            thread={taskContextThread}
            subagents={activeAsyncSubagents}
            onAsyncSubagentUpdate={runtimeState.updateAsyncSubagent}
            onAsyncSubagentStop={runtimeState.stopAsyncSubagent}
            researchActivity={taskContextResearchActivity}
            researchSessionProps={{
              onOpenCurrent: () => openThread(taskContextResearchActivity?.active_child_thread_id),
              onOpenGraph: () => openResearchSessionGraph(taskContextThread),
              onOpenLatestReport: () => {
                if (taskContextResearchActivity?.latest_report_artifact_id) {
                  handleSelection({
                    type: "artifact",
                    artifact_id: taskContextResearchActivity.latest_report_artifact_id,
                    path: taskContextResearchActivity.latest_report_path || "",
                  });
                } else if (taskContextResearchActivity?.latest_report_path) {
                  handleSelection({
                    type: "file",
                    path: taskContextResearchActivity.latest_report_path,
                  });
                }
              },
              onTogglePause: toggleResearchPause,
              canTogglePause: Boolean(
                taskContextThread?.active_research_graph_id
                && taskContextResearchActivity?.state !== "completed"
              ),
              busy: researchActionBusy,
            }}
          />
        ) : null}
        {(railDrawerOpen || planDrawerOpen) ? (
          <>
            <button
              type="button"
              className="v2-drawer-backdrop"
              aria-label="Close open panel"
              onClick={() => {
                const returnToPlan = planDrawerOpen;
                setRailDrawerOpen(false);
                setPlanDrawerOpen(false);
                window.requestAnimationFrame(() => (
                  returnToPlan
                    ? planDrawerButtonRef.current?.focus()
                    : railDrawerButtonRef.current?.focus()
                ));
              }}
            />
            <button
              ref={drawerCloseButtonRef}
              type="button"
              className="v2-drawer-close"
              aria-label="Close open panel"
              onClick={() => {
              const returnToPlan = planDrawerOpen;
              setRailDrawerOpen(false);
              setPlanDrawerOpen(false);
              window.requestAnimationFrame(() => (
                returnToPlan
                  ? planDrawerButtonRef.current?.focus()
                  : railDrawerButtonRef.current?.focus()
              ));
            }}
            >
              <X size={18} />
            </button>
          </>
        ) : null}
        <ArtifactPreviewModal
          ctx={bootstrap?.ctx || ""}
          workspaceName={workspaceName}
          item={previewItem}
          open={previewOpen}
          onClose={closePreview}
          onSelect={handleSelection}
          backItem={previewBackItem}
          onBack={() => handleSelection({ type: "agent", part: previewBackItem.part })}
        />
      </main>
    </AssistantRuntimeProvider>
  );
}
