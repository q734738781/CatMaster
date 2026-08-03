import { lazy, Suspense, useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Files, GitBranch, LogIn, LogOut, Menu, MonitorDot, Network, PanelRight, RefreshCw, ShieldAlert, ShieldCheck, UserPlus, Workflow, X } from "lucide-react";

import WorkspaceRail from "./components/WorkspaceRail";
import ThreadMessages from "./components/ThreadMessages";
import ThreadComposer from "./components/ThreadComposer";
import FilePreviewTabs from "./components/FilePreviewTabs";
import { FilesPanel, MonitorPanel, SelfEvolutionPanel } from "./components/WorkspacePanels";
import { apiFetch, useCatMasterThreadRuntime } from "./useCatMasterThreadRuntime";
import { DEFAULT_ENTRYPOINT, entrypointMeta, normalizedEntrypoints, normalizeEntrypoint } from "./entrypoints";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection";
import { artifactForSelection } from "./artifactSelection.js";
import { todoGroupsFromParts } from "./todoPanel.js";
import { displayValue, presentError, userFacingFileTitle } from "./presentation.js";

const ResearchTechTreePanel = lazy(
  () => import("./components/ResearchTechTreePanel"),
);

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
    idle: "Ready",
    created: "Ready",
    queued: "Queued",
    pending: "Waiting",
    running: "Running",
    stopping: "Stopping",
    interrupted: "Waiting for review",
    completed: "Completed",
    failed: "Needs attention",
  }[status] || displayValue(status.replace(/[_-]+/g, " "), "Ready");
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
        Entry
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
      <small>{selected.summary}</small>
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

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function ColumnResizeHandle({
  className = "",
  label,
  value,
  min,
  max,
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
    if (event.key === "ArrowLeft") next = value + 16;
    if (event.key === "ArrowRight") next = value - 16;
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
  const [previewTabs, setPreviewTabs] = useState([]);
  const [activePreviewTabId, setActivePreviewTabId] = useState("");
  const [railDrawerOpen, setRailDrawerOpen] = useState(false);
  const [inspectorDrawerOpen, setInspectorDrawerOpen] = useState(false);
  const [inspectorWidth, setInspectorWidth] = useState(() => {
    if (typeof window === "undefined") return 360;
    const saved = Number(window.localStorage.getItem("catmaster:v2:inspector-width"));
    return Number.isFinite(saved) && saved > 0 ? saved : 360;
  });
  const [error, setError] = useState("");
  const [loading, setLoading] = useState(true);
  const [selfEvolutionPayload, setSelfEvolutionPayload] = useState(null);
  const [selfEvolutionLoading, setSelfEvolutionLoading] = useState(false);
  const [selfEvolutionError, setSelfEvolutionError] = useState("");
  const selfEvolutionRequestKey = useRef(0);
  const railDrawerButtonRef = useRef(null);
  const inspectorDrawerButtonRef = useRef(null);
  const drawerCloseButtonRef = useRef(null);

  const requestedProjectSpace = useMemo(() => {
    if (boot?.project_space) return String(boot.project_space);
    if (typeof window === "undefined") return "";
    return new URLSearchParams(window.location.search).get("project_space") || "";
  }, [boot?.project_space]);
  const workspaceName = bootstrap?.workspace_name || "";
  const selfEvolutionEnabled = auth?.auth_enabled === true;
  const entrypoints = normalizedEntrypoints(bootstrap?.entrypoints);
  const activeThread = useMemo(
    () => threads.find((thread) => thread.thread_id === activeThreadId) || threads[0] || null,
    [threads, activeThreadId],
  );

  const updateThread = useCallback((thread) => {
    if (!thread?.thread_id) return;
    setThreads((prev) => {
      const next = [...prev];
      const index = next.findIndex((item) => item.thread_id === thread.thread_id);
      if (index >= 0) next[index] = { ...next[index], ...thread };
      else next.unshift(thread);
      return next;
    });
  }, []);

  const runtimeState = useCatMasterThreadRuntime({
    thread: activeThread,
    onThreadUpdate: updateThread,
    onSelectArtifact: (nextSelection) => handleSelection(nextSelection),
  });
  const todoGroups = useMemo(
    () => todoGroupsFromParts(runtimeState.todoParts),
    [runtimeState.todoParts],
  );

  useEffect(() => {
    function closeDrawers(event) {
      if (event.key !== "Escape") return;
      if (inspectorDrawerOpen) {
        setInspectorDrawerOpen(false);
        inspectorDrawerButtonRef.current?.focus();
      } else if (railDrawerOpen) {
        setRailDrawerOpen(false);
        railDrawerButtonRef.current?.focus();
      }
    }
    function closeAtDesktop() {
      if (window.innerWidth >= 1200) {
        setRailDrawerOpen(false);
        setInspectorDrawerOpen(false);
      }
    }
    window.addEventListener("keydown", closeDrawers);
    window.addEventListener("resize", closeAtDesktop);
    return () => {
      window.removeEventListener("keydown", closeDrawers);
      window.removeEventListener("resize", closeAtDesktop);
    };
  }, [inspectorDrawerOpen, railDrawerOpen]);

  useEffect(() => {
    if (!railDrawerOpen && !inspectorDrawerOpen) return undefined;
    const frame = window.requestAnimationFrame(() => drawerCloseButtonRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [inspectorDrawerOpen, railDrawerOpen]);

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
      const ws = bootPayload.workspace_name || projectSpace || "default";
      const threadPayload = await apiFetch(`/api/workspaces/${encodeURIComponent(ws)}/threads`);
      let nextThreads = Array.isArray(threadPayload.threads) ? threadPayload.threads : [];
      if (!nextThreads.length) {
        const defaultEntrypoint = normalizeEntrypoint(bootPayload.default_entrypoint || DEFAULT_ENTRYPOINT, bootPayload.entrypoints);
        const created = await apiFetch(`/api/workspaces/${encodeURIComponent(ws)}/threads`, {
          method: "POST",
          body: JSON.stringify({ title: "New thread", entrypoint: defaultEntrypoint }),
        });
        nextThreads = created.thread ? [created.thread] : [];
      }
      setThreads(nextThreads);
      setActiveThreadId((current) => current && nextThreads.some((thread) => thread.thread_id === current) ? current : (nextThreads[0]?.thread_id || ""));
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }, []);

  useEffect(() => {
    checkAuthAndBootstrap(requestedProjectSpace);
  }, [checkAuthAndBootstrap, requestedProjectSpace]);

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
      if (next) setSelection(next);
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
    const entrypoint = normalizeEntrypoint(activeThread?.entrypoint || bootstrap?.default_entrypoint || DEFAULT_ENTRYPOINT, entrypoints);
    try {
      const payload = await apiFetch(`/api/workspaces/${encodeURIComponent(workspaceName)}/threads`, {
        method: "POST",
        body: JSON.stringify({ title: "New thread", entrypoint }),
      });
      if (payload.thread) {
        setThreads((prev) => [payload.thread, ...prev]);
        setActiveThreadId(payload.thread.thread_id);
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
      setActiveThreadId(threadId);
      setActiveTab("chat");
    } catch (err) {
      setError(err);
    }
  }

  async function createWorkspace() {
    if (!bootstrap?.ctx) return;
    const name = window.prompt("New workspace name");
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

  function previewTabFromSelection(nextSelection) {
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

  function openPreviewTab(nextSelection) {
    const tab = previewTabFromSelection(nextSelection);
    if (!tab) return;
    setPreviewTabs((prev) => {
      const index = prev.findIndex((item) => item.id === tab.id);
      if (index < 0) return [...prev, tab];
      const next = [...prev];
      next[index] = { ...next[index], ...tab, preview: tab.preview || next[index].preview || null };
      return next;
    });
    setActivePreviewTabId(tab.id);
  }

  function handleSelection(nextSelection) {
    if (["file", "artifact", "activity"].includes(nextSelection?.type)) {
      openPreviewTab(nextSelection);
      if (typeof window !== "undefined" && window.innerWidth < 1200) {
        setInspectorDrawerOpen(true);
      }
    }
    setSelection(nextSelection || null);
  }

  function closePreviewTab(tabId) {
    setPreviewTabs((prev) => {
      const next = prev.filter((tab) => tab.id !== tabId);
      if (activePreviewTabId === tabId) {
        setActivePreviewTabId(next[next.length - 1]?.id || "");
      }
      return next;
    });
  }

  useEffect(() => {
    if (["file", "artifact", "activity"].includes(selection?.type)) {
      openPreviewTab(selection);
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
  const inspectorVisible = activeTab === "chat";
  const pendingEvolutionCount = Number(selfEvolutionPayload?.pending_review_count || 0);

  const inspectorMaxWidth = Math.max(300, Math.min(760, (typeof window === "undefined" ? 1366 : window.innerWidth) - 620));

  function setInspectorWidthPersisted(value) {
    const next = clampNumber(value, 280, inspectorMaxWidth);
    setInspectorWidth(next);
    window.localStorage.setItem("catmaster:v2:inspector-width", String(Math.round(next)));
  }

  function resizeInspector(event) {
    setInspectorWidthPersisted(window.innerWidth - event.clientX);
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
        className={`v2-shell v2-workspace tab-${activeTab} ${inspectorVisible ? "has-inspector" : ""} ${railDrawerOpen ? "rail-drawer-open" : ""} ${inspectorDrawerOpen ? "inspector-drawer-open" : ""}`}
        style={inspectorVisible ? { "--v2-inspector-width": `${inspectorWidth}px` } : undefined}
      >
        <WorkspaceRail
          ctx={bootstrap?.ctx || ""}
          workspaceName={workspaceName}
          workspaceChoices={bootstrap?.workspaces || []}
          threads={threads}
          activeThreadId={activeThread?.thread_id || ""}
          onWorkspaceChange={(name) => checkAuthAndBootstrap(name)}
          onCreateWorkspace={createWorkspace}
          onDeleteWorkspace={deleteWorkspace}
          onCreateThread={createThread}
          onSelectThread={(threadId) => {
            setActiveThreadId(threadId);
            handleSelection(null);
            setRailDrawerOpen(false);
          }}
          onSelectFile={(item) => {
            selectFile(item);
            setRailDrawerOpen(false);
          }}
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
              {inspectorVisible ? (
                <button
                  ref={inspectorDrawerButtonRef}
                  type="button"
                  className="v2-icon-btn"
                  aria-label="Open file inspector"
                  aria-expanded={inspectorDrawerOpen}
                  onClick={() => setInspectorDrawerOpen(true)}
                >
                  <PanelRight size={17} />
                </button>
              ) : null}
            </div>
            <div>
              <div className="v2-eyebrow">CatMaster Workspace</div>
              <h1>{activeThread?.title || "Thread"}</h1>
              <div className="v2-thread-status-strip">
                <span title={`Conversation status: ${threadStatusLabel(activeThread?.status)}`}>
                  {threadStatusLabel(activeThread?.status)}
                </span>
                <span>{entrypoints.find((item) => item.id === selectedEntrypoint)?.label || "Research"}</span>
                <span>{activeThread?.active_research_graph_id ? "Research graph attached" : "No research graph"}</span>
                <span>{runtimeState.artifacts.length} artifacts</span>
              </div>
              <nav className="v2-workspace-tabs" aria-label="Workspace tabs" role="tablist">
                <button type="button" role="tab" aria-selected={activeTab === "chat"} className={activeTab === "chat" ? "active" : ""} onClick={() => setActiveTab("chat")}>Chat</button>
                <button type="button" role="tab" aria-selected={activeTab === "monitor"} className={activeTab === "monitor" ? "active" : ""} onClick={() => setActiveTab("monitor")}><MonitorDot size={14} />Monitor</button>
                <button type="button" role="tab" aria-selected={activeTab === "hypotheses"} className={activeTab === "hypotheses" ? "active" : ""} onClick={() => setActiveTab("hypotheses")}><Network size={14} />Research Graph</button>
                {selfEvolutionEnabled ? (
                  <button type="button" role="tab" aria-selected={activeTab === "evolution"} className={activeTab === "evolution" ? "active" : ""} onClick={() => { setActiveTab("evolution"); refreshSelfEvolution(); }}>
                    <GitBranch size={14} />Skill Evolution
                    {pendingEvolutionCount > 0 ? <span className="v2-tab-badge">{pendingEvolutionCount}</span> : null}
                  </button>
                ) : null}
                <button type="button" role="tab" aria-selected={activeTab === "files"} className={activeTab === "files" ? "active" : ""} onClick={() => setActiveTab("files")}><Files size={14} />Files</button>
              </nav>
            </div>
            <div className="v2-topbar-actions">
              <EntryPointPicker
                value={selectedEntrypoint}
                entrypoints={entrypoints}
                disabled={!activeThread?.thread_id || runtimeState.isRunning}
                onChange={updateEntrypoint}
              />
              <PermissionModeToggle
                mode={permissionMode}
                disabled={!activeThread?.thread_id || runtimeState.isRunning}
                onChange={updatePermissionMode}
              />
              <button type="button" className="v2-ghost-btn" onClick={() => checkAuthAndBootstrap(workspaceName)}>
                <RefreshCw size={15} />
                Refresh
              </button>
              {auth?.auth_enabled ? (
                <button type="button" className="v2-ghost-btn" onClick={logout}>
                  <LogOut size={15} />
                  Logout
                </button>
              ) : null}
            </div>
          </header>
          <ErrorNotice error={error} />
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
                  hasMore={Boolean(runtimeState.messagePage?.truncated)}
                  onLoadOlder={runtimeState.loadOlderMessages}
                  loadingOlder={runtimeState.loadingOlder}
                  todoParts={runtimeState.todoParts}
                />
              </div>
              <ThreadComposer
                thread={activeThread}
                isRunning={runtimeState.isRunning}
                hasInterrupt={hasInterrupt}
                onSubmit={runtimeState.submitText}
              />
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
        </section>
        {inspectorVisible ? (
          <ColumnResizeHandle
            className="v2-resize-handle-inspector"
            label="Resize inspector"
            value={inspectorWidth}
            min={280}
            max={inspectorMaxWidth}
            onResize={resizeInspector}
            onResizeValue={setInspectorWidthPersisted}
          />
        ) : null}
        {inspectorVisible ? (
          <FilePreviewTabs
            ctx={bootstrap?.ctx || ""}
            workspaceName={workspaceName}
            tabs={previewTabs}
            activeTabId={activePreviewTabId}
            todoGroups={todoGroups}
            onActivate={setActivePreviewTabId}
            onClose={closePreviewTab}
          />
        ) : null}
        {(railDrawerOpen || inspectorDrawerOpen) ? (
          <>
            <button
              type="button"
              className="v2-drawer-backdrop"
              aria-label="Close open panel"
              onClick={() => {
                const returnToInspector = inspectorDrawerOpen;
                setRailDrawerOpen(false);
                setInspectorDrawerOpen(false);
                window.requestAnimationFrame(() => (
                  returnToInspector
                    ? inspectorDrawerButtonRef.current?.focus()
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
              const returnToInspector = inspectorDrawerOpen;
              setRailDrawerOpen(false);
              setInspectorDrawerOpen(false);
              window.requestAnimationFrame(() => (
                returnToInspector
                  ? inspectorDrawerButtonRef.current?.focus()
                  : railDrawerButtonRef.current?.focus()
              ));
            }}
            >
              <X size={18} />
            </button>
          </>
        ) : null}
      </main>
    </AssistantRuntimeProvider>
  );
}
