import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Files, GitBranch, LogIn, LogOut, MonitorDot, RefreshCw, ShieldAlert, ShieldCheck, UserPlus, Workflow } from "lucide-react";

import WorkspaceRail from "./components/WorkspaceRail";
import ThreadMessages from "./components/ThreadMessages";
import ThreadComposer from "./components/ThreadComposer";
import FilePreviewTabs from "./components/FilePreviewTabs";
import { FilesPanel, MonitorPanel, SelfEvolutionPanel } from "./components/WorkspacePanels";
import { apiFetch, useCatMasterThreadRuntime } from "./useCatMasterThreadRuntime";
import { DEFAULT_ENTRYPOINT, entrypointMeta, normalizedEntrypoints, normalizeEntrypoint } from "./entrypoints";
import { selectionFromHash, selectionToHash, tabFromHash } from "./inspectorSelection";
import { artifactForSelection } from "./artifactSelection.js";
import { todoGroupsFromMessages } from "./todoPanel.js";

function AuthPanel({ onReady }) {
  const [mode, setMode] = useState("login");
  const [username, setUsername] = useState("");
  const [password, setPassword] = useState("");
  const [captcha, setCaptcha] = useState(null);
  const [captchaAnswer, setCaptchaAnswer] = useState("");
  const [error, setError] = useState("");

  useEffect(() => {
    if (mode !== "register") return;
    apiFetch("/api/auth/captcha").then(setCaptcha).catch((err) => setError(err.message || String(err)));
  }, [mode]);

  async function submit(event) {
    event.preventDefault();
    setError("");
    try {
      const body = mode === "register"
        ? { username, password, captcha_id: captcha?.captcha_id || "", captcha_answer: captchaAnswer }
        : { username, password };
      await apiFetch(`/api/auth/${mode}`, { method: "POST", body: JSON.stringify(body) });
      onReady();
    } catch (err) {
      setError(err.message || String(err));
    }
  }

  return (
    <main className="v2-auth">
      <form className="v2-auth-card" onSubmit={submit}>
        <h1>CatMaster</h1>
        <input value={username} onChange={(event) => setUsername(event.target.value)} placeholder="Username" autoComplete="username" />
        <input value={password} onChange={(event) => setPassword(event.target.value)} placeholder="Password" type="password" autoComplete={mode === "register" ? "new-password" : "current-password"} />
        {mode === "register" && captcha?.question ? (
          <label className="v2-captcha">
            <span>{captcha.question}</span>
            <input value={captchaAnswer} onChange={(event) => setCaptchaAnswer(event.target.value)} placeholder="Answer" />
          </label>
        ) : null}
        {error ? <div className="v2-error">{error}</div> : null}
        <button type="submit" className="v2-primary-btn">
          {mode === "login" ? <LogIn size={15} /> : <UserPlus size={15} />}
          {mode === "login" ? "Log in" : "Register"}
        </button>
        <button type="button" className="v2-link-btn" onClick={() => setMode(mode === "login" ? "register" : "login")}>
          {mode === "login" ? "Create account" : "Use existing account"}
        </button>
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

function ColumnResizeHandle({ className = "", label, onResize }) {
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

  return (
    <div
      className={`v2-resize-handle ${className} ${dragging ? "dragging" : ""}`}
      role="separator"
      aria-label={label}
      aria-orientation="vertical"
      tabIndex={0}
      onPointerDown={startResize}
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
  const todoGroups = useMemo(() => todoGroupsFromMessages(runtimeState.messages), [runtimeState.messages]);

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
      setError(err.message || String(err));
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
    if (!selfEvolutionEnabled) return undefined;
    const timer = window.setInterval(refreshSelfEvolution, 30000);
    return () => window.clearInterval(timer);
  }, [refreshSelfEvolution, selfEvolutionEnabled]);

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
      setError(err.message || String(err));
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
      setError(err.message || String(err));
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
      setError(err.message || String(err));
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
    if (nextSelection?.type === "artifact" && nextSelection.artifact_id) {
      const artifact = artifactForSelection(nextSelection, runtimeState.artifacts);
      return {
        id: `artifact:${nextSelection.artifact_id}`,
        type: "artifact",
        artifact_id: nextSelection.artifact_id,
        path: artifact?.path || nextSelection.path || "",
        title: artifact?.title || artifact?.path || nextSelection.artifact_id,
        artifact,
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
    if (nextSelection?.type === "file" || nextSelection?.type === "artifact") {
      openPreviewTab(nextSelection);
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
    if (selection?.type === "file" || selection?.type === "artifact") {
      openPreviewTab(selection);
    }
  }, [selection?.type, selection?.path, selection?.artifact_id, runtimeState.artifacts.length]);

  function selectFile(node) {
    if (!node) return;
    handleSelection({ type: "file", path: node.path || "", node });
    setActiveTab("chat");
  }

  const hasInterrupt = runtimeState.messages.some((message) => (
    Array.isArray(message.parts) && message.parts.some((part) => part.type === "interrupt" && part.status !== "resolved")
  ));
  const permissionMode = activeThread?.meta?.permission_mode === "hitl" ? "hitl" : "auto";
  const selectedEntrypoint = normalizeEntrypoint(activeThread?.entrypoint || bootstrap?.default_entrypoint || DEFAULT_ENTRYPOINT, entrypoints);
  const inspectorVisible = activeTab === "chat";
  const pendingEvolutionCount = Number(selfEvolutionPayload?.pending_review_count || 0);

  function resizeInspector(event) {
    const maxWidth = Math.max(300, Math.min(760, window.innerWidth - 620));
    const next = clampNumber(window.innerWidth - event.clientX, 280, maxWidth);
    setInspectorWidth(next);
    window.localStorage.setItem("catmaster:v2:inspector-width", String(Math.round(next)));
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
      setError(err.message || String(err));
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
      setError(err.message || String(err));
    }
  }

  if (auth?.auth_enabled && !auth?.authenticated) {
    return <AuthPanel onReady={() => checkAuthAndBootstrap(requestedProjectSpace)} />;
  }

  return (
    <AssistantRuntimeProvider runtime={runtimeState.runtime}>
      <main
        className={`v2-workspace tab-${activeTab} ${inspectorVisible ? "has-inspector" : ""}`}
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
          }}
          onSelectFile={selectFile}
        />
        <section className="v2-center">
          <header className="v2-topbar">
            <div>
              <div className="v2-eyebrow">CatMaster Workspace</div>
              <h1>{activeThread?.title || "Thread"}</h1>
              <div className="v2-thread-status-strip">
                <span>{activeThread?.status || "idle"}</span>
                <code>{activeThread?.thread_id || "no-thread"}</code>
                <span>{runtimeState.artifacts.length} artifacts</span>
              </div>
              <nav className="v2-workspace-tabs" aria-label="Workspace tabs">
                <button type="button" className={activeTab === "chat" ? "active" : ""} onClick={() => setActiveTab("chat")}>Chat</button>
                <button type="button" className={activeTab === "monitor" ? "active" : ""} onClick={() => setActiveTab("monitor")}><MonitorDot size={14} />Monitor</button>
                {selfEvolutionEnabled ? (
                  <button type="button" className={activeTab === "evolution" ? "active" : ""} onClick={() => { setActiveTab("evolution"); refreshSelfEvolution(); }}>
                    <GitBranch size={14} />Skill Evolution
                    {pendingEvolutionCount > 0 ? <span className="v2-tab-badge">{pendingEvolutionCount}</span> : null}
                  </button>
                ) : null}
                <button type="button" className={activeTab === "files" ? "active" : ""} onClick={() => setActiveTab("files")}><Files size={14} />Files</button>
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
          {error ? <div className="v2-error">{error}</div> : null}
          {activeTab === "chat" ? (
            <>
              <div className="v2-thread-scroll">
                <ThreadMessages
                  messages={runtimeState.messages}
                  loading={loading || runtimeState.loading}
                  error={runtimeState.error}
                  onSelect={handleSelection}
                  onResume={runtimeState.resume}
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
          <ColumnResizeHandle className="v2-resize-handle-inspector" label="Resize inspector" onResize={resizeInspector} />
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
      </main>
    </AssistantRuntimeProvider>
  );
}
