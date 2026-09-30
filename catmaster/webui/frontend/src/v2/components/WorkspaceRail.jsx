import { useCallback, useEffect, useRef, useState } from "react";
import { Check, ChevronDown, ChevronRight, Files, Folder, FolderOpen, GitBranch, MessageSquare, MessageSquarePlus, MonitorDot, Network, Orbit, Pencil, Search, RefreshCw, Plus, Trash2, X } from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import { displayValue, isInternalStoragePath, presentError } from "../presentation.js";
import {
  buildResearchSessionRows,
  RELATED_RESEARCH_ACTIVITY_ID,
  researchSessionDefaultOpen,
  researchStateLabel,
} from "../researchSessions.js";

function statusLabel(value) {
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

function activityTimeLabel(value) {
  const timestamp = Number(value || 0);
  if (!timestamp) return "";
  const seconds = Math.max(0, Math.floor(Date.now() / 1000 - timestamp));
  if (seconds < 60) return "now";
  if (seconds < 3600) return `${Math.floor(seconds / 60)}m`;
  if (seconds < 86400) return `${Math.floor(seconds / 3600)}h`;
  if (seconds < 604800) return `${Math.floor(seconds / 86400)}d`;
  return new Date(timestamp * 1000).toLocaleDateString(undefined, {
    month: "short",
    day: "numeric",
  });
}

function ErrorNotice({ error }) {
  const presented = presentError(error);
  if (!presented.message) return null;
  return (
    <div className="v2-error compact" role="alert">
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

function TreeNode({ node, depth, selectedPath, onSelect, loadChildren, childrenByPath, pagesByPath, loadingByPath }) {
  const [open, setOpen] = useState(depth < 1);
  const isDirectory = node.node_type === "directory";
  const allChildren = childrenByPath[node.path || ""] || [];
  const children = allChildren.filter((child) => !isInternalStoragePath(child?.path));
  useEffect(() => {
    if (open && isDirectory) loadChildren(node.path || "");
  }, [open, isDirectory, node.path, loadChildren]);
  return (
    <div>
      <button
        type="button"
        className={`v2-tree-row ${selectedPath === node.path ? "selected" : ""}`}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
        aria-expanded={isDirectory ? open : undefined}
        aria-current={selectedPath === node.path ? "true" : undefined}
        aria-label={`${isDirectory ? (open ? "Collapse" : "Expand") : "Open"} ${node.name || "user files"}`}
        title={displayValue(node.name || node.path, "User files")}
        onClick={() => {
          if (isDirectory) setOpen((value) => !value);
          onSelect(node);
        }}
      >
        {isDirectory ? (open ? <FolderOpen size={15} /> : <Folder size={15} />) : <span className={`v2-file-dot kind-${node.preview_kind || "file"}`} />}
        <span>{node.name || "."}</span>
      </button>
      {open && isDirectory ? (
        <div>
          {children.map((child) => (
            <TreeNode
              key={child.path || child.name}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              onSelect={onSelect}
              loadChildren={loadChildren}
              childrenByPath={childrenByPath}
              pagesByPath={pagesByPath}
              loadingByPath={loadingByPath}
            />
          ))}
          {loadingByPath[node.path || ""] ? <div className="v2-empty compact" role="status">Loading this folder…</div> : null}
          {!loadingByPath[node.path || ""] && Object.hasOwn(childrenByPath, node.path || "") && !children.length ? (
            <div className="v2-empty compact">This folder is empty.</div>
          ) : null}
          {pagesByPath[node.path || ""]?.truncated ? (
            <button
              type="button"
              className="v2-file-load-more"
              onClick={() => loadChildren(node.path || "", pagesByPath[node.path || ""].next_cursor)}
            >
              Load more ({allChildren.length} of {pagesByPath[node.path || ""].total_count})
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

export default function WorkspaceRail({
  ctx,
  workspaceName,
  workspaceChoices,
  threads,
  activeThreadId,
  activeView,
  onViewChange,
  selfEvolutionEnabled,
  pendingEvolutionCount,
  onWorkspaceChange,
  onCreateWorkspace,
  onDeleteWorkspace,
  onCreateThread,
  onRenameThread,
  onSelectThread,
  onOpenResearchDecisions,
  onSelectFile,
}) {
  const [query, setQuery] = useState("");
  const [editingThreadId, setEditingThreadId] = useState("");
  const [renameDraft, setRenameDraft] = useState("");
  const [renameError, setRenameError] = useState("");
  const [renamingThreadId, setRenamingThreadId] = useState("");
  const [childrenByPath, setChildrenByPath] = useState({});
  const [pagesByPath, setPagesByPath] = useState({});
  const [selectedPath, setSelectedPath] = useState("");
  const [error, setError] = useState("");
  const [loadingByPath, setLoadingByPath] = useState({});
  const [expandedSessionIds, setExpandedSessionIds] = useState(new Set());
  const loadedPathsRef = useRef(new Set());
  const loadingPathsRef = useRef(new Set());

  const loadChildren = useCallback(async (path = "", cursor = "", force = false) => {
    if (!ctx || !workspaceName) return;
    const requestKey = `${path}\n${cursor}`;
    if (loadingPathsRef.current.has(requestKey)) return;
    if (!cursor && !force && loadedPathsRef.current.has(path)) return;
    loadingPathsRef.current.add(requestKey);
    setLoadingByPath((current) => ({ ...current, [path]: true }));
    try {
      const cursorQuery = cursor ? `&cursor=${encodeURIComponent(cursor)}` : "";
      const payload = await apiFetch(`/api/session/${encodeURIComponent(ctx)}/files/tree?path=${encodeURIComponent(path)}&project_space=${encodeURIComponent(workspaceName)}${cursorQuery}`);
      const key = payload.path || "";
      setChildrenByPath((prev) => ({
        ...prev,
        [key]: cursor
          ? [...(prev[key] || []), ...(payload.children || [])]
          : (payload.children || []),
      }));
      setPagesByPath((prev) => ({ ...prev, [key]: payload.page || {} }));
      loadedPathsRef.current.add(key);
      setError("");
    } catch (err) {
      setError(err);
    } finally {
      loadingPathsRef.current.delete(requestKey);
      setLoadingByPath((current) => ({ ...current, [path]: false }));
    }
  }, [ctx, workspaceName]);

  useEffect(() => {
    loadedPathsRef.current.clear();
    loadingPathsRef.current.clear();
    setChildrenByPath({});
    setPagesByPath({});
    setSelectedPath("");
    setLoadingByPath({});
    loadChildren("", "", true);
  }, [loadChildren]);

  useEffect(() => {
    setEditingThreadId("");
    setRenameDraft("");
    setRenameError("");
    setRenamingThreadId("");
  }, [workspaceName]);

  useEffect(() => {
    const storageKey = `catmaster:v2:research-session-folders:${workspaceName || "default"}`;
    const storedValue = window.localStorage.getItem(storageKey);
    let saved = [];
    try {
      saved = JSON.parse(storedValue || "[]");
    } catch {
      saved = [];
    }
    const next = new Set(Array.isArray(saved) ? saved : []);
    const activeThread = threads.find((thread) => thread.thread_id === activeThreadId);
    if (activeThread?.parent_thread_id) next.add(activeThread.parent_thread_id);
    if (
      activeThread?.thread_role === "research_execution"
      && !activeThread?.parent_thread_id
    ) next.add(RELATED_RESEARCH_ACTIVITY_ID);
    if (storedValue === null) {
      threads.forEach((thread) => {
        if (thread.thread_role === "research_root" && researchSessionDefaultOpen(thread)) {
          next.add(thread.thread_id);
        }
      });
    }
    setExpandedSessionIds(next);
  }, [activeThreadId, threads, workspaceName]);

  function setSessionOpen(threadId, open) {
    setExpandedSessionIds((current) => {
      const next = new Set(current);
      if (open) next.add(threadId);
      else next.delete(threadId);
      const storageKey = `catmaster:v2:research-session-folders:${workspaceName || "default"}`;
      window.localStorage.setItem(storageKey, JSON.stringify([...next]));
      return next;
    });
  }

  function beginThreadRename(thread) {
    setEditingThreadId(thread.thread_id);
    setRenameDraft(displayValue(thread.title, "New thread"));
    setRenameError("");
  }

  function cancelThreadRename() {
    setEditingThreadId("");
    setRenameDraft("");
    setRenameError("");
  }

  async function submitThreadRename(event, thread) {
    event.preventDefault();
    const nextTitle = renameDraft.trim();
    if (!nextTitle) {
      setRenameError("Thread name cannot be empty.");
      return;
    }
    if (nextTitle === String(thread.title || "").trim()) {
      cancelThreadRename();
      return;
    }
    setRenamingThreadId(thread.thread_id);
    setRenameError("");
    try {
      await onRenameThread(thread.thread_id, nextTitle);
      setEditingThreadId("");
      setRenameDraft("");
    } catch (err) {
      setRenameError(err);
    } finally {
      setRenamingThreadId("");
    }
  }

  const threadGroups = buildResearchSessionRows(threads, query);
  const allRootNodes = childrenByPath[""] || [];
  const rootNodes = allRootNodes.filter((node) => !isInternalStoragePath(node?.path));

  return (
    <aside className="v2-left-rail">
      <div className="v2-brand">
        <span className="v2-brand-mark"><Orbit size={23} strokeWidth={1.5} /></span>
        <div><strong>CatMaster</strong><small>RESEARCH WORKSPACE</small></div>
      </div>
      <div className="v2-rail-section">
        <div className="v2-section-row">
          <div className="v2-section-title">Workspace</div>
          <div className="v2-icon-row compact">
            <button type="button" className="v2-icon-btn" onClick={onCreateWorkspace} aria-label="Create workspace" title="Create workspace">
              <Plus size={15} />
            </button>
            <button type="button" className="v2-icon-btn danger" disabled={!workspaceChoices?.length} onClick={() => onDeleteWorkspace?.(workspaceName)} aria-label="Delete workspace" title="Delete workspace">
              <Trash2 size={15} />
            </button>
          </div>
        </div>
        <select className="v2-select" aria-label="Select workspace" value={workspaceName || ""} onChange={(event) => onWorkspaceChange(event.target.value)}>
          <option value="" disabled>Select a workspace</option>
          {(workspaceChoices || []).map((choice) => (
            <option key={choice.value} value={choice.value}>{choice.label}</option>
          ))}
        </select>
        <button type="button" className="v2-new-conversation" disabled={!workspaceName} onClick={onCreateThread} aria-label="New thread">
          <MessageSquarePlus size={17} /> New conversation <Plus size={15} />
        </button>
      </div>
      <nav className="v2-workspace-nav" aria-label="Workspace navigation">
        {[
          ["chat", "Chat", MessageSquare],
          ["hypotheses", "Research Graph", Network],
          ["files", "Files", Files],
          ["monitor", "Monitor", MonitorDot],
          ...(selfEvolutionEnabled ? [["evolution", "Skill Evolution", GitBranch]] : []),
        ].map(([view, label, Icon]) => (
          <button key={view} type="button" disabled={!workspaceName} className={activeView === view ? "active" : ""} aria-current={activeView === view ? "page" : undefined} onClick={() => onViewChange(view)}>
            <Icon size={17} strokeWidth={1.7} /><span>{label}</span>
            {view === "evolution" && pendingEvolutionCount > 0 ? <small>{pendingEvolutionCount}</small> : null}
          </button>
        ))}
      </nav>
      <div className="v2-rail-section grow-tight">
        <div className="v2-section-row">
          <div className="v2-section-title">Conversations</div>
        </div>
        <div className="v2-search">
          <Search size={14} />
          <input aria-label="Search threads" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search threads" />
        </div>
        <ErrorNotice error={renameError} />
        <div className="v2-thread-list">
          {threadGroups.map(({ root, children, isResearchSession, isRelatedGroup }) => {
            const rootTitle = displayValue(root.title, "New thread");
            const rootEditing = editingThreadId === root.thread_id;
            const rootRenaming = renamingThreadId === root.thread_id;
            const queryActive = Boolean(query.trim());
            const open = isResearchSession && (queryActive || expandedSessionIds.has(root.thread_id));
            const activity = root.research_activity || {};
            const rootStatus = isResearchSession
              ? researchStateLabel(activity.state)
              : statusLabel(root.status);
            return (
              <section key={root.thread_id} className={`v2-thread-group ${isResearchSession ? "research-session" : "standalone"} ${activity.action_required ? "action-required" : ""}`}>
                <div
                  className={`v2-thread-row ${isResearchSession ? "session-root" : ""} ${root.thread_id === activeThreadId ? "active" : ""} ${rootEditing ? "editing" : ""}`}
                  data-status={root.status}
                >
                  {rootEditing ? (
                    <form className="v2-thread-rename-form" onSubmit={(event) => submitThreadRename(event, root)}>
                      <input
                        autoFocus
                        aria-label={`Rename ${rootTitle}`}
                        value={renameDraft}
                        disabled={rootRenaming}
                        onChange={(event) => setRenameDraft(event.target.value)}
                        onKeyDown={(event) => {
                          if (event.key === "Escape") {
                            event.preventDefault();
                            cancelThreadRename();
                          }
                        }}
                      />
                      <button type="submit" disabled={rootRenaming} aria-label={`Save name for ${rootTitle}`} title="Save thread name">
                        <Check size={14} />
                      </button>
                      <button type="button" disabled={rootRenaming} aria-label={`Cancel renaming ${rootTitle}`} title="Cancel" onClick={cancelThreadRename}>
                        <X size={14} />
                      </button>
                    </form>
                  ) : (
                    <>
                      {isResearchSession ? (
                        <button
                          type="button"
                          className="v2-thread-folder-toggle"
                          aria-label={`${open ? "Collapse" : "Expand"} ${rootTitle}`}
                          aria-expanded={open}
                          onClick={() => setSessionOpen(root.thread_id, !open)}
                        >
                          {open ? <ChevronDown size={15} /> : <ChevronRight size={15} />}
                        </button>
                      ) : null}
                      <button
                        type="button"
                        className="v2-thread-select"
                        aria-current={root.thread_id === activeThreadId ? "true" : undefined}
                        aria-label={`${rootTitle}, ${rootStatus}`}
                        title={rootTitle}
                        onClick={() => {
                          if (!isRelatedGroup) onSelectThread(root.thread_id);
                          else setSessionOpen(root.thread_id, !open);
                        }}
                      >
                        <span>{rootTitle}</span>
                        <small>{rootStatus}</small>
                        {isResearchSession && activity.current_title ? (
                          <em title={activity.current_title}>{activity.current_title}</em>
                        ) : null}
                      </button>
                      {!isRelatedGroup ? (
                        <button
                          type="button"
                          className="v2-thread-rename-btn"
                          aria-label={`Rename ${rootTitle}`}
                          title="Rename thread"
                          onClick={() => beginThreadRename(root)}
                        >
                          <Pencil size={13} />
                        </button>
                      ) : null}
                    </>
                  )}
                </div>
                {open ? (
                  <div className="v2-thread-children" role="group" aria-label={`${rootTitle} experiment threads`}>
                    {children.map((thread) => {
                      const childTitle = displayValue(thread.title, "Experiment thread");
                      const childEditing = editingThreadId === thread.thread_id;
                      const childRenaming = renamingThreadId === thread.thread_id;
                      const childTime = activityTimeLabel(thread.updated_at);
                      return (
                        <div
                          key={thread.thread_id}
                          className={`v2-thread-row child ${thread.thread_id === activeThreadId ? "active" : ""} ${childEditing ? "editing" : ""}`}
                        >
                          {childEditing ? (
                            <form className="v2-thread-rename-form" onSubmit={(event) => submitThreadRename(event, thread)}>
                              <input
                                autoFocus
                                aria-label={`Rename ${childTitle}`}
                                value={renameDraft}
                                disabled={childRenaming}
                                onChange={(event) => setRenameDraft(event.target.value)}
                                onKeyDown={(event) => {
                                  if (event.key === "Escape") {
                                    event.preventDefault();
                                    cancelThreadRename();
                                  }
                                }}
                              />
                              <button type="submit" disabled={childRenaming} aria-label={`Save name for ${childTitle}`} title="Save thread name"><Check size={14} /></button>
                              <button type="button" disabled={childRenaming} aria-label={`Cancel renaming ${childTitle}`} title="Cancel" onClick={cancelThreadRename}><X size={14} /></button>
                            </form>
                          ) : (
                            <>
                              <button
                                type="button"
                                className="v2-thread-select"
                                aria-current={thread.thread_id === activeThreadId ? "true" : undefined}
                                aria-label={`${rootTitle} / ${childTitle}, ${statusLabel(thread.status)}`}
                                title={`${rootTitle} / ${childTitle}`}
                                onClick={() => onSelectThread(thread.thread_id)}
                              >
                                <span>{queryActive ? `${rootTitle} / ${childTitle}` : childTitle}</span>
                                <small>
                                  {statusLabel(thread.status)}
                                  {childTime ? ` · ${childTime}` : ""}
                                </small>
                              </button>
                              <button type="button" className="v2-thread-rename-btn" aria-label={`Rename ${childTitle}`} title="Rename thread" onClick={() => beginThreadRename(thread)}><Pencil size={13} /></button>
                            </>
                          )}
                        </div>
                      );
                    })}
                    {Number(activity.decision_round_count || 0) > 0 ? (
                      <button
                        type="button"
                        className="v2-thread-decisions"
                        onClick={() => onOpenResearchDecisions?.(root)}
                      >
                        {activity.decision_round_count} automatic decision {activity.decision_round_count === 1 ? "round" : "rounds"}
                      </button>
                    ) : null}
                    {!children.length && !Number(activity.decision_round_count || 0) ? (
                      <div className="v2-empty compact">Research activity will appear here.</div>
                    ) : null}
                  </div>
                ) : null}
              </section>
            );
          })}
          {!threadGroups.length ? (
            <div className="v2-empty compact">
              {query ? "No conversations match this search." : "No conversations yet. Create one to get started."}
            </div>
          ) : null}
        </div>
      </div>
      <div className="v2-rail-section grow">
        <div className="v2-section-row">
          <div className="v2-section-title">Files</div>
          <button type="button" className="v2-icon-btn" onClick={() => loadChildren("", "", true)} aria-label="Refresh files" title="Refresh files">
            <RefreshCw size={15} />
          </button>
        </div>
        <ErrorNotice error={error} />
        <div className="v2-file-tree">
          {rootNodes.map((node) => (
            <TreeNode
              key={node.path || node.name}
              node={node}
              depth={0}
              selectedPath={selectedPath}
              onSelect={(item) => {
                setSelectedPath(item.path || "");
                onSelectFile(item);
              }}
              loadChildren={loadChildren}
              childrenByPath={childrenByPath}
              pagesByPath={pagesByPath}
              loadingByPath={loadingByPath}
            />
          ))}
          {loadingByPath[""] ? <div className="v2-empty compact" role="status">Loading workspace files…</div> : null}
          {!loadingByPath[""] && Object.hasOwn(childrenByPath, "") && !rootNodes.length ? (
            <div className="v2-empty compact">No files yet. Attach a file in chat or create one through a task.</div>
          ) : null}
          {pagesByPath[""]?.truncated ? (
            <button
              type="button"
              className="v2-file-load-more"
              onClick={() => loadChildren("", pagesByPath[""].next_cursor)}
            >
              Load more ({allRootNodes.length} of {pagesByPath[""].total_count})
            </button>
          ) : null}
        </div>
      </div>
    </aside>
  );
}
