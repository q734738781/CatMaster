import { useEffect, useState } from "react";
import { Folder, FolderOpen, MessageSquarePlus, Search, RefreshCw, Plus, Trash2 } from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import { displayValue, isInternalStoragePath, presentError } from "../presentation.js";

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
  onWorkspaceChange,
  onCreateWorkspace,
  onDeleteWorkspace,
  onCreateThread,
  onSelectThread,
  onSelectFile,
}) {
  const [query, setQuery] = useState("");
  const [childrenByPath, setChildrenByPath] = useState({});
  const [pagesByPath, setPagesByPath] = useState({});
  const [selectedPath, setSelectedPath] = useState("");
  const [error, setError] = useState("");
  const [loadingByPath, setLoadingByPath] = useState({});

  const loadChildren = async (path = "", cursor = "") => {
    if (!ctx || !workspaceName) return;
    if (!cursor && childrenByPath[path] && path !== "") return;
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
      setError("");
    } catch (err) {
      setError(err);
    } finally {
      setLoadingByPath((current) => ({ ...current, [path]: false }));
    }
  };

  useEffect(() => {
    setChildrenByPath({});
    setPagesByPath({});
    setSelectedPath("");
    setLoadingByPath({});
    loadChildren("");
  }, [ctx, workspaceName]);

  const filteredThreads = threads.filter((thread) => {
    const text = String(thread.title || "").toLowerCase();
    return text.includes(query.toLowerCase());
  });
  const allRootNodes = childrenByPath[""] || [];
  const rootNodes = allRootNodes.filter((node) => !isInternalStoragePath(node?.path));

  return (
    <aside className="v2-left-rail">
      <div className="v2-rail-section">
        <div className="v2-section-row">
          <div className="v2-section-title">Workspace</div>
          <div className="v2-icon-row compact">
            <button type="button" className="v2-icon-btn" onClick={onCreateWorkspace} aria-label="Create workspace" title="Create workspace">
              <Plus size={15} />
            </button>
            <button type="button" className="v2-icon-btn danger" onClick={() => onDeleteWorkspace?.(workspaceName)} aria-label="Delete workspace" title="Delete workspace">
              <Trash2 size={15} />
            </button>
          </div>
        </div>
        <select className="v2-select" aria-label="Select workspace" value={workspaceName || ""} onChange={(event) => onWorkspaceChange(event.target.value)}>
          {(workspaceChoices || []).map((choice) => (
            <option key={choice.value} value={choice.value}>{choice.label}</option>
          ))}
        </select>
      </div>
      <div className="v2-rail-section grow-tight">
        <div className="v2-section-row">
          <div className="v2-section-title">Threads</div>
          <button type="button" className="v2-icon-btn" onClick={onCreateThread} aria-label="New thread" title="New thread">
            <MessageSquarePlus size={16} />
          </button>
        </div>
        <div className="v2-search">
          <Search size={14} />
          <input aria-label="Search threads" value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search threads" />
        </div>
        <div className="v2-thread-list">
          {filteredThreads.map((thread) => (
            <button
              type="button"
              key={thread.thread_id}
              className={`v2-thread-row ${thread.thread_id === activeThreadId ? "active" : ""}`}
              aria-current={thread.thread_id === activeThreadId ? "true" : undefined}
              aria-label={`${displayValue(thread.title, "Untitled thread")}, ${statusLabel(thread.status)}`}
              title={displayValue(thread.title, "Untitled thread")}
              onClick={() => onSelectThread(thread.thread_id)}
            >
              <span>{displayValue(thread.title, "Untitled thread")}</span>
              <small>{statusLabel(thread.status)}</small>
            </button>
          ))}
          {!filteredThreads.length ? (
            <div className="v2-empty compact">
              {query ? "No conversations match this search." : "No conversations yet. Create one to get started."}
            </div>
          ) : null}
        </div>
      </div>
      <div className="v2-rail-section grow">
        <div className="v2-section-row">
          <div className="v2-section-title">Files</div>
          <button type="button" className="v2-icon-btn" onClick={() => loadChildren("")} aria-label="Refresh files" title="Refresh files">
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
