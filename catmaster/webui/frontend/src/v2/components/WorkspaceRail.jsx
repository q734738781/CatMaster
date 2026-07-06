import { useEffect, useState } from "react";
import { Folder, FolderOpen, MessageSquarePlus, Search, RefreshCw, Plus, Trash2 } from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";

function TreeNode({ node, depth, selectedPath, onSelect, loadChildren, childrenByPath }) {
  const [open, setOpen] = useState(depth < 1);
  const isDirectory = node.node_type === "directory";
  const children = childrenByPath[node.path || ""] || [];
  useEffect(() => {
    if (open && isDirectory) loadChildren(node.path || "");
  }, [open, isDirectory, node.path, loadChildren]);
  return (
    <div>
      <button
        type="button"
        className={`v2-tree-row ${selectedPath === node.path ? "selected" : ""}`}
        style={{ paddingLeft: `${8 + depth * 14}px` }}
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
            />
          ))}
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
  const [selectedPath, setSelectedPath] = useState("");
  const [error, setError] = useState("");

  const loadChildren = async (path = "") => {
    if (!ctx || !workspaceName) return;
    if (childrenByPath[path] && path !== "") return;
    try {
      const payload = await apiFetch(`/api/session/${encodeURIComponent(ctx)}/files/tree?path=${encodeURIComponent(path)}&project_space=${encodeURIComponent(workspaceName)}`);
      setChildrenByPath((prev) => ({ ...prev, [payload.path || ""]: payload.children || [] }));
      setError("");
    } catch (err) {
      setError(err.message || String(err));
    }
  };

  useEffect(() => {
    setChildrenByPath({});
    setSelectedPath("");
    loadChildren("");
  }, [ctx, workspaceName]);

  const filteredThreads = threads.filter((thread) => {
    const text = `${thread.title || ""} ${thread.thread_id || ""}`.toLowerCase();
    return text.includes(query.toLowerCase());
  });
  const rootNodes = childrenByPath[""] || [];

  return (
    <aside className="v2-left-rail">
      <div className="v2-rail-section">
        <div className="v2-section-row">
          <div className="v2-section-title">Workspace</div>
          <div className="v2-icon-row compact">
            <button type="button" className="v2-icon-btn" onClick={onCreateWorkspace} title="Create workspace">
              <Plus size={15} />
            </button>
            <button type="button" className="v2-icon-btn danger" onClick={() => onDeleteWorkspace?.(workspaceName)} title="Delete workspace">
              <Trash2 size={15} />
            </button>
          </div>
        </div>
        <select className="v2-select" value={workspaceName || ""} onChange={(event) => onWorkspaceChange(event.target.value)}>
          {(workspaceChoices || []).map((choice) => (
            <option key={choice.value} value={choice.value}>{choice.label}</option>
          ))}
        </select>
      </div>
      <div className="v2-rail-section grow-tight">
        <div className="v2-section-row">
          <div className="v2-section-title">Threads</div>
          <button type="button" className="v2-icon-btn" onClick={onCreateThread} title="New thread">
            <MessageSquarePlus size={16} />
          </button>
        </div>
        <div className="v2-search">
          <Search size={14} />
          <input value={query} onChange={(event) => setQuery(event.target.value)} placeholder="Search threads" />
        </div>
        <div className="v2-thread-list">
          {filteredThreads.map((thread) => (
            <button
              type="button"
              key={thread.thread_id}
              className={`v2-thread-row ${thread.thread_id === activeThreadId ? "active" : ""}`}
              onClick={() => onSelectThread(thread.thread_id)}
            >
              <span>{thread.title || thread.thread_id}</span>
              <small>{thread.status}</small>
            </button>
          ))}
        </div>
      </div>
      <div className="v2-rail-section grow">
        <div className="v2-section-row">
          <div className="v2-section-title">Files</div>
          <button type="button" className="v2-icon-btn" onClick={() => loadChildren("")} title="Refresh files">
            <RefreshCw size={15} />
          </button>
        </div>
        {error ? <div className="v2-error compact">{error}</div> : null}
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
            />
          ))}
        </div>
      </div>
    </aside>
  );
}
