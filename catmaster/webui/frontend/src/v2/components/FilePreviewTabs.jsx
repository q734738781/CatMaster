import { useEffect, useMemo, useRef, useState } from "react";
import { Download, FileBox, ListChecks, RefreshCw, X } from "lucide-react";

import ArtifactRenderer from "./ArtifactRenderer";
import { apiFetch } from "../useCatMasterThreadRuntime";
import { todoSummary } from "../todoPanel.js";

function escapePath(value) {
  return encodeURIComponent(String(value || ""));
}

function tabTitle(tab) {
  const label = tab?.title || tab?.artifact?.title || tab?.artifact?.path || tab?.path || "Preview";
  return String(label).split("/").filter(Boolean).pop() || label;
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function todoStatusClass(status) {
  const value = String(status || "pending").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  return value || "pending";
}

function TodoPanel({ groups }) {
  const rows = Array.isArray(groups) ? groups : [];
  const summary = todoSummary(rows);
  return (
    <section className="v2-todo-panel" aria-label="Todo list">
      <div className="v2-todo-head">
        <div>
          <div className="v2-eyebrow">Todos</div>
          <h3>Plan</h3>
        </div>
        <small>{summary.total ? `${summary.done}/${summary.total} done` : "0 items"}</small>
      </div>
      <div className="v2-todo-groups">
        {rows.map((group) => (
          <section key={group.source} className="v2-todo-group">
            <div className="v2-todo-source">
              <span>{group.source}</span>
              <small>{group.status || "running"}</small>
            </div>
            <ol>
              {(group.rows || []).map((todo, index) => (
                <li key={`${todo.content}-${index}`} className={`status-${todoStatusClass(todo.status)}`}>
                  <span>{todo.content}</span>
                  <small>{todo.status || "pending"}</small>
                </li>
              ))}
            </ol>
          </section>
        ))}
        {!rows.length ? <div className="v2-empty compact">No todo plan yet.</div> : null}
      </div>
    </section>
  );
}

function FilePreviewBody({ ctx, workspaceName, tab }) {
  const [preview, setPreview] = useState(tab?.preview || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  const path = tab?.path || "";
  async function load() {
    if (!ctx || !path) return;
    setLoading(true);
    setError("");
    try {
      const payload = await apiFetch(`/api/session/${escapePath(ctx)}/files/content?path=${escapePath(path)}&project_space=${escapePath(workspaceName || "")}`);
      setPreview(payload);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setPreview(tab?.preview || null);
    setError("");
    if (tab?.type === "file" && !tab?.preview) load();
  }, [tab?.id, ctx, workspaceName]);

  if (tab?.type === "artifact") {
    return <ArtifactRenderer artifact={tab.artifact || { artifact_id: tab.artifact_id, path: tab.path, title: tab.title }} />;
  }
  if (loading) return <div className="v2-empty">Loading preview...</div>;
  if (error) return <div className="v2-error">{error}</div>;
  if (!preview) return <div className="v2-empty">No preview loaded.</div>;

  return (
    <div className="v2-file-tab-preview">
      <div className="v2-file-tab-actions">
        <div>
          <div className="v2-eyebrow">{preview.kind || preview.node_type || "file"}</div>
          <strong>{preview.name || preview.path || "."}</strong>
        </div>
        <div className="v2-icon-row compact">
          <button type="button" className="v2-icon-btn" onClick={load} title="Refresh"><RefreshCw size={15} /></button>
          {preview.download_url ? <a className="v2-icon-btn" href={preview.download_url} title="Download"><Download size={15} /></a> : null}
        </div>
      </div>
      <ArtifactRenderer filePreview={preview} />
    </div>
  );
}

export default function FilePreviewTabs({ ctx, workspaceName, tabs, activeTabId, todoGroups, onActivate, onClose }) {
  const shellRef = useRef(null);
  const [todoHeight, setTodoHeight] = useState(() => {
    if (typeof window === "undefined") return 220;
    const saved = Number(window.localStorage.getItem("catmaster:v2:todo-panel-height"));
    return Number.isFinite(saved) && saved > 0 ? saved : 220;
  });
  const activeTab = useMemo(
    () => (tabs || []).find((tab) => tab.id === activeTabId) || (tabs || [])[0] || null,
    [tabs, activeTabId],
  );

  function startTodoResize(event) {
    event.preventDefault();
    const shell = shellRef.current;
    if (!shell) return;
    document.body.classList.add("v2-resizing-columns");
    const move = (moveEvent) => {
      const rect = shell.getBoundingClientRect();
      const maxHeight = Math.max(150, Math.min(560, rect.height - 170));
      const next = clampNumber(rect.bottom - moveEvent.clientY, 140, maxHeight);
      setTodoHeight(next);
      window.localStorage.setItem("catmaster:v2:todo-panel-height", String(Math.round(next)));
    };
    const stop = () => {
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
    <aside ref={shellRef} className="v2-right-inspector v2-file-preview-tabs" style={{ "--v2-todo-panel-height": `${todoHeight}px` }}>
      <div className="v2-browser-tabs" role="tablist" aria-label="Open file previews">
        {(tabs || []).map((tab) => (
          <div
            key={tab.id}
            className={`v2-browser-tab ${activeTab?.id === tab.id ? "active" : ""}`}
            role="tab"
            aria-selected={activeTab?.id === tab.id}
            title={tab.path || tab.artifact?.path || tab.title || ""}
          >
            <button type="button" className="v2-browser-tab-main" onClick={() => onActivate?.(tab.id)}>
              <FileBox size={14} />
              <span>{tabTitle(tab)}</span>
            </button>
            <button
              type="button"
              className="v2-browser-tab-close"
              title="Close"
              onClick={(event) => {
                event.stopPropagation();
                onClose?.(tab.id);
              }}
            >
              <X size={13} />
            </button>
          </div>
        ))}
      </div>
      <div className="v2-browser-content">
        {activeTab ? (
          <FilePreviewBody ctx={ctx} workspaceName={workspaceName} tab={activeTab} />
        ) : (
          <div className="v2-empty">Open files or artifacts from the chat to preview them here.</div>
        )}
      </div>
      <div className="v2-todo-resize" role="separator" aria-label="Resize todo panel" aria-orientation="horizontal" tabIndex={0} onPointerDown={startTodoResize}>
        <ListChecks size={13} />
      </div>
      <TodoPanel groups={todoGroups} />
    </aside>
  );
}
