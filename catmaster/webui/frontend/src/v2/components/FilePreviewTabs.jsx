import { useEffect, useMemo, useRef, useState } from "react";
import { Clipboard, Download, FileBox, Hammer, ListChecks, RefreshCw, X } from "lucide-react";

import ArtifactRenderer from "./ArtifactRenderer";
import { apiFetch } from "../useCatMasterThreadRuntime";
import { todoSummary } from "../todoPanel.js";
import { displayValue, isInternalStoragePath, presentError, redactErrorText } from "../presentation.js";

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

function statusLabel(value) {
  const status = String(value || "updated").toLowerCase();
  return {
    created: "Starting",
    streaming: "In progress",
    running: "Running",
    queued: "Queued",
    pending: "Waiting",
    interrupted: "Waiting for review",
    resolved: "Reviewed",
    completed: "Completed",
    complete: "Completed",
    done: "Completed",
    success: "Completed",
    failed: "Failed",
    error: "Failed",
  }[status] || displayValue(status.replace(/[_-]+/g, " "), "Updated");
}

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
              <span title={displayValue(group.source, "Plan")}>{displayValue(group.source, "Plan")}</span>
              <small>{statusLabel(group.status || "running")}</small>
            </div>
            <ol>
              {(group.rows || []).map((todo, index) => (
                <li key={`${todo.content}-${index}`} className={`status-${todoStatusClass(todo.status)}`}>
                  <span title={displayValue(todo.content, "Plan item")}>{displayValue(todo.content, "Plan item")}</span>
                  <small>{statusLabel(todo.status || "pending")}</small>
                </li>
              ))}
            </ol>
          </section>
        ))}
        {!rows.length ? <div className="v2-empty compact">A step-by-step plan will appear here when the task needs one.</div> : null}
      </div>
    </section>
  );
}

function ActivityPreview({ tab }) {
  const [part, setPart] = useState(tab?.part || {});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const initial = tab?.part || {};
    setPart(initial);
    setError("");
    const ref = String(initial.detail_ref || "");
    if (!ref) return undefined;
    setLoading(true);
    apiFetch(ref)
      .then((payload) => {
        if (!cancelled && payload?.part) setPart(payload.part);
      })
      .catch((reason) => {
        if (!cancelled) setError(reason);
      })
      .finally(() => {
        if (!cancelled) setLoading(false);
      });
    return () => {
      cancelled = true;
    };
  }, [tab?.id, tab?.part]);

  const fields = Array.isArray(part?.fields) ? part.fields : [];
  const items = Array.isArray(part?.items) ? part.items : [];
  const failed = String(part?.type || "") === "error"
    || ["failed", "error"].includes(String(part?.status || "").toLowerCase());
  const visible = (value, fallback = "Not available") => {
    const text = displayValue(value, fallback);
    return failed ? redactErrorText(text) : text;
  };
  const typeLabel = {
    tool: "Tool activity",
    receipt: "Execution receipt",
    attachment: "Attachment",
  }[String(part?.type || "")] || "Activity";
  const sourceSummary = String(part?.type || "") === "tool"
    && /\bread file\b/i.test(visible(part?.title, ""))
    && Boolean(part?.summary);
  const technicalRows = [
    part?.diagnostics_ref ? ["Diagnostics reference", failed ? redactErrorText(part.diagnostics_ref) : part.diagnostics_ref] : null,
    part?.detail_ref ? ["Detail reference", failed ? redactErrorText(part.detail_ref) : part.detail_ref] : null,
    part?.id ? ["Activity ID", failed ? redactErrorText(part.id) : part.id] : null,
  ].filter(Boolean);

  return (
    <section className="v2-activity-preview" aria-label={visible(part?.title, "Activity details")}>
      <div className="v2-activity-preview-head">
        <Hammer size={17} />
        <div>
          <div className="v2-eyebrow">{typeLabel}</div>
          <h3>{visible(part?.title, "Activity details")}</h3>
          {part?.summary && !sourceSummary ? <p>{visible(part.summary)}</p> : null}
        </div>
        <small>{statusLabel(part?.status)}</small>
      </div>
      {loading ? <div className="v2-muted" role="status">Loading complete details…</div> : null}
      <ErrorNotice error={error} />
      {sourceSummary ? (
        <section className="v2-activity-source">
          <h4>File contents</h4>
          <pre>{visible(part.summary)}</pre>
        </section>
      ) : null}
      {fields.length ? (
        <dl className="v2-semantic-fields">
          {fields.map((field, index) => (
            <div key={`${field?.label || "detail"}-${index}`}>
              <dt>{visible(field?.label, "Detail")}</dt>
              <dd title={visible(field?.value, "") || undefined}>{visible(field?.value)}</dd>
            </div>
          ))}
        </dl>
      ) : null}
      {items.length ? (
        <ul className="v2-semantic-items">
          {items.map((item, index) => (
            <li key={`${item?.label || "item"}-${index}`}>
              <div>
                <span title={visible(item?.label, "Item")}>{visible(item?.label, "Item")}</span>
                {item?.summary ? <small>{visible(item.summary)}</small> : null}
              </div>
              {item?.status ? <code>{statusLabel(item.status)}</code> : null}
            </li>
          ))}
        </ul>
      ) : null}
      {!loading && !fields.length && !items.length && !part?.summary ? (
        <div className="v2-empty compact">No additional user-facing details were recorded for this activity.</div>
      ) : null}
      {technicalRows.length ? (
        <details className="v2-technical-details">
          <summary>Technical details</summary>
          <dl>
            {technicalRows.map(([label, value]) => (
              <div key={label}>
                <dt>{label}</dt>
                <dd>
                  <code>{value}</code>
                  <button
                    type="button"
                    className="v2-inline-copy"
                    aria-label={`Copy ${label}`}
                    onClick={() => navigator.clipboard?.writeText(String(value))}
                  >
                    <Clipboard size={13} />
                  </button>
                </dd>
              </div>
            ))}
          </dl>
        </details>
      ) : null}
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
      setError(err);
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
    return <ArtifactRenderer artifact={tab.artifact || { artifact_id: tab.artifact_id, path: tab.path, title: tab.title }} workspaceName={workspaceName} ctx={ctx} />;
  }
  if (tab?.type === "activity") return <ActivityPreview tab={tab} />;
  if (loading) return <div className="v2-empty" role="status">Loading file preview…</div>;
  if (error) return <ErrorNotice error={error} />;
  if (!preview) return <div className="v2-empty">This file has no preview yet. Refresh it or download the original file.</div>;

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
      <ArtifactRenderer filePreview={preview} workspaceName={workspaceName} ctx={ctx} />
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

  function setTodoHeightPersisted(value) {
    const shell = shellRef.current;
    const maxHeight = shell
      ? Math.max(150, Math.min(560, shell.getBoundingClientRect().height - 170))
      : 560;
    const next = clampNumber(value, 140, maxHeight);
    setTodoHeight(next);
    window.localStorage.setItem("catmaster:v2:todo-panel-height", String(Math.round(next)));
  }

  function resizeTodoWithKeyboard(event) {
    let next = null;
    if (event.key === "ArrowUp") next = todoHeight + 16;
    if (event.key === "ArrowDown") next = todoHeight - 16;
    if (event.key === "Home") next = 140;
    if (event.key === "End") next = 560;
    if (next === null) return;
    event.preventDefault();
    setTodoHeightPersisted(next);
  }

  return (
    <aside ref={shellRef} className="v2-right-inspector v2-file-preview-tabs" style={{ "--v2-todo-panel-height": `${todoHeight}px` }}>
      <div className="v2-browser-tabs" role="tablist" aria-label="Open file previews">
        {(tabs || []).map((tab) => (
          <div
            key={tab.id}
            className={`v2-browser-tab ${activeTab?.id === tab.id ? "active" : ""}`}
            title={
              tab.type === "file" && !isInternalStoragePath(tab.path)
                ? (tab.path || tabTitle(tab))
                : tabTitle(tab)
            }
          >
            <button
              type="button"
              className="v2-browser-tab-main"
              role="tab"
              aria-selected={activeTab?.id === tab.id}
              aria-label={`Open ${tabTitle(tab)}`}
              onClick={() => onActivate?.(tab.id)}
            >
              <FileBox size={14} />
              <span>{tabTitle(tab)}</span>
            </button>
            <button
              type="button"
              className="v2-browser-tab-close"
              aria-label={`Close ${tabTitle(tab)}`}
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
      <div
        className="v2-todo-resize"
        role="separator"
        aria-label="Resize todo panel"
        aria-orientation="horizontal"
        aria-valuemin={140}
        aria-valuemax={560}
        aria-valuenow={Math.round(todoHeight)}
        aria-valuetext={`${Math.round(todoHeight)} pixels`}
        tabIndex={0}
        onPointerDown={startTodoResize}
        onKeyDown={resizeTodoWithKeyboard}
      >
        <ListChecks size={13} />
      </div>
      <TodoPanel groups={todoGroups} />
    </aside>
  );
}
