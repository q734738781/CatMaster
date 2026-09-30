import { useEffect, useId, useMemo, useRef, useState } from "react";
import { Activity, BarChart3, Braces, ChevronRight, Clock, Cpu, Database, Download, FileBox, GitBranch, ListChecks, MessageSquare, RefreshCw, Settings2, Trash2, Upload, X } from "lucide-react";

import ArtifactRenderer from "./ArtifactRenderer";
import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  displayValue,
  humanizeKey,
  isInternalStoragePath,
  makeApiError,
  presentError,
  redactErrorText,
  userFacingFileTitle,
} from "../presentation.js";
import {
  SELF_EVOLUTION_STATUS_VALUES,
  mergeSelfEvolutionCandidates,
  mergeSelfEvolutionObservations,
  redactSelfEvolutionText,
  selfEvolutionActionDefinition,
  selfEvolutionActionEndpoint,
  selfEvolutionActionRequest,
  selfEvolutionAllowedActions,
  selfEvolutionBehaviorChange,
  selfEvolutionCandidateTitle,
  selfEvolutionCandidateVersion,
  selfEvolutionDisplayError,
  selfEvolutionEvidenceItems,
  selfEvolutionFilterCandidates,
  selfEvolutionHumanReview,
  selfEvolutionJobView,
  selfEvolutionLifecycleLabel,
  selfEvolutionObservationView,
  selfEvolutionPromotionConfirmation,
  selfEvolutionRouteLabel,
  selfEvolutionSafeText,
  selfEvolutionStatusLabel,
  selfEvolutionTextItems,
  sortSelfEvolutionCandidates,
  sortSelfEvolutionObservations,
} from "../selfEvolutionView";

function escapePath(value) {
  return encodeURIComponent(String(value || ""));
}

function formatBytes(value) {
  const bytes = Number(value || 0);
  if (!Number.isFinite(bytes) || bytes <= 0) return "0 B";
  const units = ["B", "KB", "MB", "GB"];
  let size = bytes;
  let index = 0;
  while (size >= 1024 && index < units.length - 1) {
    size /= 1024;
    index += 1;
  }
  return `${size.toFixed(index ? 1 : 0)} ${units[index]}`;
}

function formatCount(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number)) return "0";
  return Intl.NumberFormat().format(number);
}

function formatCost(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "$0";
  if (number < 0.01) return `$${number.toExponential(2)}`;
  return `$${number.toFixed(2)}`;
}

function formatDurationSec(value) {
  const seconds = Number(value || 0);
  if (!Number.isFinite(seconds) || seconds <= 0) return "0s";
  if (seconds < 60) return `${seconds.toFixed(1)}s`;
  const minutes = Math.floor(seconds / 60);
  const rest = Math.round(seconds % 60);
  return `${minutes}m ${rest}s`;
}

function formatHours(value) {
  const hours = Number(value || 0);
  if (!Number.isFinite(hours) || hours <= 0) return "0";
  if (hours < 0.01) return hours.toExponential(2);
  return hours.toFixed(2);
}

function jsonText(value) {
  if (value == null || value === "") return "";
  return typeof value === "string" ? value : JSON.stringify(value, null, 2);
}

function formatTimestamp(value) {
  const number = Number(value || 0);
  const parsed = Date.parse(String(value || ""));
  const ms = Number.isFinite(number) && number > 0
    ? (number > 100000000000 ? number : number * 1000)
    : parsed;
  if (!Number.isFinite(ms) || ms <= 0) return "-";
  return new Date(ms).toLocaleString();
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function statusLabel(value) {
  const status = String(value || "idle").toLowerCase();
  return {
    idle: "Ready",
    created: "Ready",
    streaming: "In progress",
    queued: "Queued",
    pending: "Waiting",
    running: "Running",
    stopping: "Stopping",
    interrupted: "Waiting for review",
    resolved: "Reviewed",
    completed: "Completed",
    complete: "Completed",
    success: "Completed",
    failed: "Needs attention",
    error: "Needs attention",
  }[status] || displayValue(status.replace(/[_-]+/g, " "), "Ready");
}

function statusAwareText(value, status, fallback = "Not available") {
  const text = displayValue(value, fallback);
  return ["failed", "error"].includes(String(status || "").toLowerCase())
    ? redactErrorText(text)
    : text;
}

function ErrorNotice({ error, compact = false }) {
  const presented = presentError(error);
  if (!presented.message) return null;
  return (
    <div className={`v2-error ${compact ? "compact" : ""}`} role="alert">
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

function TreeNode({
  node,
  depth,
  selectedPath,
  childrenByPath,
  loadingByPath,
  pageByPath,
  onToggle,
  onSelect,
  onLoadMore,
}) {
  const isDirectory = node.node_type === "directory";
  const open = Boolean(childrenByPath[node.path || ""]);
  const allChildren = childrenByPath[node.path || ""] || [];
  const children = allChildren.filter((child) => !isInternalStoragePath(child?.path));
  return (
    <div className="v2-file-branch">
      <div className={`v2-file-row ${selectedPath === node.path ? "selected" : ""}`} style={{ paddingLeft: `${depth * 14}px` }}>
        <button
          type="button"
          className="v2-file-toggle"
          disabled={!isDirectory}
          aria-label={isDirectory ? `${open ? "Collapse" : "Expand"} ${node.name || "directory"}` : `File ${node.name || ""}`}
          aria-expanded={isDirectory ? open : undefined}
          title={displayValue(node.name || node.path, isDirectory ? "Directory" : "File")}
          onClick={() => onToggle(node)}
        >
          {isDirectory ? (open ? "-" : "+") : ""}
        </button>
        <button
          type="button"
          className={`v2-file-label kind-${node.preview_kind || node.node_type}`}
          aria-current={selectedPath === node.path ? "true" : undefined}
          aria-label={`Open ${displayValue(node.name || node.path, "file")}`}
          title={displayValue(node.name || node.path, "File")}
          onClick={() => onSelect(node)}
        >
          <span>{node.name || "."}</span>
          {node.node_type === "file" ? <small>{formatBytes(node.size)}</small> : null}
        </button>
      </div>
      {isDirectory && open ? (
        <div>
          {loadingByPath[node.path || ""] ? <div className="v2-file-note">Loading...</div> : null}
          {!loadingByPath[node.path || ""] && !children.length ? <div className="v2-file-note">Empty directory.</div> : null}
          {children.map((child) => (
            <TreeNode
              key={child.path || child.name}
              node={child}
              depth={depth + 1}
              selectedPath={selectedPath}
              childrenByPath={childrenByPath}
              loadingByPath={loadingByPath}
              pageByPath={pageByPath}
              onToggle={onToggle}
              onSelect={onSelect}
              onLoadMore={onLoadMore}
            />
          ))}
          {pageByPath[node.path || ""]?.truncated ? (
            <button
              type="button"
              className="v2-file-load-more"
              disabled={loadingByPath[node.path || ""]}
              onClick={() => onLoadMore(node.path || "", pageByPath[node.path || ""].next_cursor)}
            >
              Load more files ({allChildren.length} of {pageByPath[node.path || ""].total_count})
            </button>
          ) : null}
        </div>
      ) : null}
    </div>
  );
}

function FilePreview({ ctx, workspaceName, preview, loading, error, onRefresh, onDelete }) {
  if (loading) return <div className="v2-empty" role="status">Loading preview…</div>;
  if (error) return <ErrorNotice error={error} />;
  if (!preview) return (
    <div className="v2-file-preview-empty">
      <FileBox size={35} strokeWidth={1.2} />
      <h3>A closer look at your work</h3>
      <p>Select a file or folder to preview reports, structures, tables and source files here.</p>
    </div>
  );
  const archiveUrl = `/api/session/${escapePath(ctx)}/files/archive?path=${escapePath(preview.path || "")}&project_space=${escapePath(workspaceName || "")}`;
  return (
    <div className="v2-files-preview">
      <div className="v2-inspector-head">
        <div>
          <div className="v2-eyebrow">{humanizeKey(preview.kind || preview.node_type || "file")}</div>
          <h3>{displayValue(preview.name || preview.path, "Workspace files")}</h3>
          <div className="v2-muted">{preview.path || "."}</div>
        </div>
        <div className="v2-icon-row">
          <button type="button" className="v2-icon-btn" onClick={onRefresh} aria-label="Refresh preview"><RefreshCw size={16} /></button>
          {preview.download_url ? <a className="v2-icon-btn" href={preview.download_url} aria-label={`Download ${preview.name || "file"}`}><Download size={16} /></a> : null}
          <a className="v2-icon-btn" href={archiveUrl} aria-label={`Download ${preview.name || "user files"} as ZIP`}><FileBox size={16} /></a>
          {preview.path ? <button type="button" className="v2-icon-btn danger" onClick={onDelete} aria-label={`Delete ${preview.name || preview.path}`}><Trash2 size={16} /></button> : null}
        </div>
      </div>
      <div className="v2-file-meta">
        <span>{humanizeKey(preview.node_type || "file")}</span>
        <span>{formatBytes(preview.size)}</span>
        {preview.mime_type ? <span>{preview.mime_type}</span> : null}
      </div>
      {preview.kind === "directory" ? (
        <>
          <div className="v2-directory-list">
            {(preview.children || []).filter((child) => !isInternalStoragePath(child?.path)).map((child) => (
              <div key={child.path || child.name}>
                <span title={displayValue(child.name || child.path, "Item")}>{displayValue(child.name || child.path, "Item")}</span>
                <small>{child.node_type === "file" ? formatBytes(child.size) : "Folder"}</small>
              </div>
            ))}
          </div>
          {preview.page?.truncated ? (
            <div className="v2-truncation-notice" role="status">
              Showing {preview.page.shown_count} of {preview.page.total_count} entries. Use the Browse tree to load the remaining entries.
            </div>
          ) : null}
        </>
      ) : null}
      {preview.kind !== "directory" ? (
        <ArtifactRenderer filePreview={preview} showHeader={false} workspaceName={workspaceName} ctx={ctx} onRefresh={onRefresh} />
      ) : null}
    </div>
  );
}

export function FilesPanel({ ctx, workspaceName, selectedFilePath = "", onSelectFile }) {
  const [childrenByPath, setChildrenByPath] = useState({});
  const [pageByPath, setPageByPath] = useState({});
  const [loadingByPath, setLoadingByPath] = useState({});
  const [selectedPath, setSelectedPath] = useState("");
  const [preview, setPreview] = useState(null);
  const [activeFileTab, setActiveFileTab] = useState("browse");
  const [treeWidth, setTreeWidth] = useState(() => {
    if (typeof window === "undefined") return 360;
    const saved = Number(window.localStorage.getItem("catmaster:v2:file-tree-width"));
    return Number.isFinite(saved) && saved > 0 ? saved : 360;
  });
  const [error, setError] = useState("");
  const [previewError, setPreviewError] = useState("");
  const [previewLoading, setPreviewLoading] = useState(false);
  const [uploadStatus, setUploadStatus] = useState("");
  const inputRef = useRef(null);
  const shellRef = useRef(null);

  async function loadTree(path = "", cursor = "") {
    if (!ctx || !workspaceName) return;
    setLoadingByPath((prev) => ({ ...prev, [path]: true }));
    try {
      const cursorQuery = cursor ? `&cursor=${escapePath(cursor)}` : "";
      const payload = await apiFetch(`/api/session/${escapePath(ctx)}/files/tree?path=${escapePath(path)}&project_space=${escapePath(workspaceName)}${cursorQuery}`);
      const key = payload.path || "";
      setChildrenByPath((prev) => ({
        ...prev,
        [key]: cursor
          ? [...(prev[key] || []), ...(payload.children || [])]
          : (payload.children || []),
      }));
      setPageByPath((prev) => ({ ...prev, [key]: payload.page || {} }));
      setError("");
    } catch (err) {
      setError(err);
    } finally {
      setLoadingByPath((prev) => ({ ...prev, [path]: false }));
    }
  }

  async function loadPreview(path = "") {
    setPreviewLoading(true);
    setPreviewError("");
    try {
      const payload = await apiFetch(`/api/session/${escapePath(ctx)}/files/content?path=${escapePath(path)}&project_space=${escapePath(workspaceName)}`);
      setPreview(payload);
      onSelectFile?.({ path, preview: payload });
    } catch (err) {
      setPreviewError(err);
    } finally {
      setPreviewLoading(false);
    }
  }

  useEffect(() => {
    setChildrenByPath({});
    setPageByPath({});
    setSelectedPath("");
    setPreview(null);
    loadTree("");
  }, [ctx, workspaceName]);

  useEffect(() => {
    const target = String(selectedFilePath || "");
    if (!ctx || !workspaceName) return;
    if (!target || previewLoading || preview?.path === target) return;
    setSelectedPath(target);
    loadPreview(target);
    const parent = target.split("/").slice(0, -1).join("/");
    if (parent) loadTree(parent);
  }, [selectedFilePath, ctx, workspaceName]);

  async function selectNode(node) {
    const path = node.path || "";
    setSelectedPath(path);
    if (node.node_type === "directory") await loadTree(path);
    await loadPreview(path);
  }

  function toggleTree(node) {
    const path = node.path || "";
    if (Object.prototype.hasOwnProperty.call(childrenByPath, path)) {
      setChildrenByPath((current) => {
        const next = { ...current };
        delete next[path];
        return next;
      });
      setPageByPath((current) => {
        const next = { ...current };
        delete next[path];
        return next;
      });
      return;
    }
    loadTree(path);
  }

  async function uploadFiles(files) {
    const list = Array.from(files || []);
    if (!list.length) return;
    const target = preview?.node_type === "directory" ? preview.path || "" : selectedPath.split("/").slice(0, -1).join("/");
    for (const file of list) {
      setUploadStatus(`Uploading ${file.name}...`);
      const response = await fetch(`/api/session/${escapePath(ctx)}/files/upload?path=${escapePath(target)}&filename=${escapePath(file.name)}&overwrite=true&project_space=${escapePath(workspaceName)}`, {
        method: "POST",
        body: file,
      });
      if (!response.ok) {
        const text = await response.text();
        throw makeApiError(response.status, text, response.headers.get("content-type") || "");
      }
    }
    setUploadStatus(`Uploaded ${list.length} file(s).`);
    await loadTree(target);
    if (selectedPath) await loadPreview(selectedPath);
  }

  async function deleteSelected() {
    if (!preview?.path) return;
    if (!window.confirm(`Delete ${preview.path}?`)) return;
    const response = await fetch(`/api/session/${escapePath(ctx)}/files/delete?path=${escapePath(preview.path)}&project_space=${escapePath(workspaceName)}`, { method: "DELETE" });
    if (!response.ok) {
      const text = await response.text();
      throw makeApiError(response.status, text, response.headers.get("content-type") || "");
    }
    const parent = preview.path.split("/").slice(0, -1).join("/");
    setPreview(null);
    setSelectedPath(parent);
    await loadTree(parent);
  }

  function startTreeResize(event) {
    event.preventDefault();
    const shell = shellRef.current;
    if (!shell) return;
    document.body.classList.add("v2-resizing-columns");
    const rect = shell.getBoundingClientRect();
    const move = (moveEvent) => {
      const maxWidth = Math.max(260, rect.width - 360);
      const next = clampNumber(moveEvent.clientX - rect.left, 240, Math.min(680, maxWidth));
      setTreeWidth(next);
      window.localStorage.setItem("catmaster:v2:file-tree-width", String(Math.round(next)));
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

  function setTreeWidthPersisted(value) {
    const shell = shellRef.current;
    const rect = shell?.getBoundingClientRect();
    const availableMax = rect ? Math.max(260, rect.width - 360) : 680;
    const maxWidth = Math.min(680, availableMax);
    const next = clampNumber(value, 240, maxWidth);
    setTreeWidth(next);
    window.localStorage.setItem("catmaster:v2:file-tree-width", String(Math.round(next)));
  }

  function resizeTreeWithKeyboard(event) {
    let next = null;
    if (event.key === "ArrowLeft") next = treeWidth - 16;
    if (event.key === "ArrowRight") next = treeWidth + 16;
    if (event.key === "Home") next = 240;
    if (event.key === "End") next = 680;
    if (next === null) return;
    event.preventDefault();
    setTreeWidthPersisted(next);
  }

  const allRoots = childrenByPath[""] || [];
  const roots = allRoots.filter((node) => !isInternalStoragePath(node?.path));
  const previewNode = (
    <FilePreview
      ctx={ctx}
      workspaceName={workspaceName}
      preview={preview}
      loading={previewLoading}
      error={previewError}
      onRefresh={() => loadPreview(selectedPath)}
      onDelete={() => deleteSelected().catch(setPreviewError)}
    />
  );

  return (
    <section className="v2-tab-panel files">
      <div className="v2-panel-toolbar">
        <div>
          <div className="v2-eyebrow">Files</div>
          <h2>Workspace Explorer</h2>
        </div>
        <div className="v2-icon-row">
          <button type="button" className="v2-ghost-btn" onClick={() => loadTree("")}><RefreshCw size={15} />Refresh</button>
          <button type="button" className="v2-ghost-btn" onClick={() => inputRef.current?.click()}><Upload size={15} />Upload</button>
          <input
            ref={inputRef}
            type="file"
            multiple
            className="v2-hidden-input"
            onChange={(event) => uploadFiles(event.target.files).catch((reason) => {
              setUploadStatus("");
              setError(reason);
            })}
          />
        </div>
      </div>
      <div className="v2-file-tabs" role="tablist" aria-label="File workspace views">
        <button type="button" className={activeFileTab === "browse" ? "active" : ""} onClick={() => setActiveFileTab("browse")} role="tab" aria-selected={activeFileTab === "browse"}>Browse</button>
        <button type="button" className={activeFileTab === "preview" ? "active" : ""} onClick={() => setActiveFileTab("preview")} role="tab" aria-selected={activeFileTab === "preview"}>Preview</button>
        <button type="button" className={activeFileTab === "uploads" ? "active" : ""} onClick={() => setActiveFileTab("uploads")} role="tab" aria-selected={activeFileTab === "uploads"}>Uploads</button>
      </div>
      <ErrorNotice error={error} />
      {uploadStatus ? <div className="v2-muted">{uploadStatus}</div> : null}
      {activeFileTab === "browse" ? (
        <div ref={shellRef} className="v2-files-shell resizable" style={{ "--v2-file-tree-width": `${treeWidth}px` }}>
          <div className="v2-files-tree">
            {roots.map((node) => (
              <TreeNode
                key={node.path || node.name}
                node={node}
                depth={0}
                selectedPath={selectedPath}
                childrenByPath={childrenByPath}
                loadingByPath={loadingByPath}
                pageByPath={pageByPath}
                onToggle={toggleTree}
                onSelect={selectNode}
                onLoadMore={loadTree}
              />
            ))}
            {pageByPath[""]?.truncated ? (
              <button
                type="button"
                className="v2-file-load-more"
                disabled={loadingByPath[""]}
                onClick={() => loadTree("", pageByPath[""].next_cursor)}
              >
                Load more files ({allRoots.length} of {pageByPath[""].total_count})
              </button>
            ) : null}
            {!loadingByPath[""] && !roots.length ? (
              <div className="v2-empty">No files yet. Upload a file or create one through a task.</div>
            ) : null}
            {loadingByPath[""] && !roots.length ? <div className="v2-empty" role="status">Loading workspace files…</div> : null}
          </div>
          <div
            className="v2-resize-handle v2-resize-handle-files"
            role="separator"
            aria-label="Resize file browser"
            aria-orientation="vertical"
            aria-valuemin={240}
            aria-valuemax={680}
            aria-valuenow={Math.round(treeWidth)}
            aria-valuetext={`${Math.round(treeWidth)} pixels`}
            tabIndex={0}
            onPointerDown={startTreeResize}
            onKeyDown={resizeTreeWithKeyboard}
          />
          {previewNode}
        </div>
      ) : null}
      {activeFileTab === "preview" ? (
        <div className="v2-files-tab-content">
          {previewNode}
        </div>
      ) : null}
      {activeFileTab === "uploads" ? (
        <div className="v2-files-tab-content v2-files-upload-panel">
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><Upload size={15} /><h3>Upload Target</h3></div>
            <div className="v2-file-meta">
              <span>{workspaceName || "workspace"}</span>
              <span>{preview?.node_type === "directory" ? preview.path || "." : selectedPath.split("/").slice(0, -1).join("/") || "."}</span>
            </div>
            <button type="button" className="v2-ghost-btn" onClick={() => inputRef.current?.click()}><Upload size={15} />Choose files</button>
            <p className="v2-muted">{uploadStatus || "Files are uploaded into the selected directory, or beside the selected file."}</p>
          </section>
        </div>
      ) : null}
    </section>
  );
}

function MonitorMetric({ icon: Icon, label, value, note }) {
  return (
    <div className="v2-monitor-metric">
      <Icon size={15} />
      <span>{label}</span>
      <strong>{value}</strong>
      {note ? <small>{note}</small> : null}
    </div>
  );
}

function ModelUsageDetails({ usage }) {
  const rows = Array.isArray(usage?.by_model) ? usage.by_model : [];
  return (
    <details className="v2-model-usage-details">
      <summary>
        <span><BarChart3 size={15} />Token details by model</span>
        <small>{rows.length ? `${formatCount(rows.length)} model label${rows.length === 1 ? "" : "s"}` : "No model detail yet"}</small>
      </summary>
      <div className="v2-model-usage-body">
        {rows.length ? (
          <div className="v2-table-wrap">
            <table className="v2-table">
              <thead>
                <tr>
                  <th scope="col">Model label</th>
                  <th scope="col">Calls</th>
                  <th scope="col">Uncached input</th>
                  <th scope="col">Cached input</th>
                  <th scope="col">Cache write</th>
                  <th scope="col">Output</th>
                  <th scope="col">Total</th>
                </tr>
              </thead>
              <tbody>
                {rows.map((row, index) => (
                  <tr key={`${row.model_label || "model"}-${index}`}>
                    <th scope="row">{displayValue(row.model_label, "Unknown model")}</th>
                    <td>{formatCount(row.calls)}</td>
                    <td>{formatCount(row.input_uncached_tokens)}</td>
                    <td>{formatCount(row.input_cached_tokens)}</td>
                    <td>{formatCount(row.input_cache_write_tokens)}</td>
                    <td>{formatCount(row.output_tokens)}</td>
                    <td>{formatCount(row.total_tokens)}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ) : <div className="v2-empty compact">Per-model token details will appear after the first completed model call.</div>}
      </div>
    </details>
  );
}

function EvolutionTextList({ items, empty = "None recorded." }) {
  const rows = selfEvolutionTextItems(items);
  return rows.length ? (
    <ul className="v2-evolution-text-list">
      {rows.map((item, index) => <li key={`${item}-${index}`}>{item}</li>)}
    </ul>
  ) : <p className="v2-muted">{empty}</p>;
}

function EvolutionEvidenceList({ items }) {
  const rows = selfEvolutionEvidenceItems(items);
  return rows.length ? (
    <div className="v2-evolution-evidence-list">
      {rows.map((item, index) => (
        <article key={`${item.title}-${index}`}>
          {item.title ? <strong>{item.title}</strong> : null}
          {item.summary ? <p>{item.summary}</p> : null}
          <footer>
            {item.sourceLabel ? <span>{item.sourceLabel}</span> : null}
            {item.href ? <a href={item.href} target="_blank" rel="noreferrer">Open source</a> : null}
            {!item.href && item.sourceRef ? (
              <details className="v2-evolution-source-ref">
                <summary>Trace reference</summary>
                <code>{item.sourceRef}</code>
              </details>
            ) : null}
          </footer>
        </article>
      ))}
    </div>
  ) : <p className="v2-muted">No evidence source was supplied.</p>;
}

function CandidateDiff({ candidate }) {
  const technical = candidate?.technical_details && typeof candidate.technical_details === "object"
    ? candidate.technical_details
    : {};
  const diffRef = String(candidate?.diff_ref || technical.diff_ref || "");
  const inlineDiff = redactSelfEvolutionText(candidate?.technical_diff || technical.diff || "");
  const technicalNotes = selfEvolutionTextItems(technical.notes || technical.summary);
  const [text, setText] = useState(inlineDiff);
  const [page, setPage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setText(inlineDiff);
    setPage(null);
    setError("");
  }, [candidate?.candidate_id, candidate?.revision, inlineDiff]);

  async function load(cursor = "") {
    if (!diffRef || loading) return;
    setLoading(true);
    setError("");
    try {
      const separator = diffRef.includes("?") ? "&" : "?";
      const url = cursor ? `${diffRef}${separator}cursor=${encodeURIComponent(cursor)}` : diffRef;
      const response = await apiFetch(url);
      const nextText = redactSelfEvolutionText(response.diff || "");
      setText((current) => (cursor ? `${current}${nextText}` : nextText));
      setPage(response.page || null);
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The reviewed diff could not be loaded. Refresh and try again."));
    } finally {
      setLoading(false);
    }
  }

  if (!diffRef && !inlineDiff && !technicalNotes.length) return null;
  return (
    <details
      className="v2-evolution-secondary"
      aria-busy={loading}
      onToggle={(event) => {
        if (event.currentTarget.open && diffRef && !page && !text && !loading) load();
      }}
    >
      <summary>Technical details and reviewed diff</summary>
      <div className="v2-evolution-technical">
        {technicalNotes.length ? <EvolutionTextList items={technicalNotes} /> : null}
        {text ? <pre>{text}</pre> : null}
        {!text && diffRef && !loading && !error ? <p className="v2-muted">Open this section to load the redacted reviewed diff.</p> : null}
        {loading ? <p className="v2-muted" role="status">Loading reviewed diff…</p> : null}
        {error ? <div className="v2-error compact" role="alert">{error}</div> : null}
        {page ? (
          <div className="v2-truncation-notice">
            <span>
              Loaded {Number(page.shown_count || text.length).toLocaleString()} of{" "}
              {Number(page.total_count || 0).toLocaleString()} characters.
            </span>
            {page.truncated && page.next_cursor ? (
              <button type="button" className="v2-ghost-btn compact" onClick={() => load(page.next_cursor)} disabled={loading}>
                Load more diff
              </button>
            ) : null}
          </div>
        ) : null}
      </div>
    </details>
  );
}

function SelfEvolutionActionDialog({
  candidate,
  action,
  workspaceName,
  busy,
  error,
  onClose,
  onSubmit,
}) {
  const titleId = useId();
  const descriptionId = useId();
  const dialogRef = useRef(null);
  const initialFocusRef = useRef(null);
  const [rationale, setRationale] = useState("");
  const [guidance, setGuidance] = useState("");
  const [scopeKind, setScopeKind] = useState("thread");
  const [scopeId, setScopeId] = useState("");
  const [confirmed, setConfirmed] = useState(false);
  const definition = selfEvolutionActionDefinition(action);
  const isIndependentReview = action === "run-review";
  const isRevision = action === "request-revision";
  const isCanary = action === "start-canary";
  const isStablePromotion = action === "promote-stable";
  const needsExactConfirmation = isCanary || isStablePromotion;
  const needsSafetyConfirmation = ["reject", "quarantine", "retire", "rollback"].includes(action);
  const scopeLabel = isCanary
    ? `${scopeKind === "run" ? "Run" : "Thread"} ${scopeId.trim() || "(select a reference)"}`
    : `Stable for every future run in ${workspaceName || "this workspace"}`;
  const review = selfEvolutionHumanReview(candidate);
  const canSubmit = Boolean(
    definition
    && (isIndependentReview || rationale.trim())
    && (!isRevision || guidance.trim())
    && (!isCanary || scopeId.trim())
    && (!(needsExactConfirmation || needsSafetyConfirmation) || confirmed),
  );

  useEffect(() => {
    setRationale("");
    setGuidance("");
    setScopeKind("thread");
    setScopeId("");
    setConfirmed(false);
    const frame = window.requestAnimationFrame(() => initialFocusRef.current?.focus());
    return () => window.cancelAnimationFrame(frame);
  }, [candidate?.candidate_id, candidate?.revision, action]);

  useEffect(() => {
    function handleKeyDown(event) {
      if (event.key === "Escape" && !busy) {
        event.preventDefault();
        onClose();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const controls = [...dialogRef.current.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), summary, a[href]',
      )].filter((element) => element.getClientRects().length > 0);
      if (!controls.length) return;
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    }
    window.addEventListener("keydown", handleKeyDown);
    return () => window.removeEventListener("keydown", handleKeyDown);
  }, [busy, onClose]);

  if (!candidate || !definition) return null;
  return (
    <div
      className="v2-evolution-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => {
        if (event.target === event.currentTarget && !busy) onClose();
      }}
    >
      <section
        ref={dialogRef}
        className="v2-evolution-dialog"
        role="dialog"
        aria-modal="true"
        aria-labelledby={titleId}
        aria-describedby={descriptionId}
        aria-busy={busy}
      >
        <header>
          <div>
            <div className="v2-eyebrow">Human decision</div>
            <h3 id={titleId}>{definition.label}</h3>
          </div>
          <button type="button" className="v2-icon-btn" aria-label={`Close ${definition.label}`} onClick={onClose} disabled={busy}>
            <X size={17} />
          </button>
        </header>
        <form
          onSubmit={(event) => {
            event.preventDefault();
            if (canSubmit) onSubmit({ rationale, guidance, scopeKind, scopeId });
          }}
        >
          <p id={descriptionId}>{definition.description}</p>
          <div className="v2-evolution-dialog-target">
            <strong>{selfEvolutionCandidateTitle(candidate)}</strong>
            <span>{selfEvolutionCandidateVersion(candidate)}</span>
          </div>
          {needsExactConfirmation ? (
            <section className="v2-evolution-confirmation" aria-label="Exact release confirmation">
              <h4>Confirm the exact release</h4>
              <dl>
                <div><dt>Version</dt><dd>{selfEvolutionCandidateVersion(candidate)}</dd></div>
                <div><dt>Target</dt><dd>{selfEvolutionCandidateTitle(candidate)}</dd></div>
                <div><dt>Scope</dt><dd>{scopeLabel}</dd></div>
              </dl>
              <h5>Reviewer concerns</h5>
              <EvolutionTextList items={review.concerns} empty="No reviewer concerns were recorded." />
            </section>
          ) : null}
          {isCanary ? (
            <div className="v2-evolution-dialog-fields two-column">
              <label>
                <span>Canary scope</span>
                <select value={scopeKind} onChange={(event) => setScopeKind(event.target.value)}>
                  <option value="thread">One thread</option>
                  <option value="run">One run</option>
                </select>
              </label>
              <label>
                <span>{scopeKind === "run" ? "Run reference" : "Thread reference"}</span>
                <input
                  value={scopeId}
                  onChange={(event) => setScopeId(event.target.value)}
                  placeholder={`Enter the exact ${scopeKind} reference`}
                  autoComplete="off"
                  required
                />
              </label>
            </div>
          ) : null}
          {isIndependentReview ? null : (
            <label className="v2-evolution-dialog-field">
              <span>{isRevision ? "Why is a new revision needed?" : "Decision rationale"}</span>
              <textarea
                ref={initialFocusRef}
                value={rationale}
                onChange={(event) => setRationale(event.target.value)}
                rows={3}
                required
                placeholder="Record the evidence-based reason for this human decision."
              />
            </label>
          )}
          {isRevision ? (
            <label className="v2-evolution-dialog-field">
              <span>Guidance for the next revision</span>
              <textarea
                value={guidance}
                onChange={(event) => setGuidance(event.target.value)}
                rows={3}
                required
                placeholder="State the boundary, evidence, or behavior that the next immutable revision must address."
              />
            </label>
          ) : null}
          {needsExactConfirmation || needsSafetyConfirmation ? (
            <label className="v2-evolution-confirm-check">
              <input
                type="checkbox"
                checked={confirmed}
                onChange={(event) => setConfirmed(event.target.checked)}
              />
              <span>
                {needsExactConfirmation
                  ? "I reviewed this exact version, target, scope, applicability boundaries, and concerns."
                  : `I understand that this action applies to ${selfEvolutionCandidateVersion(candidate)} of ${selfEvolutionCandidateTitle(candidate)}.`}
              </span>
            </label>
          ) : null}
          {isStablePromotion ? (
            <details className="v2-evolution-dialog-summary">
              <summary>Read the full release confirmation</summary>
              <p>{selfEvolutionPromotionConfirmation(candidate, workspaceName, scopeLabel)}</p>
            </details>
          ) : null}
          {error ? <div className="v2-error compact" role="alert">{error}</div> : null}
          <footer>
            <button type="button" className="v2-ghost-btn" onClick={onClose} disabled={busy}>Cancel</button>
            <button
              ref={isIndependentReview ? initialFocusRef : undefined}
              type="submit"
              className={isStablePromotion || isCanary ? "v2-primary-btn" : "v2-ghost-btn"}
              disabled={!canSubmit || busy}
            >
              {busy ? "Saving decision…" : definition.submitLabel}
            </button>
          </footer>
        </form>
      </section>
    </div>
  );
}

function SelfEvolutionCandidateCard({ candidate }) {
  const behavior = selfEvolutionBehaviorChange(candidate);
  const review = selfEvolutionHumanReview(candidate);
  const episodeCount = Array.isArray(candidate.evidence)
    ? candidate.evidence.length
    : (candidate.evidence ? 1 : 0);
  const counterexamples = [
    ...review.counterexamples,
  ].filter((item, index, rows) => rows.indexOf(item) === index);
  const applicability = selfEvolutionTextItems(candidate.applicability_boundary);
  const nonApplicability = selfEvolutionTextItems(candidate.non_applicability);
  const candidateKey = `${candidate.candidate_id}:${candidate.revision ?? candidate.version ?? ""}`;

  return (
    <details className="v2-evolution-candidate">
      <summary className="v2-evolution-card-summary">
        <div className="v2-evolution-card-main">
          <div className="v2-evolution-card-title">
            <strong>{selfEvolutionCandidateTitle(candidate)}</strong>
            <small className={`status-${String(candidate.status || "")}`}>{selfEvolutionLifecycleLabel(candidate)}</small>
          </div>
          <p>{behavior.summary}</p>
          <div className="v2-evolution-card-facts">
            <span>{selfEvolutionCandidateVersion(candidate)}</span>
            <span>{selfEvolutionRouteLabel(candidate.route)}</span>
            {candidate.content_parent_version ? <span>Parent {candidate.content_parent_version}</span> : null}
            <span>{episodeCount} complete episode observation{episodeCount === 1 ? "" : "s"}</span>
            <span>{review.recommendationLabel}</span>
            <span>Diagnostic history</span>
          </div>
        </div>
      </summary>
      <div className="v2-evolution-review">
        <section>
          <h4>What behavior will change</h4>
          <p>{behavior.summary}</p>
          {behavior.before || behavior.after || behavior.impact ? (
            <dl className="v2-evolution-change-dl">
              {behavior.before ? <div><dt>Before</dt><dd>{behavior.before}</dd></div> : null}
              {behavior.after ? <div><dt>After</dt><dd>{behavior.after}</dd></div> : null}
              {behavior.impact ? <div><dt>Expected impact</dt><dd>{behavior.impact}</dd></div> : null}
            </dl>
          ) : null}
          {review.changePoints.map((point, index) => (
            <article key={`${candidateKey}-change-${index}`} className="v2-evolution-change-point">
              <strong>{point.title}</strong>
              <dl className="v2-evolution-change-dl">
                {point.before ? <div><dt>Before</dt><dd>{point.before}</dd></div> : null}
                {point.after ? <div><dt>After</dt><dd>{point.after}</dd></div> : null}
                {point.impact ? <div><dt>Impact</dt><dd>{point.impact}</dd></div> : null}
                {point.evidence ? (
                  <div>
                    <dt>Evidence</dt>
                    <dd>{point.evidence}{point.evidenceSource ? ` · ${point.evidenceSource}` : ""}</dd>
                  </div>
                ) : null}
              </dl>
            </article>
          ))}
        </section>
        <section>
          <h4>Why it is being proposed</h4>
          {candidate.why_now ? <p>{candidate.why_now}</p> : null}
          <p>{selfEvolutionSafeText(candidate.evidence_summary, "No evidence summary was supplied.")}</p>
          <EvolutionEvidenceList items={candidate.evidence} />
        </section>
        <section>
          <h4>Where it applies — and where it must not</h4>
          <div className="v2-evolution-boundary-grid">
            <div>
              <h5>Applies when</h5>
              <EvolutionTextList items={applicability} empty="No applicability boundary was supplied." />
            </div>
            <div>
              <h5>Must not apply when</h5>
              <EvolutionTextList items={nonApplicability} empty="No non-applicability boundary was supplied." />
            </div>
          </div>
          <h5>Counterexamples</h5>
          <EvolutionTextList items={counterexamples} empty="No counterexample was supplied." />
        </section>
        <section>
          <h4>Independent review</h4>
          <p><strong>{review.recommendationLabel}</strong></p>
          <p>{review.summary}</p>
          {review.evidenceSufficiency ? <p><strong>Evidence sufficiency:</strong> {review.evidenceSufficiency}</p> : null}
          {review.scopeAssessment ? <p><strong>Scope assessment:</strong> {review.scopeAssessment}</p> : null}
          <p>
            <strong>Proportionality:</strong> {review.proportionality.label}
            {review.proportionality.explanation ? ` · ${review.proportionality.explanation}` : ""}
          </p>
          <h5>Reviewer concerns</h5>
          <EvolutionTextList items={review.concerns} empty="No reviewer concerns were recorded." />
        </section>
        <CandidateDiff candidate={candidate} />
      </div>
    </details>
  );
}

function SelfEvolutionObservationCard({ observation }) {
  const view = selfEvolutionObservationView(observation);
  return (
    <article className="v2-evolution-observation">
      <header>
        <span>{view.signalLabel}</span>
        <small>{view.statusLabel}</small>
      </header>
      <strong>{view.title}</strong>
      {view.summary ? <p>{view.summary}</p> : null}
      {view.target ? <p><strong>Initial target:</strong> {view.target}</p> : null}
      {view.resolvedTarget && view.resolvedTarget !== view.target ? (
        <p><strong>Resolved owner:</strong> {view.resolvedTarget}</p>
      ) : null}
      {view.outcome ? <p><strong>Observed outcome:</strong> {view.outcome}</p> : null}
      {view.evidence.length ? (
        <details>
          <summary>Supporting source excerpts</summary>
          <EvolutionEvidenceList items={observation.evidence || observation.evidence_refs} />
        </details>
      ) : null}
      {view.createdAt ? <time dateTime={view.createdAt}>{formatTimestamp(view.createdAt)}</time> : null}
    </article>
  );
}

function mergeSelfEvolutionJobs(current, incoming) {
  const merged = new Map();
  [...(Array.isArray(current) ? current : []), ...(Array.isArray(incoming) ? incoming : [])]
    .forEach((job) => {
      const jobId = String(job?.job_id || "");
      if (jobId) merged.set(jobId, job);
    });
  return [...merged.values()].sort((left, right) => (
    String(right?.updated_at || right?.created_at || "")
      .localeCompare(String(left?.updated_at || left?.created_at || ""))
  ));
}

function SelfEvolutionJobCard({ job, busy, onRetry }) {
  const view = selfEvolutionJobView(job);
  const resultItems = view.items;
  return (
    <article className="v2-evolution-job">
      <header>
        <div>
          <strong>{selfEvolutionSafeText(job?.title, "Learning review")}</strong>
          <small>{view.resultLabel}</small>
        </div>
        {job?.retry_allowed ? (
          <button
            type="button"
            className="v2-ghost-btn"
            disabled={busy}
            onClick={() => onRetry(job)}
          >
            <RefreshCw size={13} aria-hidden="true" />
            {busy ? "Queuing retry…" : "Retry this job"}
          </button>
        ) : null}
      </header>
      <p>{view.summary}</p>
      {resultItems.length ? (
        <div className="v2-evolution-job-items" aria-label="Learning results">
          {resultItems.map((item) => (
            <div key={item.itemRef}>
              <div className="v2-evolution-job-item-copy">
                <strong>{item.title}</strong>
                <small>{item.statusLabel}</small>
                {item.resolvedTarget && item.resolvedTarget !== item.title ? (
                  <span><strong>Resolved owner:</strong> {item.resolvedTarget}</span>
                ) : null}
                {item.change ? <span><strong>Reflected change:</strong> {item.change}</span> : null}
                {item.rationale && item.rationale !== item.change ? (
                  <span><strong>Why:</strong> {item.rationale}</span>
                ) : null}
                {item.reason && item.reason !== item.change && item.reason !== item.rationale ? <span>{item.reason}</span> : null}
                {item.evidenceRefs.length ? (
                  <details>
                    <summary>Reflection evidence handles</summary>
                    <EvolutionEvidenceList items={item.evidenceRefs} />
                  </details>
                ) : null}
                {item.error ? <span>{item.error}</span> : null}
              </div>
              {item.retryAllowed ? (
                <button
                  type="button"
                  className="v2-ghost-btn"
                  disabled={busy}
                  onClick={() => onRetry(job, item.itemRef)}
                >
                  <RefreshCw size={13} aria-hidden="true" />
                  {busy ? "Queuing retry…" : "Retry failed item"}
                </button>
              ) : null}
            </div>
          ))}
        </div>
      ) : null}
      <footer>
        <span>{Number(job?.attempt_count || 0) > 0 ? `Attempt ${formatCount(job.attempt_count)}` : "Not started"}</span>
        <time dateTime={job?.updated_at || job?.created_at || ""}>
          {formatTimestamp(job?.updated_at || job?.created_at)}
        </time>
      </footer>
    </article>
  );
}

const SKILL_EVOLUTION_MODES = [
  { value: "off", label: "Off", note: "Keep current selections; stop new evolution jobs." },
  { value: "observe", label: "Observe", note: "Learn and review, but leave approved revisions dormant." },
  { value: "auto", label: "Auto", note: "Use reviewer-approved revisions on the next run when a target follows auto." },
];

function effectiveSkillSourceLabel(value) {
  return {
    repository: "Repository",
    repository_and_workspace: "Repository + workspace",
    workspace: "Workspace",
    workspace_guidance: "Workspace guidance",
  }[String(value || "")] || "Workspace";
}

function mergeEffectiveSkills(current, incoming) {
  const merged = new Map();
  [...(Array.isArray(current) ? current : []), ...(Array.isArray(incoming) ? incoming : [])]
    .forEach((skill) => {
      const target = String(skill?.target || "");
      if (target) merged.set(target, { ...(merged.get(target) || {}), ...skill });
    });
  return [...merged.values()].sort((left, right) => (
    String(left?.target || "").localeCompare(String(right?.target || ""))
  ));
}

function SkillsManagerDialog({ ctx, workspaceName, onClose, onChanged }) {
  const [skills, setSkills] = useState([]);
  const [catalogCursor, setCatalogCursor] = useState("");
  const [totalCount, setTotalCount] = useState(0);
  const [mode, setMode] = useState("auto");
  const [modeSource, setModeSource] = useState("deployment_default");
  const [modeDraft, setModeDraft] = useState("auto");
  const [modePreview, setModePreview] = useState(null);
  const [selectedTarget, setSelectedTarget] = useState("");
  const [detail, setDetail] = useState(null);
  const [draft, setDraft] = useState(null);
  const [loading, setLoading] = useState("catalog");
  const [error, setError] = useState("");
  const [message, setMessage] = useState("");

  const selectedSummary = useMemo(
    () => skills.find((item) => item.target === selectedTarget) || null,
    [skills, selectedTarget],
  );
  const selectableVersions = useMemo(
    () => (Array.isArray(detail?.versions) ? detail.versions : []).filter((item) => item.eligible),
    [detail],
  );
  const hasTargetChanges = Boolean(detail && draft && (
    Boolean(draft.enabled) !== Boolean(detail.enabled)
    || String(draft.selected_version || "") !== String(detail.selected_version || "")
    || String(draft.update_policy || "") !== String(detail.update_policy || "")
  ));

  function catalogParams(cursor = "") {
    const params = new URLSearchParams();
    if (workspaceName) params.set("project_space", workspaceName);
    if (cursor) params.set("cursor", cursor);
    params.set("limit", "50");
    return params;
  }

  async function loadCatalog({ append = false, cursor = "" } = {}) {
    if (!ctx) return;
    setLoading(append ? "catalog-more" : "catalog");
    setError("");
    try {
      const page = await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/skills?${catalogParams(cursor).toString()}`,
      );
      const nextSkills = Array.isArray(page?.skills) ? page.skills : [];
      setSkills((current) => (append ? mergeEffectiveSkills(current, nextSkills) : nextSkills));
      setCatalogCursor(String(page?.next_cursor || ""));
      setTotalCount(Number(page?.total_count || nextSkills.length));
      setMode(String(page?.mode || "auto"));
      setModeDraft(String(page?.mode || "auto"));
      setModeSource(String(page?.source || "deployment_default"));
      setModePreview(null);
      if (!append) {
        const nextTarget = nextSkills.some((item) => item.target === selectedTarget)
          ? selectedTarget
          : String(nextSkills[0]?.target || "");
        setSelectedTarget(nextTarget);
      }
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The effective Skills catalog could not be loaded."));
    } finally {
      setLoading("");
    }
  }

  async function loadDetail(target, { versionCursor = "", historyCursor = 0, appendKind = "" } = {}) {
    if (!ctx || !target) return;
    setLoading(appendKind || "detail");
    setError("");
    const params = new URLSearchParams({ target });
    if (workspaceName) params.set("project_space", workspaceName);
    if (versionCursor) params.set("version_cursor", versionCursor);
    if (historyCursor) params.set("history_cursor", String(historyCursor));
    params.set("version_limit", appendKind === "history-more" ? "1" : "50");
    params.set("history_limit", appendKind === "versions-more" ? "1" : "50");
    try {
      const response = await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/skills/detail?${params.toString()}`,
      );
      const next = response?.skill || null;
      if (!next) throw new Error("The selected skill detail was empty.");
      if (appendKind) {
        setDetail((current) => ({
          ...current,
          ...next,
          versions: appendKind === "versions-more"
            ? [...(current?.versions || []), ...(next.versions || [])]
            : (current?.versions || []),
          history: appendKind === "history-more"
            ? [...(current?.history || []), ...(next.history || [])]
            : (current?.history || []),
          version_next_cursor: appendKind === "versions-more"
            ? String(next.version_next_cursor || "")
            : String(current?.version_next_cursor || ""),
          history_next_cursor: appendKind === "history-more"
            ? Number(next.history_next_cursor || 0)
            : Number(current?.history_next_cursor || 0),
        }));
      } else {
        setDetail(next);
        setDraft({
          enabled: Boolean(next.enabled),
          selected_version: String(next.selected_version || ""),
          update_policy: String(next.update_policy || "follow_auto"),
        });
      }
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The selected skill could not be loaded."));
    } finally {
      setLoading("");
    }
  }

  useEffect(() => {
    loadCatalog();
  }, [ctx, workspaceName]);

  useEffect(() => {
    if (selectedTarget) loadDetail(selectedTarget);
    else {
      setDetail(null);
      setDraft(null);
    }
  }, [selectedTarget]);

  useEffect(() => {
    function onKeyDown(event) {
      if (event.key === "Escape") onClose();
    }
    window.addEventListener("keydown", onKeyDown);
    return () => window.removeEventListener("keydown", onKeyDown);
  }, [onClose]);

  async function previewMode(nextMode) {
    setModeDraft(nextMode);
    setMessage("");
    if (nextMode === mode) {
      setModePreview(null);
      return;
    }
    setLoading("mode-preview");
    setError("");
    try {
      setModePreview(await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/settings/preview`,
        {
          method: "POST",
          body: JSON.stringify({ project_space: workspaceName || "", mode: nextMode }),
        },
      ));
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The mode change preview could not be prepared."));
      setModePreview(null);
    } finally {
      setLoading("");
    }
  }

  async function applyMode() {
    if (!modePreview || modeDraft === mode) return;
    setLoading("mode-save");
    setError("");
    try {
      const response = await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/settings`,
        {
          method: "PATCH",
          body: JSON.stringify({
            project_space: workspaceName || "",
            mode: modeDraft,
            expected_mode: mode,
          }),
        },
      );
      setMessage(`Workspace evolution mode is now ${String(response?.mode || modeDraft)}.`);
      await loadCatalog();
      if (selectedTarget) await loadDetail(selectedTarget);
      await onChanged?.();
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The workspace mode was not changed."));
    } finally {
      setLoading("");
    }
  }

  async function saveTarget() {
    if (!detail || !draft || !hasTargetChanges) return;
    const body = {
      project_space: workspaceName || "",
      target: detail.target,
      expected_selected_version: String(detail.selected_version || ""),
    };
    if (Boolean(draft.enabled) !== Boolean(detail.enabled)) body.enabled = Boolean(draft.enabled);
    if (String(draft.selected_version || "") !== String(detail.selected_version || "")) {
      body.selected_version = String(draft.selected_version || "");
    }
    if (String(draft.update_policy || "") !== String(detail.update_policy || "")) {
      body.update_policy = String(draft.update_policy || "");
    }
    setLoading("target-save");
    setError("");
    try {
      await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/skills/state`,
        { method: "PATCH", body: JSON.stringify(body) },
      );
      setMessage(
        draft.enabled
          ? `${detail.group} / ${detail.name} will use the saved state on the next run.`
          : `${detail.group} / ${detail.name} is disabled and will not be staged on the next run.`,
      );
      await loadDetail(detail.target);
      await loadCatalog();
      await onChanged?.();
    } catch (err) {
      setError(selfEvolutionDisplayError(err, "The exact skill state was not changed."));
    } finally {
      setLoading("");
    }
  }

  return (
    <div
      className="v2-evolution-dialog-backdrop"
      role="presentation"
      onMouseDown={(event) => { if (event.target === event.currentTarget) onClose(); }}
    >
      <section className="v2-evolution-dialog v2-skills-manager" role="dialog" aria-modal="true" aria-labelledby="v2-skills-manager-heading">
        <header>
          <div>
            <div className="v2-eyebrow">Effective state · applies to subsequent runs</div>
            <h3 id="v2-skills-manager-heading">Manage skills</h3>
            <p>{totalCount} target{totalCount === 1 ? "" : "s"} · {modeSource === "workspace" ? "workspace mode" : "deployment default"}</p>
          </div>
          <button type="button" className="v2-icon-btn" onClick={onClose} aria-label="Close skills manager"><X size={17} /></button>
        </header>
        <div className="v2-skills-manager-mode">
          <div>
            <strong>Workspace evolution mode</strong>
            <span>Auto activation uses the independent reviewer result. This control does not change thread permission mode.</span>
          </div>
          <div className="v2-skills-mode-options" role="group" aria-label="Workspace evolution mode">
            {SKILL_EVOLUTION_MODES.map((item) => (
              <button
                key={item.value}
                type="button"
                aria-pressed={modeDraft === item.value}
                onClick={() => previewMode(item.value)}
                disabled={Boolean(loading)}
                title={item.note}
              >
                {item.label}
              </button>
            ))}
          </div>
          {modePreview ? (
            <div className="v2-skills-mode-preview">
              <div>
                <strong>Preview: {modePreview.mode_before} → {modePreview.mode_after}</strong>
                <span>
                  {modePreview.changes?.length
                    ? `${modePreview.changes.length} enabled follow-auto target${modePreview.changes.length === 1 ? "" : "s"} will move to auto_head.`
                    : "No selected version will change."}
                </span>
              </div>
              {modePreview.changes?.length ? (
                <ul>{modePreview.changes.map((change) => (
                  <li key={change.target}><code>{change.target}</code> → <code>{change.selected_version_after}</code></li>
                ))}</ul>
              ) : null}
              <button type="button" className="v2-primary-btn" onClick={applyMode} disabled={Boolean(loading)}>
                {loading === "mode-save" ? "Applying…" : `Apply ${modePreview.mode_after} mode`}
              </button>
            </div>
          ) : null}
        </div>
        {error ? <div className="v2-error compact" role="alert">{error}</div> : null}
        {message ? <div className="v2-success" role="status">{message}</div> : null}
        <div className="v2-skills-manager-grid" aria-busy={Boolean(loading)}>
          <aside className="v2-skills-catalog" aria-label="Effective skills">
            <div className="v2-skills-catalog-head">
              <strong>Effective skills</strong>
              <span>{skills.length} of {totalCount} loaded</span>
            </div>
            <div className="v2-skills-catalog-list">
              {skills.map((skill) => (
                <button
                  key={skill.target}
                  type="button"
                  className={selectedTarget === skill.target ? "selected" : ""}
                  onClick={() => setSelectedTarget(skill.target)}
                >
                  <span>
                    <strong>{skill.group} / {skill.name}</strong>
                    <small>{effectiveSkillSourceLabel(skill.source)} · {skill.enabled_control && !skill.enabled ? "Disabled" : skill.selected_label || "Not selected"}</small>
                  </span>
                  <ChevronRight size={15} aria-hidden="true" />
                </button>
              ))}
              {!skills.length && loading !== "catalog" ? <div className="v2-empty">No effective skill target is available.</div> : null}
            </div>
            {catalogCursor ? (
              <button
                type="button"
                className="v2-ghost-btn"
                onClick={() => loadCatalog({ append: true, cursor: catalogCursor })}
                disabled={Boolean(loading)}
              >
                {loading === "catalog-more" ? "Loading…" : "Load more skills"}
              </button>
            ) : null}
          </aside>
          <main className="v2-skills-detail">
            {detail && draft ? (
              <>
                <header>
                  <div>
                    <div className="v2-eyebrow">{effectiveSkillSourceLabel(detail.source)}</div>
                    <h4>{detail.group} / {detail.name}</h4>
                    <p>{detail.description || "No short behavior description is recorded."}</p>
                  </div>
                  <span className={`v2-skills-enabled-state ${detail.enabled ? "enabled" : "disabled"}`}>
                    {detail.enabled_control ? (detail.enabled ? "Enabled" : "Disabled") : "Workspace guidance"}
                  </span>
                </header>
                <section className="v2-skills-controls">
                  <div className="v2-skills-control-row">
                    {detail.enabled_control ? (
                      <label className="v2-evolution-confirm-check">
                        <input
                          type="checkbox"
                          checked={Boolean(draft.enabled)}
                          onChange={(event) => setDraft((current) => ({ ...current, enabled: event.target.checked }))}
                        />
                        <span><strong>Enabled for the next run</strong><br />Disabling removes the skill from staging without deleting history.</span>
                      </label>
                    ) : null}
                    <label className="v2-evolution-dialog-field">
                      <span>Update policy</span>
                      <select
                        value={draft.update_policy}
                        onChange={(event) => setDraft((current) => ({ ...current, update_policy: event.target.value }))}
                      >
                        <option value="follow_auto">Follow auto</option>
                        <option value="pinned">Pinned</option>
                      </select>
                    </label>
                    <label className="v2-evolution-dialog-field">
                      <span>Selected exact version</span>
                      <select
                        value={draft.selected_version}
                        onChange={(event) => setDraft((current) => ({
                          ...current,
                          selected_version: event.target.value,
                          update_policy: "pinned",
                        }))}
                      >
                        {!draft.selected_version ? <option value="">Not selected</option> : null}
                        {draft.selected_version && !selectableVersions.some((version) => version.version === draft.selected_version) ? (
                          <option value={draft.selected_version}>{detail.selected_label || "Current selected version"}</option>
                        ) : null}
                        {selectableVersions.map((version) => (
                          <option key={version.version} value={version.version}>
                            {version.label}{version.version === detail.auto_head ? " · auto_head" : ""}
                          </option>
                        ))}
                      </select>
                    </label>
                  </div>
                  <div className="v2-skills-pointer-summary">
                    <span>Selected <code>{detail.selected_version || "none"}</code></span>
                    <span>auto_head <code>{detail.auto_head || "none"}</code></span>
                    <span>Latest draft <code>{detail.latest_draft || "none"}</code></span>
                  </div>
                  <div className="v2-inline-actions">
                    <button type="button" className="v2-primary-btn" onClick={saveTarget} disabled={!hasTargetChanges || Boolean(loading)}>
                      {loading === "target-save" ? "Saving…" : "Save exact state"}
                    </button>
                    {hasTargetChanges ? <span className="v2-muted">Takes effect on the next run.</span> : <span className="v2-muted">No unsaved state change.</span>}
                  </div>
                </section>
                <section className="v2-skills-version-history">
                  <div className="v2-skills-section-head"><strong>Immutable versions</strong><span>{detail.revision_count} workspace revision{detail.revision_count === 1 ? "" : "s"}</span></div>
                  <div className="v2-skills-version-list">
                    {(detail.versions || []).map((version) => (
                      <details key={version.version}>
                        <summary>
                          <span><strong>{version.label}</strong><small>{version.status}{version.selected ? " · selected" : ""}{version.auto_head ? " · auto_head" : ""}</small></span>
                          <span>{version.eligible ? "Eligible" : "Inspect only"}</span>
                        </summary>
                        <p>{version.behavior_change || version.review_summary || "No concise change summary was recorded."}</p>
                        {version.review_concerns?.length ? <EvolutionTextList items={version.review_concerns} empty="" /> : null}
                        {version.files?.length ? <p><strong>Files:</strong> {version.files.join(", ")}</p> : null}
                        {version.evidence_ids?.length ? <p><strong>Evidence:</strong> {version.evidence_ids.join(", ")}</p> : null}
                        {version.detail_ref ? <a href={version.detail_ref} target="_blank" rel="noreferrer">Open full revision record</a> : null}
                      </details>
                    ))}
                  </div>
                  {detail.version_next_cursor ? (
                    <button type="button" className="v2-ghost-btn" disabled={Boolean(loading)} onClick={() => loadDetail(detail.target, { versionCursor: detail.version_next_cursor, appendKind: "versions-more" })}>
                      {loading === "versions-more" ? "Loading…" : "Load more versions"}
                    </button>
                  ) : null}
                </section>
                <section className="v2-skills-audit-history">
                  <div className="v2-skills-section-head"><strong>State history</strong><span>Automatic, chat, and GUI transitions</span></div>
                  <div className="v2-skills-history-list">
                    {(detail.history || []).map((event) => (
                      <article key={event.sequence}>
                        <header><strong>{selfEvolutionStatusLabel(event.event)}</strong><time dateTime={event.time}>{formatTimestamp(event.time)}</time></header>
                        <p>{event.source} · {event.actor || "system"}</p>
                        <code>{event.before.selected_version || "none"} → {event.after.selected_version || "none"}</code>
                        {event.source_ref ? <p><strong>Evidence:</strong> <code>{event.source_ref}</code></p> : null}
                        {event.note ? <p>{event.note}</p> : null}
                      </article>
                    ))}
                    {!detail.history?.length ? <div className="v2-empty">No effective-state transition has been recorded yet.</div> : null}
                  </div>
                  {detail.history_next_cursor ? (
                    <button type="button" className="v2-ghost-btn" disabled={Boolean(loading)} onClick={() => loadDetail(detail.target, { historyCursor: detail.history_next_cursor, appendKind: "history-more" })}>
                      {loading === "history-more" ? "Loading…" : "Load older history"}
                    </button>
                  ) : null}
                </section>
              </>
            ) : (
              <div className="v2-empty">{selectedSummary ? "Loading exact skill state…" : "Select an effective skill."}</div>
            )}
          </main>
        </div>
      </section>
    </div>
  );
}

export function SelfEvolutionPanel({ ctx, workspaceName, payload, loading = false, error: loadError = "", onRefresh }) {
  const statusFilterId = useId();
  const workspaceRef = useRef("");
  const initialCandidateCursorRef = useRef("");
  const initialObservationCursorRef = useRef("");
  const initialJobCursorRef = useRef("");
  const candidateFilterRequestRef = useRef(0);
  const [statusFilter, setStatusFilter] = useState("all");
  const [candidates, setCandidates] = useState(() => sortSelfEvolutionCandidates(payload?.candidates));
  const [observations, setObservations] = useState(() => sortSelfEvolutionObservations(payload?.observations));
  const [jobs, setJobs] = useState(() => mergeSelfEvolutionJobs([], payload?.jobs));
  const [candidateCursor, setCandidateCursor] = useState(String(payload?.next_cursor || ""));
  const [observationCursor, setObservationCursor] = useState(String(payload?.observation_next_cursor || ""));
  const [jobCursor, setJobCursor] = useState(String(payload?.job_next_cursor || ""));
  const [pageLoading, setPageLoading] = useState("");
  const [pageError, setPageError] = useState("");
  const [managerOpen, setManagerOpen] = useState(false);
  const [busyJobId, setBusyJobId] = useState("");
  const [actionMessage, setActionMessage] = useState("");

  useEffect(() => {
    const nextWorkspace = String(workspaceName || "");
    const workspaceChanged = workspaceRef.current !== nextWorkspace;
    const incomingCandidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
    const incomingObservations = Array.isArray(payload?.observations) ? payload.observations : [];
    const nextInitialCandidateCursor = String(payload?.next_cursor || "");
    const nextInitialObservationCursor = String(payload?.observation_next_cursor || "");
    const nextInitialJobCursor = String(payload?.job_next_cursor || "");
    if (workspaceChanged) {
      candidateFilterRequestRef.current += 1;
      workspaceRef.current = nextWorkspace;
      setCandidates(sortSelfEvolutionCandidates(incomingCandidates));
      setObservations(sortSelfEvolutionObservations(incomingObservations));
      setJobs(mergeSelfEvolutionJobs([], payload?.jobs));
      setCandidateCursor(nextInitialCandidateCursor);
      setObservationCursor(nextInitialObservationCursor);
      setJobCursor(nextInitialJobCursor);
      setStatusFilter("all");
      setManagerOpen(false);
      setPageLoading("");
      setPageError("");
      setBusyJobId("");
      setActionMessage("");
    } else {
      setCandidates((current) => mergeSelfEvolutionCandidates(current, incomingCandidates));
      setObservations((current) => mergeSelfEvolutionObservations(current, incomingObservations));
      setJobs((current) => mergeSelfEvolutionJobs(current, payload?.jobs));
      setCandidateCursor((current) => (
        !current || current === initialCandidateCursorRef.current ? nextInitialCandidateCursor : current
      ));
      setObservationCursor((current) => (
        !current || current === initialObservationCursorRef.current ? nextInitialObservationCursor : current
      ));
      setJobCursor((current) => (
        !current || current === initialJobCursorRef.current ? nextInitialJobCursor : current
      ));
    }
    initialCandidateCursorRef.current = nextInitialCandidateCursor;
    initialObservationCursorRef.current = nextInitialObservationCursor;
    initialJobCursorRef.current = nextInitialJobCursor;
  }, [payload, workspaceName]);

  const visibleCandidates = useMemo(
    () => selfEvolutionFilterCandidates(candidates, statusFilter),
    [candidates, statusFilter],
  );
  const statusOptions = useMemo(
    () => [...new Set([
      ...SELF_EVOLUTION_STATUS_VALUES,
      ...candidates.map((candidate) => String(candidate?.status || "")).filter(Boolean),
    ])]
      .sort((left, right) => selfEvolutionStatusLabel(left).localeCompare(selfEvolutionStatusLabel(right))),
    [candidates],
  );
  const approvedCount = useMemo(
    () => candidates.filter((candidate) => String(candidate?.review?.recommendation || "") === "approve").length,
    [candidates],
  );
  const needsRevisionCount = useMemo(
    () => candidates.filter((candidate) => String(candidate?.review?.recommendation || "") === "needs_revision").length,
    [candidates],
  );

  async function retryJob(job, itemRef = "") {
    const jobId = String(job?.job_id || "");
    if (!ctx || !jobId || busyJobId) return;
    setBusyJobId(`${jobId}:${String(itemRef || "")}`);
    setPageError("");
    setActionMessage("");
    try {
      const response = await apiFetch(
        `/api/session/${escapePath(ctx)}/self-evolution/jobs/${escapePath(jobId)}/retry`,
        {
          method: "POST",
          body: JSON.stringify({
            project_space: workspaceName || "",
            item_ref: String(itemRef || ""),
          }),
        },
      );
      if (response?.job) setJobs((current) => mergeSelfEvolutionJobs(current, [response.job]));
      setActionMessage(
        response?.queued
          ? "The selected failed item was queued against its original run evidence."
          : "A retry for this selected item already exists; no duplicate job was created.",
      );
      await onRefresh?.();
    } catch (err) {
      setPageError(selfEvolutionDisplayError(err, "The selected job could not be retried."));
    } finally {
      setBusyJobId("");
    }
  }

  async function changeStatusFilter(nextFilter) {
    setStatusFilter(nextFilter);
    if (!ctx || pageLoading) return;
    const requestKey = candidateFilterRequestRef.current + 1;
    candidateFilterRequestRef.current = requestKey;
    setPageLoading("candidates");
    setPageError("");
    const params = new URLSearchParams();
    if (workspaceName) params.set("project_space", workspaceName);
    if (nextFilter !== "all") params.set("status", nextFilter);
    try {
      const page = await apiFetch(`/api/session/${escapePath(ctx)}/self-evolution/candidates?${params.toString()}`);
      if (candidateFilterRequestRef.current !== requestKey) return;
      setCandidates((current) => {
        const retained = nextFilter === "all"
          ? current
          : current.filter((candidate) => String(candidate?.status || "") !== nextFilter);
        return mergeSelfEvolutionCandidates(retained, page.candidates);
      });
      setObservations((current) => mergeSelfEvolutionObservations(current, page.observations));
      setCandidateCursor(String(page.next_cursor || ""));
    } catch (err) {
      if (candidateFilterRequestRef.current === requestKey) {
        setPageError(selfEvolutionDisplayError(err, "This status view could not be loaded. Refresh and try again."));
      }
    } finally {
      if (candidateFilterRequestRef.current === requestKey) setPageLoading("");
    }
  }

  async function loadMore(kind) {
    const cursor = kind === "candidates"
      ? candidateCursor
      : kind === "observations"
        ? observationCursor
        : jobCursor;
    if (!ctx || !cursor || pageLoading) return;
    setPageLoading(kind);
    setPageError("");
    const params = new URLSearchParams();
    if (workspaceName) params.set("project_space", workspaceName);
    params.set(
      kind === "candidates" ? "cursor" : kind === "observations" ? "observation_cursor" : "job_cursor",
      cursor,
    );
    if (kind === "candidates" && statusFilter !== "all") {
      params.set("status", statusFilter);
    }
    try {
      const page = await apiFetch(`/api/session/${escapePath(ctx)}/self-evolution/candidates?${params.toString()}`);
      setCandidates((current) => mergeSelfEvolutionCandidates(current, page.candidates));
      setObservations((current) => mergeSelfEvolutionObservations(current, page.observations));
      setJobs((current) => mergeSelfEvolutionJobs(current, page.jobs));
      if (kind === "candidates") setCandidateCursor(String(page.next_cursor || ""));
      else if (kind === "observations") setObservationCursor(String(page.observation_next_cursor || ""));
      else setJobCursor(String(page.job_next_cursor || ""));
    } catch (err) {
      setPageError(selfEvolutionDisplayError(err, `More ${kind} could not be loaded. Refresh and try again.`));
    } finally {
      setPageLoading("");
    }
  }

  if (payload?.enabled === false) {
    return (
      <section className="v2-tab-panel">
        <div className="v2-panel-toolbar"><div><div className="v2-eyebrow">Workspace</div><h2>Skill Evolution</h2></div></div>
        <div className="v2-evolution-scope warning">
          {selfEvolutionSafeText(payload.disabled_reason, "Skill Evolution is not available in this workspace.")}
        </div>
      </section>
    );
  }

  return (
    <section className="v2-tab-panel v2-self-evolution-panel" aria-busy={loading}>
      <div className="v2-panel-toolbar">
        <div>
          <div className="v2-eyebrow">Workspace evidence and reviewed releases</div>
          <h2>Skill Evolution</h2>
          <div className="v2-muted">{workspaceName || "No workspace selected"}</div>
        </div>
        <div className="v2-inline-actions">
          <button type="button" className="v2-primary-btn" onClick={() => setManagerOpen(true)} disabled={!ctx || !workspaceName}>
            <Settings2 size={15} />Manage skills
          </button>
          <button type="button" className="v2-ghost-btn" onClick={() => onRefresh?.()} disabled={loading || Boolean(pageLoading)}>
            <RefreshCw size={15} />Refresh
          </button>
        </div>
      </div>
      {loadError ? <div className="v2-error" role="alert">{selfEvolutionDisplayError(loadError, "Skill Evolution could not be loaded. Refresh and try again.")}</div> : null}
      {pageError ? <div className="v2-error" role="alert">{pageError}</div> : null}
      {actionMessage ? <div className="v2-success" role="status" aria-live="polite">{actionMessage}</div> : null}
      <div className="v2-evolution-scope">
        <GitBranch size={18} aria-hidden="true" />
        <div>
          <strong>Learning is automatic; exact state remains inspectable</strong>
          <span>
            Completed trajectories are reflected and independently reviewed. In auto mode, eligible follow-auto revisions take effect on the next run without a review queue.
          </span>
        </div>
      </div>
      <div className="v2-monitor-grid v2-evolution-metrics">
        <MonitorMetric icon={Settings2} label="Effective skills" value={formatCount(payload?.effective_skill_count)} note="open Manage skills for exact state" />
        <MonitorMetric icon={Activity} label="Workspace mode" value={String(payload?.mode || "auto")} note={payload?.mode_source === "workspace" ? "saved for this workspace" : "deployment default"} />
        <MonitorMetric icon={ListChecks} label="Approved revisions" value={formatCount(approvedCount)} note={`${formatCount(needsRevisionCount)} loaded revision${needsRevisionCount === 1 ? "" : "s"} need further automatic repair`} />
        <MonitorMetric icon={Clock} label="Evidence observations" value={formatCount(observations.length)} note={observationCursor ? "more available" : "all loaded"} />
      </div>
      <div className="v2-evolution-toolbar">
        <label htmlFor={statusFilterId}>
          <span>Status filter</span>
          <select
            id={statusFilterId}
            value={statusFilter}
            onChange={(event) => changeStatusFilter(event.target.value)}
            disabled={pageLoading === "candidates"}
          >
            <option value="all">All loaded revisions</option>
            {statusOptions.map((status) => <option key={status} value={status}>{selfEvolutionStatusLabel(status)}</option>)}
          </select>
        </label>
        <span aria-live="polite">{visibleCandidates.length} of {candidates.length} loaded revisions shown</span>
      </div>
      {loading && !payload ? <div className="v2-muted" role="status">Loading workspace learning state…</div> : null}
      <div className="v2-evolution-columns">
        <section className="v2-monitor-panel" aria-labelledby="v2-evolution-candidates-heading">
          <div className="v2-monitor-panel-head">
            <GitBranch size={15} aria-hidden="true" />
            <h3 id="v2-evolution-candidates-heading">Reviewed candidate revisions</h3>
          </div>
          <div className="v2-evolution-candidate-list">
            {visibleCandidates.map((candidate) => {
              const key = `${candidate.candidate_id}:${candidate.revision ?? candidate.version ?? ""}`;
              return (
                <SelfEvolutionCandidateCard
                  key={key}
                  candidate={candidate}
                />
              );
            })}
            {!visibleCandidates.length ? <div className="v2-empty">No candidate revision matches this filter.</div> : null}
          </div>
          {candidateCursor ? (
            <button
              type="button"
              className="v2-ghost-btn v2-evolution-load-more"
              onClick={() => loadMore("candidates")}
              disabled={Boolean(pageLoading)}
            >
              {pageLoading === "candidates" ? "Loading revisions…" : "Load more revisions"}
            </button>
          ) : <p className="v2-evolution-page-end">All matching revision pages are loaded.</p>}
        </section>
        <section className="v2-monitor-panel" aria-labelledby="v2-evolution-observations-heading">
          <div className="v2-monitor-panel-head">
            <Clock size={15} aria-hidden="true" />
            <h3 id="v2-evolution-observations-heading">Evidence observations</h3>
          </div>
          <p className="v2-muted">
            Observations are target-bound semantic signals from complete run evidence. The proposer and independent reviewer decide whether they support a durable change.
          </p>
          <div className="v2-evolution-observation-list">
            {observations.map((observation) => (
              <SelfEvolutionObservationCard key={observation.observation_id} observation={observation} />
            ))}
            {!observations.length ? <div className="v2-empty">No durable learning observations are available yet.</div> : null}
          </div>
          {observationCursor ? (
            <button
              type="button"
              className="v2-ghost-btn v2-evolution-load-more"
              onClick={() => loadMore("observations")}
              disabled={Boolean(pageLoading)}
            >
              {pageLoading === "observations" ? "Loading observations…" : "Load more observations"}
            </button>
          ) : <p className="v2-evolution-page-end">All observation pages are loaded.</p>}
        </section>
      </div>
      {jobs.length ? (
        <section className="v2-monitor-panel" aria-labelledby="v2-evolution-jobs-heading">
          <div className="v2-monitor-panel-head">
            <Activity size={15} aria-hidden="true" />
            <h3 id="v2-evolution-jobs-heading">Processing history</h3>
          </div>
          <p className="v2-muted">
            Every reflect remains visible, including deferred evidence, ignored findings, no-change, and existing-guidance outcomes. Failed results stay unchanged; retry creates a new job from the original evidence.
          </p>
          <div className="v2-evolution-job-list">
            {jobs.map((job) => (
              <SelfEvolutionJobCard
                key={job.job_id}
                job={job}
                busy={busyJobId.startsWith(`${String(job.job_id || "")}:`)}
                onRetry={retryJob}
              />
            ))}
          </div>
          {jobCursor ? (
            <button
              type="button"
              className="v2-ghost-btn v2-evolution-load-more"
              onClick={() => loadMore("jobs")}
              disabled={Boolean(pageLoading)}
            >
              {pageLoading === "jobs" ? "Loading reflect history…" : "Load more reflect history"}
            </button>
          ) : <p className="v2-evolution-page-end">All reflect history pages are loaded.</p>}
        </section>
      ) : null}
      {managerOpen ? (
        <SkillsManagerDialog
          ctx={ctx}
          workspaceName={workspaceName}
          onClose={() => setManagerOpen(false)}
          onChanged={onRefresh}
        />
      ) : null}
    </section>
  );
}

export function MonitorPanel({ ctx, workspaceName, thread, entrypoint }) {
  const [activeSubtab, setActiveSubtab] = useState("overview");
  const [monitor, setMonitor] = useState(null);
  const [diagnostics, setDiagnostics] = useState([]);
  const [diagnosticsPage, setDiagnosticsPage] = useState({});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const entrypointKey = String(entrypoint || thread?.entrypoint || "research");
  const projectParam = workspaceName ? `&project_space=${escapePath(workspaceName)}` : "";

  async function loadMonitor() {
    if (!ctx) return;
    setLoading(true);
    setError("");
    try {
      setMonitor(await apiFetch(`/api/session/${escapePath(ctx)}/monitor?lane=${escapePath(entrypointKey)}${projectParam}&limit=400`));
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  async function loadDiagnostics(beforeId = 0) {
    if (!ctx || !monitor?.developer_diagnostics_available) return;
    setLoading(true);
    setError("");
    try {
      const before = beforeId ? `&before_id=${beforeId}` : "";
      const payload = await apiFetch(`/api/diagnostics/session/${escapePath(ctx)}/events?limit=100${before}${projectParam}`);
      const rows = Array.isArray(payload.events) ? payload.events : [];
      setDiagnostics((current) => (beforeId ? [...rows, ...current] : rows));
      setDiagnosticsPage(payload);
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    loadMonitor();
    const timer = window.setInterval(() => {
      if (thread?.status === "running") loadMonitor();
    }, 5000);
    return () => window.clearInterval(timer);
  }, [ctx, workspaceName, thread?.status, entrypointKey]);

  useEffect(() => {
    if (activeSubtab === "diagnostics" && !diagnostics.length) loadDiagnostics();
  }, [activeSubtab, monitor?.developer_diagnostics_available]);

  const overview = monitor?.overview || {};
  const live = monitor?.live || {};
  const progress = live.progress || {};
  const progressTotal = Number(progress.total || 0);
  const progressDone = Number(progress.completed || 0);
  const overviewFailed = ["failed", "error"].includes(String(overview.status || thread?.status || "").toLowerCase());
  const overviewStatusText = displayValue(
    overview.status_text,
    monitor?.has_run ? "Run information is available." : "No run is selected.",
  );

  return (
    <section className="v2-tab-panel">
      <div className="v2-panel-toolbar">
        <div>
          <div className="v2-eyebrow">Monitor</div>
          <h2>Execution monitor</h2>
          <div className="v2-muted">{overviewFailed ? redactErrorText(overviewStatusText) : overviewStatusText}</div>
        </div>
        <button type="button" className="v2-ghost-btn" onClick={loadMonitor} disabled={loading}><RefreshCw size={15} />Refresh</button>
      </div>
      <ErrorNotice error={error} />
      <div className="v2-monitor-grid">
        <MonitorMetric icon={Activity} label="Status" value={statusLabel(overview.status || thread?.status)} note={displayValue(overview.phase, "")} />
        <MonitorMetric icon={Clock} label="Duration" value={formatDurationSec(overview.duration_sec)} note={displayValue(overview.current_task, "")} />
        <MonitorMetric icon={BarChart3} label="Model usage" value={`${formatCount(overview.total_tokens)} tokens`} note={`${formatCount(overview.llm_calls)} calls${overview.model ? ` · ${overview.model}` : ""}`} />
        <MonitorMetric icon={Cpu} label="Compute" value={`${formatHours(overview.core_hours)} core h`} note={`${formatHours(overview.node_hours)} node h`} />
        <MonitorMetric icon={Database} label="Cost" value={formatCost(overview.cost_usd)} note={`${formatCount(overview.input_tokens)} in · ${formatCount(overview.output_tokens)} out`} />
        <MonitorMetric icon={GitBranch} label="Operations" value={formatCount(overview.tool_calls)} note={`${formatCount(overview.tool_failures)} failed`} />
      </div>
      <ModelUsageDetails usage={monitor?.usage} />
      <div className="v2-subtabs" role="tablist" aria-label="Monitor views">
        {[
          ["overview", "Overview"],
          ["live", "Live"],
          ["timeline", "Timeline"],
          ...(monitor?.developer_diagnostics_available ? [["diagnostics", "Developer diagnostics"]] : []),
        ].map(([value, label]) => (
          <button
            key={value}
            type="button"
            role="tab"
            aria-selected={activeSubtab === value}
            className={activeSubtab === value ? "active" : ""}
            onClick={() => setActiveSubtab(value)}
          >
            {label}
          </button>
        ))}
      </div>
      {loading ? <div className="v2-muted">Loading monitor data…</div> : null}
      {activeSubtab === "overview" ? (
        <div className="v2-monitor-columns">
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><ListChecks size={15} /><h3>Current work</h3></div>
            <p>{displayValue(overview.current_task, "No active research step.")}</p>
            {progressTotal ? (
              <div className="v2-progress-meter">
                <div
                  role="progressbar"
                  aria-label="Research progress"
                  aria-valuemin={0}
                  aria-valuemax={progressTotal}
                  aria-valuenow={progressDone}
                >
                  <span style={{ width: `${Math.min(100, (progressDone / progressTotal) * 100)}%` }} />
                </div>
                <small>{progressDone} of {progressTotal} steps complete</small>
              </div>
            ) : null}
            <div className="v2-monitor-semantic-list">
              {(live.todos || []).map((todo, index) => (
                <div key={`${todo.label}-${index}`}>
                  <strong>{displayValue(todo.label, "Plan item")}</strong>
                  <small>{statusLabel(todo.status)}</small>
                </div>
              ))}
              {!live.todos?.length ? <div className="v2-empty compact">No active plan items.</div> : null}
            </div>
          </section>
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><FileBox size={15} /><h3>Key artifacts</h3></div>
            <div className="v2-monitor-semantic-list">
              {(monitor?.artifacts || []).map((artifact, index) => (
                <div key={`${artifact.path || artifact.title}-${index}`}>
                  <strong>{userFacingFileTitle(artifact.title, artifact.path, "Artifact")}</strong>
                  <small>{displayValue(artifact.summary || (artifact.renderer ? humanizeKey(artifact.renderer) : ""), "Result file")}</small>
                </div>
              ))}
              {!monitor?.artifacts?.length ? <div className="v2-empty compact">No artifacts recorded for this run.</div> : null}
            </div>
          </section>
        </div>
      ) : null}
      {activeSubtab === "live" ? (
        <div className="v2-monitor-columns">
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><GitBranch size={15} /><h3>Recent operations</h3></div>
            <div className="v2-monitor-semantic-list">
              {(live.tools || []).map((tool, index) => (
                <div key={`${tool.title}-${index}`}>
                  <strong>{statusAwareText(tool.title, tool.status, "Operation")}</strong>
                  <span>{statusAwareText(tool.summary, tool.status, "No additional update.")}</span>
                  <small>{statusLabel(tool.status)}</small>
                </div>
              ))}
              {!live.tools?.length ? <div className="v2-empty compact">No recent operations.</div> : null}
            </div>
          </section>
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><MessageSquare size={15} /><h3>Specialists</h3></div>
            <div className="v2-monitor-semantic-list">
              {(live.agents || []).map((agent, index) => (
                <div key={`${agent.title}-${index}`}>
                  <strong>{statusAwareText(agent.title, agent.status, "Specialist")}</strong>
                  <span>{statusAwareText(agent.summary, agent.status, "No additional update.")}</span>
                  <small>{statusLabel(agent.status)}</small>
                </div>
              ))}
              {!live.agents?.length ? <div className="v2-empty compact">No specialist activity.</div> : null}
            </div>
          </section>
        </div>
      ) : null}
      {activeSubtab === "timeline" ? (
        <div className="v2-event-list">
          {(monitor?.timeline || []).slice().reverse().map((event, index) => (
            <article key={event.id || `${event.title}-${index}`} className={`v2-public-event status-${event.status || "updated"}`}>
              <div>
                <strong>{statusAwareText(event.title, event.status, "Timeline update")}</strong>
                {event.summary ? <p>{statusAwareText(event.summary, event.status, "")}</p> : null}
              </div>
              <small>{statusLabel(event.status)}{event.timestamp ? ` · ${formatTimestamp(event.timestamp)}` : ""}</small>
              {(event.fields || []).length ? (
                <dl>{event.fields.map((field, fieldIndex) => (
                  <div key={`${field.label}-${fieldIndex}`}>
                    <dt>{statusAwareText(field.label, event.status, "Detail")}</dt>
                    <dd>{statusAwareText(field.value, event.status, "Not available")}</dd>
                  </div>
                ))}</dl>
              ) : null}
            </article>
          ))}
          {!monitor?.timeline?.length ? <div className="v2-empty">No user-facing timeline entries yet.</div> : null}
          {monitor?.page?.truncated ? (
            <div className="v2-truncation-notice" role="status">
              <span>
                Showing {Number(monitor.page.shown_count || monitor.timeline?.length || 0).toLocaleString()} of {Number(monitor.page.total_count || 0).toLocaleString()} timeline entries.
              </span>
              {monitor?.page?.full_content_ref ? (
                <a className="v2-ghost-btn compact" href={monitor.page.full_content_ref}>Open full timeline</a>
              ) : monitor?.developer_diagnostics_available ? (
                <button type="button" className="v2-ghost-btn compact" onClick={() => setActiveSubtab("diagnostics")}>Open complete diagnostics</button>
              ) : null}
            </div>
          ) : null}
        </div>
      ) : null}
      {activeSubtab === "diagnostics" && monitor?.developer_diagnostics_available ? (
        <section className="v2-developer-diagnostics" aria-label="Developer diagnostics">
          <div className="v2-diagnostics-warning" role="alert">
            <Braces size={17} />
            <div><strong>Internal developer data</strong><p>This view may contain raw event payloads, internal IDs, and paths. It is loaded only after you open this tab.</p></div>
          </div>
          {diagnosticsPage?.has_more ? (
            <button type="button" className="v2-ghost-btn" onClick={() => loadDiagnostics(diagnosticsPage.min_id)} disabled={loading}>Load older diagnostics</button>
          ) : null}
          {diagnostics.length ? (
            <div className="v2-range-label">
              Showing {diagnostics.length.toLocaleString()} loaded diagnostic entries{diagnosticsPage?.has_more ? " of an unreported total." : "."}
            </div>
          ) : null}
          <div className="v2-raw-list">
            {diagnostics.map((event, index) => (
              <details key={`${event.id || index}-${event.name || "event"}`} className="v2-raw-row">
                <summary><span>{event.name || "Internal event"}</span><small>{event.ts || ""}</small></summary>
                <pre>{jsonText(event)}</pre>
              </details>
            ))}
          </div>
        </section>
      ) : null}
    </section>
  );
}
