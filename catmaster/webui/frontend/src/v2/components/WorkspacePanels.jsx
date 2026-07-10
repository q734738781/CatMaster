import { useEffect, useMemo, useRef, useState } from "react";
import { Activity, BarChart3, Braces, Clock, Cpu, Database, Download, FileBox, Filter, GitBranch, ListChecks, MessageSquare, RefreshCw, Trash2, Upload } from "lucide-react";
import { Grid } from "gridjs-react";

import ArtifactRenderer from "./ArtifactRenderer";
import { apiFetch } from "../useCatMasterThreadRuntime";

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

function compactJson(value, max = 3200) {
  const text = jsonText(value);
  return text.length > max ? `${text.slice(0, max)}\n... truncated ...` : text;
}

function plainText(value, fallback = "-") {
  if (value == null) return fallback;
  if (typeof value === "string") {
    const text = value.trim();
    return text || fallback;
  }
  if (typeof value === "number") return Number.isFinite(value) ? String(value) : fallback;
  if (typeof value === "boolean") return value ? "true" : "false";
  if (typeof value === "object") {
    const text = jsonText(value).replace(/\s+/g, " ").trim();
    return text || fallback;
  }
  return String(value);
}

function compactPlainText(value, max = 420) {
  const text = plainText(value, "").replace(/\s+/g, " ").trim();
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function asRecord(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function formatDurationMs(value) {
  const ms = Number(value || 0);
  if (!Number.isFinite(ms) || ms <= 0) return "-";
  if (ms < 1000) return `${Math.round(ms)}ms`;
  return formatDurationSec(ms / 1000);
}

function formatMaybeDurationSec(value) {
  const seconds = Number(value || 0);
  if (!Number.isFinite(seconds) || seconds <= 0) return "-";
  return formatDurationSec(seconds);
}

function formatTimestamp(value) {
  const number = Number(value || 0);
  if (!Number.isFinite(number) || number <= 0) return "-";
  const ms = number > 100000000000 ? number : number * 1000;
  return new Date(ms).toLocaleString();
}

function formatTokenUsage(value) {
  const usage = asRecord(value);
  const total = usage.total_tokens ?? usage.totalTokens ?? usage.total;
  const input = usage.input_tokens ?? usage.prompt_tokens ?? usage.inputTokens ?? usage.prompt;
  const output = usage.output_tokens ?? usage.completion_tokens ?? usage.outputTokens ?? usage.completion;
  const chunks = [];
  if (Number(total) > 0) chunks.push(`${formatCount(total)} total`);
  if (Number(input) > 0) chunks.push(`${formatCount(input)} in`);
  if (Number(output) > 0) chunks.push(`${formatCount(output)} out`);
  return chunks.join(" / ") || "-";
}

function clampNumber(value, min, max) {
  return Math.min(max, Math.max(min, value));
}

function monitorRowsFromMap(value) {
  if (!value) return [];
  if (Array.isArray(value)) return value;
  if (typeof value === "object") {
    return Object.entries(value).map(([name, count]) => ({ name, count }));
  }
  return [];
}

function TreeNode({ node, depth, selectedPath, childrenByPath, loadingByPath, onToggle, onSelect }) {
  const isDirectory = node.node_type === "directory";
  const open = Boolean(childrenByPath[node.path || ""]);
  const children = childrenByPath[node.path || ""] || [];
  return (
    <div className="v2-file-branch">
      <div className={`v2-file-row ${selectedPath === node.path ? "selected" : ""}`} style={{ paddingLeft: `${depth * 14}px` }}>
        <button type="button" className="v2-file-toggle" disabled={!isDirectory} onClick={() => onToggle(node)}>
          {isDirectory ? (open ? "-" : "+") : ""}
        </button>
        <button type="button" className={`v2-file-label kind-${node.preview_kind || node.node_type}`} onClick={() => onSelect(node)}>
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
              onToggle={onToggle}
              onSelect={onSelect}
            />
          ))}
        </div>
      ) : null}
    </div>
  );
}

function CsvGrid({ preview }) {
  const rows = useMemo(() => {
    const text = String(preview?.preview_text || "").trim();
    if (!text) return [];
    const separator = String(preview?.path || "").toLowerCase().endsWith(".tsv") ? "\t" : ",";
    return text.split(/\r?\n/).slice(0, 200).map((line) => line.split(separator));
  }, [preview]);
  if (!rows.length) return <div className="v2-empty">CSV preview is empty.</div>;
  const columns = rows[0].map((item, index) => String(item || `Column ${index + 1}`));
  const data = rows.slice(1).map((row) => columns.map((_, index) => row[index] || ""));
  return <Grid columns={columns} data={data} search pagination={{ limit: 25 }} sort />;
}

function FilePreview({ ctx, workspaceName, preview, loading, error, onRefresh, onDelete }) {
  if (loading) return <div className="v2-empty">Loading preview...</div>;
  if (error) return <div className="v2-error">{error}</div>;
  if (!preview) return <div className="v2-empty">Select a file or directory.</div>;
  const isCsv = preview.kind === "csv" || String(preview.path || "").toLowerCase().endsWith(".csv") || String(preview.path || "").toLowerCase().endsWith(".tsv");
  const archiveUrl = `/api/session/${escapePath(ctx)}/files/archive?path=${escapePath(preview.path || "")}&project_space=${escapePath(workspaceName || "")}`;
  return (
    <div className="v2-files-preview">
      <div className="v2-inspector-head">
        <div>
          <div className="v2-eyebrow">{preview.kind || preview.node_type || "file"}</div>
          <h3>{preview.name || preview.path || "."}</h3>
          <div className="v2-muted">{preview.path || "."}</div>
        </div>
        <div className="v2-icon-row">
          <button type="button" className="v2-icon-btn" onClick={onRefresh} title="Refresh"><RefreshCw size={16} /></button>
          {preview.download_url ? <a className="v2-icon-btn" href={preview.download_url} title="Download"><Download size={16} /></a> : null}
          <a className="v2-icon-btn" href={archiveUrl} title="Download ZIP"><FileBox size={16} /></a>
          {preview.path ? <button type="button" className="v2-icon-btn danger" onClick={onDelete} title="Delete"><Trash2 size={16} /></button> : null}
        </div>
      </div>
      <div className="v2-file-meta">
        <span>{preview.node_type}</span>
        <span>{formatBytes(preview.size)}</span>
        {preview.mime_type ? <span>{preview.mime_type}</span> : null}
      </div>
      {preview.kind === "directory" ? (
        <div className="v2-directory-list">
          {(preview.children || []).map((child) => (
            <div key={child.path || child.name}><span>{child.name}</span><small>{child.node_type === "file" ? formatBytes(child.size) : "directory"}</small></div>
          ))}
        </div>
      ) : null}
      {preview.kind === "image" && preview.download_url ? <img className="v2-image-preview" src={preview.download_url} alt={preview.name || "file"} /> : null}
      {preview.kind === "structure" ? <ArtifactRenderer filePreview={preview} /> : null}
      {preview.kind === "markdown" || preview.kind === "pdf" ? <ArtifactRenderer filePreview={preview} /> : null}
      {isCsv ? <div className="v2-grid-wrap"><CsvGrid preview={preview} /></div> : null}
      {!["directory", "image", "structure", "markdown", "pdf"].includes(preview.kind) && !isCsv ? <pre className="v2-code">{preview.preview_text || "(binary file)"}</pre> : null}
      {preview.truncated ? <div className="v2-muted">Preview truncated.</div> : null}
    </div>
  );
}

export function FilesPanel({ ctx, workspaceName, selectedFilePath = "", onSelectFile }) {
  const [childrenByPath, setChildrenByPath] = useState({});
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

  async function loadTree(path = "") {
    if (!ctx || !workspaceName) return;
    setLoadingByPath((prev) => ({ ...prev, [path]: true }));
    try {
      const payload = await apiFetch(`/api/session/${escapePath(ctx)}/files/tree?path=${escapePath(path)}&project_space=${escapePath(workspaceName)}`);
      setChildrenByPath((prev) => ({ ...prev, [payload.path || ""]: payload.children || [] }));
      setError("");
    } catch (err) {
      setError(err.message || String(err));
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
      setPreviewError(err.message || String(err));
    } finally {
      setPreviewLoading(false);
    }
  }

  useEffect(() => {
    setChildrenByPath({});
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
      if (!response.ok) throw new Error(await response.text());
    }
    setUploadStatus(`Uploaded ${list.length} file(s).`);
    await loadTree(target);
    if (selectedPath) await loadPreview(selectedPath);
  }

  async function deleteSelected() {
    if (!preview?.path) return;
    if (!window.confirm(`Delete ${preview.path}?`)) return;
    const response = await fetch(`/api/session/${escapePath(ctx)}/files/delete?path=${escapePath(preview.path)}&project_space=${escapePath(workspaceName)}`, { method: "DELETE" });
    if (!response.ok) throw new Error(await response.text());
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

  const roots = childrenByPath[""] || [];
  const previewNode = (
    <FilePreview
      ctx={ctx}
      workspaceName={workspaceName}
      preview={preview}
      loading={previewLoading}
      error={previewError}
      onRefresh={() => loadPreview(selectedPath)}
      onDelete={() => deleteSelected().catch((err) => setPreviewError(err.message || String(err)))}
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
          <input ref={inputRef} type="file" multiple className="v2-hidden-input" onChange={(event) => uploadFiles(event.target.files).catch((err) => setError(err.message || String(err)))} />
        </div>
      </div>
      <div className="v2-file-tabs" role="tablist" aria-label="File workspace views">
        <button type="button" className={activeFileTab === "browse" ? "active" : ""} onClick={() => setActiveFileTab("browse")} role="tab" aria-selected={activeFileTab === "browse"}>Browse</button>
        <button type="button" className={activeFileTab === "preview" ? "active" : ""} onClick={() => setActiveFileTab("preview")} role="tab" aria-selected={activeFileTab === "preview"}>Preview</button>
        <button type="button" className={activeFileTab === "uploads" ? "active" : ""} onClick={() => setActiveFileTab("uploads")} role="tab" aria-selected={activeFileTab === "uploads"}>Uploads</button>
      </div>
      {error ? <div className="v2-error">{error}</div> : null}
      {uploadStatus ? <div className="v2-muted">{uploadStatus}</div> : null}
      {activeFileTab === "browse" ? (
        <div ref={shellRef} className="v2-files-shell resizable" style={{ "--v2-file-tree-width": `${treeWidth}px` }}>
          <div className="v2-files-tree">
            {roots.map((node) => (
              <TreeNode key={node.path || node.name} node={node} depth={0} selectedPath={selectedPath} childrenByPath={childrenByPath} loadingByPath={loadingByPath} onToggle={(item) => loadTree(item.path || "")} onSelect={selectNode} />
            ))}
            {!roots.length ? <div className="v2-empty">No files in this workspace.</div> : null}
          </div>
          <div className="v2-resize-handle v2-resize-handle-files" role="separator" aria-label="Resize file browser" aria-orientation="vertical" tabIndex={0} onPointerDown={startTreeResize} />
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

function MonitorCodeBlock({ title, text }) {
  return (
    <section className="v2-monitor-panel">
      <div className="v2-monitor-panel-head">
        <h3>{title}</h3>
      </div>
      <pre className="v2-code tall">{text || "(empty)"}</pre>
    </section>
  );
}

function SelfEvolutionPanel({ payload, onDecision, onRollback }) {
  const candidates = Array.isArray(payload?.candidates) ? payload.candidates : [];
  const jobs = Array.isArray(payload?.jobs) ? payload.jobs : [];
  return (
    <section className="v2-monitor-panel">
      <div className="v2-monitor-panel-head"><GitBranch size={15} /><h3>Self-Evolution</h3></div>
      <div className="v2-raw-list">
        {candidates.map((candidate) => {
          const status = String(candidate.status || "");
          const canDecide = status === "approved";
          const canRollback = status === "promoted";
          return (
            <details key={candidate.candidate_id} className="v2-raw-row">
              <summary>
                <span>{candidate.kind || "candidate"}</span>
                <small>{status}</small>
              </summary>
              <div className="v2-inline-actions">
                {canDecide ? (
                  <>
                    <button type="button" className="v2-ghost-btn" onClick={() => onDecision(candidate, "promote")}>Promote</button>
                    <button type="button" className="v2-ghost-btn" onClick={() => onDecision(candidate, "reject")}>Reject</button>
                  </>
                ) : null}
                {canRollback ? <button type="button" className="v2-ghost-btn" onClick={() => onRollback(candidate)}>Rollback</button> : null}
              </div>
              <pre>{jsonText({
                candidate_id: candidate.candidate_id,
                target: candidate.target,
                rationale: candidate.rationale,
                validation: candidate.validation,
                review: candidate.review,
                promotion: candidate.promotion,
              })}</pre>
            </details>
          );
        })}
        {!candidates.length ? <div className="v2-empty">No self-evolution candidates for this run.</div> : null}
        {jobs.map((job) => (
          <details key={job.job_id} className="v2-raw-row">
            <summary>
              <span>{job.trigger_kind || "learning job"}</span>
              <small>{job.status || "unknown"}</small>
            </summary>
            <pre>{jsonText({
              job_id: job.job_id,
              run_id: job.run_id,
              attempt_count: job.attempt_count,
              candidate_id: job.candidate_id,
              error: job.error,
            })}</pre>
          </details>
        ))}
        {!jobs.length ? <div className="v2-empty">No self-evolution jobs for this run.</div> : null}
      </div>
    </section>
  );
}

function MonitorList({ rows, empty = "No rows." }) {
  const normalized = monitorRowsFromMap(rows);
  if (!normalized.length) return <div className="v2-empty">{empty}</div>;
  return (
    <div className="v2-monitor-list">
      {normalized.slice(0, 40).map((row, index) => {
        const name = row.name || row.label || row.model || row.agent || row.tool || row[0] || `Row ${index + 1}`;
        const count = row.count ?? row.calls ?? row.value ?? row[1] ?? "";
        return (
          <div key={`${name}-${index}`}>
            <span>{String(name)}</span>
            <strong>{String(count)}</strong>
          </div>
        );
      })}
    </div>
  );
}

function LivePanel({ icon: Icon, title, children }) {
  return (
    <section className="v2-monitor-panel">
      <div className="v2-monitor-panel-head">
        {Icon ? <Icon size={15} /> : null}
        <h3>{title}</h3>
      </div>
      {children}
    </section>
  );
}

function LiveFieldGrid({ rows }) {
  const visibleRows = rows.filter((row) => row && (row.always || plainText(row.value, "") !== ""));
  if (!visibleRows.length) return <div className="v2-empty">No live fields yet.</div>;
  return (
    <div className="v2-live-fields">
      {visibleRows.map((row) => (
        <div key={row.label}>
          <span>{row.label}</span>
          <strong title={plainText(row.value)}>{plainText(row.value)}</strong>
        </div>
      ))}
    </div>
  );
}

function LiveProgress({ progress }) {
  const state = asRecord(progress);
  const completed = Number(state.completed || 0);
  const pending = Number(state.pending || 0);
  const failed = Number(state.failed || 0);
  const needsIntervention = Number(state.needs_intervention || 0);
  const total = Number(state.total || completed + pending + failed + needsIntervention || 0);
  const pct = total > 0 ? clampNumber((completed / total) * 100, 0, 100) : 0;
  return (
    <div className="v2-live-progress">
      <div className="v2-live-progress-bar" aria-label="Task progress">
        <span style={{ width: `${pct}%` }} />
      </div>
      <div className="v2-live-progress-counts">
        <div><span>Done</span><strong>{formatCount(completed)}</strong></div>
        <div><span>Pending</span><strong>{formatCount(pending)}</strong></div>
        <div><span>Failed</span><strong>{formatCount(failed)}</strong></div>
        <div><span>Needs input</span><strong>{formatCount(needsIntervention)}</strong></div>
        <div><span>Total</span><strong>{formatCount(total)}</strong></div>
      </div>
    </div>
  );
}

function LiveToolDetails({ toolcall, empty = "No active tool call." }) {
  const call = asRecord(toolcall);
  if (!call.tool) return <div className="v2-empty">{empty}</div>;
  const paramsText = call.params_compact || (call.params_full != null ? compactJson(call.params_full, 1400) : "");
  return (
    <>
      <LiveFieldGrid
        rows={[
          { label: "Tool", value: call.tool, always: true },
          { label: "Status", value: call.status || "running", always: true },
          { label: "Elapsed", value: formatMaybeDurationSec(call.elapsed_sec ?? call.duration_sec), always: true },
          { label: "Task", value: call.task_id },
          { label: "Step", value: call.step_id },
          { label: "Tool call", value: call.toolcall_id },
        ]}
      />
      {paramsText ? (
        <details className="v2-live-details">
          <summary>Parameters</summary>
          <pre>{paramsText}</pre>
        </details>
      ) : null}
    </>
  );
}

function LiveToolList({ rows }) {
  const normalized = Array.isArray(rows) ? rows.filter((row) => row && typeof row === "object" && row.tool) : [];
  if (!normalized.length) return <div className="v2-empty">No recent tool calls.</div>;
  return (
    <div className="v2-live-list">
      {normalized.slice().reverse().slice(0, 10).map((row, index) => (
        <div key={row.toolcall_id || `${row.tool}-${index}`} className="v2-live-row">
          <div>
            <strong>{row.tool}</strong>
            <small>{row.highlights || compactPlainText(row.params_compact, 180) || row.task_id || ""}</small>
          </div>
          <span>{row.status || "done"}</span>
          <code>{formatMaybeDurationSec(row.duration_sec ?? row.elapsed_sec)}</code>
        </div>
      ))}
    </div>
  );
}

function LiveTodoList({ liveState }) {
  const rows = Array.isArray(liveState?.todo_rows) ? liveState.todo_rows : [];
  const items = rows.length
    ? rows
    : (Array.isArray(liveState?.todo_items) ? liveState.todo_items.map((item) => ({ content: item, status: "pending" })) : []);
  const visibleItems = items.filter((item) => item && plainText(item.content || item, "")).slice(0, 8);
  if (!visibleItems.length) return <div className="v2-empty">No task list has been published.</div>;
  return (
    <div className="v2-live-list">
      {visibleItems.map((item, index) => (
        <div key={`${item.content || item}-${index}`} className="v2-live-row">
          <div>
            <strong>{plainText(item.content || item)}</strong>
          </div>
          <span>{item.status || "pending"}</span>
        </div>
      ))}
    </div>
  );
}

function liveAgentStatus(agent) {
  const state = asRecord(agent);
  const llm = asRecord(state.llm);
  if (asRecord(state.active_toolcall).tool || String(llm.status || "") === "running") return "active";
  return plainText(state.status, "idle");
}

function LiveAgentList({ agents }) {
  const rows = Object.entries(asRecord(agents))
    .map(([name, agent]) => ({ name, agent: asRecord(agent), status: liveAgentStatus(agent) }))
    .filter((row) => row.name)
    .sort((left, right) => {
      const rank = (status) => (status === "active" ? 0 : status === "completed" ? 1 : 2);
      const rankDiff = rank(left.status) - rank(right.status);
      if (rankDiff !== 0) return rankDiff;
      return Number(right.agent.last_updated_ts || 0) - Number(left.agent.last_updated_ts || 0);
    });
  if (!rows.length) return <div className="v2-empty">No subagent state captured yet.</div>;
  return (
    <div className="v2-live-list">
      {rows.slice(0, 12).map(({ name, agent, status }) => {
        const activeTool = asRecord(agent.active_toolcall);
        const llm = asRecord(agent.llm);
        const detail = activeTool.tool || llm.phase || llm.model || formatTimestamp(agent.last_updated_ts);
        return (
          <div key={name} className="v2-live-row">
            <div>
              <strong>{name}</strong>
              <small>{detail}</small>
            </div>
            <span>{status}</span>
          </div>
        );
      })}
    </div>
  );
}

function LiveLlmPanel({ llm }) {
  const state = asRecord(llm);
  const hasContent = state.model || state.phase || state.status || state.text || state.reasoning_text || Object.keys(asRecord(state.usage)).length;
  if (!hasContent) return <div className="v2-empty">No LLM call captured yet.</div>;
  return (
    <>
      <LiveFieldGrid
        rows={[
          { label: "Model", value: state.model },
          { label: "Phase", value: state.phase },
          { label: "Status", value: state.status || "idle", always: true },
          { label: "Elapsed", value: formatDurationMs(state.elapsed_ms), always: true },
          { label: "Usage", value: formatTokenUsage(state.usage), always: true },
        ]}
      />
      {state.reasoning_text ? <p className="v2-live-note"><strong>Reasoning</strong>{compactPlainText(state.reasoning_text, 700)}</p> : null}
      {state.text ? <p className="v2-live-note"><strong>Output</strong>{compactPlainText(state.text, 700)}</p> : null}
    </>
  );
}

function LiveStateView({ liveState }) {
  const state = asRecord(liveState);
  const summary = asRecord(state.live_summary);
  const headline = summary.live_headline || state.current_task_goal || state.current_phase || state.status || "No active run state.";
  const taskSummary = asRecord(state.last_task_summary);
  const journal = Array.isArray(state.journal_recent) ? state.journal_recent : [];
  return (
    <div className="v2-monitor-columns v2-live-grid">
      <LivePanel icon={Activity} title="Current Activity">
        <div className="v2-live-headline">{plainText(headline)}</div>
        {summary.live_summary ? <p className="v2-live-note">{summary.live_summary}</p> : null}
        {summary.next_expected_step ? <p className="v2-live-note"><strong>Next</strong>{summary.next_expected_step}</p> : null}
        <LiveFieldGrid
          rows={[
            { label: "Status", value: state.status || "unknown", always: true },
            { label: "Phase", value: state.current_phase || "unknown", always: true },
            { label: "Task", value: state.current_task_id },
            { label: "Run", value: state.run_id },
            { label: "Updated", value: formatTimestamp(state.last_updated_ts), always: true },
          ]}
        />
      </LivePanel>

      <LivePanel icon={Cpu} title="Active Tool">
        <LiveToolDetails toolcall={state.active_toolcall} />
      </LivePanel>

      <LivePanel icon={ListChecks} title="Progress">
        <LiveProgress progress={state.progress} />
        <LiveTodoList liveState={state} />
      </LivePanel>

      <LivePanel icon={GitBranch} title="Recent Tools">
        <LiveToolList rows={state.recent_toolcalls} />
      </LivePanel>

      <LivePanel icon={MessageSquare} title="Agents">
        <LiveAgentList agents={state.agents} />
      </LivePanel>

      <LivePanel icon={BarChart3} title="Recent LLM">
        <LiveLlmPanel llm={state.llm} />
      </LivePanel>

      {(taskSummary.summary_snippet || journal.length) ? (
        <LivePanel icon={Clock} title="Task Journal">
          {taskSummary.summary_snippet ? (
            <p className="v2-live-note"><strong>{taskSummary.outcome || "latest"}</strong>{taskSummary.summary_snippet}</p>
          ) : null}
          <div className="v2-live-list">
            {journal.slice().reverse().slice(0, 5).map((item, index) => (
              <div key={`${item.task_id || "journal"}-${index}`} className="v2-live-row">
                <div>
                  <strong>{item.task_id || `Entry ${index + 1}`}</strong>
                  <small>{item.summary_snippet || ""}</small>
                </div>
                <span>{item.outcome || ""}</span>
              </div>
            ))}
          </div>
        </LivePanel>
      ) : null}
    </div>
  );
}

function matchesMonitorFilters(event, filters) {
  if (!event || !filters) return true;
  const payload = event.payload && typeof event.payload === "object" ? event.payload : {};
  const data = event.data && typeof event.data === "object" ? event.data : {};
  const fields = {
    thread: event.thread_id || data.thread_id || payload.thread_id || "",
    run: event.run_id || data.run_id || payload.run_id || data.receipt?.run_id || "",
    agent: event.agent_name || data.agent_name || payload.agent_name || payload.agent || "",
    tool: event.tool || data.tool || data.tool_name || payload.tool || payload.tool_name || "",
    category: event.category || data.category || payload.category || "",
    channel: event.channel || data.channel || payload.channel || "",
  };
  return Object.entries(filters).every(([key, value]) => {
    const needle = String(value || "").trim().toLowerCase();
    if (!needle) return true;
    return String(fields[key] || "").toLowerCase().includes(needle);
  });
}

export function MonitorPanel({ ctx, workspaceName, thread, entrypoint, events }) {
  const [activeSubtab, setActiveSubtab] = useState("overview");
  const [observability, setObservability] = useState(null);
  const [details, setDetails] = useState(null);
  const [eventFilters, setEventFilters] = useState({ thread: "", run: "", agent: "", tool: "", category: "", channel: "" });
  const [sessionEvents, setSessionEvents] = useState([]);
  const [eventPage, setEventPage] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const threadRows = Array.isArray(events) ? events.slice(-200).reverse() : [];
  const lane = String(entrypoint || thread?.entrypoint || "research");
  const projectParam = workspaceName ? `&project_space=${escapePath(workspaceName)}` : "";
  const selectedRun = observability?.selected_run || "";

  async function loadMonitor() {
    if (!ctx) return;
    setLoading(true);
    setError("");
    try {
      const baseSuffix = `lane=${escapePath(lane)}${projectParam}`;
      const obs = await apiFetch(`/api/session/${escapePath(ctx)}/observability?${baseSuffix}&limit=600`);
      const runParam = obs?.selected_run ? `run=${escapePath(obs.selected_run)}` : "";
      const suffix = `${runParam}${projectParam ? `${runParam ? "&" : ""}${projectParam.slice(1)}` : ""}`;
      const query = suffix ? `?${suffix}` : "";
      const ev = await apiFetch(`/api/session/${escapePath(ctx)}/events${query}${query ? "&" : "?"}limit=300`);
      setObservability(obs);
      setSessionEvents(Array.isArray(ev?.events) ? ev.events : []);
      setEventPage(ev);
    } catch (err) {
      setError(err.message || String(err));
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
  }, [ctx, workspaceName, thread?.status, lane]);

  useEffect(() => {
    setDetails(null);
  }, [selectedRun, ctx, workspaceName]);

  async function loadDetails() {
    if (!ctx) return;
    setError("");
    try {
      const runParam = selectedRun ? `run=${escapePath(selectedRun)}` : "";
      const suffix = `${runParam}${projectParam ? `${runParam ? "&" : ""}${projectParam.slice(1)}` : ""}`;
      const query = suffix ? `?${suffix}` : "";
      setDetails(await apiFetch(`/api/session/${escapePath(ctx)}/details${query}`));
    } catch (err) {
      setError(err.message || String(err));
    }
  }

  async function decideSelfEvolutionCandidate(candidate, action) {
    if (!ctx || !candidate?.candidate_id) return;
    setError("");
    try {
      const body = { project_space: workspaceName, action };
      await apiFetch(`/api/session/${escapePath(ctx)}/self-evolution/candidates/${escapePath(candidate.candidate_id)}/decision`, {
        method: "POST",
        body: JSON.stringify(body),
      });
      await loadDetails();
    } catch (err) {
      setError(err.message || String(err));
    }
  }

  async function rollbackSelfEvolutionCandidate(candidate) {
    if (!ctx || !candidate?.candidate_id) return;
    setError("");
    try {
      await apiFetch(`/api/session/${escapePath(ctx)}/self-evolution/candidates/${escapePath(candidate.candidate_id)}/rollback`, {
        method: "POST",
        body: JSON.stringify({ project_space: workspaceName }),
      });
      await loadDetails();
    } catch (err) {
      setError(err.message || String(err));
    }
  }

  useEffect(() => {
    if (activeSubtab === "details" && !details) loadDetails();
  }, [activeSubtab, selectedRun, ctx, workspaceName]);

  const metrics = observability?.metrics || {};
  const usage = observability?.usage_summary || {};
  const machine = observability?.machine_time_summary || {};
  const liveState = observability?.live_state || {};
  const rawEvents = Array.isArray(observability?.raw_logs?.events) && observability.raw_logs.events.length ? observability.raw_logs.events : sessionEvents;
  const chatMessages = Array.isArray(observability?.chat_messages) ? observability.chat_messages : [];
  const filteredRawEvents = rawEvents.filter((event) => matchesMonitorFilters(event, eventFilters));
  const filteredThreadRows = threadRows.filter((event) => matchesMonitorFilters(event, eventFilters));
  const filteredSessionEvents = sessionEvents.filter((event) => matchesMonitorFilters(event, eventFilters));
  const visibleEvents = [...filteredThreadRows, ...filteredSessionEvents.slice().reverse()].slice(0, 300);

  return (
    <section className="v2-tab-panel">
      <div className="v2-panel-toolbar">
        <div>
          <div className="v2-eyebrow">Monitor</div>
          <h2>Execution Monitor</h2>
          <div className="v2-muted">{selectedRun || thread?.thread_id || "No active run selected"}</div>
        </div>
        <button type="button" className="v2-ghost-btn" onClick={() => loadMonitor()} disabled={loading}><RefreshCw size={15} />Refresh</button>
      </div>
      {error ? <div className="v2-error">{error}</div> : null}
      <div className="v2-monitor-grid">
        <MonitorMetric icon={Activity} label="Status" value={thread?.status || observability?.run_status || "idle"} note={observability?.run_status_text || ""} />
        <MonitorMetric icon={Clock} label="Duration" value={formatDurationSec(metrics.duration_sec)} note={selectedRun} />
        <MonitorMetric icon={BarChart3} label="LLM calls" value={formatCount(metrics.llm_calls || usage.calls)} note={`${formatCount(usage.total_tokens || 0)} tokens`} />
        <MonitorMetric icon={Cpu} label="Machine" value={`${formatHours(machine.core_hours)} core h`} note={`${formatHours(machine.node_hours)} node h`} />
        <MonitorMetric icon={Database} label="Cost" value={formatCost(usage.cost_usd)} note={usage.cost_source || ""} />
        <MonitorMetric icon={GitBranch} label="Tool calls" value={formatCount(metrics.tool_calls)} note={`${formatCount(metrics.tool_failures)} failed`} />
      </div>
      <div className="v2-subtabs">
        {[
          ["overview", "Overview"],
          ["live", "Live"],
          ["events", "Events"],
          ["raw", "Raw"],
          ["details", "Details"],
        ].map(([value, label]) => (
          <button key={value} type="button" className={activeSubtab === value ? "active" : ""} onClick={() => setActiveSubtab(value)}>{label}</button>
        ))}
      </div>
      <div className="v2-monitor-filters">
        <Filter size={14} />
        {[
          ["thread", "Thread"],
          ["run", "Run"],
          ["agent", "Agent"],
          ["tool", "Tool"],
          ["category", "Category"],
          ["channel", "Channel"],
        ].map(([key, label]) => (
          <input
            key={key}
            value={eventFilters[key] || ""}
            onChange={(event) => setEventFilters((prev) => ({ ...prev, [key]: event.target.value }))}
            placeholder={label}
            aria-label={`${label} filter`}
          />
        ))}
      </div>
      {loading ? <div className="v2-muted">Loading monitor data...</div> : null}
      {activeSubtab === "overview" ? (
        <div className="v2-monitor-columns">
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><MessageSquare size={15} /><h3>Models / Agents</h3></div>
            <MonitorList rows={metrics.models} empty="No model calls captured yet." />
            <MonitorList rows={metrics.agents} empty="No agent attribution captured yet." />
          </section>
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><ListChecks size={15} /><h3>Tools / Tasks</h3></div>
            <MonitorList rows={metrics.tools} empty="No tool calls captured yet." />
            <pre className="v2-code compact">{compactJson(observability?.todo_items || [], 1600)}</pre>
          </section>
        </div>
      ) : null}
      {activeSubtab === "live" ? (
        <LiveStateView liveState={liveState} />
      ) : null}
      {activeSubtab === "events" ? (
        <div className="v2-event-list">
          {visibleEvents.map((event, index) => {
            const payload = event.data || event.payload || event;
            return (
              <details key={event.seq || event.id || `${event.event || event.name}-${index}`} className="v2-event-row">
                <summary>
                  <span>{event.event || event.name || event.category || "event"}</span>
                  <small>{event.status || event.source || ""}</small>
                  <code>{compactJson(payload, 900)}</code>
                </summary>
                <pre>{jsonText(payload) || "(empty)"}</pre>
              </details>
            );
          })}
          {!visibleEvents.length ? <div className="v2-empty">No events captured yet.</div> : null}
        </div>
      ) : null}
      {activeSubtab === "raw" ? (
        <div className="v2-monitor-columns">
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><MessageSquare size={15} /><h3>Chat History</h3></div>
            <div className="v2-raw-list">
              {chatMessages.slice().reverse().slice(0, 80).map((message, index) => (
                <details key={`${message.message_id || index}-${index}`} className="v2-raw-row">
                  <summary><span>{message.role || "message"}</span><small>{message.created_at || ""}</small></summary>
                  <pre>{compactJson(message.content || message, 1600)}</pre>
                </details>
              ))}
              {!chatMessages.length ? <div className="v2-empty">No chat history for this run.</div> : null}
            </div>
          </section>
          <section className="v2-monitor-panel">
            <div className="v2-monitor-panel-head"><Braces size={15} /><h3>Raw Logs</h3></div>
            <div className="v2-raw-list">
              {filteredRawEvents.slice().reverse().slice(0, 120).map((event, index) => (
                <details key={`${event.id || event.seq || index}-${event.name || event.event}`} className="v2-raw-row">
                  <summary><span>{event.name || event.event || event.category || "event"}</span><small>{event.ts || event.created_at || ""}</small></summary>
                  <pre>{compactJson(event.payload || event, 1600)}</pre>
                </details>
              ))}
              {!filteredRawEvents.length ? <div className="v2-empty">No raw logs captured yet.</div> : null}
            </div>
          </section>
        </div>
      ) : null}
      {activeSubtab === "details" ? (
        <div className="v2-monitor-columns">
          <SelfEvolutionPanel
            payload={details?.self_evolution}
            onDecision={decideSelfEvolutionCandidate}
            onRollback={rollbackSelfEvolutionCandidate}
          />
          <MonitorCodeBlock title="Task State" text={details?.task_state || ""} />
          <MonitorCodeBlock title="Memory" text={details?.memory || ""} />
        </div>
      ) : null}
      {eventPage?.has_more ? <div className="v2-muted">Older event pages are available in the API.</div> : null}
    </section>
  );
}
