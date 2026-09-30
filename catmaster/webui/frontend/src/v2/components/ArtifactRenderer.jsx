import { lazy, Suspense, useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { Download, Expand, FileText, RefreshCw } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";

import { apiFetch } from "../useCatMasterThreadRuntime";
import { normalizeMathMarkdown } from "../markdown";
import {
  displayValue,
  formatBytes,
  humanizeKey,
  isInternalStoragePath,
  presentError,
  userFacingFileTitle,
} from "../presentation.js";
import { inferPreviewKind, parseCompleteJson, yamlOutline } from "../structuredPreview.js";

const JSMOL_SCRIPT_SRC = "/static/vendor/jsmol/JSmol.min.js";
const MatterVizBridge = lazy(() => import("../structure/MatterVizBridge"));
const StructureWorkbench = lazy(() => import("../structure/StructureWorkbench"));

function loadJSmol() {
  if (window.Jmol) return Promise.resolve(window.Jmol);
  return new Promise((resolve, reject) => {
    const existing = document.querySelector(`script[src="${JSMOL_SCRIPT_SRC}"]`);
    if (existing) {
      existing.addEventListener("load", () => resolve(window.Jmol), { once: true });
      existing.addEventListener("error", reject, { once: true });
      return;
    }
    const script = document.createElement("script");
    script.src = JSMOL_SCRIPT_SRC;
    script.async = true;
    script.onload = () => resolve(window.Jmol);
    script.onerror = reject;
    document.head.appendChild(script);
  });
}

function escapeJSmol(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function jsmolUrlSpecifier(url, fileType) {
  const normalizedUrl = String(url || "").trim();
  const normalizedType = String(fileType || "").trim();
  if (!normalizedUrl) return "";
  return normalizedType ? `${normalizedType}::${normalizedUrl}` : normalizedUrl;
}

function MarkdownContent({ text }) {
  const source = normalizeMathMarkdown(text);
  return (
    <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
      {source}
    </Markdown>
  );
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

function TableHumanView({ human }) {
  const header = Array.isArray(human?.columns) ? human.columns : [];
  const body = Array.isArray(human?.rows) ? human.rows : [];
  const pageSize = 50;
  const [pageIndex, setPageIndex] = useState(0);
  useEffect(() => {
    setPageIndex(0);
  }, [human]);
  if (!header.length) return <div className="v2-empty">The table has no visible columns.</div>;
  const pageCount = Math.max(1, Math.ceil(body.length / pageSize));
  const safePageIndex = Math.min(pageIndex, pageCount - 1);
  const start = safePageIndex * pageSize;
  const visibleRows = body.slice(start, start + pageSize);
  const shownTotal = Number(human.shown_count || body.length);
  return (
    <>
      <div className="v2-table-wrap">
      <table className="v2-table">
        <thead>
          <tr>{header.map((cell, index) => <th key={index}>{displayValue(cell, "")}</th>)}</tr>
        </thead>
        <tbody>
          {visibleRows.map((row, rowIndex) => (
            <tr key={start + rowIndex}>{header.map((_, colIndex) => <td key={colIndex}>{displayValue(row[colIndex], "")}</td>)}</tr>
          ))}
        </tbody>
      </table>
      </div>
      <div className="v2-range-label">
        {human?.total_unknown
          ? `Showing rows ${(start + 1).toLocaleString()}–${Math.min(start + visibleRows.length, shownTotal).toLocaleString()} of ${shownTotal.toLocaleString()} loaded rows; open Raw source to continue reading the file.`
          : `Showing rows ${(start + 1).toLocaleString()}–${Math.min(start + visibleRows.length, shownTotal).toLocaleString()} of ${Number(human.total_count || shownTotal).toLocaleString()}.`}
      </div>
      {pageCount > 1 ? (
        <nav className="v2-table-pagination" aria-label="Table pagination">
          <button
            type="button"
            className="v2-ghost-btn compact"
            disabled={safePageIndex === 0}
            onClick={() => setPageIndex((current) => Math.max(0, current - 1))}
          >
            Previous 50
          </button>
          <span>Page {safePageIndex + 1} of {pageCount}</span>
          <button
            type="button"
            className="v2-ghost-btn compact"
            disabled={safePageIndex >= pageCount - 1}
            onClick={() => setPageIndex((current) => Math.min(pageCount - 1, current + 1))}
          >
            Next 50
          </button>
        </nav>
      ) : null}
    </>
  );
}

function PdfPreview({ url }) {
  if (!url) {
    return (
      <div className="v2-download-card">
        <FileText size={18} />
        <span>PDF source is unavailable.</span>
      </div>
    );
  }
  return (
    <div className="v2-pdf-frame">
      <iframe title="PDF preview" src={url} />
    </div>
  );
}

function JsonHumanView({ human }) {
  if (!human || !human.kind) {
    return <div className="v2-empty">This JSON file is too large or irregular for the record view. Use Raw source or download the file.</div>;
  }
  if (human.kind === "record") {
    return (
      <div className="v2-json-human">
        <dl className="v2-semantic-fields">
          {(human.fields || []).map((field, index) => (
            <div key={`${field.label || "field"}-${index}`}>
              <dt>{displayValue(field.label, "Field")}</dt>
              <dd>{displayValue(field.value, "—")}</dd>
            </div>
          ))}
        </dl>
        {(human.collections || []).map((collection, index) => (
          <section key={`${collection.label || "collection"}-${index}`} className="v2-json-collection">
            <h4>{displayValue(collection.label, "Collection")}</h4>
            <ul>{(collection.items || []).map((item, itemIndex) => <li key={`${itemIndex}`}>{displayValue(item)}</li>)}</ul>
            {collection.truncated ? <small>Showing {collection.shown_count} of {collection.total_count} items.</small> : null}
          </section>
        ))}
      </div>
    );
  }
  if (human.kind === "list") {
    return (
      <div className="v2-json-human">
        <ul>{(human.items || []).map((item, index) => <li key={index}>{displayValue(item)}</li>)}</ul>
        {human.truncated ? <small>Showing {human.shown_count} of {human.total_count} items.</small> : null}
      </div>
    );
  }
  return <div className="v2-json-value">{displayValue(human.value, "—")}</div>;
}

function StructuredValue({ label = "", value, depth = 0 }) {
  const isArray = Array.isArray(value);
  const isObject = value !== null && typeof value === "object" && !isArray;
  if (!isArray && !isObject) {
    const scalar = value === null ? "null" : typeof value === "string" ? value : String(value);
    return (
      <div className="v2-structured-leaf">
        {label ? <span>{label}</span> : null}
        <code>{scalar}</code>
      </div>
    );
  }
  const rows = isArray ? value.map((item, index) => [`Item ${index + 1}`, item]) : Object.entries(value);
  const summary = label || (isArray ? `List · ${rows.length} items` : `Record · ${rows.length} fields`);
  return (
    <details className="v2-structured-branch" open={depth < 1}>
      <summary>
        <span>{summary}</span>
        <small>{isArray ? `${rows.length} items` : `${rows.length} fields`}</small>
      </summary>
      <div>
        {rows.map(([key, item], index) => (
          <StructuredValue
            key={`${key}-${index}`}
            label={isArray ? key : String(key)}
            value={item}
            depth={depth + 1}
          />
        ))}
        {!rows.length ? <div className="v2-muted">Empty {isArray ? "list" : "record"}</div> : null}
      </div>
    </details>
  );
}

function JsonTreeView({ preview, onOpenSource }) {
  const parsed = parseCompleteJson(preview?.preview_text || "", Boolean(preview?.page?.truncated));
  if (parsed.ok) {
    return (
      <div className="v2-structured-tree">
        <StructuredValue value={parsed.value} />
      </div>
    );
  }
  return (
    <div className="v2-structured-tree">
      <JsonHumanView human={preview?.human_view || {}} />
      <div className="v2-truncation-notice" role="status">
        <span>
          {preview?.page?.truncated
            ? `Formatted view uses ${Number(preview.page.shown_count || 0).toLocaleString()} of ${Number(preview.page.total_count || 0).toLocaleString()} source bytes.`
            : "This file could not be parsed as complete JSON."}
        </span>
        <button type="button" className="v2-ghost-btn compact" onClick={onOpenSource}>Open Raw source</button>
      </div>
    </div>
  );
}

function YamlTreeNode({ node, depth = 0 }) {
  if (!node?.children?.length) {
    return (
      <div className="v2-structured-leaf yaml" title={`Source line ${node.line}`}>
        <span>Line {node.line}</span>
        <code>{node.label}</code>
      </div>
    );
  }
  return (
    <details className="v2-structured-branch" open={depth < 1}>
      <summary>
        <span>{node.label}</span>
        <small>Line {node.line}</small>
      </summary>
      <div>
        {node.children.map((child) => <YamlTreeNode key={child.id} node={child} depth={depth + 1} />)}
      </div>
    </details>
  );
}

function YamlTreeView({ preview, onOpenSource }) {
  const rows = yamlOutline(preview?.preview_text || "");
  return (
    <div className="v2-structured-tree">
      {rows.length ? rows.map((node) => <YamlTreeNode key={node.id} node={node} />) : (
        <div className="v2-empty compact">This YAML file is empty.</div>
      )}
      {preview?.page?.truncated ? (
        <div className="v2-truncation-notice" role="status">
          <span>
            Showing the hierarchy from {Number(preview.page.shown_count || 0).toLocaleString()} of {Number(preview.page.total_count || 0).toLocaleString()} source bytes.
          </span>
          <button type="button" className="v2-ghost-btn compact" onClick={onOpenSource}>Continue in Raw source</button>
        </div>
      ) : null}
    </div>
  );
}

function SourcePreview({ preview }) {
  const [text, setText] = useState(String(preview?.preview_text || ""));
  const [page, setPage] = useState(preview?.page || {});
  const [loadedCount, setLoadedCount] = useState(
    Number(preview?.page?.range_start || 0)
      + Number(preview?.page?.shown_count || String(preview?.preview_text || "").length),
  );
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  useEffect(() => {
    setText(String(preview?.preview_text || ""));
    setPage(preview?.page || {});
    setLoadedCount(
      Number(preview?.page?.range_start || 0)
        + Number(preview?.page?.shown_count || String(preview?.preview_text || "").length),
    );
    setError("");
  }, [
    preview?.path,
    preview?.preview_text,
    preview?.page?.range_start,
    preview?.page?.shown_count,
    preview?.page?.total_count,
  ]);
  const fullRef = String(page?.full_content_ref || "");
  const hasCursor = page?.next_cursor !== "" && page?.next_cursor !== null && page?.next_cursor !== undefined;
  const canPage = Boolean(page?.truncated && hasCursor && fullRef.startsWith("/api/session/"));

  async function loadMore() {
    if (!canPage) return;
    setLoading(true);
    setError("");
    try {
      const join = fullRef.includes("?") ? "&" : "?";
      const payload = await apiFetch(`${fullRef}${join}cursor=${encodeURIComponent(String(page.next_cursor || loadedCount))}`);
      const nextText = String(payload.preview_text || "");
      setText((current) => `${current}${nextText}`);
      setLoadedCount((current) => {
        const shown = Number(payload?.page?.shown_count || nextText.length || 0);
        const total = Number(payload?.page?.total_count || page?.total_count || 0);
        return total ? Math.min(total, current + shown) : current + shown;
      });
      setPage(payload.page || {});
    } catch (err) {
      setError(err);
    } finally {
      setLoading(false);
    }
  }

  return (
    <div className="v2-source-view">
      <pre className="v2-code">{text || "(empty file)"}</pre>
      <div className="v2-truncation-notice">
        <span>
          {Number(page.total_count || 0)
            ? `Loaded ${loadedCount.toLocaleString()} of ${Number(page.total_count).toLocaleString()} bytes.`
            : `${loadedCount.toLocaleString()} bytes loaded.`}
        </span>
        {canPage ? <button type="button" className="v2-ghost-btn compact" onClick={loadMore} disabled={loading}>{loading ? "Loading…" : "Load more"}</button> : null}
        {page?.truncated && fullRef && !canPage ? <a className="v2-ghost-btn compact" href={fullRef} target="_blank" rel="noreferrer">Open full source</a> : null}
      </div>
      <ErrorNotice error={error} compact />
    </div>
  );
}

export function JSmolStructurePreview({ preview }) {
  const hostRef = useRef(null);
  const [status, setStatus] = useState("Loading structure viewer...");
  const structure = preview?.structure || {};
  useEffect(() => {
    let disposed = false;
    let observer = null;
    function labelStructureCanvases() {
      if (!hostRef.current) return;
      hostRef.current.querySelectorAll("canvas").forEach((canvas) => {
        canvas.setAttribute("role", "img");
        canvas.setAttribute(
          "aria-label",
          `Interactive structure view of ${structure.formula || preview?.name || "the selected structure"}`,
        );
      });
    }
    async function mount() {
      if (!hostRef.current || !structure?.viewer_source_url) {
        setStatus("Structure source is unavailable.");
        return;
      }
      try {
        const Jmol = await loadJSmol();
        if (disposed || !hostRef.current) return;
        const id = `catmaster_jmol_${Math.random().toString(16).slice(2)}`;
        const source = jsmolUrlSpecifier(structure.viewer_source_url, structure.viewer_source_file_type);
        const info = {
          width: "100%",
          height: 360,
          use: "HTML5",
          j2sPath: "/static/vendor/jsmol/j2s",
          disableJ2SLoadMonitor: true,
          disableInitialConsole: true,
          script: `load "${escapeJSmol(source)}"; set antialiasDisplay true; wireframe 0.12; spacefill 23%;`,
        };
        observer = new MutationObserver(labelStructureCanvases);
        observer.observe(hostRef.current, { childList: true, subtree: true });
        hostRef.current.innerHTML = Jmol.getAppletHtml(id, info);
        labelStructureCanvases();
        setStatus("");
      } catch (err) {
        setStatus(presentError(err, "Structure viewer failed to load.").message);
      }
    }
    mount();
    return () => {
      disposed = true;
      observer?.disconnect();
      if (hostRef.current) hostRef.current.innerHTML = "";
    };
  }, [structure?.viewer_source_url, structure?.viewer_source_file_type]);

  return (
    <div className="v2-structure">
      <div className="v2-structure-meta">
        <span>{structure.formula || preview?.name || "structure"}</span>
        {structure.atom_count ? <span>{structure.atom_count} atoms</span> : null}
        {structure.periodic ? <span>periodic</span> : <span>molecular</span>}
        {Array.isArray(structure.cell_lengths) && structure.cell_lengths.length
          ? <span>Cell {structure.cell_lengths.map((value) => Number(value).toFixed(2)).join(" × ")} Å</span>
          : null}
      </div>
      <div
        ref={hostRef}
        className="v2-structure-canvas"
        role="region"
        aria-label={`Structure viewer for ${structure.formula || preview?.name || "selected structure"}`}
      />
      {status ? <div className="v2-empty">{status}</div> : null}
    </div>
  );
}

function StructurePreview({ preview, workspaceName, onOpenWorkbench }) {
  const structure = preview?.structure || {};
  const [document, setDocument] = useState(null);
  const [error, setError] = useState("");
  const [loadingDocument, setLoadingDocument] = useState(false);
  const [useFallback, setUseFallback] = useState(false);
  useEffect(() => {
    let cancelled = false;
    const controller = new AbortController();
    setDocument(null);
    setError("");
    setUseFallback(false);
    if (!workspaceName || !preview?.path) {
      setLoadingDocument(false);
      return undefined;
    }
    setLoadingDocument(true);
    apiFetch("/api/structures/open", {
      method: "POST",
      signal: controller.signal,
      body: JSON.stringify({ workspace: workspaceName, path: preview.path }),
    })
      .then((payload) => {
        if (!cancelled) {
          setDocument(payload);
          setLoadingDocument(false);
        }
      })
      .catch((reason) => {
        if (!cancelled) {
          setError(reason);
          setUseFallback(Boolean(structure.viewer_source_url));
          setLoadingDocument(false);
        }
      });
    return () => {
      cancelled = true;
      controller.abort();
    };
  }, [workspaceName, preview?.path, structure.viewer_source_url]);
  return (
    <div className="v2-structure">
      <div className="v2-structure-meta">
        <span>{document?.summary?.formula || structure.formula || preview?.name || "structure"}</span>
        {document?.summary?.atom_count || structure.atom_count ? <span>{document?.summary?.atom_count || structure.atom_count} atoms</span> : null}
        <span>{document?.snapshot?.mode === "molecule" || !structure.periodic ? "molecular" : "periodic"}</span>
        {document?.summary?.space_group?.symbol ? <span>{document.summary.space_group.symbol} ({document.summary.space_group.number})</span> : null}
      </div>
      <div className="v2-structure-preview-actions">
        <button type="button" className="v2-primary-btn" onClick={onOpenWorkbench} disabled={!workspaceName}>
          <Expand size={15} /> Open Structure Workbench
        </button>
        {useFallback && document ? <button type="button" onClick={() => setUseFallback(false)}>Return to materials viewer</button> : null}
      </div>
      <div className="v2-structure-preview-canvas">
        {loadingDocument ? (
          <div className="v2-empty" role="status">Preparing the materials viewer…</div>
        ) : document && !useFallback ? (
          <Suspense fallback={<div className="v2-empty">Starting the materials viewer…</div>}>
            <MatterVizBridge
              structure={document.viewer_structure}
              readOnly
              onError={(message) => {
                setError(message);
                setUseFallback(true);
              }}
            />
          </Suspense>
        ) : useFallback && structure.viewer_source_url ? (
          <JSmolStructurePreview preview={preview} />
        ) : (
          <div className="v2-empty">This structure could not be prepared for the materials viewer.</div>
        )}
      </div>
      {error ? (
        <div className="v2-preview-fallback-note">
          <ErrorNotice error={error} compact />
          {structure.viewer_source_url ? <button type="button" onClick={() => setUseFallback(true)}>Use JSmol compatibility view</button> : null}
        </div>
      ) : null}
    </div>
  );
}

function VolumePreview({ preview, onOpenWorkbench }) {
  const volume = preview?.volume || {};
  return (
    <div className="v2-volume-preview">
      <div className="v2-volume-icon" aria-hidden="true">ρ</div>
      <div>
        <strong>{preview?.name || "Scalar field"}</strong>
        <p>{String(volume.format || "volume").toUpperCase()} · {formatBytes(volume.file_size || preview?.size || 0)}</p>
        <p>Open the workbench for structure overlay, positive/negative isosurfaces, multiple fields, and crystallographic slices.</p>
      </div>
      <button type="button" className="v2-primary-btn" onClick={onOpenWorkbench}><Expand size={15} /> Open Volume Workbench</button>
    </div>
  );
}

export default function ArtifactRenderer({ artifact, filePreview, onRefresh, showHeader = true, workspaceName = "", ctx = "" }) {
  const [preview, setPreview] = useState(filePreview || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [activeView, setActiveView] = useState("human");
  const [workbenchOpen, setWorkbenchOpen] = useState(false);
  const artifactId = artifact?.artifact_id;

  useEffect(() => {
    if (filePreview) {
      setPreview(filePreview);
      return;
    }
    let cancelled = false;
    async function load() {
      if (!artifactId) return;
      setLoading(true);
      setError("");
      try {
        const payload = await apiFetch(`/api/artifacts/${encodeURIComponent(artifactId)}/preview`);
        if (!cancelled) setPreview(payload);
      } catch (err) {
        if (!cancelled) setError(err);
      } finally {
        if (!cancelled) setLoading(false);
      }
    }
    load();
    return () => {
      cancelled = true;
    };
  }, [artifactId, filePreview]);

  const record = preview?.artifact || artifact || {};
  const displayPath = String(preview?.path || record.path || "");
  const path = displayPath.toLowerCase();
  const internalPath = isInternalStoragePath(displayPath);
  const visibleTitle = userFacingFileTitle(record.title || preview?.name, displayPath, "Artifact");
  const inferredKind = path.endsWith(".pdf") ? "pdf" : path.endsWith(".md") || path.endsWith(".markdown") || path.endsWith(".mdx") || path.endsWith(".rst") ? "markdown" : "";
  const kind = inferPreviewKind(preview?.kind || record.renderer || inferredKind || "text", path);
  const downloadUrl = record.download_url || preview?.download_url;
  const contentUrl = preview?.content_url || downloadUrl;
  const hasHumanView = ["structure", "volume", "markdown", "csv", "json", "yaml"].includes(kind);

  useEffect(() => {
    if (!workbenchOpen) return undefined;
    const previous = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    return () => {
      document.body.style.overflow = previous;
    };
  }, [workbenchOpen]);

  useEffect(() => {
    setActiveView(hasHumanView ? "human" : "source");
  }, [kind, preview?.path]);

  if (!artifact && !filePreview) {
    return <div className="v2-empty">Select a file or result from the conversation to preview it here.</div>;
  }
  return (
    <div className="v2-artifact-preview">
      {showHeader ? (
        <div className="v2-inspector-head">
          <div>
            <div className="v2-eyebrow">{humanizeKey(kind || "preview")}</div>
            <h3>{visibleTitle}</h3>
            {!internalPath && displayPath ? <div className="v2-muted">{displayPath}</div> : null}
            {Number(preview?.size || record?.size || 0) > 0 ? (
              <div className="v2-muted">{formatBytes(preview?.size || record?.size)}</div>
            ) : null}
          </div>
          <div className="v2-icon-row">
            {onRefresh ? (
              <button type="button" className="v2-icon-btn" onClick={onRefresh} aria-label="Refresh artifact preview">
                <RefreshCw size={16} />
              </button>
            ) : null}
            {downloadUrl ? (
              <a className="v2-icon-btn" href={downloadUrl} aria-label={`Download ${visibleTitle}`}>
                <Download size={16} />
              </a>
            ) : null}
          </div>
        </div>
      ) : null}
      {internalPath ? (
        <details className="v2-technical-details">
          <summary>Technical details</summary>
          <div>
            <span>Managed storage reference</span>
            <code>{displayPath}</code>
          </div>
        </details>
      ) : null}
      {record.summary ? <p className="v2-summary">{displayValue(record.summary)}</p> : null}
      {loading ? <div className="v2-empty" role="status">Loading preview…</div> : null}
      <ErrorNotice error={error} />
      {!loading && !error && preview ? (
        <>
          {hasHumanView ? (
            <div className="v2-preview-tabs" role="tablist" aria-label="File preview mode">
              <button type="button" role="tab" aria-selected={activeView === "human"} className={activeView === "human" ? "active" : ""} onClick={() => setActiveView("human")}>Human view</button>
              <button type="button" role="tab" aria-selected={activeView === "source"} className={activeView === "source" ? "active" : ""} onClick={() => setActiveView("source")}>Raw source</button>
            </div>
          ) : null}
          {kind === "image" && downloadUrl ? <img className="v2-image-preview" src={downloadUrl} alt={record.title || preview.name || "Artifact"} /> : null}
          {activeView === "human" && kind === "structure" && preview.structure && !workbenchOpen ? (
            <StructurePreview preview={preview} workspaceName={workspaceName} onOpenWorkbench={() => setWorkbenchOpen(true)} />
          ) : null}
          {activeView === "human" && kind === "volume" && preview.volume && !workbenchOpen ? <VolumePreview preview={preview} onOpenWorkbench={() => setWorkbenchOpen(true)} /> : null}
          {activeView === "human" && kind === "markdown" ? <div className="v2-markdown"><MarkdownContent text={preview.preview_text || ""} /></div> : null}
          {activeView === "human" && kind === "csv" ? <TableHumanView human={preview.human_view || {}} /> : null}
          {activeView === "human" && kind === "json" ? <JsonTreeView preview={preview} onOpenSource={() => setActiveView("source")} /> : null}
          {activeView === "human" && kind === "yaml" ? <YamlTreeView preview={preview} onOpenSource={() => setActiveView("source")} /> : null}
          {kind === "pdf" ? <PdfPreview url={contentUrl} /> : null}
          {kind === "binary" ? (
            <div className="v2-download-card">
              <FileText size={18} />
              <span>This binary file cannot be shown safely in the browser. Download the original to inspect it.</span>
              {downloadUrl ? <a className="v2-ghost-btn compact" href={downloadUrl}>Download file</a> : null}
            </div>
          ) : null}
          {!["image", "pdf", "binary"].includes(kind) && activeView === "source" ? <SourcePreview preview={preview} /> : null}
          {workbenchOpen ? createPortal(
            <Suspense fallback={<div className="v2-structure-workbench"><div className="v2-workbench-loading">Loading the full workbench…</div></div>}>
              <StructureWorkbench
                workspaceName={workspaceName}
                path={preview.path || record.path}
                preview={preview}
                onClose={() => setWorkbenchOpen(false)}
                fallback={<JSmolStructurePreview preview={preview} />}
                onSaved={() => onRefresh?.()}
              />
            </Suspense>,
            document.body,
          ) : null}
        </>
      ) : null}
    </div>
  );
}
