import { useEffect, useMemo, useRef, useState } from "react";
import { Download, FileText, RefreshCw } from "lucide-react";
import Markdown from "react-markdown";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import Papa from "papaparse";

import { apiFetch } from "../useCatMasterThreadRuntime";

const JSMOL_SCRIPT_SRC = "/static/vendor/jsmol/JSmol.min.js";

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
  return (
    <Markdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
      {String(text || "")}
    </Markdown>
  );
}

function CsvPreview({ text }) {
  const rows = useMemo(() => {
    const parsed = Papa.parse(String(text || "").trim(), { skipEmptyLines: true });
    return Array.isArray(parsed.data) ? parsed.data.slice(0, 80) : [];
  }, [text]);
  if (!rows.length) return <div className="v2-empty">CSV preview is empty.</div>;
  const header = rows[0] || [];
  const body = rows.slice(1);
  return (
    <div className="v2-table-wrap">
      <table className="v2-table">
        <thead>
          <tr>{header.map((cell, index) => <th key={index}>{String(cell || "")}</th>)}</tr>
        </thead>
        <tbody>
          {body.map((row, rowIndex) => (
            <tr key={rowIndex}>{header.map((_, colIndex) => <td key={colIndex}>{String(row[colIndex] ?? "")}</td>)}</tr>
          ))}
        </tbody>
      </table>
    </div>
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

function DocumentPreview({ preview }) {
  const text = String(preview?.preview_text || "");
  const path = String(preview?.path || preview?.name || "").toLowerCase();
  const markdownLike = path.endsWith(".md") || path.endsWith(".markdown") || path.endsWith(".mdx") || path.endsWith(".rst");
  if (markdownLike) {
    return <div className="v2-markdown"><MarkdownContent text={text} /></div>;
  }
  return <pre className="v2-code">{text || "(empty document preview)"}</pre>;
}

function StructurePreview({ preview }) {
  const hostRef = useRef(null);
  const [status, setStatus] = useState("Loading structure viewer...");
  const structure = preview?.structure || {};
  useEffect(() => {
    let disposed = false;
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
        hostRef.current.innerHTML = Jmol.getAppletHtml(id, info);
        setStatus("");
      } catch (err) {
        setStatus(err.message || "Structure viewer failed to load.");
      }
    }
    mount();
    return () => {
      disposed = true;
      if (hostRef.current) hostRef.current.innerHTML = "";
    };
  }, [structure?.viewer_source_url, structure?.viewer_source_file_type]);

  return (
    <div className="v2-structure">
      <div className="v2-structure-meta">
        <span>{structure.formula || preview?.name || "structure"}</span>
        {structure.atom_count ? <span>{structure.atom_count} atoms</span> : null}
        {structure.periodic ? <span>periodic</span> : <span>molecular</span>}
      </div>
      <div ref={hostRef} className="v2-structure-canvas" />
      {status ? <div className="v2-empty">{status}</div> : null}
      {structure.viewer_text ? (
        <details className="v2-structure-source">
          <summary>Structure source</summary>
          <pre>{String(structure.viewer_text || "").slice(0, 4000)}</pre>
        </details>
      ) : null}
    </div>
  );
}

export default function ArtifactRenderer({ artifact, filePreview, onRefresh }) {
  const [preview, setPreview] = useState(filePreview || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
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
        if (!cancelled) setError(err.message || String(err));
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
  const path = String(preview?.path || record.path || "").toLowerCase();
  const inferredKind = path.endsWith(".pdf") ? "pdf" : path.endsWith(".md") || path.endsWith(".markdown") || path.endsWith(".mdx") || path.endsWith(".rst") ? "markdown" : "";
  const kind = preview?.kind || record.renderer || inferredKind || "text";
  const downloadUrl = record.download_url || preview?.download_url;
  const contentUrl = preview?.content_url || downloadUrl;

  if (!artifact && !filePreview) {
    return <div className="v2-empty">Select an artifact or file.</div>;
  }
  return (
    <div className="v2-artifact-preview">
      <div className="v2-inspector-head">
        <div>
          <div className="v2-eyebrow">{record.renderer || preview?.kind || "preview"}</div>
          <h3>{record.title || preview?.name || record.path || "Artifact"}</h3>
          <div className="v2-muted">{record.path || preview?.path || ""}</div>
        </div>
        <div className="v2-icon-row">
          {onRefresh ? (
            <button type="button" className="v2-icon-btn" onClick={onRefresh} title="Refresh">
              <RefreshCw size={16} />
            </button>
          ) : null}
          {downloadUrl ? (
            <a className="v2-icon-btn" href={downloadUrl} title="Download">
              <Download size={16} />
            </a>
          ) : null}
        </div>
      </div>
      {record.summary ? <p className="v2-summary">{record.summary}</p> : null}
      {loading ? <div className="v2-empty">Loading preview...</div> : null}
      {error ? <div className="v2-error">{error}</div> : null}
      {!loading && !error && preview ? (
        <>
          {kind === "image" && downloadUrl ? <img className="v2-image-preview" src={downloadUrl} alt={record.title || preview.name || "Artifact"} /> : null}
          {kind === "structure" && preview.structure ? <StructurePreview preview={preview} /> : null}
          {kind === "markdown" ? <DocumentPreview preview={preview} /> : null}
          {kind === "csv" || String(preview.path || "").toLowerCase().endsWith(".csv") || String(preview.path || "").toLowerCase().endsWith(".tsv")
            ? <CsvPreview text={preview.preview_text || ""} />
            : null}
          {kind === "pdf" ? <PdfPreview url={contentUrl} /> : null}
          {!["image", "structure", "markdown", "csv", "pdf"].includes(kind) && !(String(preview.path || "").toLowerCase().endsWith(".csv")) ? (
            <DocumentPreview preview={preview} />
          ) : null}
          {preview.truncated ? <div className="v2-muted">Preview truncated for size.</div> : null}
        </>
      ) : null}
    </div>
  );
}
