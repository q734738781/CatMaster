import { useEffect, useRef, useState } from "react";
import { createPortal } from "react-dom";
import { ArrowLeft, Clipboard, Download, Eye, Hammer, RefreshCw, X } from "lucide-react";

import ArtifactRenderer from "./ArtifactRenderer";
import AsyncSubagentActivity from "./AsyncSubagentActivity";
import { apiFetch } from "../useCatMasterThreadRuntime";
import { displayValue, presentError, redactErrorText } from "../presentation.js";

function escapePath(value) {
  return encodeURIComponent(String(value || ""));
}

function previewTitle(item) {
  const label = item?.title || item?.artifact?.title || item?.artifact?.path || item?.path || "Preview";
  return String(label).split("/").filter(Boolean).pop() || label;
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

function ActivityContent({ title, source }) {
  const [content, setContent] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  async function load(cursor = "") {
    setLoading(true);
    setError("");
    try {
      const url = cursor ? `${source}${source.includes("?") ? "&" : "?"}cursor=${encodeURIComponent(cursor)}` : source;
      const result = await apiFetch(url);
      setContent((current) => ({...result, text: (cursor ? current?.text || "" : "") + result.text}));
    } catch (reason) { setError(reason); }
    finally { setLoading(false); }
  }
  return <details className="v2-activity-payload" onToggle={(event) => {
    if (event.currentTarget.open && content === null && !loading) load();
  }}>
    <summary>{title}</summary>
    {loading ? <p className="v2-muted">Loading…</p> : null}
    <ErrorNotice error={error} />
    {content ? <pre>{content.text || "No content recorded."}</pre> : null}
    {content?.page?.truncated ? <button type="button" className="v2-ghost-btn" disabled={loading} onClick={() => load(content.page.next_cursor)}>Load more</button> : null}
  </details>;
}

function ActivityPreview({ item }) {
  const [part, setPart] = useState(item?.part || {});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    let cancelled = false;
    const initial = item?.part || {};
    setPart(initial);
    setError("");
    setLoading(false);
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
  }, [item?.id, item?.part]);

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
    part?.input_ref ? ["Complete input", part.input_ref] : null,
    part?.output_ref ? ["Complete output", part.output_ref] : null,
    part?.diagnostics_ref ? ["Diagnostics reference", failed ? redactErrorText(part.diagnostics_ref) : part.diagnostics_ref] : null,
    part?.detail_ref ? ["Detail reference", failed ? redactErrorText(part.detail_ref) : part.detail_ref] : null,
    part?.id ? ["Activity ID", failed ? redactErrorText(part.id) : part.id] : null,
  ].filter(Boolean);

  return (
    <section className="v2-activity-preview" aria-label={visible(part?.title, "Activity details")}>
      <div className="v2-activity-preview-head">
        <Hammer size={17} aria-hidden="true" />
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
          {items.map((entry, index) => (
            <li key={`${entry?.label || "item"}-${index}`}>
              <div>
                <span title={visible(entry?.label, "Item")}>{visible(entry?.label, "Item")}</span>
                {entry?.summary ? <small>{visible(entry.summary)}</small> : null}
              </div>
              {entry?.status ? <code>{statusLabel(entry.status)}</code> : null}
            </li>
          ))}
        </ul>
      ) : null}
      {!loading && !fields.length && !items.length && !part?.summary ? (
        <div className="v2-empty compact">No additional user-facing details were recorded for this activity.</div>
      ) : null}
      {technicalRows.length ? (
        <div className="v2-activity-payloads">
          {part?.input_ref ? <ActivityContent key={part.input_ref} title="Input" source={part.input_ref} /> : null}
          {part?.output_ref ? <ActivityContent key={part.output_ref} title="Output" source={part.output_ref} /> : null}
        </div>
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

function PreviewBody({ ctx, workspaceName, item, onSelect }) {
  const [preview, setPreview] = useState(item?.preview || null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const path = item?.path || "";

  async function load() {
    if (!ctx || !path) return;
    setLoading(true);
    setError("");
    try {
      const payload = await apiFetch(`/api/session/${escapePath(ctx)}/files/content?path=${escapePath(path)}&project_space=${escapePath(workspaceName || "")}`);
      setPreview(payload);
    } catch (reason) {
      setError(reason);
    } finally {
      setLoading(false);
    }
  }

  useEffect(() => {
    setPreview(item?.preview || null);
    setError("");
    if (item?.type === "file" && !item?.preview) load();
  }, [item?.id, ctx, workspaceName]);

  if (item?.type === "artifact") {
    return (
      <ArtifactRenderer
        artifact={item.artifact || { artifact_id: item.artifact_id, path: item.path, title: item.title }}
        workspaceName={workspaceName}
        ctx={ctx}
      />
    );
  }
  if (item?.type === "agent") return <AsyncSubagentActivity item={item} onSelect={onSelect} />;
  if (item?.type === "activity") return <ActivityPreview item={item} />;
  if (loading) return <div className="v2-empty" role="status">Loading file preview…</div>;
  if (error) return <ErrorNotice error={error} />;
  if (!preview) return <div className="v2-empty">This file has no browser preview. Download the original file to inspect it.</div>;

  return (
    <div className="v2-file-tab-preview">
      <div className="v2-file-tab-actions">
        <div>
          <div className="v2-eyebrow">{preview.kind || preview.node_type || "file"}</div>
          <strong>{preview.name || preview.path || "."}</strong>
        </div>
        <div className="v2-icon-row compact">
          <button type="button" className="v2-icon-btn" onClick={load} aria-label="Refresh file preview"><RefreshCw size={15} /></button>
          {preview.download_url ? <a className="v2-icon-btn" href={preview.download_url} aria-label={`Download ${preview.name || "file"}`}><Download size={15} /></a> : null}
        </div>
      </div>
      <ArtifactRenderer filePreview={preview} showHeader={false} workspaceName={workspaceName} ctx={ctx} />
    </div>
  );
}

export default function ArtifactPreviewModal({ ctx, workspaceName, item, open, onClose, onSelect, backItem, onBack }) {
  const dialogRef = useRef(null);
  const closeButtonRef = useRef(null);
  const closeRef = useRef(onClose);
  closeRef.current = onClose;

  useEffect(() => {
    if (!open || !item) return undefined;
    const previousFocus = document.activeElement;
    const previousOverflow = document.body.style.overflow;
    document.body.style.overflow = "hidden";
    document.body.classList.add("v2-preview-modal-open");
    const frame = window.requestAnimationFrame(() => closeButtonRef.current?.focus());

    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        closeRef.current?.();
        return;
      }
      if (event.key !== "Tab" || !dialogRef.current) return;
      const focusable = [...dialogRef.current.querySelectorAll(
        'a[href], button:not([disabled]), input:not([disabled]), select:not([disabled]), textarea:not([disabled]), [tabindex]:not([tabindex="-1"])',
      )].filter((node) => !node.hasAttribute("hidden"));
      if (!focusable.length) {
        event.preventDefault();
        dialogRef.current.focus();
        return;
      }
      const first = focusable[0];
      const last = focusable[focusable.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };

    document.addEventListener("keydown", handleKeyDown);
    return () => {
      window.cancelAnimationFrame(frame);
      document.removeEventListener("keydown", handleKeyDown);
      document.body.style.overflow = previousOverflow;
      document.body.classList.remove("v2-preview-modal-open");
      if (previousFocus instanceof HTMLElement) {
        window.requestAnimationFrame(() => previousFocus.focus());
      }
    };
  }, [open, item?.id]);

  if (!open || !item || typeof document === "undefined") return null;
  const title = previewTitle(item);

  return createPortal(
    <div
      className="v2-preview-modal-backdrop"
      onClick={(event) => {
        if (event.target === event.currentTarget) closeRef.current?.();
      }}
    >
      <section
        ref={dialogRef}
        className={`v2-preview-modal ${item.type === "agent" ? "v2-agent-modal" : ""}`}
        role="dialog"
        aria-modal="true"
        aria-label={`Preview ${title}`}
        tabIndex={-1}
      >
        <header className="v2-preview-modal-chrome">
          {backItem ? <button type="button" className="v2-ghost-btn compact" onClick={onBack}><ArrowLeft size={15} />返回 {backItem.title}</button> : null}
          <div>
            <Eye size={16} aria-hidden="true" />
            <span>Workspace preview</span>
          </div>
          <button
            ref={closeButtonRef}
            type="button"
            className="v2-preview-modal-close"
            aria-label={`Close ${title} preview`}
            onClick={() => closeRef.current?.()}
          >
            <X size={19} />
          </button>
        </header>
        <div className="v2-preview-modal-content" key={item.id}>
          <PreviewBody ctx={ctx} workspaceName={workspaceName} item={item} onSelect={onSelect} />
        </div>
      </section>
    </div>,
    document.body,
  );
}
