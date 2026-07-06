import { useEffect, useMemo, useState } from "react";
import { Download, FileBox, RefreshCw, X } from "lucide-react";

import ArtifactRenderer from "./ArtifactRenderer";
import { apiFetch } from "../useCatMasterThreadRuntime";

function escapePath(value) {
  return encodeURIComponent(String(value || ""));
}

function tabTitle(tab) {
  const label = tab?.title || tab?.artifact?.title || tab?.artifact?.path || tab?.path || "Preview";
  return String(label).split("/").filter(Boolean).pop() || label;
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

export default function FilePreviewTabs({ ctx, workspaceName, tabs, activeTabId, onActivate, onClose }) {
  const activeTab = useMemo(
    () => (tabs || []).find((tab) => tab.id === activeTabId) || (tabs || [])[0] || null,
    [tabs, activeTabId],
  );

  return (
    <aside className="v2-right-inspector v2-file-preview-tabs">
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
    </aside>
  );
}
