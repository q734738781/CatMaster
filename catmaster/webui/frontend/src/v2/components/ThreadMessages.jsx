import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useEffect, useMemo, useRef, useState } from "react";
import { MessagePrimitive, ThreadPrimitive, useMessage } from "@assistant-ui/react";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  CheckCircle2,
  CircleAlert,
  Clipboard,
  FileBox,
  Hammer,
  Link as LinkIcon,
  ListChecks,
  LoaderCircle,
  Network,
  UserRound,
} from "lucide-react";

import { normalizeMathMarkdown } from "../markdown";
import {
  displayValue,
  isInternalStoragePath,
  presentError,
  redactErrorText,
  userFacingFileTitle,
} from "../presentation";
import { apiFetch } from "../useCatMasterThreadRuntime";

function MarkdownBlock({ text }) {
  return (
    <div className="v2-message-text">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {normalizeMathMarkdown(text)}
      </ReactMarkdown>
    </div>
  );
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
  }[status] || status.replace(/[_-]+/g, " ");
}

function FieldList({ fields, redact = false }) {
  const rows = Array.isArray(fields) ? fields : [];
  if (!rows.length) return null;
  const visible = (value, fallback = "Not available") => {
    const text = displayValue(value, fallback);
    return redact ? redactErrorText(text) : text;
  };
  return (
    <dl className="v2-semantic-fields">
      {rows.map((field, index) => (
        <div key={`${field.label || "field"}-${index}`}>
          <dt>{visible(field.label, "Detail")}</dt>
          <dd>
            {field.href && !redact ? (
              <a href={field.href} title={visible(field.value, "Open linked item")}>
                {visible(field.value, "Open linked item")}
              </a>
            ) : (
              <span title={visible(field.value, "") || undefined}>
                {visible(field.value)}
              </span>
            )}
            {field.copy_value ? (
              <button
                type="button"
                className="v2-inline-copy"
                aria-label={`Copy ${field.label || "value"}`}
                onClick={() => navigator.clipboard?.writeText(
                  redact ? visible(field.copy_value, "") : String(field.copy_value || ""),
                )}
              >
                <Clipboard size={13} />
              </button>
            ) : null}
          </dd>
        </div>
      ))}
    </dl>
  );
}

function DiagnosticsReference({ value, entries = [] }) {
  const rows = [
    ...entries,
    ...(value ? [{ label: "Diagnostics reference", value }] : []),
  ].filter((entry) => entry?.value);
  if (!rows.length) return null;
  return (
    <details className="v2-technical-details">
      <summary>Technical details</summary>
      {rows.map((entry) => (
        <div key={entry.label}>
          <span>{entry.label}</span>
          {entry.showValue ? <code>{entry.value}</code> : null}
          <button
            type="button"
            className="v2-diagnostics-ref"
            onClick={() => navigator.clipboard?.writeText(String(entry.value))}
            title={`Copy ${entry.label.toLowerCase()}`}
          >
            <Clipboard size={13} />
            Copy reference
          </button>
        </div>
      ))}
    </details>
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

function TruncationNotice({ truncation, onLoadMore, onOpenFull, loading }) {
  const total = Number(truncation.total_count || 0);
  const shown = Number(truncation.shown_count || 0);
  const sliced = Boolean(truncation?.truncated) || (total > 0 && shown < total);
  if (!sliced) return null;
  const canLoadMore = Boolean(onLoadMore && truncation?.next_cursor);
  return (
    <div className="v2-truncation-notice" role="status">
      <span>{total ? `Showing ${shown.toLocaleString()} of ${total.toLocaleString()} ${truncation.unit || "items"}.` : `Showing ${shown.toLocaleString()} ${truncation.unit || "items"}; more are available.`}</span>
      {canLoadMore ? (
        <button type="button" className="v2-ghost-btn compact" onClick={onLoadMore} disabled={loading}>
          {loading ? <LoaderCircle className="spin" size={14} /> : null}
          Load more
        </button>
      ) : null}
      {!canLoadMore && onOpenFull ? (
        <button type="button" className="v2-ghost-btn compact" onClick={onOpenFull}>
          Open full details
        </button>
      ) : null}
    </div>
  );
}

function progressTitle(part) {
  const title = displayValue(part?.title, "");
  const generic = !title || ["execution update", "progress", "update"].includes(title.toLowerCase());
  if (!generic) return title;
  const status = String(part?.status || "").toLowerCase();
  if (["completed", "complete", "done", "success"].includes(status)) return "Step completed";
  if (["failed", "error"].includes(status)) return "Step needs attention";
  return String(part?.type || "") === "reasoning" ? "Reasoning trace" : "Work in progress";
}

function LongTextPart({ part, progress = false, onSelect }) {
  const [text, setText] = useState(String(part?.text || ""));
  const [page, setPage] = useState(part?.truncation || {});
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");

  useEffect(() => {
    setText(String(part?.text || ""));
    setPage(part?.truncation || {});
  }, [
    part?.id,
    part?.text,
    part?.truncation?.shown_count,
    part?.truncation?.total_count,
    part?.truncation?.next_cursor,
  ]);

  async function loadMore() {
    const ref = String(page?.full_content_ref || part?.truncation?.full_content_ref || "");
    if (!ref) return;
    setLoading(true);
    setError("");
    try {
      const join = ref.includes("?") ? "&" : "?";
      const cursor = String(page?.next_cursor || "");
      if (!cursor) return;
      const payload = await apiFetch(`${ref}${join}cursor=${encodeURIComponent(cursor)}`);
      const nextText = String(payload.text || "");
      setText((current) => `${current}${nextText}`);
      setPage((current) => {
        const next = payload.page || {};
        const loaded = Number(current?.shown_count || 0) + Number(next?.shown_count || nextText.length || 0);
        const total = Number(next?.total_count || current?.total_count || 0);
        return {
          ...next,
          shown_count: total ? Math.min(total, loaded) : loaded,
          total_count: total,
        };
      });
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }

  if (progress) {
    return (
      <section className="v2-progress-card">
        <div className="v2-progress-head">
          <span>{progressTitle(part)}</span>
          <small>{statusLabel(part.status)}</small>
        </div>
        {text ? (
          <MarkdownBlock text={text} />
        ) : (
          <div className="v2-muted">
            {["completed", "complete", "done", "success"].includes(String(part?.status || "").toLowerCase())
              ? "This step completed without a written trace."
              : "No written trace has been recorded yet."}
          </div>
        )}
        <TruncationNotice truncation={page} onLoadMore={loadMore} loading={loading} />
        {!text && part?.detail_ref ? (
          <button
            type="button"
            className="v2-ghost-btn compact"
            onClick={() => onSelect?.({ type: "activity", part })}
          >
            Open details
          </button>
        ) : null}
        <ErrorNotice error={error} compact />
      </section>
    );
  }
  return (
    <div>
      {text ? <MarkdownBlock text={text} /> : null}
      <TruncationNotice truncation={page} onLoadMore={loadMore} loading={loading} />
      <ErrorNotice error={error} compact />
    </div>
  );
}

function ItemList({ items, ordered = false, redact = false }) {
  const rows = Array.isArray(items) ? items : [];
  if (!rows.length) return null;
  const Root = ordered ? "ol" : "ul";
  const visible = (value, fallback = "Not available") => {
    const text = displayValue(value, fallback);
    return redact ? redactErrorText(text) : text;
  };
  return (
    <Root className="v2-semantic-items">
      {rows.map((item, index) => (
        <li key={`${item.label || "item"}-${index}`}>
          <div>
            {item.href && !redact ? (
              <a href={item.href} target="_blank" rel="noreferrer" title={visible(item.label, "Open item")}>
                {visible(item.label, "Open item")}
              </a>
            ) : (
              <span title={visible(item.label, "") || undefined}>{visible(item.label, "Item")}</span>
            )}
            {item.summary ? <small>{visible(item.summary)}</small> : null}
          </div>
          {item.status ? <code>{statusLabel(item.status)}</code> : null}
        </li>
      ))}
    </Root>
  );
}

function CardActions({ actions, part, onSelect }) {
  const rows = Array.isArray(actions) ? actions : [];
  const detailTypes = new Set(["tool", "receipt", "attachment"]);
  const isArtifactWithoutOpen = String(part?.type || "") === "artifact"
    && !rows.some((action) => action?.id === "open_artifact");
  const showDetail = detailTypes.has(String(part?.type || "")) || isArtifactWithoutOpen;
  if (!rows.length && !showDetail) return null;
  return (
    <div className="v2-card-actions">
      {rows.map((action, index) => {
        if (action.id === "open_artifact") {
          return (
            <button
              key={`${action.id}-${index}`}
              type="button"
              className="v2-ghost-btn compact"
              onClick={() => onSelect?.({
                type: "artifact",
                artifact_id: part.artifact_id,
                path: part.path,
                artifact: part,
              })}
            >
              <FileBox size={14} />
              {action.label || "Open"}
            </button>
          );
        }
        const hidesErrorReference = String(part?.type || "") === "error"
          || ["failed", "error"].includes(String(part?.status || "").toLowerCase());
        if (action.href && !hidesErrorReference) {
          return (
            <button
              key={`${action.id}-${index}`}
              type="button"
              className="v2-ghost-btn compact"
              onClick={() => onSelect?.({ type: "file", path: action.href })}
            >
              <LinkIcon size={14} />
              {displayValue(action.label, "Open")}
            </button>
          );
        }
        if (action.id === "focus_composer") {
          return (
            <button
              key={`${action.id}-${index}`}
              type="button"
              className="v2-primary-btn compact"
              onClick={() => {
                const composer = document.querySelector(".v2-composer textarea");
                composer?.focus();
                composer?.scrollIntoView?.({ block: "nearest" });
              }}
            >
              {displayValue(action.label, "Review and try again")}
            </button>
          );
        }
        return null;
      })}
      {showDetail ? (
        <button
          type="button"
          className="v2-ghost-btn compact"
          onClick={() => {
            if (String(part?.type || "") === "artifact") {
              onSelect?.({
                type: "artifact",
                artifact_id: part.artifact_id,
                path: part.path,
                artifact: part,
              });
              return;
            }
            onSelect?.({ type: "activity", part });
          }}
        >
          Open details
        </button>
      ) : null}
    </div>
  );
}

function SemanticCard({ part, icon: Icon = Hammer, className = "", onSelect }) {
  const isError = String(part.type || "") === "error"
    || ["failed", "error"].includes(String(part.status || "").toLowerCase());
  const internalPath = isInternalStoragePath(part.path);
  const publicTitle = String(part.type || "") === "artifact"
    ? userFacingFileTitle(part.title, part.path, "Artifact")
    : displayValue(part.title, "Activity");
  const longFileContent = String(part.type || "") === "tool"
    && /\bread file\b/i.test(publicTitle)
    && displayValue(part.summary, "").length > 240;
  const visible = (value, fallback) => {
    const text = displayValue(value, fallback);
    return isError ? redactErrorText(text) : text;
  };
  return (
    <section
      className={`v2-semantic-card ${className} status-${String(part.status || "updated").toLowerCase()}`}
      role={isError ? "alert" : "region"}
      aria-label={publicTitle}
    >
      <div className="v2-semantic-card-head">
        <Icon size={16} />
        <div>
          <strong>{isError ? redactErrorText(publicTitle) : publicTitle}</strong>
          {part.summary ? (
            <p>{longFileContent ? "File contents are available in details." : visible(part.summary, "")}</p>
          ) : null}
        </div>
        <small>{statusLabel(part.status)}</small>
      </div>
      <FieldList
        fields={(Array.isArray(part.fields) ? part.fields : []).filter((field) => !isInternalStoragePath(field?.value))}
        redact={isError}
      />
      <ItemList items={part.items} redact={isError} />
      <CardActions actions={part.actions} part={part} onSelect={onSelect} />
      <TruncationNotice
        truncation={part.truncation || {}}
        onOpenFull={part.detail_ref ? () => onSelect?.({ type: "activity", part }) : null}
      />
      <DiagnosticsReference
        value={part.diagnostics_ref}
        entries={internalPath ? [{ label: "Managed storage reference", value: part.path, showValue: true }] : []}
      />
    </section>
  );
}

function ReviewField({ field, value, onChange }) {
  if (field.input_type === "boolean") {
    return (
      <label className="v2-review-field checkbox">
        <input
          type="checkbox"
          checked={String(value) === "true" || value === true}
          onChange={(event) => onChange(event.target.checked)}
        />
        <span>{field.label}</span>
      </label>
    );
  }
  const Input = field.input_type === "textarea" ? "textarea" : "input";
  return (
    <label className="v2-review-field">
      <span>{field.label}{field.required ? " *" : ""}</span>
      <Input
        type={field.input_type === "number" ? "number" : undefined}
        rows={field.input_type === "textarea" ? 4 : undefined}
        value={value ?? ""}
        required={field.required}
        onChange={(event) => onChange(event.target.value)}
      />
    </label>
  );
}

function InterruptCard({ part, onResume }) {
  const actions = Array.isArray(part.actions) ? part.actions : [];
  const actionIds = useMemo(
    () => [...new Set(actions.map((action) => String(action.id || "")).filter(Boolean))],
    [actions],
  );
  const [selected, setSelected] = useState({});
  const [values, setValues] = useState({});
  const [error, setError] = useState("");
  const resolved = part.status === "resolved";
  const cardRef = useRef(null);
  const validationErrorRef = useRef(null);
  const wasPendingRef = useRef(!resolved);

  useEffect(() => {
    if (!resolved) {
      wasPendingRef.current = true;
      if (!document.activeElement || document.activeElement === document.body) {
        cardRef.current?.focus();
      }
      return;
    }
    if (wasPendingRef.current) {
      wasPendingRef.current = false;
      window.requestAnimationFrame(() => {
        document.querySelector(".v2-composer textarea")?.focus();
      });
    }
  }, [resolved]);

  useEffect(() => {
    if (error) validationErrorRef.current?.focus();
  }, [error]);

  function choose(action) {
    const id = String(action.id || "");
    const defaults = {};
    (action.fields || []).forEach((field) => {
      defaults[field.name] = field.value ?? "";
    });
    setSelected((current) => ({ ...current, [id]: action.decision }));
    setValues((current) => ({
      ...current,
      [id]: {
        ...(current[id] || {}),
        ...defaults,
      },
    }));
    setError("");
  }

  async function submit() {
    if (actionIds.some((id) => !selected[id])) {
      setError("Choose one decision for every pending action.");
      return;
    }
    const reviews = [];
    for (const id of actionIds) {
      const action = actions.find((item) => String(item.id) === id && item.decision === selected[id]);
      const fields = { ...(values[id] || {}) };
      const missing = (action?.fields || []).find((field) => field.required && !String(fields[field.name] ?? "").trim());
      if (missing) {
        setError(`${missing.label} is required.`);
        return;
      }
      if (action?.confirmation && !window.confirm(action.confirmation)) return;
      reviews.push({
        action_id: id,
        decision: selected[id],
        fields,
        reason: String(fields.reason || ""),
      });
    }
    setError("");
    await onResume?.(reviews);
  }

  return (
    <section
      ref={cardRef}
      className={`v2-interrupt-card status-${part.status || "pending"}`}
      role="region"
      aria-label={part.title || "Review required"}
      tabIndex={resolved ? undefined : -1}
    >
      <div className="v2-semantic-card-head">
        <CircleAlert size={17} />
        <div>
          <strong>{part.title || "Your decision is required"}</strong>
          <p>{part.summary || "The task is paused until you decide."}</p>
        </div>
        <small>{statusLabel(part.status)}</small>
      </div>
      <ItemList items={part.items} ordered />
      {!resolved ? actionIds.map((id, index) => {
        const variants = actions.filter((action) => String(action.id) === id);
        const active = variants.find((action) => action.decision === selected[id]);
        return (
          <fieldset key={id} className="v2-review-action">
            <legend>{part.items?.[index]?.label || `Action ${index + 1}`}</legend>
            <div className="v2-interrupt-row">
              {variants.map((action) => (
                <button
                  key={`${id}-${action.decision}`}
                  type="button"
                  className={`v2-review-choice kind-${action.kind || "secondary"} ${selected[id] === action.decision ? "active" : ""}`}
                  aria-pressed={selected[id] === action.decision}
                  onClick={() => choose(action)}
                >
                  {action.label}
                </button>
              ))}
            </div>
            {active?.fields?.length ? (
              <div className="v2-review-fields">
                {active.fields.map((field) => (
                  <ReviewField
                    key={field.name}
                    field={field}
                    value={values[id]?.[field.name] ?? field.value ?? ""}
                    onChange={(value) => setValues((current) => ({
                      ...current,
                      [id]: { ...(current[id] || {}), [field.name]: value },
                    }))}
                  />
                ))}
              </div>
            ) : null}
          </fieldset>
        );
      }) : null}
      {error ? <div ref={validationErrorRef} className="v2-error compact" role="alert" tabIndex={-1}>{error}</div> : null}
      {!resolved && actionIds.length ? (
        <button type="button" className="v2-primary-btn" onClick={submit}>Submit decisions</button>
      ) : null}
      <DiagnosticsReference value={part.diagnostics_ref} />
    </section>
  );
}

function RenderProjectedPart({ part, onSelect, onResume }) {
  const type = String(part?.type || "unknown");
  if (type === "text") return <LongTextPart part={part} />;
  if (type === "reasoning") return <LongTextPart part={part} progress onSelect={onSelect} />;
  if (type === "progress") {
    if (part.items?.length) return <SemanticCard part={part} icon={ListChecks} className="progress" onSelect={onSelect} />;
    return <LongTextPart part={part} progress onSelect={onSelect} />;
  }
  if (type === "artifact" || type === "attachment") return <SemanticCard part={part} icon={FileBox} className="artifact" onSelect={onSelect} />;
  if (type === "tool") return <SemanticCard part={part} icon={Hammer} className="tool" onSelect={onSelect} />;
  if (type === "receipt") return <SemanticCard part={part} icon={Network} className="receipt" onSelect={onSelect} />;
  if (type === "interrupt") return <InterruptCard part={part} onResume={onResume} />;
  if (type === "error") return <SemanticCard part={part} icon={CircleAlert} className="error" onSelect={onSelect} />;
  if (type === "citations") return <SemanticCard part={part} icon={LinkIcon} className="citations" onSelect={onSelect} />;
  return <SemanticCard part={part} icon={CircleAlert} className="unknown" onSelect={onSelect} />;
}

function partFromAssistantContent(part) {
  if (part?.type === "data") return part.data || {};
  return {
    id: "",
    type: "unknown",
    status: "unsupported",
    title: "This activity cannot be displayed yet",
    summary: "The record remains available to developer diagnostics.",
    fields: [],
    actions: [],
    items: [],
  };
}

function CatMasterMessage({ onSelect, onResume }) {
  const message = useMessage();
  const role = String(message?.role || "assistant");
  const status = message?.status?.type || message?.status || "";
  const projectedMessage = message?.metadata?.custom?.catmaster || {};
  const initialParts = (Array.isArray(message?.content) ? message.content : []).map(partFromAssistantContent);
  const [additionalParts, setAdditionalParts] = useState([]);
  const [partsPage, setPartsPage] = useState(projectedMessage.parts_page || {});
  const [loadingParts, setLoadingParts] = useState(false);
  const [partsError, setPartsError] = useState("");
  const parts = [...initialParts, ...additionalParts];

  useEffect(() => {
    setAdditionalParts([]);
    setPartsPage(projectedMessage.parts_page || {});
    setPartsError("");
  }, [message?.id]);

  async function loadMoreParts() {
    const ref = String(partsPage?.full_content_ref || "");
    const cursor = String(partsPage?.next_cursor || "");
    if (!ref || !cursor || loadingParts) return;
    setLoadingParts(true);
    setPartsError("");
    try {
      const join = ref.includes("?") ? "&" : "?";
      const payload = await apiFetch(`${ref}${join}cursor=${encodeURIComponent(cursor)}`);
      const rows = Array.isArray(payload.parts) ? payload.parts : [];
      setAdditionalParts((current) => {
        const seen = new Set([...initialParts, ...current].map((part) => part.id));
        return [...current, ...rows.filter((part) => !seen.has(part.id))];
      });
      setPartsPage(payload.page || {});
    } catch (err) {
      setPartsError(err.message || String(err));
    } finally {
      setLoadingParts(false);
    }
  }

  if (!parts.length && role === "assistant") return null;
  return (
    <MessagePrimitive.Root asChild>
      <article className={`v2-message role-${role} status-${status}`}>
        <div className="v2-message-avatar">{role === "user" ? <UserRound size={17} /> : <Bot size={17} />}</div>
        <div className="v2-message-body">
          <div className="v2-message-meta">
            <span>{role === "user" ? "You" : "CatMaster"}</span>
            <small>{statusLabel(status)}</small>
          </div>
          <div className="v2-message-parts">
            {parts.map((part, index) => (
              <RenderProjectedPart
                key={part.id || `${part.type || "part"}-${index}`}
                part={part}
                onSelect={onSelect}
                onResume={onResume}
              />
            ))}
            {partsPage?.truncated ? (
              <TruncationNotice
                truncation={partsPage}
                onLoadMore={loadMoreParts}
                loading={loadingParts}
              />
            ) : null}
            <ErrorNotice error={partsError} compact />
          </div>
        </div>
      </article>
    </MessagePrimitive.Root>
  );
}

export default function ThreadMessages({
  threadId = "",
  messages,
  loading,
  error,
  onSelect,
  onResume,
  hasMore = false,
  onLoadOlder,
  loadingOlder = false,
}) {
  const viewportRef = useRef(null);
  const [preservingHistoryAnchor, setPreservingHistoryAnchor] = useState(false);
  const preservingHistoryRef = useRef(false);
  const followBottomRef = useRef(true);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport) return undefined;
    const updateFollowState = () => {
      if (preservingHistoryRef.current) return;
      followBottomRef.current = (
        Math.abs(viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight) <= 2
        || viewport.scrollHeight <= viewport.clientHeight
      );
    };
    const content = viewport.querySelector(".v2-thread-messages");
    const observer = new ResizeObserver(() => {
      if (!preservingHistoryRef.current && followBottomRef.current) {
        viewport.scrollTop = viewport.scrollHeight;
      }
    });
    viewport.addEventListener("scroll", updateFollowState, { passive: true });
    if (content) observer.observe(content);
    updateFollowState();
    return () => {
      viewport.removeEventListener("scroll", updateFollowState);
      observer.disconnect();
    };
  }, []);

  useEffect(() => {
    const viewport = viewportRef.current;
    if (!viewport || !threadId) return;
    followBottomRef.current = true;
    window.requestAnimationFrame(() => {
      viewport.scrollTop = viewport.scrollHeight;
    });
  }, [threadId]);

  async function loadOlder() {
    const viewport = viewportRef.current;
    if (!viewport) {
      await onLoadOlder?.();
      return;
    }
    const viewportRect = viewport.getBoundingClientRect();
    const anchor = [...viewport.querySelectorAll("[data-message-id]")].find((element) => {
      const rect = element.getBoundingClientRect();
      return rect.bottom > viewportRect.top + 1;
    });
    const anchorId = anchor?.getAttribute("data-message-id") || "";
    const anchorOffset = anchor ? anchor.getBoundingClientRect().top - viewportRect.top : 0;
    const previousHeight = viewport.scrollHeight;
    const previousTop = viewport.scrollTop;
    const previousMessageCount = viewport.querySelectorAll("[data-message-id]").length;
    preservingHistoryRef.current = true;
    setPreservingHistoryAnchor(true);
    try {
      await onLoadOlder?.();
      let attempt = 0;
      let stableFrames = 0;
      let observedPrepend = false;
      let lastHeight = previousHeight;
      const restoreAnchor = () => {
        const messageCount = viewport.querySelectorAll("[data-message-id]").length;
        const currentHeight = viewport.scrollHeight;
        if (messageCount > previousMessageCount || currentHeight !== previousHeight) {
          observedPrepend = true;
        }
        const nextAnchor = anchorId
          ? viewport.querySelector(`[data-message-id="${CSS.escape(anchorId)}"]`)
          : null;
        let offsetError = Number.POSITIVE_INFINITY;
        if (nextAnchor) {
          const nextOffset = nextAnchor.getBoundingClientRect().top - viewport.getBoundingClientRect().top;
          offsetError = nextOffset - anchorOffset;
          if (Math.abs(offsetError) > 0.25) viewport.scrollTop += offsetError;
        } else if (observedPrepend && attempt === 0) {
          viewport.scrollTop = previousTop + Math.max(0, viewport.scrollHeight - previousHeight);
        }
        const heightStable = Math.abs(currentHeight - lastHeight) <= 0.5;
        const anchorStable = nextAnchor && Math.abs(offsetError) <= 0.5;
        stableFrames = observedPrepend && heightStable && anchorStable ? stableFrames + 1 : 0;
        lastHeight = currentHeight;
        attempt += 1;
        // React and assistant-ui may commit the prepended window after the
        // request promise resolves. Wait for the DOM change, then require a
        // short quiet period so late Markdown/font layout cannot move the
        // reader's visual anchor.
        if (attempt < 180 && (!observedPrepend || stableFrames < 12)) {
          window.requestAnimationFrame(restoreAnchor);
          return;
        }
        viewport.dispatchEvent(new Event("scroll"));
        preservingHistoryRef.current = false;
        followBottomRef.current = (
          Math.abs(viewport.scrollHeight - viewport.scrollTop - viewport.clientHeight) <= 2
        );
        setPreservingHistoryAnchor(false);
      };
      window.requestAnimationFrame(restoreAnchor);
    } catch (loadError) {
      preservingHistoryRef.current = false;
      setPreservingHistoryAnchor(false);
      throw loadError;
    }
  }
  if (loading) return <div className="v2-empty" role="status">Loading this conversation…</div>;
  if (error) return <ErrorNotice error={error} />;
  if (!messages.length) {
    return (
      <div className="v2-empty">
        <strong>This conversation is ready.</strong>
        <span>Describe the research question or task in the composer below.</span>
      </div>
    );
  }
  return (
    <ThreadPrimitive.Root className="v2-thread-root">
      <ThreadPrimitive.Viewport
        ref={viewportRef}
        className="v2-thread-viewport"
        autoScroll={false}
        scrollToBottomOnInitialize={false}
        scrollToBottomOnRunStart={false}
        scrollToBottomOnThreadSwitch={false}
        data-preserving-history-anchor={preservingHistoryAnchor ? "true" : undefined}
      >
        <div className="v2-thread-messages">
          {hasMore ? (
            <button type="button" className="v2-load-history" onClick={loadOlder} disabled={loadingOlder}>
              {loadingOlder ? <LoaderCircle className="spin" size={15} /> : <CheckCircle2 size={15} />}
              Load earlier messages
            </button>
          ) : null}
          <ThreadPrimitive.Messages>
            {() => <CatMasterMessage onSelect={onSelect} onResume={onResume} />}
          </ThreadPrimitive.Messages>
        </div>
        <ThreadPrimitive.ScrollToBottom className="v2-new-messages">
          New messages
        </ThreadPrimitive.ScrollToBottom>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
