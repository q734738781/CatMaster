import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { createContext, useContext, useEffect, useMemo, useRef, useState } from "react";
import { MessagePrimitive, ThreadPrimitive, useMessage } from "@assistant-ui/react";
import ReactMarkdown from "react-markdown";
import {
  Bot,
  CheckCircle2,
  CircleAlert,
  Clipboard,
  ChevronDown,
  FileBox,
  Hammer,
  Link as LinkIcon,
  ListChecks,
  LoaderCircle,
  Maximize2,
  Network,
  UserRound,
} from "lucide-react";

import { normalizeMathMarkdown } from "../markdown";
import ConversationWelcome from "./ConversationWelcome";
import AsyncSubagentCard from "./AsyncSubagentCard";
import {
  workspaceMarkdownUrlTransform,
  workspacePathFromSandboxHref,
  workspacePathFromInlineImageUrl,
} from "../workspaceLinks";
import {
  hasVisibleTurnPresentation,
  isLongActivityGroup,
  latestTodoParts,
  organizeTurnParts,
  todoCardProjection,
  withCanonicalTodoParts,
} from "../activityPresentation";
import { entrypointMeta } from "../entrypoints";
import {
  displayValue,
  isInternalStoragePath,
  presentError,
  redactErrorText,
  userFacingFileTitle,
} from "../presentation";
import { apiFetch } from "../useCatMasterThreadRuntime";

const MarkdownWorkspaceContext = createContext({ threadId: "", onSelect: null });

function MarkdownImage({ src = "", alt = "", title = "", threadId = "", onSelect, node: _node, ...props }) {
  const [failed, setFailed] = useState(false);
  const workspacePath = workspacePathFromInlineImageUrl(src, threadId);
  const sourceName = workspacePath.split("/").at(-1) || "";
  const caption = String(title || alt || sourceName).trim();
  const openSource = () => onSelect?.({ type: "file", path: workspacePath });

  useEffect(() => setFailed(false), [src]);

  if (failed) {
    return (
      <span className="v2-markdown-image-fallback" role="status">
        <span className="v2-markdown-image-fallback-icon">
          <CircleAlert size={17} aria-hidden="true" />
        </span>
        <span className="v2-markdown-image-fallback-copy">
          <strong>Image unavailable</strong>
          <span>{caption || "The referenced image could not be displayed."}</span>
        </span>
        {workspacePath ? (
          <button type="button" onClick={openSource}>
            Open file
          </button>
        ) : null}
      </span>
    );
  }

  const image = (
    <img
      {...props}
      src={src}
      alt={alt}
      title={title || undefined}
      loading="lazy"
      decoding="async"
      onError={() => setFailed(true)}
    />
  );

  return (
    <span className="v2-markdown-figure" role="group" aria-label={caption || "Inline image"}>
      {workspacePath ? (
        <button
          type="button"
          className="v2-markdown-image-stage v2-markdown-image-button"
          aria-label={`Open ${caption || "image"} in a large preview`}
          onClick={openSource}
        >
          {image}
        </button>
      ) : (
        <span className="v2-markdown-image-stage">{image}</span>
      )}
      {caption || workspacePath ? (
        <span className="v2-markdown-image-footer">
          <span className="v2-markdown-image-caption">{caption || sourceName}</span>
          {workspacePath ? (
            <button type="button" className="v2-markdown-image-open" onClick={openSource} title={workspacePath}>
              <Maximize2 size={13} aria-hidden="true" />
              View larger
            </button>
          ) : null}
        </span>
      ) : null}
    </span>
  );
}

function MarkdownBlock({ text, onSelect }) {
  const workspaceContext = useContext(MarkdownWorkspaceContext);
  const threadId = String(workspaceContext.threadId || "");
  const select = onSelect || workspaceContext.onSelect;
  return (
    <div className="v2-message-text">
      <ReactMarkdown
        remarkPlugins={[remarkGfm, remarkMath]}
        rehypePlugins={[rehypeKatex]}
        urlTransform={(url, key, node) => workspaceMarkdownUrlTransform(url, key, node, threadId)}
        components={{
          a({ href, children, node: _node, ...props }) {
            const path = workspacePathFromSandboxHref(href);
            if (!path) return <a href={href} {...props}>{children}</a>;
            return (
              <a
                href={`#inspect=file&path=${encodeURIComponent(path)}`}
                {...props}
                onClick={(event) => {
                  event.preventDefault();
                  select?.({ type: "file", path });
                }}
              >
                {children}
              </a>
            );
          },
          img(props) {
            return <MarkdownImage {...props} threadId={threadId} onSelect={select} />;
          },
        }}
      >
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
    interrupted: "Interrupted",
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
  if (String(part?.type || "") === "reasoning") return "Reasoning trace";
  const status = String(part?.status || "").toLowerCase();
  if (["completed", "complete", "done", "success"].includes(status)) return "Step completed";
  if (["failed", "error"].includes(status)) return "Step needs attention";
  return "Work in progress";
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
          <MarkdownBlock text={text} onSelect={onSelect} />
        ) : (
          <div className="v2-muted">
            {["completed", "complete", "done", "success"].includes(String(part?.status || "").toLowerCase())
              ? "This step completed without a written trace."
              : "No written trace has been recorded yet."}
          </div>
        )}
        <TruncationNotice truncation={page} onLoadMore={loadMore} loading={loading} />
        {part?.detail_ref ? (
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
      {text ? <MarkdownBlock text={text} onSelect={onSelect} /> : null}
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

function CardActions({ actions, part, onSelect, onContinueFromCheckpoint }) {
  const rows = Array.isArray(actions) ? actions : [];
  const [pendingAction, setPendingAction] = useState("");
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
        if (action.id === "continue_from_checkpoint") {
          if (!onContinueFromCheckpoint) return null;
          const pending = pendingAction === action.id;
          return (
            <button
              key={`${action.id}-${index}`}
              type="button"
              className="v2-primary-btn compact"
              disabled={pending}
              onClick={async () => {
                setPendingAction(action.id);
                try {
                  await onContinueFromCheckpoint();
                } catch {
                  // The thread runtime presents the structured API error.
                } finally {
                  setPendingAction("");
                }
              }}
            >
              {pending ? <LoaderCircle className="spin" size={14} /> : null}
              {displayValue(action.label, "Continue from checkpoint")}
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

function SemanticCard({
  part,
  icon: Icon = Hammer,
  className = "",
  onSelect,
  onContinueFromCheckpoint,
}) {
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
      <CardActions
        actions={part.actions}
        part={part}
        onSelect={onSelect}
        onContinueFromCheckpoint={onContinueFromCheckpoint}
      />
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

function RenderProjectedPart({
  part,
  onSelect,
  onResume,
  onContinueFromCheckpoint,
}) {
  const type = String(part?.type || "unknown");
  if (type === "subagent") return <AsyncSubagentCard part={part} onSelect={onSelect} />;
  if (type === "text") return <LongTextPart part={part} onSelect={onSelect} />;
  if (type === "reasoning") return <LongTextPart part={part} progress onSelect={onSelect} />;
  if (type === "progress") {
    if (part.items?.length) return <SemanticCard part={part} icon={ListChecks} className="progress" onSelect={onSelect} />;
    return <LongTextPart part={part} progress onSelect={onSelect} />;
  }
  if (type === "artifact" || type === "attachment") return <SemanticCard part={part} icon={FileBox} className="artifact" onSelect={onSelect} />;
  if (type === "tool") return <SemanticCard part={part} icon={Hammer} className="tool" onSelect={onSelect} />;
  if (type === "receipt") return <SemanticCard part={part} icon={Network} className="receipt" onSelect={onSelect} />;
  if (type === "interrupt") return <InterruptCard part={part} onResume={onResume} />;
  if (type === "error") return (
    <SemanticCard
      part={part}
      icon={CircleAlert}
      className="error"
      onSelect={onSelect}
      onContinueFromCheckpoint={onContinueFromCheckpoint}
    />
  );
  if (type === "citations") return <SemanticCard part={part} icon={LinkIcon} className="citations" onSelect={onSelect} />;
  return <SemanticCard part={part} icon={CircleAlert} className="unknown" onSelect={onSelect} />;
}

function TurnPlanOverview({ parts, onSelect }) {
  const rows = Array.isArray(parts) ? parts : [];
  if (!rows.length) return null;
  const items = rows.flatMap((part) => (Array.isArray(part.items) ? part.items : []));
  const complete = items.filter((item) => ["done", "completed", "complete"].includes(String(item.status || "").toLowerCase())).length;
  return (
    <details className="v2-turn-plan" aria-label="Current plan">
      <summary className="v2-turn-plan-head">
        <ListChecks size={17} />
        <div>
          <strong>Plan</strong>
          <small>{items.find((item) => item.status === "in_progress")?.label || "View steps"}</small>
        </div>
        <code>{complete}/{items.length} done</code>
      </summary>
      <div className="v2-turn-plan-groups">
        {rows.map((part) => (
          <section key={part.id} className="v2-turn-plan-group">
            <div className="v2-turn-plan-source">
              <strong>{displayValue(part.plan_source, "CatMaster")}</strong>
            </div>
            <ItemList items={part.items} ordered />
            <TruncationNotice
              truncation={part.truncation || {}}
              onOpenFull={part.detail_ref ? () => onSelect?.({ type: "activity", part }) : null}
            />
          </section>
        ))}
      </div>
    </details>
  );
}

function TurnProgressOverview({ parts }) {
  const rows = Array.isArray(parts) ? parts : [];
  if (!rows.length) return null;
  const latest = rows.at(-1);
  const earlier = rows.slice(0, -1).reverse();
  const latestText = String(latest?.text || latest?.summary || "").trim();
  return (
    <section className="v2-progress-card v2-research-updates" aria-label="Latest research update">
      <div className="v2-progress-head">
        <span>{latest.title || "Research update"}</span>
        <small>{rows.length === 1 ? "Latest phase" : `${rows.length} phase updates`}</small>
      </div>
      {latestText ? <MarkdownBlock text={latestText} /> : <div className="v2-muted">A research phase update was recorded.</div>}
      {earlier.length ? (
        <details className="v2-progress-history">
          <summary>{earlier.length} earlier {earlier.length === 1 ? "update" : "updates"}</summary>
          <div className="v2-progress-history-list">
            {earlier.map((part) => (
              <div key={part.id} className="v2-progress-history-item">
                <MarkdownBlock text={String(part?.text || part?.summary || "A research phase update was recorded.")} />
              </div>
            ))}
          </div>
        </details>
      ) : null}
    </section>
  );
}

function activityPartLabel(part, groupTitle) {
  const type = String(part?.type || "");
  const semanticProgress = type === "progress" && !part?.items?.length
    ? displayValue(part?.summary, "")
    : "";
  const label = semanticProgress || (["reasoning", "progress"].includes(type)
    ? progressTitle(part)
    : displayValue(part?.title, "Activity"));
  const prefix = `${String(groupTitle || "").trim()} · `;
  if (prefix.trim() && label.toLowerCase().startsWith(prefix.toLowerCase())) {
    return label.slice(prefix.length) || "Activity";
  }
  if (label.toLowerCase() === String(groupTitle || "").trim().toLowerCase()) {
    return String(part?.type || "") === "progress" ? "Progress" : label;
  }
  return label;
}

function ActivityGroup({ group, onSelect, onResume, onContinueFromCheckpoint }) {
  const isLong = isLongActivityGroup(group);
  const [expanded, setExpanded] = useState(!isLong);
  const userChanged = useRef(false);
  const previousLong = useRef(isLong);

  useEffect(() => {
    if (previousLong.current !== isLong && !userChanged.current) {
      setExpanded(!isLong);
    }
    previousLong.current = isLong;
  }, [isLong]);

  const failed = ["failed", "interrupted"].includes(group.status);
  const running = group.status === "running";
  const StatusIcon = failed ? CircleAlert : running ? LoaderCircle : CheckCircle2;
  const latestLabel = activityPartLabel(group.activePart, group.title);
  return (
    <section
      className={`v2-activity-group status-${group.status}`}
      data-activity-group={group.id}
      aria-label={`${group.title} activity`}
    >
      <button
        type="button"
        className="v2-activity-group-toggle"
        aria-expanded={expanded}
        onClick={() => {
          userChanged.current = true;
          setExpanded((value) => !value);
        }}
      >
        <ChevronDown className={expanded ? "expanded" : ""} size={16} aria-hidden="true" />
        <StatusIcon className={running ? "spin" : ""} size={15} aria-hidden="true" />
        <strong>{group.title}</strong>
        <span title={latestLabel}>{latestLabel}</span>
        <small title={`${group.parts.length} activities · ${statusLabel(group.status)}`}>
          {group.parts.length}{" · "}{statusLabel(group.status)}
        </small>
      </button>
      {expanded ? (
        <div className="v2-activity-list">
          {group.parts.map((part, index) => (
            <RenderProjectedPart
              key={part.id || `${part.type || "part"}-${index}`}
              part={part}
              onSelect={onSelect}
              onResume={onResume}
              onContinueFromCheckpoint={onContinueFromCheckpoint}
            />
          ))}
        </div>
      ) : null}
    </section>
  );
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

function CatMasterMessage({
  activityView = false,
  currentTodoMessageId,
  todoOwnerMessageIds,
  todoPartsByMessageId,
  latestMessageId,
  onSelect,
  onResume,
  onContinueFromCheckpoint,
}) {
  const message = useMessage();
  const role = String(message?.role || "assistant");
  const status = message?.status?.type || message?.status || "";
  const projectedMessage = message?.metadata?.custom?.catmaster || {};
  const turnEntrypoint = String(projectedMessage?.meta?.entrypoint || "").trim();
  const turnEntrypointLabel = turnEntrypoint
    ? entrypointMeta(turnEntrypoint).label
    : "";
  const initialParts = (Array.isArray(message?.content) ? message.content : []).map(partFromAssistantContent);
  const [additionalParts, setAdditionalParts] = useState([]);
  const [partsPage, setPartsPage] = useState(projectedMessage.parts_page || {});
  const [loadingParts, setLoadingParts] = useState(false);
  const [partsError, setPartsError] = useState("");
  const parts = [...initialParts, ...additionalParts];
  const ownsTodoCard = todoOwnerMessageIds.has(message?.id);
  const projectedTodoParts = todoPartsByMessageId.get(message?.id) || [];
  const visibleParts = activityView || role !== "assistant"
    ? parts
    : !ownsTodoCard
      ? withCanonicalTodoParts(parts, [])
      : message?.id === currentTodoMessageId
        ? withCanonicalTodoParts(parts, projectedTodoParts)
        : withCanonicalTodoParts(
          parts,
          latestTodoParts([...projectedTodoParts, ...parts]),
        );
  const presentation = activityView ? { planParts: [], semanticProgressParts: [], activityGroups: [], contentParts: visibleParts } : organizeTurnParts(
    visibleParts,
  );
  const continueLatestFailure = (
    message?.id === latestMessageId && onContinueFromCheckpoint
  )
    ? () => onContinueFromCheckpoint?.(message.id)
    : null;

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

  if (
    role === "assistant"
    && !hasVisibleTurnPresentation(presentation)
    && !partsPage?.truncated
    && !partsError
  ) return null;
  if (projectedMessage.origin === "runtime") {
    return (
      <MessagePrimitive.Root asChild>
        <details className="v2-runtime-notice">
          <summary>{projectedMessage.notification_title || "Background specialist finished"}<span>View notification</span></summary>
          <div className="v2-message-text">{parts.map((part) => part.text || "").join("\n")}</div>
        </details>
      </MessagePrimitive.Root>
    );
  }
  return (
    <MessagePrimitive.Root asChild>
      <article data-message-id={message.id} className={`v2-message role-${role} message-status-${status}`}>
        <div className="v2-message-avatar">{role === "user" ? <UserRound size={17} /> : <Bot size={17} />}</div>
        <div className="v2-message-body">
          <div className="v2-message-meta">
            <span>
              {projectedMessage.source || (role === "user" ? "You" : "CatMaster")}
              {role === "assistant" && turnEntrypointLabel
                ? ` · ${turnEntrypointLabel}`
                : ""}
            </span>
            {role === "assistant" ? <small>{status === "complete" ? "Reply complete" : statusLabel(status)}</small> : null}
          </div>
          <div className="v2-message-parts">
            {status === "running" && !parts.some((part) => part.text || part.summary || part.items?.length || part.type === "tool") ? (
              <div className="v2-thinking-placeholder"><LoaderCircle className="spin" size={14} /> Working on your request</div>
            ) : null}
            <TurnProgressOverview parts={presentation.semanticProgressParts} />
            <TurnPlanOverview parts={presentation.planParts} onSelect={onSelect} />
            {presentation.activityGroups.map((group) => (
              <ActivityGroup
                key={group.id}
                group={group}
                onSelect={onSelect}
                onResume={onResume}
                onContinueFromCheckpoint={continueLatestFailure}
              />
            ))}
            {presentation.contentParts.map((part, index) => (
              <RenderProjectedPart
                key={part.id || `${part.type || "part"}-${index}`}
                part={part}
                onSelect={onSelect}
                onResume={onResume}
                onContinueFromCheckpoint={continueLatestFailure}
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
  activityView = false,
  threadId = "",
  messages = [],
  loading,
  error,
  onSelect,
  onResume,
  onContinueFromCheckpoint,
  hasMore = false,
  onLoadOlder,
  loadingOlder = false,
  todoParts = [],
}) {
  const viewportRef = useRef(null);
  const [preservingHistoryAnchor, setPreservingHistoryAnchor] = useState(false);
  const preservingHistoryRef = useRef(false);
  const followBottomRef = useRef(true);
  const todoProjection = useMemo(
    () => todoCardProjection(messages, todoParts),
    [messages, todoParts],
  );
  const markdownWorkspace = useMemo(
    () => ({ threadId, onSelect }),
    [threadId, onSelect],
  );
  const latestMessageId = messages.at(-1)?.id || "";
  const hasMessages = messages.length > 0;
  const viewportReady = !loading && !error && hasMessages;

  useEffect(() => {
    if (!viewportReady) return undefined;
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
  }, [threadId, viewportReady]);

  useEffect(() => {
    if (!viewportReady) return;
    const viewport = viewportRef.current;
    if (!viewport || !threadId) return;
    followBottomRef.current = true;
    window.requestAnimationFrame(() => {
      viewport.scrollTop = viewport.scrollHeight;
    });
  }, [threadId, viewportReady]);

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
    if (activityView) return <div className="v2-empty" role="status">此视图暂无已记录的内容。运行中的子任务会继续更新。</div>;
    return <ConversationWelcome />;
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
          <MarkdownWorkspaceContext.Provider value={markdownWorkspace}>
            <ThreadPrimitive.Messages>
              {() => (
                <CatMasterMessage
                  activityView={activityView}
                  currentTodoMessageId={todoProjection.currentOwnerId}
                  todoOwnerMessageIds={todoProjection.ownerMessageIds}
                  todoPartsByMessageId={todoProjection.partsByMessageId}
                  latestMessageId={latestMessageId}
                  onSelect={onSelect}
                  onResume={onResume}
                  onContinueFromCheckpoint={onContinueFromCheckpoint}
                />
              )}
            </ThreadPrimitive.Messages>
          </MarkdownWorkspaceContext.Provider>
        </div>
        <ThreadPrimitive.ScrollToBottom className="v2-new-messages">
          New messages
        </ThreadPrimitive.ScrollToBottom>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
