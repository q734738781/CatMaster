import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import rehypeKatex from "rehype-katex";
import { useMemo, useState } from "react";
import { MessagePrimitive, ThreadPrimitive, useMessage } from "@assistant-ui/react";
import ReactMarkdown from "react-markdown";
import { Bot, CircleAlert, FileBox, Hammer, Network, UserRound } from "lucide-react";

import { interruptActions, repeatInterruptDecision } from "../interruptPayload";

function MarkdownBlock({ text }) {
  return (
    <div className="v2-message-text">
      <ReactMarkdown remarkPlugins={[remarkGfm, remarkMath]} rehypePlugins={[rehypeKatex]}>
        {String(text || "")}
      </ReactMarkdown>
    </div>
  );
}

function InterruptActions({ part, onResume }) {
  const actions = useMemo(() => interruptActions(part), [part]);
  const [text, setText] = useState("");
  const [editJson, setEditJson] = useState(() => JSON.stringify(actions.length === 1 ? actions[0] : actions, null, 2));
  const [error, setError] = useState("");
  const resolved = part.status === "resolved";
  if (resolved) return null;
  const rejectPayload = text.trim() ? { type: "reject", message: text.trim() } : { type: "reject" };
  return (
    <div className="v2-interrupt-actions">
      <div className="v2-interrupt-row">
        <button type="button" onClick={() => onResume(repeatInterruptDecision(part, { type: "approve" }))}>Approve</button>
        <button type="button" onClick={() => onResume(repeatInterruptDecision(part, rejectPayload))}>Reject</button>
      </div>
      <textarea
        value={text}
        onChange={(event) => setText(event.target.value)}
        placeholder="Response or rejection note"
        rows={2}
      />
      <div className="v2-interrupt-row">
        <button type="button" disabled={!text.trim()} onClick={() => onResume(repeatInterruptDecision(part, { type: "respond", message: text.trim() }))}>Respond</button>
      </div>
      <details className="v2-interrupt-edit">
        <summary>Edit action</summary>
        <textarea
          value={editJson}
          onChange={(event) => {
            setEditJson(event.target.value);
            setError("");
          }}
          rows={5}
        />
        {error ? <div className="v2-error compact">{error}</div> : null}
        <button
          type="button"
          onClick={() => {
            try {
              const edited = JSON.parse(editJson || "{}");
              const editedActions = Array.isArray(edited) ? edited : [edited];
              if (editedActions.length !== actions.length) {
                throw new Error(`Expected ${actions.length} edited action(s).`);
              }
              editedActions.forEach((item) => {
                if (!item || typeof item !== "object" || !item.name || !item.args || typeof item.args !== "object") {
                  throw new Error("Each edited action must include name and args.");
                }
              });
              onResume(editedActions.map((item) => ({ type: "edit", edited_action: item })));
            } catch (err) {
              setError(err.message || String(err));
            }
          }}
        >
          Submit edit
        </button>
      </details>
    </div>
  );
}

function ReasoningPart({ text }) {
  return (
    <details className="v2-reasoning">
      <summary>Progress</summary>
      <MarkdownBlock text={text} />
    </details>
  );
}

function ToolPart({ toolName, toolCallId, args, result, artifact, status }) {
  const normalizedStatus = status?.type || (result === undefined ? "running" : "completed");
  const subagentSource = String(artifact?.subagent_source || "").trim();
  const displayName = subagentSource ? `${subagentSource} · ${toolName || "Tool call"}` : (toolName || "Tool call");
  const part = {
    id: toolCallId,
    type: "tool-call",
    status: normalizedStatus,
    tool: toolName,
    meta: {
      tool_call_id: toolCallId,
      tool: toolName,
      input: args || {},
      output: result,
      artifact,
    },
  };
  const inputText = JSON.stringify(part.meta.input || {}, null, 2);
  const outputText = result === undefined ? "" : (typeof result === "string" ? result : JSON.stringify(result, null, 2));
  return (
    <details className={`v2-tool-card status-${normalizedStatus}`}>
      <summary>
        <Hammer size={16} />
        <span>{displayName}</span>
        <small>{normalizedStatus}</small>
      </summary>
      <div className="v2-tool-card-body">
        <label>Input</label>
        <pre className="v2-code compact">{inputText}</pre>
        {outputText ? (
          <>
            <label>Output</label>
            <pre className="v2-code compact">{outputText}</pre>
          </>
        ) : null}
      </div>
    </details>
  );
}

function DataPart({ name, data, status, onSelect, onResume }) {
  const type = String(data?.type || name || "data").replace(/^catmaster-/, "");
  if (type === "artifact") {
    return (
      <button type="button" className="v2-part-card artifact" onClick={() => onSelect({ type: "artifact", artifact_id: data.artifact_id, path: data.path, artifact: data })}>
        <FileBox size={16} />
        <span>{data.title || data.path || "Artifact"}</span>
        <small>{data.renderer || "file"}</small>
      </button>
    );
  }
  if (type === "interrupt") {
    const part = { ...data, status: data.status || status?.type || "pending", meta: data.meta || data };
    return (
      <div className="v2-interrupt-card">
        <div className="v2-part-card interrupt">
          <CircleAlert size={16} />
          <span>{part.meta?.title || "Review required"}</span>
          <small>{part.status || "pending"}</small>
        </div>
        <InterruptActions part={part} onResume={onResume} />
      </div>
    );
  }
  if (type === "receipt") {
    return (
      <details className="v2-subagent-card">
        <summary>
          <FileBox size={16} />
          <span>{data.meta?.remote_context_id || data.meta?.submission_hash || data.text || "Remote receipt"}</span>
          <small>{data.status || status?.type || "updated"}</small>
        </summary>
        <pre className="v2-code compact">{JSON.stringify(data.meta || data, null, 2)}</pre>
      </details>
    );
  }
  if (type === "subagent" || type === "trace") {
    const source = data.meta?.source || data.source || "internal";
    const preview = String(data.text || "").trim();
    return (
      <details className="v2-subagent-card">
        <summary>
          <Network size={16} />
          <span>{source}</span>
          <small>{data.status || status?.type || "running"}</small>
        </summary>
        <div className="v2-subagent-body">
          {preview ? <MarkdownBlock text={preview} /> : <span>No visible trace text.</span>}
        </div>
      </details>
    );
  }
  return <pre className="v2-code compact">{JSON.stringify(data, null, 2)}</pre>;
}

function partActivityKind(part) {
  if (part?.type === "tool-call") return "tool";
  if (part?.type !== "data") return "";
  const type = String(part.data?.type || part.name || "data").replace(/^catmaster-/, "");
  return ["receipt", "subagent", "trace"].includes(type) ? type : "";
}

function compactText(value, max = 140) {
  const text = String(value || "").replace(/\s+/g, " ").trim();
  if (!text) return "";
  return text.length > max ? `${text.slice(0, max)}...` : text;
}

function toolSelectionPart(part) {
  return {
    id: part.toolCallId,
    type: "tool-call",
    status: part.status?.type || part.status || (part.result === undefined ? "running" : "completed"),
    tool: part.toolName,
    meta: {
      tool_call_id: part.toolCallId,
      tool: part.toolName,
      input: part.args || {},
      output: part.result,
      artifact: part.artifact,
    },
  };
}

function activityLabel(part) {
  const kind = partActivityKind(part);
  if (kind === "tool") {
    const source = String(part.artifact?.subagent_source || "").trim();
    const toolTitle = part.toolName || "Tool call";
    return {
      icon: Hammer,
      title: source ? `${source} · ${toolTitle}` : toolTitle,
      status: part.status?.type || part.status || (part.result === undefined ? "running" : "completed"),
      detail: compactText(JSON.stringify(part.args || {})),
    };
  }
  const data = part.data || {};
  if (kind === "receipt") {
    return {
      icon: FileBox,
      title: data.meta?.remote_context_id || data.meta?.submission_hash || data.text || "Remote receipt",
      status: data.status || part.status?.type || "updated",
      detail: compactText(data.meta?.receipt_rel || data.meta?.status_message || ""),
    };
  }
  const source = data.meta?.source || data.source || kind;
  return {
    icon: Network,
    title: source,
    status: data.status || part.status?.type || "running",
    detail: compactText(data.text || data.summary || ""),
  };
}

function ActivityRow({ part }) {
  const label = activityLabel(part);
  const Icon = label.icon;
  const kind = partActivityKind(part);
  if (kind === "tool") {
    return (
      <ToolPart
        toolName={part.toolName}
        toolCallId={part.toolCallId}
        args={part.args}
        result={part.result}
        artifact={part.artifact}
        status={part.status}
      />
    );
  }
  const data = part.data || {};
  return (
    <details className={`v2-activity-row status-${label.status}`}>
      <summary>
        <Icon size={15} />
        <span>{label.title}</span>
        {label.detail ? <small>{label.detail}</small> : null}
        <code>{label.status}</code>
      </summary>
      <pre className="v2-code compact">{JSON.stringify(data.meta || data, null, 2)}</pre>
    </details>
  );
}

function ActivityGroup({ parts, onSelect }) {
  const counts = parts.reduce((acc, part) => {
    const kind = partActivityKind(part) || "event";
    acc[kind] = (acc[kind] || 0) + 1;
    const status = activityLabel(part).status;
    if (["failed", "error", "incomplete"].includes(String(status))) acc.failed += 1;
    if (["running", "streaming"].includes(String(status))) acc.running += 1;
    return acc;
  }, { tool: 0, receipt: 0, trace: 0, subagent: 0, failed: 0, running: 0 });
  const summary = [
    counts.tool ? `${counts.tool} tools` : "",
    counts.subagent || counts.trace ? `${counts.subagent + counts.trace} traces` : "",
    counts.receipt ? `${counts.receipt} receipts` : "",
  ].filter(Boolean).join(" / ");
  const status = counts.failed ? `${counts.failed} failed` : counts.running ? `${counts.running} running` : "complete";
  return (
    <details className="v2-activity-group">
      <summary>
        <Hammer size={16} />
        <span>Activity</span>
        <small>{summary || `${parts.length} events`}</small>
        <code>{status}</code>
      </summary>
      <div className="v2-activity-list">
        {parts.map((part, index) => (
          <ActivityRow key={part.toolCallId || part.data?.id || part.data?.meta?.submission_hash || `${part.name || part.type}-${index}`} part={part} />
        ))}
      </div>
    </details>
  );
}

function groupMessageParts(parts) {
  const groups = [];
  let activity = [];
  const flushActivity = () => {
    if (!activity.length) return;
    if (activity.length <= 2) {
      activity.forEach((part) => groups.push({ type: "part", part }));
    } else {
      groups.push({ type: "activity", parts: activity });
    }
    activity = [];
  };
  (Array.isArray(parts) ? parts : []).forEach((part) => {
    if (partActivityKind(part)) {
      activity.push(part);
    } else {
      flushActivity();
      groups.push({ type: "part", part });
    }
  });
  flushActivity();
  return groups;
}

function RenderMessagePart({ part, onSelect, onResume }) {
  if (part?.type === "text") return <MarkdownBlock text={part.text} />;
  if (part?.type === "reasoning") return <ReasoningPart text={part.text} />;
  if (part?.type === "tool-call") return <ToolPart {...part} />;
  if (part?.type === "data") return <DataPart {...part} onSelect={onSelect} onResume={onResume} />;
  return <pre className="v2-code compact">{JSON.stringify(part, null, 2)}</pre>;
}

function CatMasterMessage({ onSelect, onResume }) {
  const message = useMessage();
  const role = String(message?.role || "assistant");
  const status = message?.status?.type || message?.status || "";
  const contentGroups = groupMessageParts(message?.content);
  return (
    <MessagePrimitive.Root asChild>
      <article className={`v2-message role-${role} status-${status}`}>
        <div className="v2-message-avatar">{role === "user" ? <UserRound size={17} /> : <Bot size={17} />}</div>
        <div className="v2-message-body">
          <div className="v2-message-meta">
            <span>{role === "user" ? "You" : "CatMaster"}</span>
            <small>{status}</small>
          </div>
          <div className="v2-message-parts">
            {contentGroups.map((group, index) => (
              group.type === "activity"
                ? <ActivityGroup key={`activity-${index}`} parts={group.parts} onSelect={onSelect} />
                : <RenderMessagePart key={group.part?.toolCallId || group.part?.data?.id || `${group.part?.type || "part"}-${index}`} part={group.part} onSelect={onSelect} onResume={onResume} />
            ))}
          </div>
        </div>
      </article>
    </MessagePrimitive.Root>
  );
}

export default function ThreadMessages({ messages, loading, error, onSelect, onResume }) {
  if (loading) return <div className="v2-empty">Loading thread...</div>;
  if (error) return <div className="v2-error">{error}</div>;
  if (!messages.length) {
    return <div className="v2-empty">Start a thread from the composer.</div>;
  }
  return (
    <ThreadPrimitive.Root>
      <ThreadPrimitive.Viewport className="v2-thread-viewport" autoScroll>
        <div className="v2-thread-messages">
          <ThreadPrimitive.Messages>
            {() => <CatMasterMessage onSelect={onSelect} onResume={onResume} />}
          </ThreadPrimitive.Messages>
        </div>
      </ThreadPrimitive.Viewport>
    </ThreadPrimitive.Root>
  );
}
