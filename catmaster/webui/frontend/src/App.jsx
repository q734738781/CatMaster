import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "katex/dist/katex.min.css";

function escapePath(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return encodeURIComponent(String(value));
}

function isRunActive(status) {
  return ["running", "starting", "interrupting", "awaiting_human_feedback"].includes(String(status || "").trim());
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  return response.json();
}

function formatTime(ts) {
  if (!ts) {
    return "";
  }
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch {
    return "";
  }
}

function joinItems(items) {
  return (items || []).filter(Boolean).join(" · ");
}

function compactText(value, maxChars = 700) {
  const text = String(value || "").trim();
  if (!text) {
    return "";
  }
  if (text.length <= maxChars) {
    return text;
  }
  return `${text.slice(0, Math.max(0, maxChars - 3)).trimEnd()}...`;
}

function normalizeComparableText(value) {
  return String(value || "")
    .toLowerCase()
    .replace(/[`*_#>-]+/g, " ")
    .replace(/\s+/g, " ")
    .trim();
}

function compactComparableText(value) {
  return normalizeComparableText(value).replace(/[^a-z0-9\u4e00-\u9fff]+/g, "");
}

function normalizeStatusToken(value) {
  return String(value || "").trim().toLowerCase().replaceAll("-", "_");
}

function todoStatusRank(status) {
  const normalized = normalizeStatusToken(status);
  if (normalized === "in_progress" || normalized === "running") {
    return 0;
  }
  if (normalized === "pending" || normalized === "queued") {
    return 1;
  }
  if (normalized === "completed" || normalized === "done" || normalized === "success") {
    return 2;
  }
  return 3;
}

function toolStatusRank(status) {
  const normalized = normalizeStatusToken(status);
  if (normalized === "running" || normalized === "in_progress") {
    return 0;
  }
  if (normalized === "error" || normalized === "failure" || normalized === "validation_failed" || normalized === "interrupted") {
    return 2;
  }
  return 1;
}

function shouldRecordEvent(event) {
  const name = String(event?.name || "");
  return !["LLM_TOKEN_DELTA", "LLM_REASONING_DELTA"].includes(name);
}

function resolveThinkingAgent(snapshot, agentTab = "ALL") {
  const live = snapshot?.live_state || {};
  const agents = live?.agents && typeof live.agents === "object" ? live.agents : {};
  if (agentTab !== "ALL") {
    const selected = agents[agentTab];
    if (selected && typeof selected === "object") {
      const llm = selected?.llm && typeof selected.llm === "object" ? selected.llm : {};
      const hasText = String(llm.reasoning_text || llm.text || "").trim();
      if (hasText) {
        return { name: agentTab, state: selected };
      }
    }
  }
  const rows = Object.entries(agents)
    .map(([name, state]) => ({ name, state: state && typeof state === "object" ? state : {} }))
    .map((row) => {
      const llm = row.state?.llm && typeof row.state.llm === "object" ? row.state.llm : {};
      const text = String(llm.reasoning_text || llm.text || "").trim();
      return { ...row, hasText: text };
    })
    .filter(({ hasText }) => hasText)
    .sort((left, right) => {
      const leftLlm = left.state?.llm && typeof left.state.llm === "object" ? left.state.llm : {};
      const rightLlm = right.state?.llm && typeof right.state.llm === "object" ? right.state.llm : {};
      const leftRunning = normalizeStatusToken(leftLlm.status) === "running" ? 0 : 1;
      const rightRunning = normalizeStatusToken(rightLlm.status) === "running" ? 0 : 1;
      if (leftRunning !== rightRunning) {
        return leftRunning - rightRunning;
      }
      return Number(right.state?.last_updated_ts || 0) - Number(left.state?.last_updated_ts || 0);
    });
  if (rows.length) {
    return rows[0];
  }
  if (agentTab !== "ALL") {
    const selected = agents[agentTab];
    if (selected && typeof selected === "object") {
      return { name: agentTab, state: selected };
    }
  }
  return null;
}

function buildLiveAssistantMessage(snapshot, agentTab = "ALL") {
  const live = snapshot?.live_state || {};
  const status = String(snapshot?.run_status || live.status || "").trim();
  const isActive = isRunActive(status);
  if (!isActive) {
    return null;
  }
  const thinkingAgent = resolveThinkingAgent(snapshot, agentTab);
  const llm = thinkingAgent?.state?.llm && typeof thinkingAgent.state.llm === "object"
    ? thinkingAgent.state.llm
    : (snapshot?.llm || live.llm || {});
  const graph = snapshot?.graph || {};
  let reasoningText = compactText(llm.reasoning_text || "", 1400);
  let draftText = compactText(llm.text || graph.text_preview || "", 1400);
  const reasoningCompact = compactComparableText(reasoningText);
  const draftCompact = compactComparableText(draftText);
  if (reasoningCompact && draftCompact) {
    if (reasoningCompact.includes(draftCompact)) {
      draftText = "";
    } else if (draftCompact.includes(reasoningCompact)) {
      reasoningText = "";
    }
  }
  if (!reasoningText && !draftText) {
    return null;
  }

  const parts = [reasoningText, draftText].filter(Boolean);

  return {
    role: "assistant",
    kind: "live_assistant",
    badge: "thinking",
    status: thinkingAgent?.name && thinkingAgent.name !== "ALL" ? `${thinkingAgent.name}` : (status || "running"),
    content: parts.join("\n\n"),
  };
}

function mergeChatMessages(messages, liveMessage) {
  const rows = Array.isArray(messages) ? [...messages] : [];
  if (liveMessage) {
    rows.push(liveMessage);
  }
  return rows;
}

function formatPromptContent(prompt) {
  if (!prompt) {
    return "";
  }
  const payload = prompt.payload || {};
  const todo = Array.isArray(payload.todo) ? payload.todo.filter(Boolean) : [];
  const parts = [];
  if (payload.proposal_description) {
    parts.push(String(payload.proposal_description).trim());
  }
  if (todo.length) {
    parts.push(`Todo\n${todo.map((item, index) => `${index + 1}. ${item}`).join("\n")}`);
  }
  if (payload.guidance) {
    parts.push(String(payload.guidance).trim());
  }
  return parts.filter(Boolean).join("\n\n");
}

function buildProposalMessage(snapshot) {
  const prompt = snapshot?.prompt || null;
  if (prompt?.kind === "proposal_review") {
    const content = formatPromptContent(prompt);
    if (!content) {
      return null;
    }
    return {
      role: "assistant",
      kind: "proposal",
      badge: "proposal",
      status: "awaiting review",
      content,
    };
  }
  return null;
}

function buildTodoMessage(snapshot) {
  if (snapshot?.prompt?.kind === "proposal_review") {
    return null;
  }
  const items = Array.isArray(snapshot?.todo_items) ? snapshot.todo_items.filter(Boolean) : [];
  if (!items.length) {
    return null;
  }
  return {
    role: "assistant",
    kind: "proposal",
    badge: "todo",
    status: "checklist",
    content: `Todo\n${items.map((item, index) => `${index + 1}. ${item}`).join("\n")}`,
  };
}

function eventToChatMessage(event) {
  if (!event || typeof event !== "object") {
    return null;
  }
  const name = String(event.name || "");
  const payload = event.payload || {};
  const tsText = formatTime(event.ts);
  if (name === "RUN_START") {
    const lane = String(payload.entrypoint || payload.lane || "").trim();
    return {
      role: "assistant",
      kind: "run_event",
      badge: "run",
      status: tsText,
      content: lane ? `Started ${lane} run.` : "Started run.",
    };
  }
  if (name === "LLM_CALL_END") {
    const preview = compactText(payload.text_preview || "", 900);
    const tools = Array.isArray(payload.tool_calls) ? payload.tool_calls.filter(Boolean) : [];
    if (!preview && !tools.length) {
      return null;
    }
    const parts = [];
    if (preview) {
      parts.push(preview);
    }
    if (tools.length) {
      parts.push(`Tool plan: ${tools.join(", ")}`);
    }
    return {
      role: "assistant",
      kind: "llm_step",
      badge: "llm",
      status: tsText,
      content: parts.join("\n\n"),
    };
  }
  if (name === "TOOL_CALL_START") {
    const tool = String(payload.tool || "").trim();
    if (!tool) {
      return null;
    }
    const params = compactText(payload.params_compact || "", 320);
    return {
      role: "assistant",
      kind: "tool_event",
      badge: "tool",
      status: tsText,
      content: [`Running ${tool}.`, params ? `Args: ${params}` : ""].filter(Boolean).join("\n"),
    };
  }
  if (name === "TOOL_CALL_END") {
    const tool = String(payload.tool || "").trim();
    if (!tool) {
      return null;
    }
    const status = String(payload.status || "done").trim();
    const summary = compactText(payload.highlights || "", 420);
    return {
      role: "assistant",
      kind: "tool_event",
      badge: status === "success" ? "done" : status,
      status: tsText,
      content: [`${tool}: ${status}.`, summary].filter(Boolean).join("\n"),
    };
  }
  if (name === "PROMPT_REQUESTED") {
    return {
      role: "assistant",
      kind: "run_event",
      badge: "pause",
      status: tsText,
      content: "Waiting for human feedback.",
    };
  }
  if (name === "RUN_END") {
    const status = String(payload.status || "done").trim();
    return {
      role: "assistant",
      kind: "run_event",
      badge: status,
      status: tsText,
      content: `Run finished with status: ${status}.`,
    };
  }
  return null;
}

function buildEventMessages(events) {
  const seen = new Set();
  return (Array.isArray(events) ? events : [])
    .slice(-40)
    .map((event) => {
      const key = String(event?.seq || `${event?.name || "event"}-${event?.ts || ""}`);
      if (seen.has(key)) {
        return null;
      }
      seen.add(key);
      return eventToChatMessage(event);
    })
    .filter(Boolean);
}

function messageMatchesResult(message, resultText) {
  if (!message || typeof message !== "object") {
    return false;
  }
  if (String(message.role || "") !== "assistant") {
    return false;
  }
  const messageText = compactText(message.content || "", 1800);
  const left = compactComparableText(messageText);
  const right = compactComparableText(resultText);
  if (!left || !right) {
    return false;
  }
  return left === right || left.includes(right) || right.includes(left);
}

function decoratePersistedMessages(snapshot, persistedMessages) {
  const rows = Array.isArray(persistedMessages) ? persistedMessages.map((message) => ({ ...message })) : [];
  const resultText = compactText(snapshot?.result_text || "", 1800);
  if (!resultText) {
    return rows;
  }
  let matchIndex = -1;
  rows.forEach((message, index) => {
    if (messageMatchesResult(message, resultText)) {
      matchIndex = index;
    }
  });
  if (matchIndex < 0) {
    return rows;
  }
  const matched = rows[matchIndex] || {};
  rows[matchIndex] = {
    ...matched,
    kind: "result",
    badge: "result",
    status: matched.status || String(snapshot?.run_status || "done"),
  };
  return rows;
}

function buildResultFallbackMessage(snapshot, persistedMessages) {
  const resultText = compactText(snapshot?.result_text || "", 1800);
  if (!resultText) {
    return null;
  }
  const hasAssistantResult = (Array.isArray(persistedMessages) ? persistedMessages : [])
    .some((message) => messageMatchesResult(message, resultText));
  if (hasAssistantResult) {
    return null;
  }
  return {
    role: "assistant",
    kind: "result",
    badge: "result",
    status: String(snapshot?.run_status || "done"),
    content: resultText,
  };
}

function buildChatTimeline(snapshot, events, liveMessage) {
  const persistedMessages = decoratePersistedMessages(
    snapshot,
    Array.isArray(snapshot?.chat_messages) ? snapshot.chat_messages : [],
  );
  const rows = [...persistedMessages];
  const proposalMessage = buildProposalMessage(snapshot);
  if (proposalMessage) {
    rows.push(proposalMessage);
  }
  const fallbackResult = buildResultFallbackMessage(snapshot, persistedMessages);
  if (fallbackResult) {
    rows.push(fallbackResult);
  }
  return mergeChatMessages(rows, liveMessage);
}

function agentVisualStatus(agent) {
  if (!agent || typeof agent !== "object") {
    return "idle";
  }
  if (agent.active_toolcall) {
    return "active";
  }
  const llmStatus = String(agent?.llm?.status || "").trim();
  if (llmStatus === "running") {
    return "active";
  }
  const hasHistory = (Array.isArray(agent.recent_toolcalls) && agent.recent_toolcalls.length)
    || (Array.isArray(agent.todo_rows) && agent.todo_rows.length)
    || String(agent?.llm?.text || agent?.llm?.reasoning_text || "").trim();
  return hasHistory ? "completed" : "idle";
}

function normalizeAgentTodoRows(agentName, agent) {
  const rows = Array.isArray(agent?.todo_rows) ? agent.todo_rows : [];
  return rows
    .filter((row) => row && row.content)
    .map((row, index) => ({
      ...row,
      status: row.status,
      agent_name: agentName,
      agent_sort_key: agentName,
      row_index: index,
    }));
}

function toolTraceRowsForAgent(agentName, agent) {
  const rows = [];
  const visualStatus = agentVisualStatus(agent);
  if (agent?.active_toolcall?.tool) {
    rows.push({
      ...agent.active_toolcall,
      agent_name: agentName,
      sort_ts: Number(agent.active_toolcall.started_ts || 0),
      agent_sort_key: agentName,
      local_index: -1,
    });
  }
  const recent = Array.isArray(agent?.recent_toolcalls) ? agent.recent_toolcalls : [];
  recent.forEach((row, index) => {
    if (!row?.tool) {
      return;
    }
    rows.push({
      ...row,
      agent_name: agentName,
      sort_ts: Number(row.ended_ts || row.started_ts || 0),
      agent_sort_key: agentName,
      local_index: index,
    });
  });
  if (!rows.length && visualStatus !== "idle") {
    const active = visualStatus === "active";
    const sortTs = Number(
      active
        ? agent?.started_ts || agent?.last_updated_ts || 0
        : agent?.completed_ts || agent?.last_updated_ts || 0,
    );
    rows.push({
      tool: "subagent",
      status: active ? "running" : "success",
      highlights: active ? `${agentName} is running.` : `${agentName} completed.`,
      params_compact: "",
      agent_name: agentName,
      sort_ts: sortTs,
      agent_sort_key: agentName,
      local_index: -1,
      toolcall_id: `subagent:${agentName}:${sortTs || 0}:${visualStatus}`,
    });
  }
  return rows;
}

function agentTabs(live) {
  const agents = live?.agents && typeof live.agents === "object" ? live.agents : {};
  const rows = Object.entries(agents)
    .map(([name, value]) => {
      const agent = value && typeof value === "object" ? value : {};
      const status = agentVisualStatus(agent);
      return {
        name,
        status,
        lastUpdatedTs: Number(agent.last_updated_ts || 0),
      };
    })
    .filter((row) => row.name);
  rows.sort((left, right) => {
    const leftRank = left.status === "active" ? 0 : left.status === "completed" ? 1 : 2;
    const rightRank = right.status === "active" ? 0 : right.status === "completed" ? 1 : 2;
    if (leftRank !== rightRank) {
      return leftRank - rightRank;
    }
    return (right.lastUpdatedTs || 0) - (left.lastUpdatedTs || 0);
  });
  return [
    { name: "ALL", status: rows.some((row) => row.status === "active") ? "active" : "completed", lastUpdatedTs: 0 },
    ...rows,
  ];
}

function aggregateAgentTodos(live) {
  const agents = live?.agents && typeof live.agents === "object" ? live.agents : {};
  return Object.entries(agents)
    .flatMap(([agentName, agent]) => normalizeAgentTodoRows(agentName, agent))
    .sort((left, right) => {
      const leftRank = todoStatusRank(left.status);
      const rightRank = todoStatusRank(right.status);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      const agentCmp = String(left.agent_sort_key || "").localeCompare(String(right.agent_sort_key || ""));
      if (agentCmp !== 0) {
        return agentCmp;
      }
      return Number(left.row_index || 0) - Number(right.row_index || 0);
    });
}

function aggregateAgentToolTrace(live) {
  const agents = live?.agents && typeof live.agents === "object" ? live.agents : {};
  const rows = Object.entries(agents).flatMap(([agentName, agent]) => toolTraceRowsForAgent(agentName, agent));
  const seen = new Set();
  return rows
    .sort((left, right) => {
      const leftRank = toolStatusRank(left.status);
      const rightRank = toolStatusRank(right.status);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      const tsDiff = Number(right.sort_ts || 0) - Number(left.sort_ts || 0);
      if (tsDiff !== 0) {
        return tsDiff;
      }
      const agentCmp = String(left.agent_sort_key || "").localeCompare(String(right.agent_sort_key || ""));
      if (agentCmp !== 0) {
        return agentCmp;
      }
      return Number(left.local_index || 0) - Number(right.local_index || 0);
    })
    .filter((row) => {
      const key = String(row.toolcall_id || `${row.agent_name}:${row.tool}:${row.sort_ts}`);
      if (seen.has(key)) {
        return false;
      }
      seen.add(key);
      return true;
    });
}

const LANE_GUIDE = {
  experiment: {
    title: "Experiment",
    summary: "Run bounded computational execution and return concise evidence with files.",
    subagents: ["task_worker_agent", "literature_agent"],
  },
  research: {
    title: "Research",
    summary: "Coordinate broader investigation and delegate only when experiment or writing work is needed.",
    subagents: ["experiment_specialist", "writing_specialist", "litreview_agent"],
  },
  writing: {
    title: "Writing",
    summary: "Draft or revise deliverables from existing evidence and compile when needed.",
    subagents: ["writing_worker_agent"],
  },
};

function formatCount(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const numeric = Number(value);
  if (Number.isNaN(numeric)) {
    return String(value);
  }
  return numeric.toLocaleString();
}

function StatusPill({ status }) {
  return <span className={`status-pill status-${String(status || "idle").replaceAll("_", "-")}`}>{status || "idle"}</span>;
}

function MetricCard({ label, value, note }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value || "-"}</div>
      {note ? <div className="metric-note">{note}</div> : null}
    </div>
  );
}

function RunCard({ card, active, onSelect }) {
  return (
    <button type="button" className={`run-card ${active ? "active" : ""}`} onClick={() => onSelect(card.run_name)}>
      <div className="run-card-header">
        <div>
          <h3>{card.headline || card.run_name}</h3>
          <p>{joinItems([card.status, card.model_name, card.start_time])}</p>
        </div>
        <span className="run-card-id">{card.run_name}</span>
      </div>
      <p className="run-card-summary">{card.summary || "No summary yet."}</p>
      {(card.next_actions || []).length ? (
        <ul className="run-card-actions">
          {(card.next_actions || []).slice(0, 3).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
    </button>
  );
}

function EventFeed({ events }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const node = containerRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [events]);

  return (
    <div ref={containerRef} className="feed-list">
      {(events || []).slice(-120).map((event) => {
        const payload = event.payload || {};
        const title = joinItems([event.category, event.name, payload.tool || payload.model || payload.node]);
        const body =
          payload.text ||
          payload.summary_snippet ||
          payload.reasoning_text ||
          payload.error ||
          payload.text_preview ||
          payload.goal ||
          payload.status ||
          "";
        return (
          <article key={event.seq || `${event.name}-${event.ts}`} className="feed-item">
            <div className="feed-meta">
              <span>{title}</span>
              <span>{formatTime(event.ts)}</span>
            </div>
            <p>{body || "(no body)"}</p>
          </article>
        );
      })}
    </div>
  );
}

function normalizeMathMarkdown(text) {
  const source = String(text || "");
  if (!source) {
    return "";
  }
  return source
    .replace(/\\\[((?:.|\n)*?)\\\]/g, (_match, expr) => `\n$$\n${String(expr || "").trim()}\n$$\n`)
    .replace(/\\\(((?:.|\n)*?)\\\)/g, (_match, expr) => `$${String(expr || "").trim()}$`);
}

const REMARK_PLUGINS = [remarkGfm, remarkMath];
const REHYPE_PLUGINS = [rehypeKatex];

function MarkdownContent({ text }) {
  const source = normalizeMathMarkdown(text);
  const rendered = useMemo(
    () => (
      <Markdown remarkPlugins={REMARK_PLUGINS} rehypePlugins={REHYPE_PLUGINS}>{source}</Markdown>
    ),
    [source],
  );
  return <div className="chat-content md-body">{rendered}</div>;
}

function ChatThread({ messages }) {
  const threadRef = useRef(null);

  useEffect(() => {
    const node = threadRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages]);

  return (
    <div ref={threadRef} className="chat-thread">
      {(messages || []).map((message, index) => (
        <article
          key={`${message.role || "assistant"}-${message.kind || "chat"}-${index}`}
          className={`chat-bubble ${message.role || "assistant"} ${message.kind || "chat"}`}
        >
          <div className="chat-bubble-header">
            <div className="chat-role">{message.role || "assistant"}</div>
            {message.badge ? <div className="chat-badge">{message.badge}</div> : null}
          </div>
          {message.status ? <div className="chat-status">{message.status}</div> : null}
          <MarkdownContent text={message.content} />
        </article>
      ))}
    </div>
  );
}

function PromptPanel({ prompt, value, onChange, onSubmit, disabled }) {
  if (!prompt) {
    return null;
  }
  const payload = prompt.payload || {};
  const body = [
    payload.proposal_description,
    payload.report_text,
    payload.guidance,
    Array.isArray(payload.todo) && payload.todo.length
      ? payload.todo.map((item, index) => `${index + 1}. ${item}`).join("\n")
      : "",
    payload.report_path ? `report: ${payload.report_path}` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
  return (
    <section className="prompt-panel">
      <div className="section-label">Human Input Required</div>
      <div className="prompt-meta">{joinItems([prompt.kind, payload.run_id, payload.prompt_id || prompt.prompt_id])}</div>
      <pre className="code-pane">{body || "(empty prompt payload)"}</pre>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Provide feedback, approval, or revised guidance."
        disabled={disabled}
      />
      <button type="button" onClick={onSubmit} disabled={disabled}>
        Submit Feedback
      </button>
    </section>
  );
}

function ArtifactPanel({ details }) {
  const artifacts = details?.artifacts || [];
  if (!artifacts.length) {
    return null;
  }
  return (
    <div className="artifact-grid">
      <div className="section-label">Artifacts</div>
      {artifacts.slice(0, 24).map((row, index) => (
        <article key={`${row.path || "artifact"}-${index}`} className="artifact-card">
          <div className="section-label">{joinItems([row.kind, row.type])}</div>
          <h4>{row.path || "(unknown path)"}</h4>
          <p>{row.description || "No description."}</p>
        </article>
      ))}
    </div>
  );
}

function MonitorTabs({ tab, onChange }) {
  const tabs = [
    ["result", "Result"],
    ["task", "Run State"],
    ["trace", "Trace"],
    ["artifacts", "Artifacts"],
  ];
  return (
    <div className="tab-row">
      {tabs.map(([value, label]) => (
        <button
          key={value}
          type="button"
          className={`tab-btn ${tab === value ? "active" : ""}`}
          onClick={() => onChange(value)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function CodePane({ title, text, helper }) {
  return (
    <section className="control-stack">
      <div className="section-head">
        <div>
          <div className="section-label">{helper || "Details"}</div>
          <h3 className="section-title">{title}</h3>
        </div>
      </div>
      <pre className="code-pane tall">{text || "(empty)"}</pre>
    </section>
  );
}

function UsagePanel({ usage }) {
  const rows = [
    ["Input", usage?.input_tokens],
    ["Cached input", usage?.input_cached_tokens],
    ["Output", usage?.output_tokens],
    ["Reasoning", usage?.reasoning_tokens],
    ["Total", usage?.total_tokens],
  ];
  return (
    <section className="control-stack">
      <div className="section-head">
        <div>
          <div className="section-label">Usage</div>
          <h3 className="section-title">Token totals</h3>
        </div>
      </div>
      <div className="usage-grid">
        {rows.map(([label, value]) => (
          <div key={label} className="usage-cell">
            <div className="usage-label">{label}</div>
            <div className="usage-value">{formatCount(value)}</div>
          </div>
        ))}
      </div>
    </section>
  );
}

function TodoPanel({ todos }) {
  const rows = (Array.isArray(todos) ? todos.filter((item) => item && item.content) : [])
    .map((item, index) => ({ ...item, __sort_index: index }))
    .sort((left, right) => {
      const leftRank = todoStatusRank(left.status);
      const rightRank = todoStatusRank(right.status);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      return (left.__sort_index || 0) - (right.__sort_index || 0);
    });
  return (
    <section className="todo-panel">
      <div className="section-head">
        <div>
          <div className="section-label">Todo</div>
        </div>
      </div>
      {rows.length ? (
        <div className="todo-list">
          {rows.map((item, index) => (
            <article key={`${item.content}-${index}`} className={`todo-item status-${String(item.status || "pending").replaceAll("_", "-")}`}>
              <div className="todo-item-head">
                <span className="todo-status-pill">{item.status || "pending"}</span>
                {item.agent_name ? <span className="todo-agent-pill">{item.agent_name}</span> : null}
              </div>
              <div className="todo-item-text">{item.content}</div>
            </article>
          ))}
        </div>
      ) : (
        <div className="todo-empty">No live todo list yet.</div>
      )}
    </section>
  );
}

function ToolTracePanel({ activeToolcall, recentToolcalls }) {
  const rows = (Array.isArray(recentToolcalls) ? recentToolcalls.filter((item) => item && item.tool) : [])
    .sort((left, right) => {
      const leftRank = toolStatusRank(left.status);
      const rightRank = toolStatusRank(right.status);
      if (leftRank !== rightRank) {
        return leftRank - rightRank;
      }
      const leftTs = Number(left.started_ts || left.ended_ts || left.sort_ts || 0);
      const rightTs = Number(right.started_ts || right.ended_ts || right.sort_ts || 0);
      if (rightTs !== leftTs) {
        return rightTs - leftTs;
      }
      return String(left.agent_name || "").localeCompare(String(right.agent_name || ""));
    });

  return (
    <section className="tool-trace-panel">
      <div className="section-head">
        <div>
          <div className="section-label">Tool trace</div>
        </div>
      </div>
      {rows.length ? (
        <div className="tool-trace-list">
          {rows.map((item, index) => {
            const status = String(item.status || "running").trim() || "running";
            const summary = compactText(item.highlights || item.params_compact || "", 420);
            return (
              <article key={`${item.toolcall_id || item.tool}-${index}`} className={`tool-trace-item status-${status.replaceAll("_", "-")}`}>
                <div className="tool-trace-head">
                  <div className="tool-trace-title">
                    {String(item.tool || "").trim()}
                    {item.agent_name ? <span className="tool-trace-agent">{item.agent_name}</span> : null}
                  </div>
                  <span className="tool-trace-status">{status}</span>
                </div>
                {summary ? <div className="tool-trace-body">{summary}</div> : null}
              </article>
            );
          })}
        </div>
      ) : (
        <div className="todo-empty">No tool calls yet.</div>
      )}
    </section>
  );
}

function MemoryDrawer({ open, workspaceName, loading, error, text, onRefresh, onClose }) {
  if (!open) {
    return null;
  }
  return (
    <>
      <button type="button" className="memory-drawer-backdrop" aria-label="Close memory viewer" onClick={onClose} />
      <aside className="memory-drawer">
        <div className="memory-drawer-header">
          <div>
            <div className="section-label">Persistent memory</div>
            <h3>{workspaceName ? `${workspaceName} memory` : "Project memory"}</h3>
          </div>
          <div className="inline-actions">
            <button type="button" className="ghost-btn" onClick={onRefresh} disabled={loading}>
              Refresh
            </button>
            <button type="button" className="ghost-btn" onClick={onClose}>
              Close
            </button>
          </div>
        </div>
        {error ? <div className="memory-drawer-note error">{error}</div> : null}
        {!error && loading && !text ? <div className="memory-drawer-note">Loading memory...</div> : null}
        <pre className="code-pane memory-card">{text || "No persistent memory recorded yet."}</pre>
      </aside>
    </>
  );
}

function App({ boot }) {
  const view = boot?.view === "monitor" ? "monitor" : "home";
  const [snapshot, setSnapshot] = useState(null);
  const [details, setDetails] = useState(null);
  const [ctx, setCtx] = useState("");
  const [lane, setLane] = useState("experiment");
  const [selectedRun, setSelectedRun] = useState("");
  const [workspaceRoot, setWorkspaceRoot] = useState("");
  const [workspaceName, setWorkspaceName] = useState("");
  const [search, setSearch] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [events, setEvents] = useState([]);
  const [promptResponse, setPromptResponse] = useState("");
  const [monitorTab, setMonitorTab] = useState("result");
  const [agentTab, setAgentTab] = useState("ALL");
  const [streamNonce, setStreamNonce] = useState(0);
  const [memoryOpen, setMemoryOpen] = useState(false);
  const [memoryPanel, setMemoryPanel] = useState({
    text: "",
    error: "",
    loading: false,
    workspace: "",
  });
  const [form, setForm] = useState({
    prompt: "",
    run_mode: "new_run",
    resume_run_name: "",
    proposal_review: false,
  });
  const deferredSearch = useDeferredValue(search);
  const eventSourceRef = useRef(null);
  const latestSeqRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams(window.location.search);
    const nextLane = params.get("lane") || "experiment";
    if (!params.get("lane")) {
      params.set("lane", nextLane);
    }
    setLane(nextLane);
    (async () => {
      try {
        const data = await apiFetch(`/api/bootstrap?${params.toString()}`);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setCtx(data.ctx || "");
          setWorkspaceRoot(data.workspace_root || "");
          setSelectedRun(data.selected_run || "");
          setSnapshot(data);
          setStatusMessage(data.status_message || "");
          setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
          latestSeqRef.current = Number(data.runtime?.seq || 0);
        });
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(String(error?.message || error));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (view !== "monitor" || !ctx || !selectedRun) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const data = await apiFetch(`/api/session/${escapePath(ctx)}/details?run=${escapePath(selectedRun)}`);
        if (!cancelled) {
          startTransition(() => {
            setDetails(data);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(String(error?.message || error));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ctx, selectedRun, view]);

  useEffect(() => {
    if (!ctx) {
      return undefined;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const source = new EventSource(`/api/session/${escapePath(ctx)}/stream?last_seq=${escapePath(latestSeqRef.current)}`);
    eventSourceRef.current = source;

    source.onmessage = (message) => {
      const data = JSON.parse(message.data || "{}");
      const event = data.event || {};
      const runtime = data.runtime || {};
      latestSeqRef.current = Number(runtime.seq || event.seq || latestSeqRef.current || 0);
      const streamRunName = data.active_run || data.selected_run || runtime.run_name || "";
      const streamActive = Boolean(runtime.active) && Boolean(streamRunName);
      const shouldApplyStream = view === "home"
        ? streamActive
        : (Boolean(streamRunName) && (!selectedRun || streamRunName === selectedRun));
      if (shouldApplyStream) {
        startTransition(() => {
          if (streamRunName) {
            setSelectedRun(streamRunName);
          }
          if (shouldRecordEvent(event)) {
            setEvents((prev) => [...prev, event].slice(-120));
          }
          setSnapshot((prev) => {
            if (!prev) {
              return prev;
            }
            return {
              ...prev,
              active_run: data.active_run || prev.active_run || "",
              selected_run: streamRunName || prev.selected_run || "",
              runtime,
              live_state: runtime.live_state || prev.live_state || {},
              llm: runtime.llm || prev.llm || {},
              graph: runtime.graph || prev.graph || {},
              prompt: runtime.prompt ?? prev.prompt ?? null,
              usage_summary: data.usage_summary || runtime.usage_totals || prev.usage_summary || {},
              chat_messages: data.chat_messages || prev.chat_messages || [],
              cards: data.cards || prev.cards || [],
              todo_items: data.todo_items || prev.todo_items || [],
              result_text: data.result_text ?? prev.result_text ?? "",
              proposal: data.proposal ?? prev.proposal ?? "",
              can_submit_prompt: Boolean(runtime.prompt),
              run_status: data.run_status || prev.run_status,
              run_status_text: data.run_status_text || prev.run_status_text,
            };
          });
        });
      }
      if (view === "monitor" && ["RUN_SNAPSHOT_READY", "PROMPT_REQUESTED", "PROMPT_RESOLVED"].includes(String(event.name || ""))) {
        const nextRun = streamRunName || selectedRun;
        if (nextRun) {
          apiFetch(`/api/session/${escapePath(ctx)}/details?run=${escapePath(nextRun)}`)
            .then((detailData) => {
              startTransition(() => {
                setDetails(detailData);
              });
            })
            .catch(() => {});
        }
      }
    };

    source.onerror = () => {
      source.close();
      eventSourceRef.current = null;
      window.setTimeout(() => {
        if (ctx) {
          latestSeqRef.current = latestSeqRef.current || 0;
          setStatusMessage((prev) => prev || "Stream disconnected. Reconnecting.");
          setStreamNonce((value) => value + 1);
        }
      }, 1500);
    };

    return () => {
      source.close();
      eventSourceRef.current = null;
    };
  }, [ctx, selectedRun, streamNonce]);

  useEffect(() => {
    if (view !== "home" || !memoryOpen || !ctx) {
      return;
    }
    let cancelled = false;
    const workspaceLabel = String(snapshot?.workspace_name || "");
    startTransition(() => {
      setMemoryPanel((prev) => ({
        ...prev,
        loading: true,
        error: "",
      }));
    });
    apiFetch(`/api/session/${escapePath(ctx)}/memory?run=${escapePath(selectedRun || "")}`)
      .then((data) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setMemoryPanel({
            text: String(data.memory || "").trim(),
            error: "",
            loading: false,
            workspace: workspaceLabel,
          });
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setMemoryPanel((prev) => ({
            ...prev,
            loading: false,
            error: String(error?.message || error),
            workspace: workspaceLabel,
          }));
        });
      });
    return () => {
      cancelled = true;
    };
  }, [ctx, memoryOpen, selectedRun, snapshot?.workspace_name, view]);

  useEffect(() => {
    const live = snapshot?.live_state || {};
    const tabs = agentTabs(live).map((item) => item.name);
    if (!tabs.includes(agentTab)) {
      setAgentTab("ALL");
    }
  }, [agentTab, snapshot?.live_state]);

  async function refreshSnapshot(runName = selectedRun) {
    if (!ctx) {
      return;
    }
    const data = await apiFetch(
      `/api/session/${escapePath(ctx)}/snapshot?lane=${escapePath(lane)}&run=${escapePath(runName || "")}`,
    );
    startTransition(() => {
      setSnapshot(data);
      setSelectedRun(data.selected_run || "");
      setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
      latestSeqRef.current = Number(data.runtime?.seq || latestSeqRef.current || 0);
    });
  }

  async function refreshMemoryPanel() {
    if (!ctx) {
      return;
    }
    startTransition(() => {
      setMemoryPanel((prev) => ({
        ...prev,
        loading: true,
        error: "",
      }));
    });
    try {
      const data = await apiFetch(`/api/session/${escapePath(ctx)}/memory?run=${escapePath(selectedRun || "")}`);
      startTransition(() => {
        setMemoryPanel({
          text: String(data.memory || "").trim(),
          error: "",
          loading: false,
          workspace: String(snapshot?.workspace_name || ""),
        });
      });
    } catch (error) {
      startTransition(() => {
        setMemoryPanel((prev) => ({
          ...prev,
          loading: false,
          error: String(error?.message || error),
          workspace: String(snapshot?.workspace_name || ""),
        }));
      });
    }
  }

  async function postAndApply(url, payload, { loadDetails = false } = {}) {
    if (!ctx) {
      return;
    }
    const data = await apiFetch(url, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    startTransition(() => {
      setSnapshot(data);
      setStatusMessage(data.status_message || "");
      setWorkspaceRoot(data.workspace_root || workspaceRoot);
      setSelectedRun(data.selected_run || data.runtime?.run_name || "");
      setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
      latestSeqRef.current = Number(data.runtime?.seq || latestSeqRef.current || 0);
      if (data.selected_run || data.runtime?.run_name) {
        setForm((prev) => ({ ...prev, resume_run_name: data.selected_run || data.runtime?.run_name || "" }));
      }
    });
    if (loadDetails && data.selected_run) {
      const detailData = await apiFetch(`/api/session/${escapePath(ctx)}/details?run=${escapePath(data.selected_run)}`);
      startTransition(() => {
        setDetails(detailData);
      });
    }
  }

  async function handleWorkspaceRefresh() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/refresh`, {
      root_path: workspaceRoot,
      lane,
    });
  }

  async function handleWorkspaceOpen() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/open`, {
      root_path: workspaceRoot,
      workspace: snapshot?.workspace_name || "",
      lane,
    }, { loadDetails: view === "monitor" });
  }

  async function handleWorkspaceCreate() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/create`, {
      root_path: workspaceRoot,
      workspace: workspaceName,
      lane,
    }, { loadDetails: view === "monitor" });
    setWorkspaceName("");
  }

  async function handleRunSelect(runName) {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/select`, {
      run_name: runName,
      lane,
    }, { loadDetails: view === "monitor" });
  }

  async function handleChatCreate() {
    await postAndApply(`/api/session/${escapePath(ctx)}/chat/create`, {
      lane,
    });
  }

  async function handleChatSelect(sessionId) {
    await postAndApply(`/api/session/${escapePath(ctx)}/chat/select`, {
      session_id: sessionId,
      lane,
    });
  }

  async function handleStartRun() {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/start`, {
      ...form,
      prompt: form.prompt,
      lane,
      resume_run_name: form.resume_run_name || selectedRun,
    }, { loadDetails: view === "monitor" });
    setForm((prev) => ({ ...prev, prompt: "" }));
  }

  async function handleInterrupt() {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/interrupt`, { lane });
  }

  async function handlePromptSubmit() {
    const prompt = snapshot?.prompt;
    if (!prompt) {
      return;
    }
    await postAndApply(`/api/session/${escapePath(ctx)}/prompt/respond`, {
      prompt_id: prompt.prompt_id || prompt.payload?.prompt_id || "",
      text: promptResponse,
      lane,
      run_name: selectedRun,
    }, { loadDetails: view === "monitor" });
    setPromptResponse("");
  }

  const workspaceOptions = snapshot?.workspaces || [];
  const chatSessionOptions = snapshot?.chat_sessions || [];
  const runOptions = snapshot?.runs || [];
  const cards = (snapshot?.cards || []).filter((card) => {
    if (!deferredSearch.trim()) {
      return true;
    }
    return JSON.stringify(card).toLowerCase().includes(deferredSearch.trim().toLowerCase());
  });
  const live = snapshot?.live_state || {};
  const llm = snapshot?.llm || live.llm || {};
  const graph = snapshot?.graph || {};
  const usage = snapshot?.usage_summary || {};
  const visibleEvents = view === "monitor" ? events : [];
  const liveAssistantMessage = buildLiveAssistantMessage(snapshot, agentTab);
  const chatMessages = buildChatTimeline(snapshot, events, liveAssistantMessage);
  const laneGuide = LANE_GUIDE[lane] || LANE_GUIDE.experiment;
  const todoRows = Array.isArray(live?.todo_rows) && live.todo_rows.length
    ? live.todo_rows
    : (Array.isArray(snapshot?.todo_items) ? snapshot.todo_items.filter(Boolean).map((item) => ({ content: item, status: "pending" })) : []);
  const activeToolcall = live?.active_toolcall || null;
  const recentToolcalls = Array.isArray(live?.recent_toolcalls) ? live.recent_toolcalls : [];
  const availableAgentTabs = agentTabs(live);
  const agentStates = live?.agents && typeof live.agents === "object" ? live.agents : {};
  const selectedAgentState = agentTab !== "ALL" && agentStates[agentTab] && typeof agentStates[agentTab] === "object"
    ? agentStates[agentTab]
    : null;
  const displayedTodos = agentTab === "ALL"
    ? aggregateAgentTodos(live)
    : normalizeAgentTodoRows(agentTab, selectedAgentState);
  const displayedToolTrace = agentTab === "ALL"
    ? aggregateAgentToolTrace(live)
    : toolTraceRowsForAgent(agentTab, selectedAgentState)
      .sort((left, right) => {
        const leftRank = toolStatusRank(left.status);
        const rightRank = toolStatusRank(right.status);
        if (leftRank !== rightRank) {
          return leftRank - rightRank;
        }
        const tsDiff = Number(right.sort_ts || 0) - Number(left.sort_ts || 0);
        if (tsDiff !== 0) {
          return tsDiff;
        }
        return Number(left.local_index || 0) - Number(right.local_index || 0);
      });

  return (
    <main className={`app-shell view-${view}`}>
      <header className="topbar">
        <div className="topbar-brand">
          <div className="topbar-logo">C</div>
          <span className="topbar-title">CatMaster</span>
          <span className="topbar-subtitle">
            {view === "home" ? "Cockpit" : "Monitor"}
          </span>
        </div>
        <nav className="topbar-nav">
          <a className={view === "home" ? "active" : ""} href={snapshot?.ctx ? `/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}` : "/"}>
            Home
          </a>
          <a
            className={view === "monitor" ? "active" : ""}
            href={snapshot?.ctx ? `/monitor/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}&run=${escapePath(selectedRun)}` : "/monitor/"}
          >
            Monitor
          </a>
        </nav>
      </header>

      <div className="status-bar">
        <StatusPill status={snapshot?.run_status} />
        <span className="status-bar-text">{snapshot?.run_status_text || "No active run."}</span>
        {statusMessage ? <span className="status-bar-message">{statusMessage}</span> : null}
      </div>

      <div className={`layout ${view}`}>
        <aside className="left-rail">
          <div className="control-stack">
            <div className="section-head">
              <div className="section-label">Workspace</div>
              <button type="button" className="ghost-btn" onClick={handleWorkspaceRefresh}>Refresh</button>
            </div>
            <label>
              <span>Root</span>
              <input value={workspaceRoot} onChange={(event) => setWorkspaceRoot(event.target.value)} placeholder="Project-space root" />
            </label>
            <label>
              <span>Current workspace</span>
              <select
                value={snapshot?.workspace_name || ""}
                onChange={(event) => {
                  const next = event.target.value;
                  startTransition(() => {
                    setSnapshot((prev) => (prev ? { ...prev, workspace_name: next } : prev));
                  });
                }}
              >
                <option value="">(select workspace)</option>
                {workspaceOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="btn-row">
              <button type="button" onClick={handleWorkspaceOpen}>Open</button>
              <button type="button" className="ghost-btn" onClick={() => setWorkspaceName(snapshot?.workspace_name || "")}>Mirror</button>
            </div>
            <label>
              <span>New workspace</span>
              <input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} placeholder="new workspace" />
            </label>
            <button type="button" onClick={handleWorkspaceCreate}>Create Workspace</button>
          </div>

          <div className="divider" />

          <div className="control-stack">
            <div className="section-head">
              <div className="section-label">Chat Sessions</div>
              <button type="button" className="ghost-btn" onClick={handleChatCreate}>New Chat</button>
            </div>
            <label>
              <span>Current chat</span>
              <select
                value={snapshot?.current_chat_session || ""}
                onChange={(event) => handleChatSelect(event.target.value)}
              >
                <option value="">(select chat session)</option>
                {chatSessionOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="divider" />

          <div className="control-stack">
            <div className="section-label">Runs</div>
            <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter runs..." />
            <label>
              <span>Select run</span>
              <select value={selectedRun} onChange={(event) => handleRunSelect(event.target.value)}>
                <option value="">(select run)</option>
                {runOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
          </div>

          <div className="run-list">
            {cards.map((card) => (
              <RunCard key={card.run_name} card={card} active={card.run_name === selectedRun} onSelect={handleRunSelect} />
            ))}
          </div>
        </aside>

        <section className="center-stage">
          <div className="center-content">
            <div className="center-header">
              <div className="center-header-left">
                <h2>{view === "home" ? "Conversation" : "Execution Stream"}</h2>
                <span className="section-label">{laneGuide.title} lane</span>
              </div>
              <div className="inline-actions">
                {view === "home" ? (
                  <button
                    type="button"
                    className={`ghost-btn ${memoryOpen ? "active" : ""}`}
                    onClick={() => setMemoryOpen((prev) => !prev)}
                  >
                    {memoryOpen ? "Hide Memory" : "Memory"}
                  </button>
                ) : null}
                <button type="button" className="ghost-btn danger" onClick={handleInterrupt}>
                  Interrupt
                </button>
                {view === "monitor" ? (
                  <button type="button" className="ghost-btn" onClick={() => refreshSnapshot(selectedRun)}>
                    Refresh
                  </button>
                ) : null}
              </div>
            </div>

            {view === "monitor" ? (
              <div className="metrics-grid">
                <MetricCard label="Phase" value={live.current_phase || snapshot?.run_status} />
                <MetricCard label="Graph node" value={graph.node || live.current_node} />
                <MetricCard label="Task" value={live.current_task_goal || live.current_task_id} />
                <MetricCard label="Tool" value={live.active_toolcall?.tool || "-"} note={live.active_toolcall?.status || ""} />
                <MetricCard label="Output tokens" value={usage.output_tokens || usage.outputTokens || llm.usage?.output_tokens} />
                <MetricCard label="Reasoning tokens" value={usage.reasoning_tokens || llm.usage?.reasoning_tokens || "-"} />
              </div>
            ) : null}

            <PromptPanel
              prompt={snapshot?.prompt}
              value={promptResponse}
              onChange={setPromptResponse}
              onSubmit={handlePromptSubmit}
              disabled={!snapshot?.can_submit_prompt}
            />

            {view === "home" ? (
              <>
                <p className="lane-info">{laneGuide.summary}</p>
                <ChatThread messages={chatMessages} />
              </>
            ) : (
              <div className="monitor-grid">
                <div className="section-label">Events</div>
                <EventFeed events={visibleEvents} />
              </div>
            )}
          </div>

          {view === "home" ? (
            <div className="composer">
              <div className="composer-fields">
                <label>
                  <span>Lane</span>
                  <select value={lane} onChange={(event) => setLane(event.target.value)}>
                    {["experiment", "research", "writing"].map((item) => (
                      <option key={item} value={item}>{item}</option>
                    ))}
                  </select>
                </label>
                <label>
                  <span>Run mode</span>
                  <select
                    value={form.run_mode}
                    onChange={(event) => setForm((prev) => ({ ...prev, run_mode: event.target.value }))}
                  >
                    <option value="new_run">new_run</option>
                    <option value="resume_selected_run">resume_selected_run</option>
                  </select>
                </label>
              </div>
              <textarea
                value={form.prompt}
                onChange={(event) => setForm((prev) => ({ ...prev, prompt: event.target.value }))}
                placeholder={`Ask the ${laneGuide.title} lane to do one clear thing...`}
              />
              <div className="composer-fields">
                <label>
                  <span>Resume run</span>
                  <select
                    value={form.resume_run_name}
                    onChange={(event) => setForm((prev) => ({ ...prev, resume_run_name: event.target.value }))}
                  >
                    <option value="">(use selected run)</option>
                    {runOptions.map((item) => (
                      <option key={item.value} value={item.value}>{item.label}</option>
                    ))}
                  </select>
                </label>
                <label className="toggle-line">
                  <input
                    type="checkbox"
                    checked={form.proposal_review}
                    onChange={(event) => setForm((prev) => ({ ...prev, proposal_review: event.target.checked }))}
                  />
                  <span>Review proposal before execution</span>
                </label>
              </div>
              <div className="btn-row">
                <button type="button" onClick={handleStartRun}>Start Run</button>
                <button
                  type="button"
                  className="ghost-btn"
                  onClick={() => setForm((prev) => ({ ...prev, resume_run_name: selectedRun }))}
                >
                  Use selected run for resume
                </button>
              </div>
            </div>
          ) : null}
        </section>

        {view === "monitor" ? (
          <aside className="right-rail">
            <div className="section-head">
              <div>
                <div className="section-label">Run details</div>
                <h3 className="section-title">Artifacts & Traces</h3>
              </div>
              <button type="button" className="ghost-btn" onClick={() => refreshSnapshot(selectedRun)}>
                Pull latest
              </button>
            </div>

            <MonitorTabs tab={monitorTab} onChange={setMonitorTab} />
            {monitorTab === "result" ? (
              <CodePane title="Recorded result" helper="run_state result" text={snapshot?.result_text || ""} />
            ) : null}
            {monitorTab === "task" ? (
              <CodePane title="Run state" helper="run_state.json" text={details?.task_state || ""} />
            ) : null}
            {monitorTab === "trace" ? (
              <CodePane
                title="Trace bundle"
                helper="event/tool/patch trace"
                text={[details?.trace_event, details?.trace_tool, details?.trace_patch].filter(Boolean).join("\n\n")}
              />
            ) : null}
            {monitorTab === "artifacts" ? (
              <CodePane title="Artifacts JSON" helper="run_state.artifacts" text={JSON.stringify(details?.artifacts || [], null, 2)} />
            ) : null}
            <UsagePanel usage={usage} />
            <ArtifactPanel details={details} />
          </aside>
        ) : null}

        {view === "home" ? (
          <aside className="right-rail home-sidecar">
            <div className="agent-tabs" role="tablist" aria-label="Live agent trace filters">
              {availableAgentTabs.map((item) => (
                <button
                  key={item.name}
                  type="button"
                  className={`agent-tab ${agentTab === item.name ? "active" : ""} status-${item.status}`}
                  onClick={() => setAgentTab(item.name)}
                >
                  <span>{item.name}</span>
                  {item.name !== "ALL" ? <span className="agent-tab-status">{item.status}</span> : null}
                </button>
              ))}
            </div>
            <TodoPanel todos={agentTab === "ALL" ? (displayedTodos.length ? displayedTodos : todoRows) : displayedTodos} />
            <ToolTracePanel activeToolcall={activeToolcall} recentToolcalls={agentTab === "ALL" ? (displayedToolTrace.length ? displayedToolTrace : recentToolcalls) : displayedToolTrace} />
          </aside>
        ) : null}
      </div>

      {view === "home" ? (
        <MemoryDrawer
          open={memoryOpen}
          workspaceName={memoryPanel.workspace || snapshot?.workspace_name || ""}
          loading={memoryPanel.loading}
          error={memoryPanel.error}
          text={memoryPanel.text}
          onRefresh={refreshMemoryPanel}
          onClose={() => setMemoryOpen(false)}
        />
      ) : null}
    </main>
  );
}

export default App;
