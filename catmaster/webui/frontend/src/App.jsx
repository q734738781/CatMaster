import {
  startTransition,
  useDeferredValue,
  useEffect,
  useMemo,
  useRef,
  useState,
} from "react";
import { Grid } from "gridjs-react";
import Papa from "papaparse";
import Markdown from "react-markdown";
import rehypeKatex from "rehype-katex";
import remarkGfm from "remark-gfm";
import remarkMath from "remark-math";
import "gridjs/dist/theme/mermaid.css";
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
  return !["LLM_CALL_START", "LLM_TOKEN_DELTA", "LLM_REASONING_DELTA"].includes(name);
}

function getAgentExecutionState(agent) {
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

function resolveThinkingAgent(snapshot, agentTab = "ALL") {
  const live = snapshot?.live_state || {};
  const agents = live?.agents && typeof live.agents === "object" ? live.agents : {};
  if (agentTab !== "ALL") {
    const selected = agents[agentTab];
    if (selected && typeof selected === "object" && getAgentExecutionState(selected) === "active") {
      const llm = selected?.llm && typeof selected.llm === "object" ? selected.llm : {};
      const hasText = String(llm.reasoning_text || llm.text || "").trim();
      if (hasText) {
        return { name: agentTab, state: selected };
      }
    }
  }
  const rows = Object.entries(agents)
    .map(([name, state]) => ({ name, state: state && typeof state === "object" ? state : {} }))
    .filter((row) => getAgentExecutionState(row.state) === "active")
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
    if (selected && typeof selected === "object" && getAgentExecutionState(selected) === "active") {
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
    : null;
  const globalLlm = snapshot?.llm || live.llm || {};
  const activeLlm = llm || (normalizeStatusToken(globalLlm?.status) === "running" ? globalLlm : null);
  if (!activeLlm) {
    return null;
  }
  const graph = snapshot?.graph || {};
  let reasoningText = compactText(activeLlm.reasoning_text || "", 1400);
  let draftText = compactText(activeLlm.text || graph.text_preview || "", 1400);
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
  if (prompt.kind === "proposal_review" && Number(payload.revision_count || 0) > 0) {
    parts.push(`Revised proposal ${Number(payload.revision_count)}`);
  }
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
    const revisionCount = Number(prompt?.payload?.revision_count || 0);
    return {
      role: "assistant",
      kind: "proposal",
      badge: revisionCount > 0 ? "revised proposal" : "proposal",
      status: revisionCount > 0 ? `revision ${revisionCount} awaiting approval` : "awaiting approval",
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
  const resultText = String(snapshot?.result_text || "").trim();
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
  const resultText = String(snapshot?.result_text || "").trim();
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
  return getAgentExecutionState(agent);
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
    subagents: ["materials_worker", "ml_worker", "literature_agent"],
  },
  research: {
    title: "Research",
    summary: "Coordinate broader investigation and delegate experiment, writing, literature, or publication-grade peer review only when needed.",
    subagents: ["experiment_specialist", "writing_specialist", "peer_review_specialist", "litreview_agent"],
  },
  writing: {
    title: "Writing",
    summary: "Draft or revise deliverables from existing evidence and compile when needed.",
    subagents: ["writing_worker_agent", "literature_agent"],
  },
  peer_review: {
    title: "Peer Review",
    summary: "Act like a journal editor: locate the manuscript PDF, collect reviewer-style reports, and return an editor decision plus raw reviewer comments.",
    subagents: [],
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

function formatCost(value) {
  if (value === null || value === undefined || value === "") {
    return "-";
  }
  const numeric = Number(value);
  if (!Number.isFinite(numeric)) {
    return String(value);
  }
  return `$${numeric.toFixed(4)}`;
}

function formatCostNote(summary) {
  if (!summary || typeof summary !== "object") {
    return "";
  }
  const source = String(summary.cost_source || "").trim();
  const missing = Number(summary.missing_cost_calls || 0);
  if (source && missing > 0) {
    return `${source} · ${missing} missing`;
  }
  return source;
}

function formatBytes(value) {
  const numeric = Number(value);
  if (!Number.isFinite(numeric) || numeric < 0) {
    return "-";
  }
  if (numeric < 1024) {
    return `${numeric} B`;
  }
  const units = ["KB", "MB", "GB", "TB"];
  let size = numeric / 1024;
  let unitIndex = 0;
  while (size >= 1024 && unitIndex < units.length - 1) {
    size /= 1024;
    unitIndex += 1;
  }
  const digits = size >= 100 ? 0 : size >= 10 ? 1 : 2;
  return `${size.toFixed(digits)} ${units[unitIndex]}`;
}

function formatDateTime(value) {
  if (!value) {
    return "";
  }
  try {
    return new Date(Number(value) * 1000).toLocaleString();
  } catch {
    return "";
  }
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
  const hiddenNames = new Set(["LLM_CALL_START", "LLM_TOKEN_DELTA", "LLM_REASONING_DELTA"]);

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
        if (hiddenNames.has(String(event?.name || ""))) {
          return null;
        }
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
        if (!body) {
          return null;
        }
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
const JSMOL_SCRIPT_SRC = "/static/vendor/jsmol/JSmol.min.js";
const JSMOL_J2S_PATH = "/static/vendor/jsmol/j2s";
const STRUCTURE_DISPLAY_OPTIONS = [
  ["ball-stick", "Ball-stick"],
  ["spacefill", "Spacefill"],
  ["wireframe", "Wireframe"],
];
const STRUCTURE_SUPERCELL_OPTIONS = [
  ["1x1x1", "1x1x1"],
  ["2x2x1", "2x2x1"],
  ["3x3x1", "3x3x1"],
  ["2x2x2", "2x2x2"],
  ["3x3x3", "3x3x3"],
];
const STRUCTURE_MEASUREMENT_OPTIONS = [
  ["inspect", "Inspect atoms"],
  ["distance", "Distance"],
  ["angle", "Angle"],
  ["torsion", "Dihedral"],
];
const STRUCTURE_VECTOR_RADIUS_OPTIONS = [
  ["0.05", "0.05"],
  ["0.1", "0.1"],
  ["0.15", "0.15"],
  ["0.2", "0.2"],
];

function defaultStructureDisplayMode() {
  return "ball-stick";
}

function defaultStructureSupercell() {
  return "1x1x1";
}

function defaultStructureMeasurementMode() {
  return "inspect";
}

function defaultStructureVectorRadius() {
  return "0.05";
}

function defaultStructureVectorScale() {
  return "0.5";
}

function defaultStructureVibrationScale() {
  return "0.2";
}

function defaultStructureVibrationPeriod() {
  return "1";
}

function loadExternalScriptOnce(src, globalName) {
  const stateKey = `__catmaster_script_${globalName || src}`;
  if (window[stateKey]) {
    return window[stateKey];
  }
  window[stateKey] = new Promise((resolve, reject) => {
    const resolveLoadedScript = () => {
      if (globalName && !window[globalName]) {
        reject(new Error(`Loaded script did not expose ${globalName}: ${src}`));
        return;
      }
      resolve(globalName ? window[globalName] : true);
    };
    if (globalName && window[globalName]) {
      resolve(window[globalName]);
      return;
    }
    const existing = document.querySelector(`script[data-catmaster-src="${src}"]`);
    if (existing) {
      existing.addEventListener("load", resolveLoadedScript, { once: true });
      existing.addEventListener("error", () => reject(new Error(`Failed to load script: ${src}`)), { once: true });
      if (existing.dataset.catmasterLoaded === "true") {
        resolveLoadedScript();
      }
      return;
    }
    const script = document.createElement("script");
    script.src = src;
    script.async = true;
    script.dataset.catmasterSrc = src;
    script.onload = () => {
      script.dataset.catmasterLoaded = "true";
      resolveLoadedScript();
    };
    script.onerror = () => reject(new Error(`Failed to load script: ${src}`));
    document.head.appendChild(script);
  });
  return window[stateKey];
}

function escapeJSmolString(value) {
  return String(value || "").replace(/\\/g, "\\\\").replace(/"/g, '\\"');
}

function getJSmolPropertyAsArray(applet, key, value = "") {
  if (!window.Jmol) {
    return null;
  }
  if (typeof window.Jmol.getPropertyAsArray === "function") {
    return window.Jmol.getPropertyAsArray(applet, key, value);
  }
  if (typeof window.Jmol.getPropertyAsJSON === "function") {
    const raw = window.Jmol.getPropertyAsJSON(applet, key, value);
    return raw ? JSON.parse(raw) : null;
  }
  return null;
}

function normalizeSupercell(value) {
  const parts = String(value || defaultStructureSupercell())
    .split("x")
    .map((item) => Math.max(1, Math.min(3, Number.parseInt(item, 10) || 1)));
  while (parts.length < 3) {
    parts.push(1);
  }
  return parts.slice(0, 3);
}

function buildJSmolPackedCellToken(value) {
  const [ax, by, cz] = normalizeSupercell(value);
  return `{${ax} ${by} ${cz}} packed`;
}

function buildJSmolUrlSpecifier(url, fileType) {
  const normalizedUrl = String(url || "").trim();
  const normalizedType = String(fileType || "").trim();
  if (!normalizedUrl) {
    return "";
  }
  return normalizedType ? `${normalizedType}::${normalizedUrl}` : normalizedUrl;
}

function measurementScriptCommand(mode) {
  if (mode === "angle") {
    return "set picking MEASURE ANGLE";
  }
  if (mode === "torsion") {
    return "set picking MEASURE TORSION";
  }
  return "set picking MEASURE DISTANCE";
}

function jmolElementSelector(symbol) {
  const normalized = String(symbol || "").trim().replace(/[^A-Za-z]/g, "");
  return normalized ? `_${normalized}` : "all";
}

function formatJmolPoint(coords) {
  const [x, y, z] = Array.isArray(coords) ? coords : [0, 0, 0];
  return `{${Number(x || 0).toFixed(6)} ${Number(y || 0).toFixed(6)} ${Number(z || 0).toFixed(6)}}`;
}

function buildCustomUnitCellScript(structure, visible) {
  const ids = [
    "cm_uc_1", "cm_uc_2", "cm_uc_3", "cm_uc_4",
    "cm_uc_5", "cm_uc_6", "cm_uc_7", "cm_uc_8",
    "cm_uc_9", "cm_uc_10", "cm_uc_11", "cm_uc_12",
  ];
  const cleanup = ids.map((id) => `draw ${id} delete`);
  const vectors = Array.isArray(structure?.cell_vectors) ? structure.cell_vectors : [];
  if (!visible || vectors.length !== 3) {
    return cleanup;
  }
  const [a, b, c] = vectors.map((vector) => [
    Number(vector?.[0] || 0),
    Number(vector?.[1] || 0),
    Number(vector?.[2] || 0),
  ]);
  const add = (lhs, rhs) => [lhs[0] + rhs[0], lhs[1] + rhs[1], lhs[2] + rhs[2]];
  const origin = [0, 0, 0];
  const ab = add(a, b);
  const ac = add(a, c);
  const bc = add(b, c);
  const abc = add(ab, c);
  const edges = [
    [origin, a],
    [origin, b],
    [origin, c],
    [a, ab],
    [a, ac],
    [b, ab],
    [b, bc],
    [c, ac],
    [c, bc],
    [ab, abc],
    [ac, abc],
    [bc, abc],
  ];
  return [
    ...cleanup,
    "color draw [72,90,115]",
    ...edges.map(([start, end], index) => `draw ${ids[index]} line ${formatJmolPoint(start)} ${formatJmolPoint(end)}`),
  ];
}

function vibrationModeByIndex(structure, modeIndex) {
  const modes = Array.isArray(structure?.vibration_modes) ? structure.vibration_modes : [];
  return modes.find((mode) => String(mode.mode_index) === String(modeIndex)) || modes[0] || null;
}

function collectNativeVibrationFrameMap(modelInfo, vibrationModes = []) {
  const explicitMap = {};
  const modelEntries = [];
  const roots = [];
  if (Array.isArray(modelInfo)) {
    roots.push(...modelInfo);
  } else if (modelInfo && typeof modelInfo === "object") {
    roots.push(modelInfo);
  }
  const stack = [...roots];
  while (stack.length) {
    const node = stack.pop();
    if (!node || typeof node !== "object") {
      continue;
    }
    const models = Array.isArray(node.models) ? node.models : null;
    if (models) {
      for (const model of models) {
        if (!model || typeof model !== "object") {
          continue;
        }
        const modelIndex = Number.parseInt(String(model.modelIndex ?? model.modelNumberIndex ?? ""), 10);
        const modelProperties = model.modelProperties && typeof model.modelProperties === "object" ? model.modelProperties : {};
        const modeValue = model.vibrationalMode ?? modelProperties.vibrationalMode ?? modelProperties.Mode;
        const modeNumber = Number.parseInt(String(modeValue ?? ""), 10);
        if (!Number.isFinite(modelIndex)) {
          continue;
        }
        modelEntries.push({ modelIndex, model });
        if (Number.isFinite(modeNumber)) {
          explicitMap[String(modeNumber)] = modelIndex + 1;
        }
      }
    }
    for (const value of Object.values(node)) {
      if (Array.isArray(value)) {
        stack.push(...value);
      } else if (value && typeof value === "object") {
        stack.push(value);
      }
    }
  }
  if (Object.keys(explicitMap).length) {
    return explicitMap;
  }
  if (!Array.isArray(vibrationModes) || !vibrationModes.length || modelEntries.length < vibrationModes.length) {
    return explicitMap;
  }
  const sortedEntries = [...modelEntries].sort((lhs, rhs) => lhs.modelIndex - rhs.modelIndex);
  const tailEntries = sortedEntries.slice(-vibrationModes.length);
  return Object.fromEntries(
    vibrationModes.map((mode, index) => [String(mode.mode_number), tailEntries[index].modelIndex + 1]),
  );
}

function buildStructureStyleScript({
  periodic,
  displayMode,
  showUnitCell,
  measurementMode,
  showAxes,
  filterElement,
  highlightElement,
  customUnitCellScript,
}) {
  const styleScript = [];
  if (displayMode === "spacefill") {
    styleScript.push("select all", "spacefill 100%", "wireframe off");
  } else if (displayMode === "wireframe") {
    styleScript.push("select all", "spacefill off", periodic ? "wireframe 0.12" : "wireframe 0.15");
  } else {
    styleScript.push(
      "select all",
      periodic ? "spacefill 18%" : "spacefill 23%",
      periodic ? "wireframe 0.12" : "wireframe 0.15",
    );
  }
  styleScript.push(filterElement && filterElement !== "all" ? `display ${jmolElementSelector(filterElement)}` : "display all");
  if (customUnitCellScript?.length) {
    styleScript.push("unitcell off", ...customUnitCellScript);
  } else {
    styleScript.push(showUnitCell && periodic ? "unitcell on" : "unitcell off");
  }
  styleScript.push(showAxes ? "axes on" : "axes off", "color cpk", "selectionHalos off");
  if (highlightElement && highlightElement !== "all") {
    styleScript.push(`select ${jmolElementSelector(highlightElement)}`, "color [255,140,0]");
  }
  styleScript.push("select all");
  if (measurementMode && measurementMode !== "inspect") {
    styleScript.push("set pickingStyle MEASURE ON", measurementScriptCommand(measurementMode));
  } else {
    styleScript.push("measure DELETE");
    styleScript.push("set picking IDENT");
  }
  return styleScript;
}

function buildJSmolLoadScript(structure, options = {}) {
  const body = String(structure?.viewer_text || "").replace(/\r\n/g, "\n");
  const periodic = Boolean(structure?.periodic);
  const displayMode = options.displayMode || defaultStructureDisplayMode();
  const showUnitCell = options.showUnitCell ?? periodic;
  const measurementMode = options.measurementMode || defaultStructureMeasurementMode();
  const customUnitCellScript = buildCustomUnitCellScript(
    structure,
    Boolean(options.showUnitCell)
      && Boolean(structure?.supports_vibration)
      && String(structure?.viewer_source_file_type || "") === "Xyz",
  );
  const packSpecifier = periodic ? ` ${buildJSmolPackedCellToken(options.supercell)}` : "";
  const loadCommand = options.loadCommand || (
    structure?.viewer_source_mode === "url" && structure?.viewer_source_url
      ? `load "${escapeJSmolString(buildJSmolUrlSpecifier(structure.viewer_source_url, structure?.viewer_source_file_type))}"${packSpecifier}`
      : [
        'load DATA "model"',
        body,
        `END "model"${packSpecifier}`,
      ].join("\n")
  );
  return [
    loadCommand,
    "background white",
    "set antialiasDisplay true",
    "set zoomlarge false",
    "frank off",
    ...buildStructureStyleScript({
      periodic,
      displayMode,
      showUnitCell,
      measurementMode,
      showAxes: Boolean(options.showAxes),
      filterElement: options.filterElement || "all",
      highlightElement: options.highlightElement || "all",
      customUnitCellScript,
    }),
    "zoom 120",
  ].join("\n");
}

function buildStructureControlScript(structure, options = {}) {
  const customUnitCellScript = buildCustomUnitCellScript(
    structure,
    Boolean(options.showUnitCell)
      && Boolean(structure?.supports_vibration)
      && String(structure?.viewer_source_file_type || "") === "Xyz",
  );
  return buildStructureStyleScript({
    periodic: Boolean(structure?.periodic),
    displayMode: options.displayMode || defaultStructureDisplayMode(),
    showUnitCell: options.showUnitCell ?? Boolean(structure?.periodic),
    measurementMode: options.measurementMode || defaultStructureMeasurementMode(),
    showAxes: Boolean(options.showAxes),
    filterElement: options.filterElement || "all",
    highlightElement: options.highlightElement || "all",
    customUnitCellScript,
  }).join("\n");
}

function buildStructureResetScript(structure, options = {}) {
  return [
    "reset",
    "zoom 120",
  ].join("\n");
}

function buildStructureVibrationScript({
  modeIndex,
  vectorsVisible,
  vectorRadius,
  vectorScale,
  vibrationScale,
  vibrationPeriod,
  vibrationPlaying,
  nativeModeLoad,
  nativeFrameNumber,
}) {
  const frameNumber = Math.max(1, (Number.parseInt(String(modeIndex || "0"), 10) || 0) + 1);
  return [
    ...(nativeModeLoad
      ? (Number.isFinite(nativeFrameNumber) && nativeFrameNumber > 0 ? [`frame ${nativeFrameNumber}`] : [])
      : [`frame ${frameNumber}`]),
    vectorsVisible ? `vectors ${String(vectorRadius || defaultStructureVectorRadius())}` : "vectors off",
    `set vectorScale ${String(vectorScale || defaultStructureVectorScale())}`,
    `set vibrationScale ${String(vibrationScale || defaultStructureVibrationScale())}`,
    `set vibrationPeriod ${String(vibrationPeriod || defaultStructureVibrationPeriod())}`,
    vibrationPlaying ? "vibration on" : "vibration off",
  ].join("\n");
}

function parseJSmolAtomIndex(value) {
  const normalized = Number.parseInt(String(value ?? "").trim(), 10);
  return Number.isFinite(normalized) ? normalized : null;
}

function parseJSmolAtomInfo(rawValue, atomIndex) {
  const text = String(rawValue || "").trim();
  if (!text) {
    return null;
  }
  const match = text.match(/^(.*?)\s+#(\d+)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)\s+(-?\d+(?:\.\d+)?)(?:\s|$)/);
  return {
    label: (match?.[1] || text).trim(),
    atomNumber: match ? Number.parseInt(match[2], 10) : null,
    index: atomIndex,
    x: match ? Number.parseFloat(match[3]) : null,
    y: match ? Number.parseFloat(match[4]) : null,
    z: match ? Number.parseFloat(match[5]) : null,
    raw: text,
  };
}

function parseJSmolMeasurement(args) {
  const atoms = String(args?.[1] || args?.[2] || "").trim();
  const status = String(args?.[3] || "").trim();
  const rawValue = Array.isArray(args?.[4]) ? args[4][0] : args?.[4];
  const numericValue = Number.parseFloat(String(rawValue ?? ""));
  const bracketMatch = atoms.match(/\[([^\]]+)\]/);
  const atomLabels = bracketMatch
    ? bracketMatch[1].split(",").map((item) => item.trim()).filter(Boolean)
    : [];
  const typeByCount = { 2: "distance", 3: "angle", 4: "torsion" };
  const type = typeByCount[atomLabels.length] || "distance";
  return {
    atoms,
    atomLabels,
    status: status === "measureCompleted" ? "completed" : "pending",
    value: Number.isFinite(numericValue) ? numericValue : null,
    type,
  };
}

function formatStructureMeasurement(measurement) {
  if (!measurement) {
    return "Measurement mode is off.";
  }
  if (measurement.value === null) {
    return measurement.status === "pending" ? "Select the remaining atoms to finish the measurement." : "Measurement unavailable.";
  }
  const labels = {
    distance: "Distance",
    angle: "Angle",
    torsion: "Dihedral",
  };
  const unit = measurement.type === "distance" ? "A" : "deg";
  return `${labels[measurement.type] || "Measurement"}: ${measurement.value.toFixed(3)} ${unit}`;
}

function formatStructureCoordinate(value) {
  return Number.isFinite(value) ? value.toFixed(3) : "-";
}

function formatElementOptionLabel(element, counts) {
  const count = Number(counts?.[element] || 0);
  return count > 0 ? `${element} (${count})` : element;
}

function downloadStructureViewport(hostNode, filenameBase = "structure-view") {
  const canvas = hostNode?.querySelector("canvas");
  if (!(canvas instanceof HTMLCanvasElement)) {
    throw new Error("JSmol canvas is not ready yet.");
  }
  const anchor = document.createElement("a");
  anchor.download = `${String(filenameBase || "structure-view").replace(/\.[^.]+$/, "")}.png`;
  if (typeof canvas.toBlob === "function") {
    canvas.toBlob((blob) => {
      if (!blob) {
        return;
      }
      const url = URL.createObjectURL(blob);
      anchor.href = url;
      anchor.click();
      setTimeout(() => URL.revokeObjectURL(url), 0);
    }, "image/png");
    return;
  }
  anchor.href = canvas.toDataURL("image/png");
  anchor.click();
}

function isCsvPreview(preview) {
  const mimeType = String(preview?.mime_type || "").toLowerCase();
  const path = String(preview?.path || preview?.name || "").toLowerCase();
  return mimeType.includes("csv") || path.endsWith(".csv");
}

function buildCsvPreviewModel(text) {
  const parsed = Papa.parse(String(text || ""), {
    skipEmptyLines: "greedy",
  });
  const sourceRows = Array.isArray(parsed.data) ? parsed.data.filter((row) => Array.isArray(row) && row.length) : [];
  if (!sourceRows.length) {
    return {
      columns: [],
      rows: [],
      errors: Array.isArray(parsed.errors) ? parsed.errors : [],
    };
  }
  const width = sourceRows.reduce((maxWidth, row) => Math.max(maxWidth, row.length), 0);
  const normalizeRow = (row) => Array.from({ length: width }, (_value, index) => String(row?.[index] ?? ""));
  const normalizedRows = sourceRows.map(normalizeRow);
  const hasHeaderRow = normalizedRows.length > 1;
  const headerRow = hasHeaderRow
    ? normalizedRows[0]
    : Array.from({ length: width }, (_value, index) => `Column ${index + 1}`);
  const dataRows = hasHeaderRow ? normalizedRows.slice(1) : normalizedRows;
  const duplicateLabels = {};
  const columns = [
    "#",
    ...headerRow.map((label, index) => {
      const baseLabel = String(label || "").trim() || `Column ${index + 1}`;
      const nextCount = Number(duplicateLabels[baseLabel] || 0) + 1;
      duplicateLabels[baseLabel] = nextCount;
      return nextCount > 1 ? `${baseLabel} (${nextCount})` : baseLabel;
    }),
  ];
  const rows = dataRows.map((row, index) => [String(index + 1), ...row]);
  return {
    columns,
    rows,
    errors: Array.isArray(parsed.errors) ? parsed.errors : [],
  };
}

function CsvPreview({ preview }) {
  const csvModel = useMemo(() => buildCsvPreviewModel(preview?.preview_text || ""), [preview?.preview_text]);

  if (!csvModel.columns.length) {
    return <div className="memory-drawer-note">CSV preview is empty.</div>;
  }

  return (
    <div className="files-csv-preview">
      <div className="files-csv-meta">
        <span>{csvModel.rows.length} row(s)</span>
        <span>{Math.max(0, csvModel.columns.length - 1)} column(s)</span>
      </div>
      <Grid
        key={`${preview?.path || preview?.name || "csv"}-${preview?.modified_ts || ""}`}
        data={csvModel.rows}
        columns={csvModel.columns}
        sort
        search={csvModel.rows.length > 10}
        pagination={csvModel.rows.length > 25 ? { enabled: true, limit: 25 } : false}
        fixedHeader
        height="420px"
        className={{
          container: "files-csv-grid",
          table: "files-csv-grid-table",
        }}
      />
      {csvModel.errors.length ? (
        <div className="memory-drawer-note">
          CSV preview parsed with {csvModel.errors.length} warning(s); incomplete trailing rows may be caused by preview truncation.
        </div>
      ) : null}
    </div>
  );
}

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
  const isProposalReview = prompt.kind === "proposal_review";
  const revisionCount = Number(payload.revision_count || 0);
  const placeholder = isProposalReview
    ? 'Type "approve" to continue, or enter feedback to request a revised proposal.'
    : "Provide feedback, approval, or revised guidance.";
  const body = [
    isProposalReview && revisionCount > 0 ? `Revised proposal ${revisionCount}` : "",
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
    <section className={`prompt-panel ${isProposalReview ? "proposal-review" : ""}`}>
      <div className="section-label">{isProposalReview ? "Proposal Review Required" : "Human Input Required"}</div>
      <div className="prompt-meta">{joinItems([prompt.kind, payload.run_id, payload.prompt_id || prompt.prompt_id])}</div>
      <pre className="code-pane">{body || "(empty prompt payload)"}</pre>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder={placeholder}
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

function StructureViewer({ structure }) {
  const hostRef = useRef(null);
  const appletRef = useRef(null);
  const loadedSupercellRef = useRef(defaultStructureSupercell());
  const [viewerError, setViewerError] = useState("");
  const [viewerReady, setViewerReady] = useState(false);
  const [displayMode, setDisplayMode] = useState(defaultStructureDisplayMode());
  const [showUnitCell, setShowUnitCell] = useState(Boolean(structure?.periodic));
  const [showAxes, setShowAxes] = useState(false);
  const [supercell, setSupercell] = useState(defaultStructureSupercell());
  const [filterElement, setFilterElement] = useState("all");
  const [highlightElement, setHighlightElement] = useState("all");
  const [interactionMode, setInteractionMode] = useState(defaultStructureMeasurementMode());
  const [pickedAtom, setPickedAtom] = useState(null);
  const [measurement, setMeasurement] = useState(null);
  const [animationPlaying, setAnimationPlaying] = useState(false);
  const [vibrationPlaying, setVibrationPlaying] = useState(false);
  const [selectedVibrationMode, setSelectedVibrationMode] = useState("0");
  const [vectorsVisible, setVectorsVisible] = useState(true);
  const [vectorRadius, setVectorRadius] = useState(defaultStructureVectorRadius());
  const [vectorScale, setVectorScale] = useState(defaultStructureVectorScale());
  const [vibrationScale, setVibrationScale] = useState(defaultStructureVibrationScale());
  const [vibrationPeriod, setVibrationPeriod] = useState(defaultStructureVibrationPeriod());
  const [nativeVibrationFrames, setNativeVibrationFrames] = useState({});
  const elementOptions = Array.isArray(structure?.elements) ? structure.elements : [];
  const supportsAnimation = Boolean(structure?.supports_animation);
  const supportsVibration = Boolean(structure?.supports_vibration);
  const supportsSupercell = Boolean(structure?.periodic) && !supportsAnimation && !supportsVibration;
  const supportsUnitCell = Boolean(structure?.periodic);
  const vibrationModes = Array.isArray(structure?.vibration_modes) ? structure.vibration_modes : [];
  const activeVibrationMode = vibrationModeByIndex(structure, selectedVibrationMode);
  const isNativeVibrationSource = String(structure?.viewer_source_file_type || "") === "VaspOutcar";
  const activeNativeFrame = activeVibrationMode
    ? nativeVibrationFrames[String(activeVibrationMode.mode_number)] ?? null
    : null;

  useEffect(() => {
    loadedSupercellRef.current = defaultStructureSupercell();
    setViewerReady(false);
    setDisplayMode(defaultStructureDisplayMode());
    setShowUnitCell(Boolean(structure?.periodic));
    setShowAxes(false);
    setSupercell(defaultStructureSupercell());
    setFilterElement("all");
    setHighlightElement("all");
    setInteractionMode(defaultStructureMeasurementMode());
    setPickedAtom(null);
    setMeasurement(null);
    setAnimationPlaying(false);
    setVibrationPlaying(false);
    setSelectedVibrationMode("0");
    setVectorsVisible(true);
    setVectorRadius(defaultStructureVectorRadius());
    setVectorScale(defaultStructureVectorScale());
    setVibrationScale(defaultStructureVibrationScale());
    setVibrationPeriod(defaultStructureVibrationPeriod());
    setNativeVibrationFrames({});
    setViewerError("");
  }, [structure, supportsVibration]);

  useEffect(() => {
    if (!viewerReady || !appletRef.current || !window.Jmol) {
      return;
    }
    window.Jmol.script(
      appletRef.current,
      buildStructureControlScript(structure, {
        displayMode,
        showUnitCell: showUnitCell && supportsUnitCell,
        showAxes,
        filterElement,
        highlightElement,
        measurementMode: interactionMode,
      }),
    );
  }, [displayMode, filterElement, highlightElement, interactionMode, showAxes, showUnitCell, structure, supportsUnitCell, viewerReady]);

  useEffect(() => {
    if (!viewerReady || !appletRef.current || !window.Jmol || (!structure?.viewer_text && !(structure?.viewer_source_mode === "url" && structure?.viewer_source_url))) {
      return;
    }
    if (loadedSupercellRef.current === supercell) {
      return;
    }
    loadedSupercellRef.current = supercell;
    setPickedAtom(null);
    setMeasurement(null);
    window.Jmol.script(
      appletRef.current,
      buildJSmolLoadScript(structure, {
        displayMode,
        showUnitCell: showUnitCell && supportsUnitCell,
        showAxes,
        supercell,
        filterElement,
        highlightElement,
        measurementMode: interactionMode,
      }),
    );
  }, [displayMode, filterElement, highlightElement, interactionMode, showAxes, showUnitCell, structure, supercell, supportsUnitCell, viewerReady]);

  useEffect(() => {
    if (!viewerReady || !appletRef.current || !window.Jmol || !supportsVibration || !isNativeVibrationSource) {
      return;
    }
    let cancelled = false;
    let attempts = 0;
    const maxAttempts = 10;

    const collectFrames = () => {
      if (cancelled || !appletRef.current) {
        return;
      }
      try {
        const modelInfo = getJSmolPropertyAsArray(appletRef.current, "modelInfo", "");
        const frameMap = collectNativeVibrationFrameMap(modelInfo, vibrationModes);
        if (Object.keys(frameMap).length || attempts >= maxAttempts) {
          setNativeVibrationFrames(frameMap);
          return;
        }
      } catch (_error) {
        if (attempts >= maxAttempts) {
          setNativeVibrationFrames({});
          return;
        }
      }
      attempts += 1;
      window.setTimeout(collectFrames, 150);
    };

    collectFrames();
    return () => {
      cancelled = true;
    };
  }, [isNativeVibrationSource, structure, supportsVibration, vibrationModes, viewerReady]);

  useEffect(() => {
    if (!supportsVibration || !viewerReady || !appletRef.current || !window.Jmol || !vibrationModes.length) {
      return;
    }
    if (isNativeVibrationSource && !Number.isFinite(activeNativeFrame)) {
      return;
    }
    const vibrationScript = buildStructureVibrationScript({
      modeIndex: selectedVibrationMode,
      vectorsVisible,
      vectorRadius,
      vectorScale,
      vibrationScale,
      vibrationPeriod,
      vibrationPlaying,
      nativeModeLoad: isNativeVibrationSource,
      nativeFrameNumber: activeNativeFrame,
    });
    const redrawScript = !isNativeVibrationSource
      ? buildStructureControlScript(structure, {
        displayMode,
        showUnitCell: showUnitCell && supportsUnitCell,
        showAxes,
        filterElement,
        highlightElement,
        measurementMode: interactionMode,
      })
      : "";
    window.Jmol.script(
      appletRef.current,
      redrawScript ? `${vibrationScript}\n${redrawScript}` : vibrationScript,
    );
  }, [
    activeNativeFrame,
    displayMode,
    filterElement,
    highlightElement,
    interactionMode,
    isNativeVibrationSource,
    selectedVibrationMode,
    showAxes,
    showUnitCell,
    supportsVibration,
    supportsUnitCell,
    structure,
    vectorRadius,
    vectorScale,
    vectorsVisible,
    vibrationModes.length,
    vibrationPeriod,
    vibrationPlaying,
    vibrationScale,
    viewerReady,
  ]);

  useEffect(() => {
    let cancelled = false;
    let appletId = "";
    let pickCallbackName = "";
    let measureCallbackName = "";

    async function renderStructure() {
      if (!hostRef.current || (!structure?.viewer_text && !(structure?.viewer_source_mode === "url" && structure?.viewer_source_url))) {
        return;
      }
      try {
        const Jmol = await loadExternalScriptOnce(JSMOL_SCRIPT_SRC, "Jmol");
        if (cancelled || !hostRef.current) {
          return;
        }
        appletId = `catmaster_jmol_${Math.random().toString(36).slice(2, 10)}`;
        pickCallbackName = `${appletId}_pick_callback`;
        measureCallbackName = `${appletId}_measure_callback`;
        loadedSupercellRef.current = defaultStructureSupercell();
        window[pickCallbackName] = (...args) => {
          if (cancelled) {
            return;
          }
          setPickedAtom(parseJSmolAtomInfo(args[1], parseJSmolAtomIndex(args[2])));
        };
        window[measureCallbackName] = (...args) => {
          if (cancelled) {
            return;
          }
          setMeasurement(parseJSmolMeasurement(args));
        };
        hostRef.current.innerHTML = "";
        Jmol.setDocument(0);
        const info = {
          width: "100%",
          height: "100%",
          debug: false,
          color: "#ffffff",
          addSelectionOptions: false,
          use: "HTML5",
          j2sPath: JSMOL_J2S_PATH,
          disableJ2SLoadMonitor: true,
          disableInitialConsole: true,
          pickCallback: pickCallbackName,
          measureCallback: measureCallbackName,
          script: buildJSmolLoadScript(structure, {
            displayMode: defaultStructureDisplayMode(),
            showUnitCell: Boolean(structure?.periodic),
            showAxes: false,
            supercell: defaultStructureSupercell(),
            filterElement: "all",
            highlightElement: "all",
            measurementMode: defaultStructureMeasurementMode(),
          }),
          readyFunction: () => {
            if (!cancelled) {
              appletRef.current = window[appletId] || Jmol._applets?.[appletId] || null;
              setViewerReady(true);
              setViewerError("");
              if (String(structure?.viewer_source_file_type || "") === "VaspPoscar") {
                window.setTimeout(() => {
                  if (cancelled || !appletRef.current || !window.Jmol) {
                    return;
                  }
                  window.Jmol.script(appletRef.current, buildStructureResetScript(structure));
                }, 1000);
              }
            }
          },
        };
        hostRef.current.innerHTML = Jmol.getAppletHtml(appletId, info);
        const shell = hostRef.current.firstElementChild;
        if (shell instanceof HTMLElement) {
          shell.style.width = "100%";
          shell.style.height = "100%";
          shell.style.position = "relative";
        }
        hostRef.current.querySelectorAll("[id$='_appletdiv'], [id$='_infotablediv']").forEach((element) => {
          if (element instanceof HTMLElement) {
            element.style.maxWidth = "100%";
            element.style.maxHeight = "100%";
          }
        });
        setViewerError("");
      } catch (error) {
        if (!cancelled) {
          setViewerReady(false);
          setViewerError(String(error?.message || error));
        }
      }
    }

    renderStructure();
    return () => {
      cancelled = true;
      setViewerReady(false);
      appletRef.current = null;
      if (hostRef.current) {
        hostRef.current.innerHTML = "";
      }
      if (appletId && window.Jmol?._applets?.[appletId]) {
        delete window.Jmol._applets[appletId];
      }
      if (appletId && window[appletId]) {
        delete window[appletId];
      }
      if (pickCallbackName && window[pickCallbackName]) {
        delete window[pickCallbackName];
      }
      if (measureCallbackName && window[measureCallbackName]) {
        delete window[measureCallbackName];
      }
    };
  }, [structure, supportsVibration]);

  return (
    <div className="file-structure-viewer-shell">
      <div className="file-structure-toolbar">
        <div className="file-structure-toolbar-group">
          <label className="file-structure-field">
            <span>Display</span>
            <select value={displayMode} onChange={(event) => setDisplayMode(event.target.value)}>
              {STRUCTURE_DISPLAY_OPTIONS.map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <label className={`file-structure-field ${!structure?.periodic ? "disabled" : ""}`}>
            <span>Supercell</span>
            <select
              value={supercell}
              onChange={(event) => setSupercell(event.target.value)}
              disabled={!supportsSupercell}
            >
              {STRUCTURE_SUPERCELL_OPTIONS.map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className={`ghost-btn ${showUnitCell ? "active" : ""}`}
            disabled={!supportsUnitCell}
            onClick={() => setShowUnitCell((value) => !value)}
          >
            Unit cell
          </button>
          <button
            type="button"
            className={`ghost-btn ${showAxes ? "active" : ""}`}
            onClick={() => setShowAxes((value) => !value)}
          >
            Axes
          </button>
        </div>
        <div className="file-structure-toolbar-group">
          <label className="file-structure-field">
            <span>Filter</span>
            <select value={filterElement} onChange={(event) => setFilterElement(event.target.value)}>
              <option value="all">All elements</option>
              {elementOptions.map((element) => (
                <option key={element} value={element}>{formatElementOptionLabel(element, structure?.element_counts)}</option>
              ))}
            </select>
          </label>
          <label className="file-structure-field">
            <span>Highlight</span>
            <select value={highlightElement} onChange={(event) => setHighlightElement(event.target.value)}>
              <option value="all">None</option>
              {elementOptions.map((element) => (
                <option key={element} value={element}>{formatElementOptionLabel(element, structure?.element_counts)}</option>
              ))}
            </select>
          </label>
        </div>
        <div className="file-structure-toolbar-group">
          {supportsAnimation ? (
            <>
              <button
                type="button"
                className="ghost-btn"
                disabled={!viewerReady || !window.Jmol || !appletRef.current}
                onClick={() => {
                  setAnimationPlaying(false);
                  window.Jmol.script(appletRef.current, "animation OFF; frame PREVIOUS");
                }}
              >
                Prev frame
              </button>
              <button
                type="button"
                className={`ghost-btn ${animationPlaying ? "active" : ""}`}
                disabled={!viewerReady || !window.Jmol || !appletRef.current}
                onClick={() => {
                  const next = !animationPlaying;
                  setAnimationPlaying(next);
                  setVibrationPlaying(false);
                  window.Jmol.script(
                    appletRef.current,
                    next ? "vibration off; animation MODE LOOP; animation FPS 10; animation ON" : "animation OFF",
                  );
                }}
              >
                {animationPlaying ? "Pause" : "Play"}
              </button>
              <button
                type="button"
                className="ghost-btn"
                disabled={!viewerReady || !window.Jmol || !appletRef.current}
                onClick={() => {
                  setAnimationPlaying(false);
                  window.Jmol.script(appletRef.current, "animation OFF; frame NEXT");
                }}
              >
                Next frame
              </button>
            </>
          ) : null}
          {supportsVibration ? (
            <>
              <label className="file-structure-field">
                <span>Mode</span>
                <select value={selectedVibrationMode} onChange={(event) => setSelectedVibrationMode(event.target.value)}>
                  {vibrationModes.map((mode) => (
                    <option key={mode.mode_index} value={String(mode.mode_index)}>{mode.label}</option>
                  ))}
                </select>
              </label>
              <button
                type="button"
                className={`ghost-btn ${vibrationPlaying ? "active" : ""}`}
                disabled={!viewerReady || !window.Jmol || !appletRef.current || !vibrationModes.length}
                onClick={() => {
                  setAnimationPlaying(false);
                  setVibrationPlaying((value) => !value);
                }}
              >
                {vibrationPlaying ? "Stop vibration" : "Play vibration"}
              </button>
              <button
                type="button"
                className={`ghost-btn ${vectorsVisible ? "active" : ""}`}
                disabled={!viewerReady || !window.Jmol || !appletRef.current || !vibrationModes.length}
                onClick={() => setVectorsVisible((value) => !value)}
              >
                {vectorsVisible ? "Hide vectors" : "Show vectors"}
              </button>
              <label className="file-structure-field">
                <span>Vector radius</span>
                <select value={vectorRadius} onChange={(event) => setVectorRadius(event.target.value)}>
                  {STRUCTURE_VECTOR_RADIUS_OPTIONS.map(([value, label]) => (
                    <option key={value} value={value}>{label}</option>
                  ))}
                </select>
              </label>
              <label className="file-structure-field">
                <span>Vector scale</span>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={vectorScale}
                  onChange={(event) => setVectorScale(event.target.value)}
                />
              </label>
              <label className="file-structure-field">
                <span>Amplitude</span>
                <input
                  type="number"
                  min="0"
                  step="0.1"
                  value={vibrationScale}
                  onChange={(event) => setVibrationScale(event.target.value)}
                />
              </label>
              <label className="file-structure-field">
                <span>Period</span>
                <input
                  type="number"
                  min="0.1"
                  step="0.1"
                  value={vibrationPeriod}
                  onChange={(event) => setVibrationPeriod(event.target.value)}
                />
              </label>
              <button
                type="button"
                className="ghost-btn"
                disabled={!viewerReady || !window.Jmol || !appletRef.current || !vibrationModes.length}
                onClick={() => {
                  setAnimationPlaying(false);
                  setVibrationPlaying(false);
                  setVectorsVisible(true);
                  setVectorRadius(defaultStructureVectorRadius());
                  setVectorScale(defaultStructureVectorScale());
                  setVibrationScale(defaultStructureVibrationScale());
                  setVibrationPeriod(defaultStructureVibrationPeriod());
                }}
              >
                Reset vibration
              </button>
            </>
          ) : null}
        </div>
        <div className="file-structure-toolbar-group">
          <label className="file-structure-field">
            <span>Measure</span>
            <select value={interactionMode} onChange={(event) => {
              setMeasurement(null);
              setInteractionMode(event.target.value);
            }}>
              {STRUCTURE_MEASUREMENT_OPTIONS.map(([value, label]) => (
                <option key={value} value={value}>{label}</option>
              ))}
            </select>
          </label>
          <button
            type="button"
            className="ghost-btn"
            disabled={!viewerReady || !window.Jmol || !appletRef.current}
            onClick={() => {
              setMeasurement(null);
              setInteractionMode(defaultStructureMeasurementMode());
              setVibrationPlaying(false);
              window.Jmol.script(appletRef.current, "measure DELETE; set picking IDENT");
            }}
          >
            Clear measures
          </button>
          <button
            type="button"
            className="ghost-btn"
            disabled={!viewerReady || !window.Jmol || !appletRef.current}
            onClick={() => window.Jmol.script(
              appletRef.current,
              buildStructureResetScript(structure, {
                displayMode,
                showUnitCell: showUnitCell && supportsUnitCell,
                showAxes,
                filterElement,
                highlightElement,
                measurementMode: interactionMode,
              }),
            )}
          >
            Reset view
          </button>
          <button
            type="button"
            className="ghost-btn"
            disabled={!viewerReady}
            onClick={() => {
              try {
                downloadStructureViewport(hostRef.current, structure?.formula || structure?.viewer_format || "structure");
              } catch (error) {
                setViewerError(String(error?.message || error));
              }
            }}
          >
            Export PNG
          </button>
        </div>
      </div>
      <div ref={hostRef} className="file-structure-viewer" />
      <div className="file-structure-status-grid">
        <div className="file-structure-status-card">
          <div className="section-label">Picked Atom</div>
          {pickedAtom ? (
            <div className="file-structure-status-body">
              <div className="file-structure-status-strong">{pickedAtom.label}</div>
              <div>Atom #{pickedAtom.atomNumber ?? "-"}</div>
              <div>
                ({formatStructureCoordinate(pickedAtom.x)}, {formatStructureCoordinate(pickedAtom.y)}, {formatStructureCoordinate(pickedAtom.z)})
              </div>
            </div>
          ) : (
            <div className="file-structure-status-body muted">Click an atom to inspect its label and coordinates.</div>
          )}
        </div>
        <div className="file-structure-status-card">
          <div className="section-label">Measurement</div>
          <div className="file-structure-status-body">
            <div className="file-structure-status-strong">
              {interactionMode === "inspect" ? "Measurement mode off" : `${interactionMode} mode on`}
            </div>
            <div>{formatStructureMeasurement(measurement)}</div>
            {measurement?.atoms ? <div className="muted">{measurement.atoms}</div> : null}
          </div>
        </div>
        <div className="file-structure-status-card">
          <div className="section-label">Selection</div>
          <div className="file-structure-status-body">
            <div className="file-structure-status-strong">
              {filterElement === "all" ? "Showing all elements" : `Only ${filterElement}`}
            </div>
            <div>{highlightElement === "all" ? "No highlighted element" : `Highlighting ${highlightElement}`}</div>
            <div>{showAxes ? "Axes visible" : "Axes hidden"}</div>
          </div>
        </div>
        {structure?.periodic ? (
          <div className="file-structure-status-card">
            <div className="section-label">Cell</div>
            <div className="file-structure-status-body">
              <div className="file-structure-status-strong">
                {showUnitCell ? "Unit cell visible" : "Unit cell hidden"}
              </div>
              {Array.isArray(structure?.cell_lengths) && structure.cell_lengths.length === 3 ? (
                <div>
                  a={formatStructureCoordinate(structure.cell_lengths[0])} b={formatStructureCoordinate(structure.cell_lengths[1])} c={formatStructureCoordinate(structure.cell_lengths[2])}
                </div>
              ) : null}
              {Array.isArray(structure?.cell_angles) && structure.cell_angles.length === 3 ? (
                <div>
                  alpha={formatStructureCoordinate(structure.cell_angles[0])} beta={formatStructureCoordinate(structure.cell_angles[1])} gamma={formatStructureCoordinate(structure.cell_angles[2])}
                </div>
              ) : null}
            </div>
          </div>
        ) : null}
        {(supportsAnimation || supportsVibration) ? (
          <div className="file-structure-status-card">
          <div className="section-label">Dynamics</div>
          <div className="file-structure-status-body">
            <div className="file-structure-status-strong">
              {supportsAnimation ? `${structure?.frame_count || 0} trajectory frame(s)` : "No trajectory"}
            </div>
            {supportsAnimation ? <div>{animationPlaying ? "Trajectory playing" : "Trajectory paused"}</div> : null}
            {supportsVibration ? (
              <div>
                {activeVibrationMode ? activeVibrationMode.label : `${vibrationModes.length} vibration mode(s) available`}
              </div>
            ) : null}
            {supportsVibration ? <div>{vibrationPlaying ? "Vibration running" : "Vibration paused"}</div> : null}
            {supportsVibration ? <div>{vectorsVisible ? "Vectors visible" : "Vectors hidden"}</div> : null}
            {supportsVibration ? <div>radius {vectorRadius} / vector {vectorScale} / amplitude {vibrationScale} / period {vibrationPeriod}</div> : null}
            {structure?.frames_truncated ? <div className="muted">Animation source was capped for preview size.</div> : null}
          </div>
        </div>
      ) : null}
      </div>
      {viewerError ? <div className="memory-drawer-note error">{viewerError}</div> : null}
    </div>
  );
}

function FileTreeNode({
  node,
  depth,
  expandedDirs,
  treeNodes,
  treeLoading,
  onToggle,
  onSelect,
  selectedPath,
}) {
  const isDirectory = node.node_type === "directory";
  const expanded = Boolean(expandedDirs[node.path]);
  const selected = selectedPath === node.path;
  const children = treeNodes[node.path] || [];
  const loading = Boolean(treeLoading[node.path]);

  return (
    <div className="file-tree-branch">
      <div className={`file-tree-row ${selected ? "selected" : ""}`} style={{ paddingLeft: `${depth * 16}px` }}>
        <button
          type="button"
          className={`file-tree-toggle ${!isDirectory ? "leaf" : ""}`}
          onClick={() => {
            if (isDirectory) {
              onToggle(node);
            }
          }}
          disabled={!isDirectory}
          aria-label={isDirectory ? (expanded ? "Collapse directory" : "Expand directory") : "File"}
        >
          {isDirectory ? (expanded ? "-" : "+") : ""}
        </button>
        <button
          type="button"
          className={`file-tree-label kind-${node.preview_kind || node.node_type}`}
          onClick={() => onSelect(node)}
        >
          <span className="file-tree-name">{node.name}</span>
          {node.node_type === "file" ? <span className="file-tree-size">{formatBytes(node.size)}</span> : null}
        </button>
      </div>
      {isDirectory && expanded ? (
        <div className="file-tree-children">
          {loading ? <div className="file-tree-note">Loading...</div> : null}
          {!loading && !children.length ? <div className="file-tree-note">Empty directory.</div> : null}
          {!loading
            ? children.map((child) => (
              <FileTreeNode
                key={child.path || child.name}
                node={child}
                depth={depth + 1}
                expandedDirs={expandedDirs}
                treeNodes={treeNodes}
                treeLoading={treeLoading}
                onToggle={onToggle}
                onSelect={onSelect}
                selectedPath={selectedPath}
              />
            ))
            : null}
        </div>
      ) : null}
    </div>
  );
}

function FileTree({
  treeNodes,
  expandedDirs,
  treeLoading,
  selectedPath,
  error,
  onToggle,
  onSelect,
}) {
  const roots = treeNodes[""] || [];
  return (
    <section className="files-panel files-tree-panel">
      <div className="section-head">
        <div>
          <div className="section-label">Tree</div>
          <h3 className="section-title">Workspace files</h3>
        </div>
      </div>
      {error ? <div className="memory-drawer-note error">{error}</div> : null}
      {!error && !roots.length && !treeLoading[""] ? (
        <div className="file-tree-note">No workspace files available yet.</div>
      ) : null}
      <div className="file-tree">
        {(roots || []).map((node) => (
          <FileTreeNode
            key={node.path || node.name}
            node={node}
            depth={0}
            expandedDirs={expandedDirs}
            treeNodes={treeNodes}
            treeLoading={treeLoading}
            onToggle={onToggle}
            onSelect={onSelect}
            selectedPath={selectedPath}
          />
        ))}
      </div>
    </section>
  );
}

function FilePreviewPanel({ preview, loading, error, onRefresh }) {
  const directoryChildren = Array.isArray(preview?.children) ? preview.children : [];
  const structure = preview?.structure && typeof preview.structure === "object" ? preview.structure : null;
  const csvPreview = isCsvPreview(preview);

  return (
    <section className="files-panel files-preview-panel">
      <div className="section-head">
        <div>
          <div className="section-label">Preview</div>
          <h3 className="section-title">{preview?.name || "Select a file"}</h3>
        </div>
        <div className="inline-actions">
          <button type="button" className="ghost-btn" onClick={onRefresh} disabled={loading}>
            Refresh
          </button>
          {preview?.node_type === "file" && preview?.download_url ? (
            <a className="ghost-btn file-download-link" href={preview.download_url}>
              Download
            </a>
          ) : null}
        </div>
      </div>

      {error ? <div className="memory-drawer-note error">{error}</div> : null}
      {!error && loading ? <div className="memory-drawer-note">Loading preview...</div> : null}
      {!error && !loading && !preview ? (
        <div className="memory-drawer-note">Choose a file from the tree to inspect it here.</div>
      ) : null}

      {preview ? (
        <>
          <div className="file-meta-row">
            <span className="file-meta-pill">{preview.path || "."}</span>
            <span className="file-meta-pill">{preview.node_type}</span>
            <span className="file-meta-pill">{formatBytes(preview.size)}</span>
            {preview.mime_type ? <span className="file-meta-pill">{preview.mime_type}</span> : null}
            {preview.modified_ts ? <span className="file-meta-pill">{formatDateTime(preview.modified_ts)}</span> : null}
          </div>

          {preview.kind === "directory" ? (
            <div className="file-directory-preview">
              {directoryChildren.length ? (
                directoryChildren.map((child) => (
                  <div key={child.path || child.name} className="file-directory-row">
                    <span>{child.name}</span>
                    <span>{child.node_type === "file" ? formatBytes(child.size) : "directory"}</span>
                  </div>
                ))
              ) : (
                <div className="file-tree-note">Directory is empty.</div>
              )}
            </div>
          ) : null}

          {preview.kind === "image" && preview.download_url ? (
            <div className="file-image-wrap">
              <img className="file-image-preview" src={preview.download_url} alt={preview.name || "Selected file"} />
            </div>
          ) : null}

          {preview.kind === "structure" && structure ? (
            <div className="file-structure-panel">
              <div className="file-structure-meta">
                <div className="usage-cell">
                  <div className="usage-label">Formula</div>
                  <div className="usage-value">{structure.formula || "-"}</div>
                </div>
                <div className="usage-cell">
                  <div className="usage-label">Atoms</div>
                  <div className="usage-value">{formatCount(structure.atom_count)}</div>
                </div>
                <div className="usage-cell">
                  <div className="usage-label">Periodic</div>
                  <div className="usage-value">{structure.periodic ? "yes" : "no"}</div>
                </div>
              </div>
              <StructureViewer structure={structure} />
            </div>
          ) : null}

          {preview.kind === "markdown" && preview.preview_text ? (
            <div className="files-markdown-preview">
              <MarkdownContent text={preview.preview_text} />
            </div>
          ) : null}

          {csvPreview && preview.preview_text ? (
            <CsvPreview preview={preview} />
          ) : null}

          {preview.kind !== "image" && preview.kind !== "directory" && !(preview.kind === "markdown" && preview.preview_text) && !csvPreview ? (
            <pre className="code-pane tall">{preview.preview_text || "(binary file)"}</pre>
          ) : null}

          {preview.truncated ? <div className="memory-drawer-note">Preview truncated for large file size.</div> : null}
        </>
      ) : null}
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

function MemoryDrawer({ open, workspaceName, loading, error, text, source, onSourceChange, onRefresh, onClose }) {
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
            <button type="button" className={`ghost-btn ${source === "all" ? "active" : ""}`} onClick={() => onSourceChange("all")} disabled={loading}>
              All
            </button>
            <button type="button" className={`ghost-btn ${source === "langmem" ? "active" : ""}`} onClick={() => onSourceChange("langmem")} disabled={loading}>
              LangMem
            </button>
            <button type="button" className={`ghost-btn ${source === "instruction" ? "active" : ""}`} onClick={() => onSourceChange("instruction")} disabled={loading}>
              AGENTS
            </button>
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
  const view = ["home", "monitor", "files"].includes(boot?.view) ? boot.view : "home";
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
    source: "all",
  });
  const [treeNodes, setTreeNodes] = useState({});
  const [treeLoading, setTreeLoading] = useState({});
  const [expandedDirs, setExpandedDirs] = useState({ "": true });
  const [selectedFilePath, setSelectedFilePath] = useState("");
  const [filePreview, setFilePreview] = useState(null);
  const [fileTreeError, setFileTreeError] = useState("");
  const [filePreviewError, setFilePreviewError] = useState("");
  const [filePreviewLoading, setFilePreviewLoading] = useState(false);
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
    const memorySource = String(memoryPanel.source || "all");
    startTransition(() => {
      setMemoryPanel((prev) => ({
        ...prev,
        loading: true,
        error: "",
      }));
    });
    apiFetch(`/api/session/${escapePath(ctx)}/memory?run=${escapePath(selectedRun || "")}&source=${escapePath(memorySource)}`)
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
            source: String(data.source || memorySource || "all"),
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
  }, [ctx, memoryOpen, selectedRun, snapshot?.workspace_name, view, memoryPanel.source]);

  useEffect(() => {
    if (view !== "files" || !ctx) {
      return;
    }
    let cancelled = false;
    startTransition(() => {
      setTreeNodes({});
      setTreeLoading({ "": true });
      setExpandedDirs({ "": true });
      setSelectedFilePath("");
      setFilePreview(null);
      setFileTreeError("");
      setFilePreviewError("");
      setFilePreviewLoading(false);
    });
    apiFetch(`/api/session/${escapePath(ctx)}/files/tree`)
      .then((data) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setTreeNodes({ "": Array.isArray(data.children) ? data.children : [] });
          setTreeLoading({});
          setFileTreeError("");
        });
      })
      .catch((error) => {
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setTreeLoading({});
          setFileTreeError(String(error?.message || error));
        });
      });
    return () => {
      cancelled = true;
    };
  }, [ctx, snapshot?.workspace_path, view]);

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
    const memorySource = String(memoryPanel.source || "all");
    startTransition(() => {
      setMemoryPanel((prev) => ({
        ...prev,
        loading: true,
        error: "",
      }));
    });
    try {
      const data = await apiFetch(`/api/session/${escapePath(ctx)}/memory?run=${escapePath(selectedRun || "")}&source=${escapePath(memorySource)}`);
      startTransition(() => {
        setMemoryPanel({
          text: String(data.memory || "").trim(),
          error: "",
          loading: false,
          workspace: String(snapshot?.workspace_name || ""),
          source: String(data.source || memorySource || "all"),
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

  async function loadDirectory(path = "", { force = false } = {}) {
    const targetPath = String(path || "");
    if (!ctx) {
      return;
    }
    if (!force && Array.isArray(treeNodes[targetPath])) {
      return;
    }
    startTransition(() => {
      setTreeLoading((prev) => ({ ...prev, [targetPath]: true }));
      setFileTreeError("");
    });
    try {
      const data = await apiFetch(`/api/session/${escapePath(ctx)}/files/tree?path=${escapePath(targetPath)}`);
      startTransition(() => {
        setTreeNodes((prev) => ({ ...prev, [targetPath]: Array.isArray(data.children) ? data.children : [] }));
        setTreeLoading((prev) => ({ ...prev, [targetPath]: false }));
      });
    } catch (error) {
      startTransition(() => {
        setTreeLoading((prev) => ({ ...prev, [targetPath]: false }));
        setFileTreeError(String(error?.message || error));
      });
    }
  }

  async function loadFilePreview(path) {
    const targetPath = String(path || "");
    if (!ctx || !targetPath) {
      return;
    }
    startTransition(() => {
      setSelectedFilePath(targetPath);
      setFilePreviewLoading(true);
      setFilePreviewError("");
    });
    try {
      const data = await apiFetch(`/api/session/${escapePath(ctx)}/files/content?path=${escapePath(targetPath)}`);
      startTransition(() => {
        setFilePreview(data);
        setFilePreviewLoading(false);
      });
    } catch (error) {
      startTransition(() => {
        setFilePreview(null);
        setFilePreviewLoading(false);
        setFilePreviewError(String(error?.message || error));
      });
    }
  }

  async function handleDirectoryToggle(node) {
    if (!node || node.node_type !== "directory") {
      return;
    }
    const nextExpanded = !expandedDirs[node.path];
    startTransition(() => {
      setExpandedDirs((prev) => ({ ...prev, [node.path]: nextExpanded }));
    });
    if (nextExpanded) {
      await loadDirectory(node.path);
    }
  }

  async function handleFileSelect(node) {
    if (!node) {
      return;
    }
    if (node.node_type === "directory") {
      if (!expandedDirs[node.path]) {
        startTransition(() => {
          setExpandedDirs((prev) => ({ ...prev, [node.path]: true }));
        });
        await loadDirectory(node.path);
      }
      await loadFilePreview(node.path);
      return;
    }
    await loadFilePreview(node.path);
  }

  async function refreshFilesView() {
    await loadDirectory("", { force: true });
    if (selectedFilePath) {
      await loadFilePreview(selectedFilePath);
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
  const viewSubtitle = view === "home" ? "Cockpit" : view === "monitor" ? "Monitor" : "Files";
  const homeHref = snapshot?.ctx
    ? `/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}`
    : "/";
  const monitorHref = snapshot?.ctx
    ? `/monitor/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}&run=${escapePath(selectedRun)}`
    : "/monitor/";
  const filesHref = snapshot?.ctx
    ? `/files/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}&run=${escapePath(selectedRun)}`
    : "/files/";
  const centerTitle = view === "home" ? "Conversation" : view === "monitor" ? "Execution Stream" : "Workspace Explorer";

  return (
    <main className={`app-shell view-${view}`}>
      <header className="topbar">
        <div className="topbar-brand">
          <div className="topbar-logo">C</div>
          <span className="topbar-title">CatMaster</span>
          <span className="topbar-subtitle">{viewSubtitle}</span>
        </div>
        <nav className="topbar-nav">
          <a className={view === "home" ? "active" : ""} href={homeHref}>
            Home
          </a>
          <a className={view === "monitor" ? "active" : ""} href={monitorHref}>
            Monitor
          </a>
          <a className={view === "files" ? "active" : ""} href={filesHref}>
            Files
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
          {view === "files" ? (
            <div className="center-content files-content">
              <div className="center-header">
                <div className="center-header-left">
                  <h2>{centerTitle}</h2>
                  <span className="section-label">{snapshot?.workspace_name || "No workspace"}</span>
                </div>
                <div className="inline-actions">
                  <button type="button" className="ghost-btn" onClick={refreshFilesView}>
                    Refresh
                  </button>
                </div>
              </div>
              <div className="files-workspace-path">{snapshot?.workspace_path || "Open a project space to browse files."}</div>
              <div className="files-shell">
                <FileTree
                  treeNodes={treeNodes}
                  expandedDirs={expandedDirs}
                  treeLoading={treeLoading}
                  selectedPath={selectedFilePath}
                  error={fileTreeError}
                  onToggle={handleDirectoryToggle}
                  onSelect={handleFileSelect}
                />
                <FilePreviewPanel
                  preview={filePreview}
                  loading={filePreviewLoading}
                  error={filePreviewError}
                  onRefresh={() => {
                    if (selectedFilePath) {
                      loadFilePreview(selectedFilePath);
                    }
                  }}
                />
              </div>
            </div>
          ) : (
            <>
              <div className="center-content">
                <div className="center-header">
                  <div className="center-header-left">
                    <h2>{centerTitle}</h2>
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
                    <MetricCard label="Cost" value={formatCost(usage.cost_usd)} note={formatCostNote(usage)} />
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
                        {["experiment", "research", "writing", "peer_review"].map((item) => (
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
            </>
          )}
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
          source={memoryPanel.source}
          onSourceChange={(source) => setMemoryPanel((prev) => ({ ...prev, source }))}
          onRefresh={refreshMemoryPanel}
          onClose={() => setMemoryOpen(false)}
        />
      ) : null}
    </main>
  );
}

export default App;
