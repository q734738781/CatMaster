import { useCallback, useEffect, useMemo, useRef, useState } from "react";
import {
  Background,
  Controls,
  Handle,
  MiniMap,
  Position,
  ReactFlow,
  ReactFlowProvider,
  useReactFlow,
} from "@xyflow/react";
import "@xyflow/react/dist/style.css";
import ELK from "elkjs/lib/elk.bundled.js";
import {
  Archive,
  ArrowRight,
  CirclePlus,
  ExternalLink,
  Focus,
  Link2,
  Network,
  Play,
  RefreshCw,
  Search,
  Trash2,
  Unlink,
  X,
} from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  boundedResearchGraph,
  evidenceStateLabel,
  executionLaneLabel,
  experimentStateLabel,
  relationLabel,
} from "../researchTechTree";
import { RESEARCH_GRAPH_ACTIVITY_EVENTS } from "../researchSessions.js";
import ResearchCollaborationPanel from "./ResearchCollaborationPanel";

const elk = new ELK();
const NODE_COLORS = {
  hypothesis: "#7c3aed",
  experiment: "#0284c7",
  result: "#15803d",
};
const SOURCE_KINDS = ["note", "artifact", "run", "doi", "url", "thread", "message"];
const SOURCE_KIND_LABELS = {
  note: "Workspace note",
  artifact: "Artifact",
  run: "Research run",
  doi: "DOI",
  url: "Web page",
  thread: "Thread",
  message: "Message",
};
const EMPTY_FORM = {};

function splitLines(value) {
  return String(value || "")
    .split(/\r?\n/)
    .map((item) => item.trim())
    .filter(Boolean);
}

function bandLabel(value) {
  return {
    none: "None",
    low: "Low",
    medium: "Medium",
    high: "High",
  }[String(value || "").toLowerCase()] || "Not specified";
}

function recommendationLabel(node) {
  return node.recommended ? "Recommended next experiment" : "";
}

function nodeMetaLabel(node) {
  if (node.kind === "hypothesis") {
    return node.body?.importance
      ? `${bandLabel(node.body.importance)} importance`
      : "";
  }
  if (node.kind === "experiment") {
    return [
      node.body?.estimated_compute_cost
        ? `${bandLabel(node.body.estimated_compute_cost)} compute`
        : "",
    ].filter(Boolean).join(" · ");
  }
  return "";
}

function countLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

function graphNextStepLabel(item) {
  const labels = [];
  if (item.frontier?.length) {
    labels.push(`Ready next: ${item.frontier.map((node) => node.title).join(", ")}`);
  }
  if (item.external_handoffs?.length) {
    labels.push(`${countLabel(item.external_handoffs.length, "external handoff")} awaiting laboratory or collaborator results`);
  }
  return labels.length ? `${labels.join(" · ")}.` : "No experiment is ready to run.";
}

function ResearchNodeCard({ data, selected }) {
  const node = data.node;
  const threadFocused = data.threadFocused === true;
  const provisional = node.provisional === true;
  const kindLabel = {
    hypothesis: provisional ? "Proposed hypothesis" : "Hypothesis",
    experiment: provisional ? "Proposed experiment" : "Experiment proposal",
    result: "Result",
  }[node.kind] || "Research node";
  const recommendation = recommendationLabel(node);
  const recommended = node.recommended === true;
  const state = provisional
    ? (recommendation ? `${recommendation} · Temporary planning branch` : "Temporary planning branch")
    : recommendation
      ? `${recommendation} · ${node.kind === "experiment" ? experimentStateLabel(node.state, node.body?.execution_lane) : evidenceStateLabel(node.evidence_state)}`
      : node.kind === "hypothesis"
        ? evidenceStateLabel(node.evidence_state)
        : node.kind === "experiment"
          ? experimentStateLabel(node.state, node.body?.execution_lane)
          : "Result recorded";
  const metadata = nodeMetaLabel(node);
  return (
    <button
      type="button"
      className={`v2-rg-node kind-${node.kind} ${provisional ? "provisional" : ""} ${recommended ? "recommended" : ""} ${threadFocused ? "thread-focused" : ""} ${selected ? "selected" : ""}`}
      aria-label={`${kindLabel}: ${node.title}. ${state}${metadata ? `. ${metadata}` : ""}`}
      data-research-node-id={node.node_id}
      title={node.title}
      onClick={data.onSelect}
    >
      <Handle type="target" position={Position.Left} className="v2-rg-handle" />
      <span className="v2-rg-node-kind">{kindLabel}{threadFocused ? " · Thread focus" : ""}</span>
      <strong>{node.title}</strong>
      <small>{metadata ? `${state} · ${metadata}` : state}</small>
      <Handle type="source" position={Position.Right} className="v2-rg-handle" />
    </button>
  );
}

const NODE_TYPES = { researchNode: ResearchNodeCard };

async function layoutGraph(nodes, edges) {
  if (!nodes.length) return { nodes: [], edges: [] };
  const graph = {
    id: "research-graph",
    layoutOptions: {
      "elk.algorithm": "layered",
      "elk.direction": "RIGHT",
      "elk.spacing.nodeNode": "54",
      "elk.layered.spacing.nodeNodeBetweenLayers": "96",
      "elk.layered.nodePlacement.strategy": "NETWORK_SIMPLEX",
      "elk.edgeRouting": "ORTHOGONAL",
    },
    children: nodes.map((node) => ({
      id: node.id,
      width: 284,
      height: 138,
    })),
    edges: edges.map((edge) => ({
      id: edge.id,
      sources: [edge.source],
      targets: [edge.target],
    })),
  };
  const laidOut = await elk.layout(graph);
  const positions = new Map(
    (laidOut.children || []).map((node) => [
      node.id,
      { x: Number(node.x || 0), y: Number(node.y || 0) },
    ]),
  );
  return {
    nodes: nodes.map((node) => ({
      ...node,
      position: positions.get(node.id) || { x: 0, y: 0 },
    })),
    edges,
  };
}

function formatUpdated(value) {
  const timestamp = Number(value || 0) * 1000;
  if (!timestamp) return "Update time unavailable";
  return `Updated ${new Intl.DateTimeFormat(undefined, {
    dateStyle: "medium",
    timeStyle: "short",
  }).format(new Date(timestamp))}`;
}

function GraphModal({ title, children, onClose }) {
  const dialogRef = useRef(null);
  const returnFocusRef = useRef(null);
  const onCloseRef = useRef(onClose);
  onCloseRef.current = onClose;

  useEffect(() => {
    returnFocusRef.current = document.activeElement;
    const dialog = dialogRef.current;
    const focusable = dialog?.querySelector(
      'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
    );
    (focusable || dialog)?.focus();
    const handleKeyDown = (event) => {
      if (event.key === "Escape") {
        event.preventDefault();
        onCloseRef.current();
        return;
      }
      if (event.key !== "Tab" || !dialog) return;
      const controls = [...dialog.querySelectorAll(
        'button:not([disabled]), input:not([disabled]), textarea:not([disabled]), select:not([disabled]), a[href], [tabindex]:not([tabindex="-1"])',
      )];
      if (!controls.length) {
        event.preventDefault();
        dialog.focus();
        return;
      }
      const first = controls[0];
      const last = controls[controls.length - 1];
      if (event.shiftKey && document.activeElement === first) {
        event.preventDefault();
        last.focus();
      } else if (!event.shiftKey && document.activeElement === last) {
        event.preventDefault();
        first.focus();
      }
    };
    window.addEventListener("keydown", handleKeyDown);
    return () => {
      window.removeEventListener("keydown", handleKeyDown);
      const previous = returnFocusRef.current;
      if (previous instanceof HTMLElement && previous.isConnected) {
        window.requestAnimationFrame(() => previous.focus());
      }
    };
  }, []);

  return (
    <div className="v2-rg-modal-backdrop" role="presentation" onMouseDown={onClose}>
      <section
        ref={dialogRef}
        className="v2-rg-modal"
        role="dialog"
        aria-modal="true"
        aria-label={title}
        tabIndex={-1}
        onMouseDown={(event) => event.stopPropagation()}
      >
        <header>
          <h3>{title}</h3>
          <button type="button" className="v2-icon-btn" aria-label={`Close ${title}`} onClick={onClose}>
            <X size={17} />
          </button>
        </header>
        {children}
      </section>
    </div>
  );
}

function Field({ label, children, hint = "" }) {
  return (
    <label className="v2-rg-field">
      <span>{label}</span>
      {children}
      {hint ? <small>{hint}</small> : null}
    </label>
  );
}

function OptionalDetails({ children, label = "Optional details" }) {
  return (
    <details className="v2-rg-optional-details">
      <summary>{label}</summary>
      <div>{children}</div>
    </details>
  );
}

function OptionalSourceFields({ form, setForm, hint = "" }) {
  return (
    <>
      <Field label="Source type (optional)">
        <select
          value={form.ref_kind || "note"}
          onChange={(event) => setForm({ ...form, ref_kind: event.target.value })}
        >
          {SOURCE_KINDS.map((kind) => (
            <option value={kind} key={kind}>{SOURCE_KIND_LABELS[kind]}</option>
          ))}
        </select>
      </Field>
      <Field
        label="Source identifier (optional)"
        hint={hint || "Use a DOI, URL, workspace note path, artifact, run, thread, or message. More sources can be attached later."}
      >
        <input
          value={form.ref_id || ""}
          onChange={(event) => setForm({ ...form, ref_id: event.target.value })}
        />
      </Field>
    </>
  );
}

function sourceRefs(form) {
  const refId = String(form.ref_id || "").trim();
  return refId ? [{ ref_kind: form.ref_kind || "note", ref_id: refId }] : [];
}

function launchActivityLabel(value) {
  return {
    running: "Running",
    waiting_continue: "Waiting to continue",
    waiting_review: "Waiting for review",
    operationally_incomplete: "Operationally incomplete",
  }[String(value || "running")] || "Running";
}

function ReferenceList({ refs, onOpenThread, onOpenReference, onOpenDiscussion }) {
  if (!refs?.length) return <p className="v2-muted">No source is attached.</p>;
  return (
    <ul className="v2-rg-ref-list">
      {refs.map((ref) => {
        const key = `${ref.ref_kind}:${ref.ref_id}`;
        const label = ref.available ? ref.label : "Source unavailable";
        if (ref.href) {
          return (
            <li key={key}>
              <a href={ref.href} target="_blank" rel="noreferrer">
                <ExternalLink size={13} /> {label}
              </a>
            </li>
          );
        }
        if (ref.discussion_id) {
          return <li key={key}><button type="button" className="v2-link-btn" onClick={() => onOpenDiscussion?.(ref)}>
            <ExternalLink size={13} /> {label}</button></li>;
        }
        if (ref.thread_id) {
          return (
            <li key={key}>
              <button type="button" className="v2-link-btn" onClick={() => onOpenThread?.(ref.thread_id)}>
                <ExternalLink size={13} /> {label}
              </button>
            </li>
          );
        }
        if (ref.artifact_id) {
          return (
            <li key={key}>
              <button
                type="button"
                className="v2-link-btn"
                onClick={() => onOpenReference?.({ type: "artifact", artifact_id: ref.artifact_id })}
              >
                <ExternalLink size={13} /> {label}
              </button>
            </li>
          );
        }
        if (ref.path) {
          return (
            <li key={key}>
              <button
                type="button"
                className="v2-link-btn"
                onClick={() => onOpenReference?.({ type: "file", path: ref.path })}
              >
                <ExternalLink size={13} /> {label}
              </button>
            </li>
          );
        }
        return <li key={key} className={!ref.available ? "unavailable" : ""}>{label}</li>;
      })}
    </ul>
  );
}

function GraphCanvas({ payload, selectedNodeId, threadFocusNodeId, onSelectNode }) {
  const [density, setDensity] = useState(25);
  const [focusOnly, setFocusOnly] = useState(false);
  const [query, setQuery] = useState("");
  const [flowNodes, setFlowNodes] = useState([]);
  const [flowEdges, setFlowEdges] = useState([]);
  const [layoutVersion, setLayoutVersion] = useState(0);
  const layoutNodeIdsRef = useRef([]);
  const selectedNodeIdRef = useRef(selectedNodeId);
  const onSelectNodeRef = useRef(onSelectNode);
  selectedNodeIdRef.current = selectedNodeId;
  onSelectNodeRef.current = onSelectNode;
  const instance = useReactFlow();
  const focusNodeId = focusOnly ? selectedNodeId : "";
  const visible = useMemo(
    () => boundedResearchGraph(payload, {
      limit: density,
      focusNodeId,
      hops: 2,
      query,
    }),
    [density, focusNodeId, payload, query],
  );

  useEffect(() => {
    let cancelled = false;
    const nodes = visible.nodes.map((node) => ({
      id: node.node_id,
      type: "researchNode",
      position: { x: 0, y: 0 },
      data: {
        node,
        threadFocused: node.node_id === threadFocusNodeId,
        onSelect: () => onSelectNodeRef.current(node.node_id),
      },
      selected: node.node_id === selectedNodeIdRef.current,
      draggable: false,
      connectable: false,
      focusable: false,
      ariaLabel: `${node.kind}: ${node.title}`,
    }));
    const visibleById = new Map(
      visible.nodes.map((node) => [String(node.node_id), node]),
    );
    const edges = visible.edges.map((edge) => ({
      id: `${edge.source_node_id}:${edge.relation}:${edge.target_node_id}`,
      source: edge.source_node_id,
      target: edge.target_node_id,
      type: "smoothstep",
      label: relationLabel(edge.relation),
      className: `v2-rg-edge relation-${edge.relation}`,
      markerEnd: { type: "arrowclosed", width: 18, height: 18 },
      focusable: true,
      ariaLabel: `${
        visibleById.get(String(edge.source_node_id))?.title || "Source node"
      } ${relationLabel(edge.relation)} ${
        visibleById.get(String(edge.target_node_id))?.title || "Target node"
      }`,
    }));
    layoutGraph(nodes, edges).then((next) => {
      if (cancelled) return;
      layoutNodeIdsRef.current = next.nodes.map((node) => node.id);
      setFlowNodes(next.nodes);
      setFlowEdges(next.edges);
      setLayoutVersion((value) => value + 1);
    });
    return () => {
      cancelled = true;
    };
  }, [payload?.graph?.revision, threadFocusNodeId, visible]);

  useEffect(() => {
    setFlowNodes((nodes) => nodes.map((node) => ({
      ...node,
      selected: node.id === selectedNodeId,
    })));
  }, [selectedNodeId]);

  useEffect(() => {
    if (
      !layoutVersion
      || !instance?.viewportInitialized
      || !layoutNodeIdsRef.current.length
    ) return undefined;
    let cancelled = false;
    const frame = window.requestAnimationFrame(() => {
      if (cancelled) return;
      void instance.fitView({
        nodes: layoutNodeIdsRef.current.map((id) => ({ id })),
        padding: 0.22,
        minZoom: 0.15,
        maxZoom: 1,
        duration: 240,
      });
    });
    return () => {
      cancelled = true;
      window.cancelAnimationFrame(frame);
    };
  }, [instance, layoutVersion]);

  return (
    <div className="v2-rg-canvas-shell">
      <div className="v2-rg-canvas-toolbar">
        <label className="v2-rg-node-search">
          <Search size={14} aria-hidden="true" />
          <input
            type="search"
            value={query}
            placeholder="Find a hypothesis, experiment, or result"
            aria-label="Find a research node"
            onChange={(event) => setQuery(event.target.value)}
          />
        </label>
        <label>
          Show
          <select value={density} onChange={(event) => setDensity(Number(event.target.value))} aria-label="Research graph node density">
            <option value={5}>up to 5 nodes</option>
            <option value={25}>up to 25 nodes</option>
            <option value={100}>up to 100 nodes</option>
          </select>
        </label>
        <button
          type="button"
          className={`v2-ghost-btn ${focusOnly ? "active" : ""}`}
          aria-pressed={focusOnly}
          disabled={!selectedNodeId}
          onClick={() => setFocusOnly((value) => !value)}
        >
          <Focus size={14} /> {focusOnly ? "Show full graph" : "Focus neighborhood"}
        </button>
        <span>
          {query
            ? `Showing ${visible.nodes.length} of ${visible.matchingCount} matching nodes across ${visible.totalCount}`
            : `Showing ${visible.nodes.length} of ${visible.totalCount} nodes`}
          {visible.omittedCount ? ` · ${visible.omittedCount} available outside this view` : ""}
        </span>
      </div>
      <div className="v2-rg-flow" role="region" aria-label="Research knowledge graph">
        <ReactFlow
          nodes={flowNodes}
          edges={flowEdges}
          nodeTypes={NODE_TYPES}
          nodesFocusable={false}
          nodesDraggable={false}
          nodesConnectable={false}
          minZoom={0.15}
          maxZoom={2.5}
          proOptions={{ hideAttribution: true }}
        >
          <Background gap={22} size={1} />
          <Controls
            showInteractive={false}
            fitViewOptions={{ padding: 0.22, minZoom: 0.15, maxZoom: 1 }}
          />
          <MiniMap
            pannable
            zoomable
            style={{ width: 120, height: 90 }}
            nodeColor={(node) => NODE_COLORS[node.data?.node?.kind] || "#64748b"}
            ariaLabel="Research graph minimap"
          />
        </ReactFlow>
      </div>
      <p className="v2-rg-keyboard-help">
        Tab reaches nodes and controls; Enter selects a node. Use arrow keys on graph controls, or drag and wheel to pan and zoom.
      </p>
    </div>
  );
}

function ResearchGraphPanelContent({
  workspaceName,
  thread,
  onOpenThread,
  onThreadUpdate,
  onOpenReference,
}) {
  const [catalog, setCatalog] = useState([]);
  const [selectedGraphId, setSelectedGraphId] = useState("");
  const [payload, setPayload] = useState(null);
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [loading, setLoading] = useState(false);
  const [mutating, setMutating] = useState(false);
  const [historyLoading, setHistoryLoading] = useState(false);
  const [historyError, setHistoryError] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [streamStatus, setStreamStatus] = useState("");
  const [modal, setModal] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const [collaborationRefresh, setCollaborationRefresh] = useState(0);
  const [discussionFocus, setDiscussionFocus] = useState(null);
  const eventIdRef = useRef(0);
  const refreshTimerRef = useRef(0);
  const inspectorCloseRef = useRef(null);
  const inspectorReturnFocusRef = useRef(null);

  const activeGraphId = String(thread?.active_research_graph_id || "");
  const graphId = selectedGraphId || activeGraphId;
  const threadFocusNodeId = activeGraphId === graphId
    ? String(thread?.research_focus_node_id || "")
    : "";
  const graph = payload?.graph || null;
  const displayPayload = useMemo(() => {
    const planning = payload?.planning_preview || {};
    const planningSummary = String(payload?.planning_preview?.summary || "");
    const recommendedId = String(planning.recommended_experiment_id || "");
    const decorateNode = (node) => {
      const nodeId = String(node.node_id || "");
      return {
        ...node,
        recommended: Boolean(recommendedId && nodeId === recommendedId),
        planning_reason: recommendedId && nodeId === recommendedId ? planningSummary : "",
      };
    };
    const durableNodes = (Array.isArray(payload?.nodes) ? payload.nodes : []).map(decorateNode);
    const durableEdges = Array.isArray(payload?.edges) ? payload.edges : [];
    const previewNodes = Array.isArray(payload?.planning_preview?.nodes)
      ? payload.planning_preview.nodes.map(decorateNode)
      : [];
    const previewEdges = Array.isArray(payload?.planning_preview?.edges)
      ? payload.planning_preview.edges
      : [];
    const durableIds = new Set(durableNodes.map((node) => String(node.node_id || "")));
    const nodes = [
      ...durableNodes,
      ...previewNodes.filter((node) => !durableIds.has(String(node.node_id || ""))),
    ];
    const nodeIds = new Set(nodes.map((node) => String(node.node_id || "")));
    const seenEdges = new Set();
    const edges = [...durableEdges, ...previewEdges].filter((edge) => {
      const source = String(edge.source_node_id || "");
      const target = String(edge.target_node_id || "");
      const key = `${source}:${edge.relation}:${target}`;
      if (!nodeIds.has(source) || !nodeIds.has(target) || seenEdges.has(key)) return false;
      seenEdges.add(key);
      return true;
    });
    return { ...payload, nodes, edges };
  }, [payload]);
  const selectedNode = useMemo(
    () => (displayPayload?.nodes || []).find((node) => node.node_id === selectedNodeId)
      || displayPayload?.nodes?.[0]
      || null,
    [displayPayload?.nodes, selectedNodeId],
  );
  const revisedRelatedRecords = useMemo(() => {
    if (!["experiment", "hypothesis"].includes(selectedNode?.kind)) return [];
    const edges = payload?.edges || [];
    const revisedIds = new Set(edges.filter((edge) => edge.relation === "revises").map((edge) => edge.target_node_id));
    const relations = selectedNode.kind === "experiment" ? ["tests"] : ["supports", "opposes", "inconclusive", "suggests"];
    const inputIds = new Set(edges.filter((edge) => relations.includes(edge.relation) && edge.target_node_id === selectedNode.node_id).map((edge) => edge.source_node_id));
    return (payload?.nodes || []).filter((node) => inputIds.has(node.node_id) && revisedIds.has(node.node_id));
  }, [payload, selectedNode]);

  const openInspector = useCallback(() => {
    inspectorReturnFocusRef.current = document.activeElement;
    setInspectorOpen(true);
  }, []);

  const closeInspector = useCallback(() => {
    const previous = inspectorReturnFocusRef.current;
    setInspectorOpen(false);
    window.requestAnimationFrame(() => {
      if (previous instanceof HTMLElement && previous.isConnected && !previous.closest(".v2-rg-inspector")) {
        previous.focus();
        return;
      }
      const nodeButton = selectedNodeId
        ? document.querySelector(
          `[data-research-node-id="${CSS.escape(selectedNodeId)}"]`,
        )
        : null;
      (nodeButton || document.querySelector(".v2-rg-mobile-inspector-trigger"))?.focus();
    });
  }, [selectedNodeId]);

  useEffect(() => {
    setSelectedGraphId(activeGraphId);
    setInspectorOpen(false);
  }, [activeGraphId, thread?.thread_id]);

  useEffect(() => {
    if (!inspectorOpen) return undefined;
    if (window.matchMedia("(max-width: 760px)").matches) {
      inspectorCloseRef.current?.focus();
    }
    const closeOnEscape = (event) => {
      if (event.key === "Escape") closeInspector();
    };
    window.addEventListener("keydown", closeOnEscape);
    return () => window.removeEventListener("keydown", closeOnEscape);
  }, [closeInspector, inspectorOpen]);

  const loadCatalog = useCallback(async () => {
    if (!workspaceName) return [];
    const query = thread?.thread_id
      ? `?include_archived=true&thread_id=${encodeURIComponent(thread.thread_id)}`
      : "?include_archived=true";
    const next = await apiFetch(
      `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs${query}`,
    );
    const rows = Array.isArray(next.graphs) ? next.graphs : [];
    setCatalog(rows);
    if (!activeGraphId) {
      const activeRows = rows.filter((item) => !item.archived);
      setSelectedGraphId((current) => {
        if (current && rows.some((item) => item.graph_id === current)) return current;
        return activeRows.length === 1 ? activeRows[0].graph_id : "";
      });
    }
    return rows;
  }, [activeGraphId, thread?.thread_id, workspaceName]);

  const refreshGraph = useCallback(async (targetGraphId = graphId) => {
    if (!workspaceName || !targetGraphId) {
      setPayload(null);
      return null;
    }
    const threadQuery = thread?.thread_id
      ? `?thread_id=${encodeURIComponent(thread.thread_id)}`
      : "";
    const next = await apiFetch(
      `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(targetGraphId)}${threadQuery}`,
    );
    setPayload(next);
    setHistoryError("");
    setSelectedNodeId((current) => (
      current && next.nodes?.some((node) => node.node_id === current)
        ? current
        : next.nodes?.[0]?.node_id || ""
    ));
    return next;
  }, [graphId, thread?.thread_id, workspaceName]);

  async function loadOlderHistory() {
    if (historyLoading) return;
    const targetGraphId = String(payload?.graph?.graph_id || "");
    const historyRef = String(payload?.mutation_history_ref || "");
    const visibleHistory = Array.isArray(payload?.mutation_history)
      ? payload.mutation_history
      : [];
    const beforeEventId = Number(visibleHistory.at(-1)?.event_id || 0);
    if (!targetGraphId || !historyRef || !beforeEventId) {
      setHistoryError("Older graph history is unavailable.");
      return;
    }
    setHistoryLoading(true);
    setHistoryError("");
    try {
      const separator = historyRef.includes("?") ? "&" : "?";
      const next = await apiFetch(
        `${historyRef}${separator}before_event_id=${encodeURIComponent(beforeEventId)}&limit=50`,
      );
      const olderEvents = Array.isArray(next?.events) ? next.events : [];
      setPayload((current) => {
        if (String(current?.graph?.graph_id || "") !== targetGraphId) return current;
        const currentHistory = Array.isArray(current?.mutation_history)
          ? current.mutation_history
          : [];
        const knownEventIds = new Set(currentHistory.map((entry) => entry.event_id));
        return {
          ...current,
          mutation_history: [
            ...currentHistory,
            ...olderEvents.filter((entry) => !knownEventIds.has(entry.event_id)),
          ],
          mutation_history_has_more: Boolean(next?.page?.truncated),
        };
      });
    } catch (err) {
      setHistoryError(err.message || String(err));
    } finally {
      setHistoryLoading(false);
    }
  }

  const refreshAll = useCallback(async () => {
    setLoading(true);
    setError("");
    try {
      await loadCatalog();
      await refreshGraph();
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [loadCatalog, refreshGraph]);

  useEffect(() => {
    refreshAll();
  }, [refreshAll]);

  useEffect(() => {
    if (!graphId || !workspaceName) return undefined;
    eventIdRef.current = 0;
    setStreamStatus("Live updates connected");
    const stream = new EventSource(
      `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graphId)}/stream`,
    );
    const handleEvent = (event) => {
      const eventId = Number(event.lastEventId || 0);
      if (eventId && eventId <= eventIdRef.current) return;
      if (eventId) eventIdRef.current = eventId;
      window.clearTimeout(refreshTimerRef.current);
      refreshTimerRef.current = window.setTimeout(() => {
        setCollaborationRefresh((value) => value + 1);
        refreshGraph(graphId).catch((err) => setError(err.message || String(err)));
        loadCatalog().catch(() => {});
      }, 80);
    };
    RESEARCH_GRAPH_ACTIVITY_EVENTS.forEach((type) => stream.addEventListener(type, handleEvent));
    stream.onopen = () => setStreamStatus("Live updates connected");
    stream.onerror = () => setStreamStatus("Reconnecting live updates…");
    return () => {
      window.clearTimeout(refreshTimerRef.current);
      RESEARCH_GRAPH_ACTIVITY_EVENTS.forEach((type) => stream.removeEventListener(type, handleEvent));
      stream.close();
    };
  }, [graphId, loadCatalog, refreshGraph, workspaceName]);

  async function mutate(request, successMessage = "", options = {}) {
    setMutating(true);
    setError("");
    setNotice("");
    try {
      const next = await request();
      if (next?.graph) setPayload(next);
      if (next?.thread && options.updateThread !== false) {
        onThreadUpdate?.(next.thread);
      }
      if (next?.node?.node_id) {
        setSelectedNodeId(next.node.node_id);
        openInspector();
      }
      if (next?.deleted_result?.node_id) {
        setSelectedNodeId(next.nodes?.[0]?.node_id || "");
      }
      setNotice(successMessage);
      setModal("");
      setForm(EMPTY_FORM);
      await loadCatalog();
      return next;
    } catch (err) {
      const message = err.message || String(err);
      setError(message);
      if (/changed in another thread|revision/i.test(message)) {
        await refreshGraph().catch(() => {});
      }
      return null;
    } finally {
      setMutating(false);
    }
  }

  async function bindGraph(targetGraphId, focusNodeId = "") {
    if (!thread?.thread_id) return;
    const next = await mutate(
      () => apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/active-research-graph`, {
        method: "PUT",
        body: JSON.stringify({ graph_id: targetGraphId, focus_node_id: focusNodeId }),
      }),
      targetGraphId ? "Research Graph attached to this thread." : "Research Graph detached from this thread.",
    );
    if (next?.thread) {
      setSelectedGraphId(targetGraphId);
      if (!targetGraphId) setPayload(null);
    }
  }

  async function setThreadFocus(nodeId = "") {
    if (!thread?.thread_id || activeGraphId !== graph?.graph_id) return;
    await mutate(
      () => apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/active-research-graph`, {
        method: "PUT",
        body: JSON.stringify({ graph_id: graph.graph_id, focus_node_id: nodeId }),
      }),
      nodeId ? "Thread focus updated." : "Thread focus cleared.",
    );
  }

  function openModal(kind, defaults = {}) {
    setForm(defaults);
    setModal(kind);
    setError("");
    setNotice("");
  }

  async function submitModal(event) {
    event.preventDefault();
    if (!workspaceName) return;
    const base = `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs`;
    if (modal === "new_graph") {
      const created = await mutate(
        () => apiFetch(`${base}${thread?.thread_id ? `?thread_id=${encodeURIComponent(thread.thread_id)}` : ""}`, {
          method: "POST",
          body: JSON.stringify({
            question: form.question || "",
            title: form.title || "",
            completion_criterion: form.completion_criterion || "",
            decision_preferences: form.decision_preferences || "",
            initial_hypotheses: (form.initial_hypotheses || [])
              .filter((item) => String(item?.claim || "").trim())
              .map((item) => ({
                title: item.title || "",
                claim: item.claim || "",
                rationale: item.rationale || "",
                predictions: splitLines(item.predictions),
                importance: item.importance || "",
                refs: sourceRefs(item),
              })),
          }),
        }),
        "Research Graph created.",
      );
      const createdId = created?.graph?.graph_id;
      if (createdId) {
        setSelectedGraphId(createdId);
        setPayload(created);
        if (form.attach !== false && thread?.thread_id) await bindGraph(createdId);
      }
      return;
    }
    if (!graph?.graph_id) return;
    const graphBase = `${base}/${encodeURIComponent(graph.graph_id)}`;
    if (modal === "graph") {
      await mutate(
        () => apiFetch(graphBase, {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: graph.revision,
            title: form.title || graph.title,
            question: form.question || graph.question,
            completion_criterion: form.completion_criterion || graph.completion_criterion,
            decision_preferences: form.decision_preferences ?? graph.decision_preferences,
          }),
        }),
        "Research goal updated.",
      );
    } else if (modal === "hypothesis") {
      await mutate(
        () => apiFetch(`${graphBase}/hypotheses`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            title: form.title || "",
            claim: form.claim || "",
            rationale: form.rationale || "",
            predictions: splitLines(form.predictions),
            importance: form.importance || "",
            suggested_by_result_ids: form.suggested_by_result_id ? [form.suggested_by_result_id] : [],
            refs: sourceRefs(form),
          }),
        }),
        "Hypothesis added.",
      );
    } else if (modal === "experiment") {
      await mutate(
        () => apiFetch(`${graphBase}/experiments`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            title: form.title || "",
            objective: form.objective || "",
            plan_summary: form.plan_summary || "",
            decision_rule: form.decision_rule || "",
            execution_lane: form.execution_lane || "experiment",
            estimated_compute_cost: form.estimated_compute_cost || "",
            state: form.state || "draft",
            tests_hypothesis_ids: form.tests_hypothesis_ids || [],
            depends_on_experiment_ids: [],
            refs: sourceRefs(form),
          }),
        }),
        "Experiment proposal added.",
      );
    } else if (modal === "result") {
      const judgments = Object.entries(form.judgments || {})
        .filter(([, relation]) => relation)
        .map(([hypothesis_node_id, relation]) => ({ hypothesis_node_id, relation }));
      await mutate(
        () => apiFetch(`${graphBase}/results`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            title: form.title || "",
            summary: form.summary || "",
            methods: form.methods || "",
            conclusion: form.conclusion || "",
            experiment_node_id: form.experiment_node_id || "",
            judgments,
            refs: sourceRefs(form),
          }),
        }),
        "Observation or result recorded.",
      );
    } else if (modal === "judgment" && selectedNode?.kind === "result") {
      await mutate(
        () => apiFetch(
          `${graphBase}/results/${encodeURIComponent(selectedNode.node_id)}/judgments/${encodeURIComponent(form.hypothesis_node_id || "")}`,
          {
            method: "PUT",
            body: JSON.stringify({
              expected_revision: graph.revision,
              relation: form.relation || "unjudged",
              scope: form.scope || "",
              rationale: form.rationale || "",
            }),
          },
        ),
        form.relation === "unjudged"
          ? "Result left unjudged for this hypothesis."
          : "Result judgment updated.",
      );
    } else if (modal === "edit" && selectedNode) {
      const body = selectedNode.kind === "hypothesis"
        ? {
          claim: form.claim || "",
          rationale: form.rationale || "",
          predictions: splitLines(form.predictions),
          importance: form.importance || "",
        }
        : selectedNode.kind === "experiment"
          ? {
            objective: form.objective || "",
            plan_summary: form.plan_summary || "",
            decision_rule: form.decision_rule || "",
            execution_lane: form.execution_lane || "experiment",
            estimated_compute_cost: form.estimated_compute_cost || "",
            blocking_reason: selectedNode.body?.blocking_reason || "",
          }
          : { ...selectedNode.body, summary: form.summary || "", methods: form.methods || "", conclusion: form.conclusion || "" };
      await mutate(
        () => apiFetch(`${graphBase}/nodes/${encodeURIComponent(selectedNode.node_id)}`, {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: graph.revision,
            expected_node_revision: selectedNode.revision,
            title: form.title || selectedNode.title,
            state: selectedNode.kind === "experiment" ? (form.state || selectedNode.state) : "",
            body,
          }),
        }),
        "Confirmed edit saved.",
      );
    } else if (modal === "delete_result" && selectedNode?.kind === "result") {
      await mutate(
        () => apiFetch(`${graphBase}/results/${encodeURIComponent(selectedNode.node_id)}`, {
          method: "DELETE",
          body: JSON.stringify({
            expected_revision: graph.revision,
            expected_node_revision: selectedNode.revision,
            reason: form.reason || "",
          }),
        }),
        "Result deleted and recorded in graph history.",
      );
    } else if (modal === "ref" && selectedNode) {
      await mutate(
        () => apiFetch(`${graphBase}/refs`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            node_id: selectedNode.node_id,
            ref_kind: form.ref_kind || "note",
            ref_id: form.ref_id || "",
          }),
        }),
        "Source attached.",
      );
    } else if (modal === "dependency" && selectedNode) {
      await mutate(
        () => apiFetch(`${graphBase}/edges`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            source_node_id: selectedNode.node_id,
            target_node_id: form.target_node_id || "",
            relation: "depends_on",
          }),
        }),
        "Experiment dependency added.",
      );
    } else if (modal === "blocked" && selectedNode) {
      await mutate(
        () => apiFetch(`${graphBase}/experiments/${encodeURIComponent(selectedNode.node_id)}/blocked`, {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            reason: form.reason || "",
          }),
        }),
        "Experiment marked blocked with its reason.",
      );
    }
  }

  function editSelected() {
    if (!selectedNode) return;
    const body = selectedNode.body || {};
    openModal("edit", {
      title: selectedNode.title,
      state: selectedNode.state,
      claim: body.claim,
      rationale: body.rationale,
      predictions: (body.predictions || []).join("\n"),
      importance: body.importance || "",
      objective: body.objective,
      plan_summary: body.plan_summary,
      decision_rule: body.decision_rule,
      execution_lane: body.execution_lane,
      estimated_compute_cost: body.estimated_compute_cost || "",
      summary: body.summary,
      methods: body.methods,
      conclusion: body.conclusion,
  });
}

  async function launchExperiment(replicate) {
    if (!selectedNode || !graph) return;
    const next = await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}/experiments/${encodeURIComponent(selectedNode.node_id)}/launch${thread?.thread_id ? `?thread_id=${encodeURIComponent(thread.thread_id)}` : ""}`,
        {
          method: "POST",
          body: JSON.stringify({ expected_revision: graph.revision, replicate }),
        },
      ),
      replicate ? "Replicate thread started." : "Experiment thread started.",
    );
    if (next?.thread) {
      onOpenThread?.(next.thread.thread_id);
    }
  }

  async function toggleCompletion() {
    if (!graph) return;
    await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({
            expected_revision: graph.revision,
            completed: !graph.completed,
          }),
        },
      ),
      graph.completed
        ? "Research Graph reopened."
        : "Research Graph completion criterion marked satisfied.",
    );
  }

  function resultDefaults(experimentNode = null) {
    return {
      experiment_node_id: experimentNode?.node_id || "",
      judgments: {},
      ref_kind: "note",
    };
  }

  function resultJudgment(resultNodeId, hypothesisNodeId) {
    return (payload?.edges || []).find((edge) => (
      edge.source_node_id === resultNodeId
      && edge.target_node_id === hypothesisNodeId
      && ["supports", "opposes", "inconclusive"].includes(edge.relation)
    ))?.relation || "unjudged";
  }

  function judgmentDetails(resultNodeId, hypothesisNodeId) {
    return (payload?.edges || []).find((edge) => edge.source_node_id === resultNodeId
      && edge.target_node_id === hypothesisNodeId
      && ["supports", "opposes", "inconclusive"].includes(edge.relation)) || {};
  }

  function openJudgmentEditor(hypothesisNodeId = "") {
    if (selectedNode?.kind !== "result") return;
    const targetId = hypothesisNodeId || hypotheses[0]?.node_id || "";
    openModal("judgment", {
      ...judgmentDetails(selectedNode.node_id, targetId),
      hypothesis_node_id: targetId,
      relation: targetId
        ? resultJudgment(selectedNode.node_id, targetId)
        : "unjudged",
    });
  }

  const hypotheses = (payload?.nodes || []).filter((node) => node.kind === "hypothesis");
  const experiments = (payload?.nodes || []).filter((node) => node.kind === "experiment");
  const resultExperiments = experiments.filter((node) => (
    ["ready", "running", "has_results"].includes(node.state)
  ));
  const results = (payload?.nodes || []).filter((node) => node.kind === "result");

  return (
    <section className="v2-tab-panel v2-research-graph-panel">
      <div className="v2-panel-toolbar">
        <div>
          <h2>Workspace Research Graph</h2>
          <p className="v2-muted">
            Shared hypotheses, experiments, and results across this workspace. Detailed notes, files, and run receipts remain in their own workspace stores.
          </p>
        </div>
        <div className="v2-research-tree-toolbar-actions">
          <button type="button" className="v2-primary-btn" onClick={() => openModal("new_graph", {
            attach: true,
            completion_criterion: "",
            initial_hypotheses: [],
          })}>
            <CirclePlus size={14} /> New graph
          </button>
          <button type="button" className="v2-ghost-btn" onClick={refreshAll} disabled={loading}>
            <RefreshCw size={14} className={loading ? "v2-spin" : ""} /> Refresh
          </button>
        </div>
      </div>

      {error ? <div className="v2-error" role="alert">{error}</div> : null}
      {notice ? <div className="v2-rg-notice" role="status">{notice}</div> : null}

      <details className="v2-rg-catalog-disclosure" open={!graph}>
        <summary>Research graphs <small>{catalog.length} in this workspace</small></summary>
        <div className="v2-rg-catalog" aria-label="Research Graph catalog">
        {catalog.length ? catalog.map((item) => (
          <article
            key={item.graph_id}
            className={`v2-rg-catalog-card ${item.graph_id === graphId ? "selected" : ""} ${item.archived ? "archived" : ""}`}
          >
            <div>
              <span>
                {item.archived ? "Archived" : item.completed ? "Completed" : "Active"}
                {item.bound_to_current_thread ? " · Attached to this thread" : ""}
              </span>
              <h3>{item.title}</h3>
              <p>{item.question}</p>
              <small>
                {countLabel(item.counts.hypotheses, "hypothesis", "hypotheses")} · {countLabel(item.counts.experiments, "experiment")} · {countLabel(item.counts.results, "result")}
                {" · "}{countLabel(item.bound_thread_count, "attached thread")} · {formatUpdated(item.updated_at)}
              </small>
              <small>
                {graphNextStepLabel(item)}
              </small>
            </div>
            <button type="button" className="v2-ghost-btn" onClick={() => setSelectedGraphId(item.graph_id)}>
              {item.graph_id === graphId ? "Selected" : "Open"}
            </button>
          </article>
        )) : (
          <div className="v2-rg-empty">
            <Network size={24} />
            <strong>No Research Graph yet</strong>
            <p>Create one from a research question and optional seed hypotheses. Ordinary one-off chat does not require a graph.</p>
          </div>
        )}
      </div>
      </details>

      {graph ? (
        <>
          <header className="v2-rg-header">
            <div>
              <div className="v2-eyebrow">
                {graph.archived
                  ? "Archived · read only"
                  : graph.completed
                    ? "Completion criterion satisfied"
                    : activeGraphId === graph.graph_id
                      ? "Attached to this thread"
                      : "Open, not attached"}
              </div>
              <h3>{graph.title}</h3>
              <details className="v2-rg-goal-disclosure">
                <summary>Research question & completion criteria</summary>
              <p>{graph.question}</p>
              <div className="v2-rg-goal">
                <strong>Completion criterion</strong>
                <span>{graph.completion_criterion}</span>
              </div>
              <div className="v2-rg-goal">
                <strong>Decision preferences</strong>
                <span>{graph.decision_preferences || "No explicit user preference recorded."}</span>
              </div>
              </details>
              <div className="v2-rg-summary">
                <span>{countLabel(graph.counts.hypotheses, "hypothesis", "hypotheses")}</span>
                <span>{countLabel(graph.counts.experiments, "experiment")}</span>
                <span>{countLabel(graph.counts.results, "result")}</span>
                {graph.counts.external_handoffs ? (
                  <span>{countLabel(graph.counts.external_handoffs, "external handoff")} awaiting results</span>
                ) : null}
                <span>{streamStatus}</span>
              </div>
            </div>
            <div className="v2-rg-header-actions">
              {activeGraphId === graph.graph_id ? (
                <button type="button" className="v2-ghost-btn" onClick={() => bindGraph("")} disabled={mutating}>
                  <Unlink size={14} /> Detach
                </button>
              ) : (
                <button type="button" className="v2-primary-btn" onClick={() => bindGraph(graph.graph_id)} disabled={mutating || !thread?.thread_id}>
                  <Link2 size={14} /> Attach to thread
                </button>
              )}
              {activeGraphId === graph.graph_id && threadFocusNodeId ? (
                <button type="button" className="v2-ghost-btn" onClick={() => setThreadFocus("")} disabled={mutating}>
                  <Focus size={14} /> Clear thread focus
                </button>
              ) : null}
              <button
                type="button"
                className="v2-ghost-btn"
                onClick={toggleCompletion}
                disabled={mutating || graph.archived || (!graph.completed && !graph.counts.results)}
                title={!graph.completed && !graph.counts.results ? "Record at least one Result before completing the graph." : ""}
              >
                {graph.completed ? "Reopen research" : "Mark criterion satisfied"}
              </button>
              <button
                type="button"
                className="v2-ghost-btn"
                onClick={() => openModal("graph", {
                  title: graph.title,
                  question: graph.question,
                  completion_criterion: graph.completion_criterion,
                  decision_preferences: graph.decision_preferences || "",
                })}
                disabled={mutating || graph.archived}
              >
                Edit research goal
              </button>
              <button
                type="button"
                className="v2-ghost-btn"
                disabled={mutating}
                onClick={() => mutate(
                  () => apiFetch(
                    `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}`,
                    {
                      method: "PATCH",
                      body: JSON.stringify({ expected_revision: graph.revision, archived: !graph.archived }),
                    },
                  ),
                  graph.archived ? "Graph restored." : "Graph archived.",
                )}
              >
                <Archive size={14} /> {graph.archived ? "Restore" : "Archive"}
              </button>
            </div>
          </header>

          <div className="v2-rg-add-actions" aria-label="Add scientific input">
            <span className="v2-rg-input-label">Add scientific input</span>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("hypothesis")}><CirclePlus size={14} /> Hypothesis or idea</button>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("result", resultDefaults())}><CirclePlus size={14} /> Observation or result</button>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("experiment")}><CirclePlus size={14} /> Experiment proposal</button>
          </div>

          {(payload?.decisions || []).length ? (
            <details className="v2-rg-confirmation">
              <summary>Research decisions and open premises ({payload.decisions.length})</summary>
              {payload.decisions.map((decision) => (
                <section key={decision.decision_id}>
                  <h4>{decision.body.problem} · {decision.body.disposition}</h4>
                  <p>{decision.body.reason}</p>
                  {decision.review.assessment ? <p>Independent reconsideration: {decision.review.assessment}</p> : null}
                  {decision.review.remedy_experiment_id ? <button type="button" className="v2-link-btn" onClick={() => setSelectedNodeId(decision.review.remedy_experiment_id)}>Open recommended validation</button> : null}
                  {decision.body.validation_result_ids?.map((nodeId) => <button key={nodeId} type="button" className="v2-link-btn" onClick={() => setSelectedNodeId(nodeId)}>Validation result</button>)}
                  {decision.body.exception ? <p>Exception: {decision.body.exception} · {decision.body.reason}</p> : null}
                  <p>Resume when: {decision.body.resume_when || decision.review.resume_when || "Not recorded"}</p>
                </section>
              ))}
            </details>
          ) : null}
          {payload?.planning_preview ? (
            <section className="v2-rg-plan-summary" aria-label="Revision-bound experiment selection">
              <div>
                <strong>Current revision selection</strong>
                <p>{payload.planning_preview.summary}</p>
                {payload.planning_preview.status === "no_change" ? (
                  <p className="v2-muted">No new route was justified for this graph revision.</p>
                ) : ["pending", "comparing"].includes(payload.planning_preview.status) ? (
                  <p className="v2-muted">
                    Comparing {payload.planning_preview.candidate_experiment_ids?.length || 0} ready Experiment{payload.planning_preview.candidate_experiment_ids?.length === 1 ? "" : "s"} in fresh isolated pairs.
                  </p>
                ) : payload.planning_preview.status === "recommended" ? (
                  <>
                    <p>
                      Recommended next Experiment: {displayPayload.nodes.find((node) => node.node_id === payload.planning_preview.recommended_experiment_id)?.title || payload.planning_preview.recommended_experiment_id}
                    </p>
                    {payload.planning_preview.unresolved_tradeoff ? (
                      <p className="v2-muted">Remaining tradeoff: {payload.planning_preview.unresolved_tradeoff}</p>
                    ) : null}
                  </>
                ) : payload.planning_preview.status === "wait" ? (
                  <p className="v2-muted">
                    {graph.external_handoffs?.length && !payload.planning_preview.candidate_experiment_ids?.length
                      ? "No internal Experiment will launch while the external handoff awaits a laboratory or collaborator Result."
                      : "No Experiment clearly beats waiting in this revision."}
                    {payload.planning_preview.unresolved_tradeoff ? ` ${payload.planning_preview.unresolved_tradeoff}` : ""}
                  </p>
                ) : payload.planning_preview.nodes?.length ? (
                  <p className="v2-muted">The complete staged H/E set will be admitted when this planning turn finishes.</p>
                ) : null}
                {payload.planning_preview.pair_records?.length ? (
                  <>
                    <ul>
                      {payload.planning_preview.pair_records.map((row) => (
                        <li key={`${row.sequence}:${row.purpose}`}>
                          <span>
                            {displayPayload.nodes.find((node) => node.node_id === row.candidate_a_id)?.title || (row.candidate_a_id === "__wait__" ? "Wait" : row.candidate_a_id)} vs {displayPayload.nodes.find((node) => node.node_id === row.candidate_b_id)?.title || (row.candidate_b_id === "__wait__" ? "Wait" : row.candidate_b_id)}: {row.outcome}. {row.reason}
                          </span>
                          {row.decisive_source_refs?.length ? (
                            <div className="v2-muted">Decisive sources: {row.decisive_source_refs.join(", ")}</div>
                          ) : null}
                          {row.unresolved_tradeoff ? (
                            <div className="v2-muted">Unresolved tradeoff: {row.unresolved_tradeoff}</div>
                          ) : null}
                        </li>
                      ))}
                    </ul>
                  </>
                ) : null}
                {payload.planning_preview.decisive_source_refs?.length ? (
                  <p className="v2-muted">Final decisive sources: {payload.planning_preview.decisive_source_refs.join(", ")}</p>
                ) : null}
              </div>
            </section>
          ) : null}

          <div className="v2-rg-workspace">
            <GraphCanvas
              payload={displayPayload}
              selectedNodeId={selectedNodeId}
              threadFocusNodeId={threadFocusNodeId}
              onSelectNode={(nodeId) => {
                setSelectedNodeId(nodeId);
                openInspector();
              }}
            />
            <button
              type="button"
              className="v2-rg-mobile-inspector-trigger"
              onClick={openInspector}
              disabled={!selectedNode}
            >
              View selected node details
            </button>
            {inspectorOpen ? (
              <button
                type="button"
                className="v2-rg-inspector-backdrop"
                aria-label="Close node details"
                onClick={closeInspector}
              />
            ) : null}
            <aside
              className={`v2-rg-inspector ${inspectorOpen ? "open" : ""}`}
              aria-label="Research node inspector"
            >
              <header className="v2-rg-inspector-drawer-head">
                <strong>Node details</strong>
                <button
                  ref={inspectorCloseRef}
                  type="button"
                  className="v2-icon-btn"
                  aria-label="Close node details"
                  onClick={closeInspector}
                >
                  <X size={17} />
                </button>
              </header>
              {selectedNode ? (
                <>
                  <div className="v2-eyebrow">{selectedNode.kind}</div>
                  <h3>{selectedNode.title}</h3>
                  {!selectedNode.provisional && threadFocusNodeId === selectedNode.node_id ? (
                    <p className="v2-rg-state">Current focus for this thread</p>
                  ) : null}
                  {selectedNode.provisional || selectedNode.recommended ? (
                    <div className="v2-rg-provisional-note">
                      <strong>
                        {selectedNode.provisional
                          ? (selectedNode.recommended ? "Recommended temporary route" : "Temporary planning branch")
                          : "Recommended next experiment"}
                      </strong>
                      <p>
                        {selectedNode.planning_reason
                          || (selectedNode.provisional
                            ? "This branch has not been added to the durable Research Graph."
                            : "Recommended from the current graph evidence.")}
                      </p>
                    </div>
                  ) : null}
                  {selectedNode.kind === "hypothesis" ? (
                    <>
                      <p>{selectedNode.body.claim}</p>
                      <h4>Importance</h4>
                      <p className={selectedNode.body.importance ? "" : "v2-muted"}>
                        {bandLabel(selectedNode.body.importance)}
                      </p>
                      <h4>Rationale</h4>
                      <p>{selectedNode.body.rationale || "No rationale recorded."}</p>
                      <h4>Predictions</h4>
                      {selectedNode.body.predictions?.length ? (
                        <ul>{selectedNode.body.predictions.map((item) => <li key={item}>{item}</li>)}</ul>
                      ) : <p className="v2-muted">No predictions recorded.</p>}
                      <p className="v2-rg-state">
                        {selectedNode.provisional ? "Not yet part of the durable graph" : evidenceStateLabel(selectedNode.evidence_state)}
                      </p>
                    </>
                  ) : null}
                  {selectedNode.kind === "experiment" ? (
                    <>
                      <h4>Objective</h4>
                      <p>{selectedNode.body.objective}</p>
                      <h4>Plan</h4>
                      <p className={selectedNode.body.plan_summary ? "" : "v2-muted"}>
                        {selectedNode.body.plan_summary || "Not specified yet; this proposal remains a draft."}
                      </p>
                      <h4>Decision rule</h4>
                      <p className={selectedNode.body.decision_rule ? "" : "v2-muted"}>
                        {selectedNode.body.decision_rule || "Not specified yet; add one before marking the experiment ready."}
                      </p>
                      {selectedNode.body.estimated_compute_cost ? (
                        <><h4>Estimated compute cost</h4><p>{bandLabel(selectedNode.body.estimated_compute_cost)}</p></>
                      ) : null}
                      {selectedNode.state === "blocked" && selectedNode.body.blocking_reason ? (
                        <>
                          <h4>Why it is blocked</h4>
                          <p>{selectedNode.body.blocking_reason}</p>
                        </>
                      ) : null}
                      <p className="v2-rg-state">
                        {selectedNode.provisional ? "Temporary proposal" : experimentStateLabel(selectedNode.state, selectedNode.body.execution_lane)} · {executionLaneLabel(selectedNode.body.execution_lane)}
                      </p>
                      {!selectedNode.provisional && selectedNode.body.execution_lane === "external" ? (
                        <div className="v2-rg-provisional-note">
                          <strong>External experiment handoff</strong>
                          <p>CatMaster will not launch this Experiment. Record the laboratory or collaborator Result here when it becomes available.</p>
                        </div>
                      ) : null}
                      {selectedNode.active_launch ? (
                        <p className="v2-rg-state">
                          Launch · {launchActivityLabel(selectedNode.active_launch.activity)}
                        </p>
                      ) : null}
                    </>
                  ) : null}
                  {selectedNode.kind === "result" ? (
                    <>
                      <h4>Methods</h4>
                      <p>{selectedNode.body.methods || "Not recorded"}</p>
                      <h4>Results</h4>
                      <p>{selectedNode.body.summary}</p>
                      <h4>Conclusion and next question</h4>
                      <p>{selectedNode.body.conclusion || "Not recorded"}</p>
                      <h4>Effect on hypotheses</h4>
                      {hypotheses.some((node) => resultJudgment(selectedNode.node_id, node.node_id) !== "unjudged") ? (
                        <ul>
                          {hypotheses
                            .filter((node) => resultJudgment(selectedNode.node_id, node.node_id) !== "unjudged")
                            .map((node) => (
                              <li key={node.node_id}>
                                {relationLabel(resultJudgment(selectedNode.node_id, node.node_id))}: {node.title}
                                {judgmentDetails(selectedNode.node_id, node.node_id).scope ? <p>Scope: {judgmentDetails(selectedNode.node_id, node.node_id).scope}</p> : null}
                                {judgmentDetails(selectedNode.node_id, node.node_id).rationale ? <p>{judgmentDetails(selectedNode.node_id, node.node_id).rationale}</p> : null}
                              </li>
                            ))}
                        </ul>
                      ) : <p className="v2-muted">Not yet judged against a hypothesis.</p>}
                    </>
                  ) : null}
                  {(payload?.edges || []).filter((edge) => edge.relation === "revises" && [edge.source_node_id, edge.target_node_id].includes(selectedNode.node_id)).map((edge) => {
                    const otherId = edge.source_node_id === selectedNode.node_id ? edge.target_node_id : edge.source_node_id;
                    return <section key={`revision-${otherId}`} className="v2-rg-confirmation">
                      <h4>Scientific revision · {edge.action}</h4>
                      <p>{edge.source_node_id === selectedNode.node_id ? "Revises the earlier record:" : "Revised by:"}</p>
                      <button type="button" className="v2-link-btn" onClick={() => setSelectedNodeId(otherId)}>{(payload.nodes || []).find((node) => node.node_id === otherId)?.title || otherId}</button>
                      <p>{edge.scope}</p><p>{edge.rationale}</p>
                    </section>;
                  })}
                  <h4>Sources</h4>
                  {revisedRelatedRecords.length ? <section className="v2-rg-confirmation">
                    <p>{selectedNode.kind === "experiment" ? "A tested claim has a scientific revision. Read its scope before reusing this experiment." : "A linked Result has a scientific revision. Read its scope when interpreting the recorded judgments."}</p>
                    {revisedRelatedRecords.map((node) => <button key={node.node_id} type="button" className="v2-link-btn" onClick={() => setSelectedNodeId(node.node_id)}>{node.title}</button>)}
                  </section> : null}
                  <ReferenceList refs={selectedNode.refs} onOpenThread={onOpenThread} onOpenReference={onOpenReference}
                    onOpenDiscussion={(ref) => {
                      setSelectedGraphId(ref.discussion_graph_id);
                      setDiscussionFocus({ id: ref.discussion_id });
                      closeInspector();
                    }} />
                  <div className="v2-rg-node-actions">
                    {selectedNode.provisional ? (
                      <p className="v2-muted">Admission is automatic for the complete staged branch set; no per-route approval is needed.</p>
                    ) : (
                      <>
                        {activeGraphId === graph.graph_id && threadFocusNodeId !== selectedNode.node_id ? (
                          <button type="button" className="v2-primary-btn" disabled={mutating} onClick={() => setThreadFocus(selectedNode.node_id)}>
                            <Focus size={13} /> Set as thread focus
                          </button>
                        ) : null}
                        <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={editSelected}>Edit with confirmation</button>
                        <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("ref", { ref_kind: "note" })}><Link2 size={13} /> Add source</button>
                      </>
                    )}
                    {!selectedNode.provisional && selectedNode.kind === "hypothesis" ? (
                      <>
                        <button type="button" className="v2-primary-btn" disabled={mutating || graph.archived} onClick={() => openModal("experiment", { tests_hypothesis_ids: [selectedNode.node_id], state: "draft" })}>
                          Develop experiment proposal <ArrowRight size={13} />
                        </button>
                        <button
                          type="button"
                          className="v2-ghost-btn"
                          onClick={() => {
                            const evidenceIds = (payload.edges || [])
                              .filter((edge) => (
                                edge.target_node_id === selectedNode.node_id
                                && ["supports", "opposes", "inconclusive"].includes(edge.relation)
                              ))
                              .map((edge) => edge.source_node_id);
                            if (evidenceIds.length) {
                              setSelectedNodeId(evidenceIds[0]);
                              setNotice(`Opened ${evidenceIds.length} linked result${evidenceIds.length === 1 ? "" : "s"}; use the graph to inspect the others.`);
                            } else {
                              setNotice("No supporting, opposing, or inconclusive result is recorded yet.");
                            }
                          }}
                        >
                          <Search size={13} /> Open supporting and opposing results
                        </button>
                      </>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.state === "ready" && selectedNode.body.execution_lane !== "external" ? (
                      <button type="button" className="v2-primary-btn" onClick={() => launchExperiment(false)} disabled={mutating || graph.archived}><Play size={13} /> Run</button>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.state === "draft" ? (
                      <button
                        type="button"
                        className="v2-primary-btn"
                        disabled={mutating || graph.archived}
                        onClick={() => {
                          editSelected();
                          setForm((current) => ({ ...current, state: "ready" }));
                        }}
                      >
                        Prepare and mark ready
                      </button>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.state === "has_results" && selectedNode.body.execution_lane !== "external" ? (
                      <button type="button" className="v2-primary-btn" onClick={() => launchExperiment(true)} disabled={mutating || graph.archived}><Play size={13} /> Run replicate</button>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.body.execution_lane === "external" && ["ready", "has_results"].includes(selectedNode.state) ? (
                      <button type="button" className="v2-primary-btn" disabled={mutating || graph.archived} onClick={() => openModal("result", resultDefaults(selectedNode))}>Record external result</button>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.active_launch?.thread_id ? (
                      <button type="button" className="v2-ghost-btn" onClick={() => onOpenThread?.(selectedNode.active_launch.thread_id)}><ExternalLink size={13} /> Open active launch</button>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "experiment" ? (
                      <>
                        <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("dependency")}>Add dependency</button>
                        {!["has_results", "blocked"].includes(selectedNode.state) ? (
                          <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("blocked")}>Mark blocked</button>
                        ) : null}
                        {["ready", "running", "has_results"].includes(selectedNode.state) && selectedNode.body.execution_lane !== "external" ? (
                          <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("result", resultDefaults(selectedNode))}>Record result</button>
                        ) : null}
                      </>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "result" ? (
                      <>
                        <button type="button" className="v2-primary-btn" disabled={mutating || graph.archived} onClick={() => openModal("hypothesis", { suggested_by_result_id: selectedNode.node_id })}>
                          Add next hypothesis yourself <ArrowRight size={13} />
                        </button>
                        <button
                          type="button"
                          className="v2-ghost-btn"
                          onClick={() => openJudgmentEditor()}
                          disabled={mutating || graph.archived || !hypotheses.length}
                        >
                          Set or clear hypothesis effect
                        </button>
                        <button
                          type="button"
                          className="v2-ghost-btn"
                          disabled={mutating || graph.archived}
                          onClick={() => openModal("experiment", {
                            state: "draft",
                            tests_hypothesis_ids: (payload.edges || [])
                              .filter((edge) => (
                                edge.source_node_id === selectedNode.node_id
                                && ["supports", "opposes", "inconclusive"].includes(edge.relation)
                              ))
                              .map((edge) => edge.target_node_id),
                          })}
                        >
                          Develop follow-up experiment
                        </button>
                        <button
                          type="button"
                          className="v2-ghost-btn danger"
                          disabled={mutating || graph.archived}
                          onClick={() => openModal("delete_result")}
                        >
                          <Trash2 size={13} /> Delete result with audit reason
                        </button>
                      </>
                    ) : null}
                  </div>
                </>
              ) : <p>Select a node to inspect its full scientific content and sources.</p>}
            </aside>
          </div>

          <div className="v2-rg-legend" aria-label="Relationship legend">
            {["tests", "produces", "supports", "opposes", "inconclusive", "suggests", "depends_on"].map((relation) => (
              <span key={relation} className={`relation-${relation}`}>{relationLabel(relation)}</span>
            ))}
          </div>
          <ResearchCollaborationPanel key={`${workspaceName}:${graphId}`} workspaceName={workspaceName}
            graphId={graphId} selectedNodeId={selectedNodeId} refreshKey={collaborationRefresh}
            discussionFocus={discussionFocus}
            onOpenThread={onOpenThread} onSelectNode={(nodeId) => { setSelectedNodeId(nodeId); openInspector(); }} />

          <details className="v2-rg-history">
            <summary>Recent graph changes ({payload?.mutation_history?.length || 0})</summary>
            {payload?.mutation_history?.length ? (
              <ol>
                {payload.mutation_history.map((entry) => {
                  const change = String(entry.payload?.change || "graph.updated").replaceAll("_", " ").replaceAll(".", " · ");
                  const deleted = entry.payload?.details?.deleted_result;
                  const reason = String(entry.payload?.details?.reason || "");
                  return (
                    <li key={entry.event_id}>
                      <strong>{deleted?.title ? `Result retracted: ${deleted.title}` : change}</strong>
                      <span>{formatUpdated(entry.created_at)}{reason ? ` · ${reason}` : ""}</span>
                    </li>
                  );
                })}
              </ol>
            ) : <p className="v2-muted">No graph mutations are recorded.</p>}
            {payload?.mutation_history_has_more ? (
              <button
                type="button"
                className="v2-ghost-btn"
                onClick={loadOlderHistory}
                disabled={historyLoading}
              >
                {historyLoading ? "Loading older changes…" : "Load older changes"}
              </button>
            ) : null}
            {historyError ? <p className="v2-error" role="alert">{historyError}</p> : null}
          </details>
        </>
      ) : null}

      {modal ? (
        <GraphModal
          title={{
            new_graph: "Create Research Graph",
            graph: "Edit research goal",
            hypothesis: "Add hypothesis",
            experiment: "Add experiment proposal",
            result: "Record observation or result",
            judgment: "Set hypothesis effect",
            edit: "Confirm scientific node edit",
            ref: "Attach source",
            dependency: "Add experiment dependency",
            blocked: "Mark experiment blocked",
            delete_result: "Delete result",
          }[modal]}
          onClose={() => setModal("")}
        >
          <form className="v2-rg-form" onSubmit={submitModal}>
            {modal === "new_graph" ? (
              <>
                <Field label="Research question"><textarea required value={form.question || ""} onChange={(event) => setForm({ ...form, question: event.target.value })} /></Field>
                <OptionalDetails label="Optional setup and initial hypotheses">
                  <Field label="Title" hint="The question is used when left empty."><input value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                  <Field label="Completion criterion" hint="Leave empty to use the default: a defensible answer supported by recorded results and traceable sources.">
                    <textarea value={form.completion_criterion || ""} onChange={(event) => setForm({ ...form, completion_criterion: event.target.value })} />
                  </Field>
                  <Field label="Decision preferences" hint="Optional stable priorities explicitly supplied by the user, such as preferring a low-cost discriminating check. Leave empty when none were stated.">
                    <textarea value={form.decision_preferences || ""} onChange={(event) => setForm({ ...form, decision_preferences: event.target.value })} />
                  </Field>
                  <fieldset className="v2-rg-seed-list">
                  <legend>Initial hypotheses (optional)</legend>
                  {(form.initial_hypotheses || []).map((item, index) => (
                    <section key={`seed-${index}`} className="v2-rg-seed-card">
                      <div className="v2-rg-seed-card-header">
                        <strong>Hypothesis {index + 1}</strong>
                        <button
                          type="button"
                          className="v2-link-btn"
                          onClick={() => setForm({
                            ...form,
                            initial_hypotheses: form.initial_hypotheses.filter((_, itemIndex) => itemIndex !== index),
                          })}
                        >
                          Remove
                        </button>
                      </div>
                      <Field label="Title (optional)"><input value={item.title || ""} onChange={(event) => setForm({
                        ...form,
                        initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                          itemIndex === index ? { ...seed, title: event.target.value } : seed
                        )),
                      })} /></Field>
                      <Field label="Falsifiable claim"><textarea value={item.claim || ""} onChange={(event) => setForm({
                        ...form,
                        initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                          itemIndex === index ? { ...seed, claim: event.target.value } : seed
                        )),
                      })} /></Field>
                      <Field label="Rationale (optional)"><textarea value={item.rationale || ""} onChange={(event) => setForm({
                        ...form,
                        initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                          itemIndex === index ? { ...seed, rationale: event.target.value } : seed
                        )),
                      })} /></Field>
                      <Field label="Observable predictions (optional)" hint="One prediction per line."><textarea value={item.predictions || ""} onChange={(event) => setForm({
                        ...form,
                        initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                          itemIndex === index ? { ...seed, predictions: event.target.value } : seed
                        )),
                      })} /></Field>
                      <Field label="Relative importance" hint="Scientific importance within this graph, not confidence that it is true.">
                        <select value={item.importance || ""} onChange={(event) => setForm({
                          ...form,
                          initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                            itemIndex === index ? { ...seed, importance: event.target.value } : seed
                          )),
                        })}>
                          <option value="">Not specified</option>
                          <option value="low">Low</option>
                          <option value="medium">Medium</option>
                          <option value="high">High</option>
                        </select>
                      </Field>
                      <OptionalSourceFields
                        form={item}
                        setForm={(nextSeed) => setForm({
                          ...form,
                          initial_hypotheses: form.initial_hypotheses.map((seed, itemIndex) => (
                            itemIndex === index ? nextSeed : seed
                          )),
                        })}
                        hint="Attach the literature, note, or other source that motivated this hypothesis. More sources can be attached later."
                      />
                    </section>
                  ))}
                  <button
                    type="button"
                    className="v2-ghost-btn"
                    onClick={() => setForm({
                      ...form,
                      initial_hypotheses: [
                        ...(form.initial_hypotheses || []),
                        {
                          title: "",
                          claim: "",
                          rationale: "",
                          predictions: "",
                          importance: "",
                          ref_kind: "note",
                          ref_id: "",
                        },
                      ],
                    })}
                  >
                    <CirclePlus size={13} /> {(form.initial_hypotheses || []).length ? "Add another hypothesis" : "Add initial hypothesis"}
                  </button>
                  </fieldset>
                  <label className="v2-rg-checkbox"><input type="checkbox" checked={form.attach !== false} onChange={(event) => setForm({ ...form, attach: event.target.checked })} /> Attach to this thread</label>
                </OptionalDetails>
              </>
            ) : null}
            {modal === "graph" ? (
              <>
                <Field label="Title"><input required value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                <Field label="Research question"><textarea required value={form.question || ""} onChange={(event) => setForm({ ...form, question: event.target.value })} /></Field>
                <Field label="Completion criterion" hint="State what recorded evidence would make this research question sufficiently answered."><textarea required value={form.completion_criterion || ""} onChange={(event) => setForm({ ...form, completion_criterion: event.target.value })} /></Field>
                <Field label="Decision preferences" hint="Only record stable comparison preferences explicitly stated by the user; this is not a score or approval rule."><textarea value={form.decision_preferences || ""} onChange={(event) => setForm({ ...form, decision_preferences: event.target.value })} /></Field>
              </>
            ) : null}
            {modal === "hypothesis" ? (
              <>
                <Field label="Falsifiable claim"><textarea required value={form.claim || ""} onChange={(event) => setForm({ ...form, claim: event.target.value })} /></Field>
                <OptionalDetails>
                  <Field label="Title"><input value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                  <Field label="Rationale"><textarea value={form.rationale || ""} onChange={(event) => setForm({ ...form, rationale: event.target.value })} /></Field>
                  <Field label="Observable predictions" hint="One prediction per line."><textarea value={form.predictions || ""} onChange={(event) => setForm({ ...form, predictions: event.target.value })} /></Field>
                  <Field label="Relative importance" hint="Optional; leave unspecified when it has not been assessed."><select value={form.importance || ""} onChange={(event) => setForm({ ...form, importance: event.target.value })}><option value="">Not specified</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></Field>
                  <OptionalSourceFields form={form} setForm={setForm} />
                </OptionalDetails>
              </>
            ) : null}
            {modal === "experiment" ? (
              <>
                <Field label="Objective"><textarea required value={form.objective || ""} onChange={(event) => setForm({ ...form, objective: event.target.value })} /></Field>
                <OptionalDetails label="Optional planning and execution details">
                  <Field label="Title"><input value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                  <Field label="Plan summary" hint={form.execution_lane === "external" ? "For an external handoff, give the laboratory or collaborator an actionable preparation or measurement plan." : "Required only when this proposal is marked ready to run."}><textarea required={(form.state || "draft") === "ready"} value={form.plan_summary || ""} onChange={(event) => setForm({ ...form, plan_summary: event.target.value })} /></Field>
                  <Field label="Decision rule" hint={form.execution_lane === "external" ? "State how the returned observation changes the tested hypotheses." : "Required only when this proposal is marked ready to run."}><textarea required={(form.state || "draft") === "ready"} value={form.decision_rule || ""} onChange={(event) => setForm({ ...form, decision_rule: event.target.value })} /></Field>
                  <fieldset className="v2-rg-judgments">
                  <legend>Hypotheses tested</legend>
                  {hypotheses.length ? hypotheses.map((node) => (
                    <label key={node.node_id} className="v2-rg-check-row">
                      <input
                        type="checkbox"
                        checked={(form.tests_hypothesis_ids || []).includes(node.node_id)}
                        onChange={(event) => {
                          const current = new Set(form.tests_hypothesis_ids || []);
                          if (event.target.checked) current.add(node.node_id);
                          else current.delete(node.node_id);
                          setForm({ ...form, tests_hypothesis_ids: [...current] });
                        }}
                      />
                      <span>{node.title}</span>
                    </label>
                  )) : <p className="v2-muted">No hypothesis exists yet; this proposal can be linked later.</p>}
                  </fieldset>
                  <Field label="Execution lane"><select value={form.execution_lane || "experiment"} onChange={(event) => setForm({ ...form, execution_lane: event.target.value })}><option value="experiment">Experiment</option><option value="research">Research</option><option value="literature_review">Literature review</option><option value="external">External lab / collaborator</option></select></Field>
                  <Field label="Estimated compute cost" hint="Optional; leave unspecified rather than inventing an estimate."><select value={form.estimated_compute_cost || ""} onChange={(event) => setForm({ ...form, estimated_compute_cost: event.target.value })}><option value="">Not specified</option><option value="none">None</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></Field>
                  <Field label="Readiness"><select value={form.state || "draft"} onChange={(event) => setForm({ ...form, state: event.target.value })}><option value="draft">Draft</option><option value="ready">{experimentStateLabel("ready", form.execution_lane)}</option></select></Field>
                  <OptionalSourceFields form={form} setForm={setForm} />
                </OptionalDetails>
              </>
            ) : null}
            {modal === "result" ? (
              <>
                <Field label="Methods" hint="Evidence/data, comparison, analysis and validation actually used."><textarea value={form.methods || ""} onChange={(event) => setForm({ ...form, methods: event.target.value })} /></Field>
                <Field label="Conclusion and next question" hint="What this method supports, leaves open, and what could distinguish the alternatives."><textarea value={form.conclusion || ""} onChange={(event) => setForm({ ...form, conclusion: event.target.value })} /></Field>
                <Field label="Observed or derived result" hint="Separate observation, derived analysis, and causal interpretation. Include modality, applicable conditions, or provenance when they affect meaning; do not assign a global evidence grade."><textarea required value={form.summary || ""} onChange={(event) => setForm({ ...form, summary: event.target.value })} /></Field>
                <OptionalDetails label="Optional provenance and interpretation">
                  <Field label="Title"><input value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                  <Field label="Producing experiment" hint="Leave empty for a literature finding, collaborator result, historical observation, or other evidence obtained outside this graph.">
                  <select value={form.experiment_node_id || ""} onChange={(event) => setForm({ ...form, experiment_node_id: event.target.value })}>
                    <option value="">No Research Graph experiment</option>
                    {resultExperiments.map((node) => <option key={node.node_id} value={node.node_id}>{node.title}</option>)}
                  </select>
                  </Field>
                  <fieldset className="v2-rg-judgments">
                  <legend>Effect on hypotheses</legend>
                  {hypotheses.map((node) => (
                    <label key={node.node_id}><span>{node.title}</span><select value={form.judgments?.[node.node_id] || ""} onChange={(event) => setForm({ ...form, judgments: { ...(form.judgments || {}), [node.node_id]: event.target.value } })}><option value="">Not judged</option><option value="supports">Supports</option><option value="opposes">Opposes</option><option value="inconclusive">Inconclusive</option></select></label>
                  ))}
                  </fieldset>
                  <OptionalSourceFields
                    form={form}
                    setForm={setForm}
                    hint="Attach the DOI, URL, note, artifact, run, thread, or message that preserves the observation. More sources can be attached later."
                  />
                </OptionalDetails>
              </>
            ) : null}
            {modal === "judgment" && selectedNode?.kind === "result" ? (
              <>
                <Field
                  label="Hypothesis"
                  hint="Choose the hypothesis whose interpretation should be added, replaced, or cleared."
                >
                  <select
                    required
                    value={form.hypothesis_node_id || ""}
                    onChange={(event) => {
                      const hypothesisNodeId = event.target.value;
                      setForm({
                        ...form,
                        ...judgmentDetails(selectedNode.node_id, hypothesisNodeId),
                        scope: judgmentDetails(selectedNode.node_id, hypothesisNodeId).scope || "",
                        rationale: judgmentDetails(selectedNode.node_id, hypothesisNodeId).rationale || "",
                        hypothesis_node_id: hypothesisNodeId,
                        relation: hypothesisNodeId
                          ? resultJudgment(selectedNode.node_id, hypothesisNodeId)
                          : "unjudged",
                      });
                    }}
                  >
                    <option value="">Choose a hypothesis</option>
                    {hypotheses.map((node) => (
                      <option key={node.node_id} value={node.node_id}>{node.title}</option>
                    ))}
                  </select>
                </Field>
                <Field
                  label="Effect"
                  hint="Unjudged clears the existing interpretation for this result-hypothesis pair without deleting either node."
                >
                  <select
                    value={form.relation || "unjudged"}
                    onChange={(event) => setForm({ ...form, relation: event.target.value })}
                  >
                    <option value="supports">Supports</option>
                    <option value="opposes">Opposes</option>
                    <option value="inconclusive">Inconclusive</option>
                    <option value="unjudged">Not judged / clear existing effect</option>
                  </select>
                </Field>
                <Field label="Applicable scope"><textarea value={form.scope || ""} onChange={(event) => setForm({ ...form, scope: event.target.value })} /></Field>
                <Field label="Scientific reason"><textarea value={form.rationale || ""} onChange={(event) => setForm({ ...form, rationale: event.target.value })} /></Field>
              </>
            ) : null}
            {modal === "edit" && selectedNode ? (
              <>
                <p className="v2-rg-confirmation">You are editing shared cross-thread scientific state. Review the complete fields before saving.</p>
                <Field label="Title"><input required value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                {selectedNode.kind === "hypothesis" ? <><Field label="Claim"><textarea required value={form.claim || ""} onChange={(event) => setForm({ ...form, claim: event.target.value })} /></Field><Field label="Rationale"><textarea value={form.rationale || ""} onChange={(event) => setForm({ ...form, rationale: event.target.value })} /></Field><Field label="Predictions"><textarea value={form.predictions || ""} onChange={(event) => setForm({ ...form, predictions: event.target.value })} /></Field><Field label="Relative importance" hint="Optional; leave unspecified when it has not been assessed."><select value={form.importance || ""} onChange={(event) => setForm({ ...form, importance: event.target.value })}><option value="">Not specified</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></Field></> : null}
                {selectedNode.kind === "experiment" ? (
                  <>
                    <Field label="Objective">
                      <textarea
                        required
                        value={form.objective || ""}
                        onChange={(event) => setForm({ ...form, objective: event.target.value })}
                      />
                    </Field>
                    <Field
                      label="Plan summary (optional for a draft)"
                      hint={form.execution_lane === "external" ? "For an external handoff, give the laboratory or collaborator an actionable preparation or measurement plan." : "Required before this proposal can be marked ready to run."}
                    >
                      <textarea
                        required={(form.state || selectedNode.state) === "ready"}
                        value={form.plan_summary || ""}
                        onChange={(event) => setForm({ ...form, plan_summary: event.target.value })}
                      />
                    </Field>
                    <Field
                      label="Decision rule (optional for a draft)"
                      hint={form.execution_lane === "external" ? "State how the returned observation changes the tested hypotheses." : "Required before this proposal can be marked ready to run."}
                    >
                      <textarea
                        required={(form.state || selectedNode.state) === "ready"}
                        value={form.decision_rule || ""}
                        onChange={(event) => setForm({ ...form, decision_rule: event.target.value })}
                      />
                    </Field>
                    <Field label="Execution lane">
                      <select value={form.execution_lane || "experiment"} onChange={(event) => setForm({ ...form, execution_lane: event.target.value })}>
                        <option value="experiment">Experiment</option>
                        <option value="research">Research</option>
                        <option value="literature_review">Literature review</option>
                        <option value="external">External lab / collaborator</option>
                      </select>
                    </Field>
                    <Field label="Estimated compute cost">
                      <select value={form.estimated_compute_cost || ""} onChange={(event) => setForm({ ...form, estimated_compute_cost: event.target.value })}>
                        <option value="">Not specified</option>
                        <option value="none">None</option>
                        <option value="low">Low</option>
                        <option value="medium">Medium</option>
                        <option value="high">High</option>
                      </select>
                    </Field>
                    <Field
                      label="Readiness"
                      hint={["running", "has_results"].includes(selectedNode.state)
                        ? "Execution and result states change through their dedicated actions."
                        : "Drafts need only an objective; ready experiments also need a plan and decision rule."}
                    >
                      <select
                        value={form.state || selectedNode.state}
                        disabled={["running", "has_results"].includes(selectedNode.state)}
                        onChange={(event) => setForm({ ...form, state: event.target.value })}
                      >
                        {(selectedNode.state === "blocked"
                          ? ["blocked", "draft", "ready"]
                          : ["running", "has_results"].includes(selectedNode.state)
                            ? [selectedNode.state]
                            : ["draft", "ready"]
                        ).map((state) => (
                          <option key={state} value={state}>{experimentStateLabel(state, form.execution_lane)}</option>
                        ))}
                      </select>
                    </Field>
                  </>
                ) : null}
                {selectedNode.kind === "result" ? <>
                  <Field label="Methods"><textarea value={form.methods || ""} onChange={(event) => setForm({ ...form, methods: event.target.value })} /></Field>
                  <Field label="Observed or derived result"><textarea required value={form.summary || ""} onChange={(event) => setForm({ ...form, summary: event.target.value })} /></Field>
                  <Field label="Conclusion and next question"><textarea value={form.conclusion || ""} onChange={(event) => setForm({ ...form, conclusion: event.target.value })} /></Field>
                </> : null}
              </>
            ) : null}
            {modal === "ref" ? (
              <>
                <Field label="Source type"><select value={form.ref_kind || "note"} onChange={(event) => setForm({ ...form, ref_kind: event.target.value })}>{SOURCE_KINDS.map((kind) => <option value={kind} key={kind}>{SOURCE_KIND_LABELS[kind]}</option>)}</select></Field>
                <Field label="Source identifier" hint="Notes must be existing workspace file paths; messages may use thread_id:message_id."><input required value={form.ref_id || ""} onChange={(event) => setForm({ ...form, ref_id: event.target.value })} /></Field>
              </>
            ) : null}
            {modal === "dependency" ? <Field label="Depends on experiment"><select required value={form.target_node_id || ""} onChange={(event) => setForm({ ...form, target_node_id: event.target.value })}><option value="">Choose an earlier experiment</option>{experiments.filter((node) => node.node_id !== selectedNode?.node_id).map((node) => <option key={node.node_id} value={node.node_id}>{node.title}</option>)}</select></Field> : null}
            {modal === "blocked" ? <Field label="Concrete blocking reason"><textarea required value={form.reason || ""} onChange={(event) => setForm({ ...form, reason: event.target.value })} /></Field> : null}
            {modal === "delete_result" ? (
              <>
                <p className="v2-rg-confirmation">This removes the Result and its graph relations. The deleted title, source refs, relations, and your reason are recorded in recent graph history.</p>
                <Field label="Deletion reason"><textarea required value={form.reason || ""} onChange={(event) => setForm({ ...form, reason: event.target.value })} /></Field>
              </>
            ) : null}
            <footer>
              <button type="button" className="v2-ghost-btn" onClick={() => setModal("")}>Cancel</button>
              <button type="submit" className={modal === "delete_result" ? "v2-primary-btn danger" : "v2-primary-btn"} disabled={mutating}>{modal === "edit" ? "Save confirmed edit" : modal === "delete_result" ? "Delete result" : "Save"}</button>
            </footer>
          </form>
        </GraphModal>
      ) : null}
    </section>
  );
}

export default function ResearchTechTreePanel(props) {
  return (
    <ReactFlowProvider>
      <ResearchGraphPanelContent {...props} />
    </ReactFlowProvider>
  );
}
