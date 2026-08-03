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
  Bot,
  CirclePlus,
  ExternalLink,
  Focus,
  Link2,
  Network,
  Pause,
  Play,
  RefreshCw,
  Search,
  Unlink,
  X,
} from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  boundedResearchGraph,
  evidenceStateLabel,
  executionLaneLabel,
  experimentStateLabel,
  orchestrationModeLabel,
  relationLabel,
} from "../researchTechTree";

const elk = new ELK();
const GRAPH_EVENT_TYPES = [
  "research_graph.updated",
  "research_graph.planning_started",
  "research_graph.planning_attached",
  "research_graph.planning_preview",
  "research_graph.planning_finished",
  "research_graph.planning_no_change",
  "research_graph.planning_stale",
];
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

function scoreLabel(value) {
  if (value === null || value === undefined || value === "") return "";
  const parsed = Number(value);
  return Number.isFinite(parsed) ? parsed.toFixed(2) : "";
}

function recommendationLabel(node) {
  if (node.innovation_recommended && node.conservative_recommended) {
    return "Innovation and conservative recommendation";
  }
  if (node.innovation_recommended) return "Innovation recommendation";
  if (node.conservative_recommended) return "Conservative recommendation";
  if (node.proposer_recommended) return "Proposer route";
  return "";
}

function nodeRankLabel(node) {
  if (node.kind === "hypothesis") {
    return node.body?.importance
      ? `${bandLabel(node.body.importance)} importance`
      : "";
  }
  if (node.kind === "experiment") {
    return [
      scoreLabel(node.innovation_score)
        ? `${scoreLabel(node.innovation_score)} innovation`
        : "",
      scoreLabel(node.conservative_score)
        ? `${scoreLabel(node.conservative_score)} conservative`
        : "",
      node.body?.estimated_compute_cost
        ? `${bandLabel(node.body.estimated_compute_cost)} compute`
        : "",
    ].filter(Boolean).join(" · ");
  }
  return "";
}

function ResearchNodeCard({ data, selected }) {
  const node = data.node;
  const provisional = node.provisional === true;
  const kindLabel = {
    hypothesis: provisional ? "Proposed hypothesis" : "Hypothesis",
    experiment: provisional ? "Proposed experiment" : "Experiment proposal",
    result: "Result",
  }[node.kind] || "Research node";
  const recommendation = recommendationLabel(node);
  const recommended = Boolean(node.innovation_recommended || node.conservative_recommended);
  const state = provisional
    ? (recommendation ? `${recommendation} · Temporary planning branch` : "Temporary planning branch")
    : recommendation
      ? `${recommendation} · ${node.kind === "experiment" ? experimentStateLabel(node.state) : evidenceStateLabel(node.evidence_state)}`
      : node.kind === "hypothesis"
        ? evidenceStateLabel(node.evidence_state)
        : node.kind === "experiment"
          ? experimentStateLabel(node.state)
          : "Result recorded";
  const rank = nodeRankLabel(node);
  return (
    <button
      type="button"
      className={`v2-rg-node kind-${node.kind} ${provisional ? "provisional" : ""} ${recommended ? "recommended" : ""} ${selected ? "selected" : ""}`}
      aria-label={`${kindLabel}: ${node.title}. ${state}${rank ? `. ${rank}` : ""}`}
      data-research-node-id={node.node_id}
      title={node.title}
      onClick={data.onSelect}
    >
      <Handle type="target" position={Position.Left} className="v2-rg-handle" />
      <span className="v2-rg-node-kind">{kindLabel}</span>
      <strong>{node.title}</strong>
      <small>{rank ? `${state} · ${rank}` : state}</small>
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

function ReferenceList({ refs, onOpenThread, onOpenReference }) {
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

function GraphCanvas({ payload, selectedNodeId, onSelectNode }) {
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
  }, [payload?.graph?.revision, visible]);

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
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [streamStatus, setStreamStatus] = useState("");
  const [modal, setModal] = useState("");
  const [form, setForm] = useState(EMPTY_FORM);
  const [inspectorOpen, setInspectorOpen] = useState(false);
  const eventIdRef = useRef(0);
  const refreshTimerRef = useRef(0);
  const inspectorCloseRef = useRef(null);
  const inspectorReturnFocusRef = useRef(null);

  const activeGraphId = String(thread?.active_research_graph_id || "");
  const graphId = selectedGraphId || activeGraphId;
  const graph = payload?.graph || null;
  const displayPayload = useMemo(() => {
    const planning = payload?.planning_preview || {};
    const planningSummary = String(payload?.planning_preview?.summary || "");
    const evaluations = new Map(
      (Array.isArray(planning.experiment_evaluations)
        ? planning.experiment_evaluations
        : []).map((row) => [String(row.experiment_id || ""), row]),
    );
    const innovationId = String(planning.innovation_recommendation || "");
    const conservativeId = String(planning.conservative_recommendation || "");
    const proposerId = String(planning.proposer_recommended_target_id || "");
    const decorateNode = (node) => {
      const nodeId = String(node.node_id || "");
      return {
        ...node,
        ...(evaluations.get(nodeId) || {}),
        innovation_recommended: Boolean(innovationId && nodeId === innovationId),
        conservative_recommended: Boolean(conservativeId && nodeId === conservativeId),
        proposer_recommended: Boolean(proposerId && nodeId === proposerId),
        planning_reason: proposerId && nodeId === proposerId ? planningSummary : "",
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
    setSelectedNodeId((current) => (
      current && next.nodes?.some((node) => node.node_id === current)
        ? current
        : next.nodes?.[0]?.node_id || ""
    ));
    return next;
  }, [graphId, thread?.thread_id, workspaceName]);

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
        refreshGraph(graphId).catch((err) => setError(err.message || String(err)));
        loadCatalog().catch(() => {});
      }, 80);
    };
    GRAPH_EVENT_TYPES.forEach((type) => stream.addEventListener(type, handleEvent));
    stream.onopen = () => setStreamStatus("Live updates connected");
    stream.onerror = () => setStreamStatus("Reconnecting live updates…");
    return () => {
      window.clearTimeout(refreshTimerRef.current);
      GRAPH_EVENT_TYPES.forEach((type) => stream.removeEventListener(type, handleEvent));
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
        () => apiFetch(base, {
          method: "POST",
          body: JSON.stringify({
            question: form.question || "",
            title: form.title || "",
            completion_criterion: form.completion_criterion || "",
            orchestration_mode: form.orchestration_mode || "manual",
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
          : { summary: form.summary || "" };
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
  });
}

function countLabel(count, singular, plural = `${singular}s`) {
  return `${count} ${count === 1 ? singular : plural}`;
}

  async function launchExperiment(replicate) {
    if (!selectedNode || !graph) return;
    const next = await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}/experiments/${encodeURIComponent(selectedNode.node_id)}/launch`,
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

  async function planNextStep(focusNodeId = "") {
    if (!graph) return;
    await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}/plan`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            focus_node_id: focusNodeId,
          }),
        },
      ),
      "Research is developing the next scientific step.",
      { updateThread: false },
    );
  }

  async function materializeProposal(proposalId) {
    if (!graph || !payload?.planning_preview?.planning_id || !proposalId) return;
    await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}/plans/${encodeURIComponent(payload.planning_preview.planning_id)}/materialize`,
        {
          method: "POST",
          body: JSON.stringify({
            expected_revision: graph.revision,
            proposal_id: proposalId,
          }),
        },
      ),
      "The selected route is now part of the Research Graph.",
    );
  }

  async function toggleAutomation() {
    if (!graph) return;
    const nextMode = graph.orchestration_mode === "auto" ? "manual" : "auto";
    await mutate(
      () => apiFetch(
        `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graph.graph_id)}`,
        {
          method: "PATCH",
          body: JSON.stringify({ expected_revision: graph.revision, orchestration_mode: nextMode }),
        },
      ),
      nextMode === "auto"
        ? "Automatic orchestration enabled."
        : "Automatic orchestration stopped; current work is left intact.",
    );
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

  function openJudgmentEditor(hypothesisNodeId = "") {
    if (selectedNode?.kind !== "result") return;
    const targetId = hypothesisNodeId || hypotheses[0]?.node_id || "";
    openModal("judgment", {
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
            orchestration_mode: "manual",
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

      <div className="v2-rg-catalog" aria-label="Research Graph catalog">
        {catalog.length ? catalog.map((item) => (
          <article
            key={item.graph_id}
            className={`v2-rg-catalog-card ${item.graph_id === graphId ? "selected" : ""} ${item.archived ? "archived" : ""}`}
          >
            <div>
              <span>
                {item.archived ? "Archived" : item.completed ? "Completed" : "Active"} · {orchestrationModeLabel(item.orchestration_mode)}
                {item.bound_to_current_thread ? " · Attached to this thread" : ""}
              </span>
              <h3>{item.title}</h3>
              <p>{item.question}</p>
              <small>
                {countLabel(item.counts.hypotheses, "hypothesis", "hypotheses")} · {countLabel(item.counts.experiments, "experiment")} · {countLabel(item.counts.results, "result")}
                {" · "}{countLabel(item.bound_thread_count, "attached thread")} · {formatUpdated(item.updated_at)}
              </small>
              <small>
                {item.frontier?.length
                  ? `Ready next: ${item.frontier.map((node) => node.title).join(", ")}`
                  : "No experiment is ready to run."}
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
              <p>{graph.question}</p>
              <div className="v2-rg-goal">
                <strong>Completion criterion</strong>
                <span>{graph.completion_criterion}</span>
              </div>
              <div className="v2-rg-summary">
                <span>{countLabel(graph.counts.hypotheses, "hypothesis", "hypotheses")}</span>
                <span>{countLabel(graph.counts.experiments, "experiment")}</span>
                <span>{countLabel(graph.counts.results, "result")}</span>
                <span>{streamStatus}</span>
              </div>
            </div>
            <div className="v2-rg-header-actions">
              {activeGraphId === graph.graph_id ? (
                <button type="button" className="v2-ghost-btn" onClick={() => bindGraph("")} disabled={mutating}>
                  <Unlink size={14} /> Detach
                </button>
              ) : (
                <button type="button" className="v2-primary-btn" onClick={() => bindGraph(graph.graph_id, selectedNodeId)} disabled={mutating || !thread?.thread_id}>
                  <Link2 size={14} /> Attach to thread
                </button>
              )}
              <button type="button" className="v2-ghost-btn" onClick={toggleAutomation} disabled={mutating || graph.archived || graph.completed}>
                {graph.orchestration_mode === "auto" ? <Pause size={14} /> : <Bot size={14} />}
                {graph.orchestration_mode === "auto" ? "Use manual orchestration" : "Enable automatic orchestration"}
              </button>
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
            <button
              type="button"
              className="v2-primary-btn"
              disabled={mutating || graph.archived || graph.completed}
              onClick={() => planNextStep("")}
            >
              <Bot size={14} /> {(payload?.nodes || []).length
                ? "Ask Research to update routes"
                : "Ask Research to propose starting routes"}
            </button>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("hypothesis")}><CirclePlus size={14} /> Hypothesis or idea</button>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("result", resultDefaults())}><CirclePlus size={14} /> Observation or result</button>
            <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("experiment")}><CirclePlus size={14} /> Experiment proposal</button>
          </div>

          {payload?.planning_preview ? (
            <section className="v2-rg-plan-summary" aria-label="Evidence-aware research planning">
              <div>
                <strong>Evidence-aware route planning</strong>
                <p>{payload.planning_preview.summary}</p>
                {payload.planning_preview.experiment_evaluations?.length ? (
                  <>
                    <p>
                      Innovation recommendation: {displayPayload.nodes.find((node) => node.node_id === payload.planning_preview.innovation_recommendation)?.title || "No selection"}
                      {" · "}
                      Conservative recommendation: {displayPayload.nodes.find((node) => node.node_id === payload.planning_preview.conservative_recommendation)?.title || "No selection"}
                    </p>
                    <ul>
                      {payload.planning_preview.experiment_evaluations.map((row) => (
                        <li key={row.experiment_id}>
                          {displayPayload.nodes.find((node) => node.node_id === row.experiment_id)?.title || row.experiment_id}: innovation {scoreLabel(row.innovation_score)}, conservative {scoreLabel(row.conservative_score)}
                        </li>
                      ))}
                    </ul>
                    {payload.planning_preview.evaluation_memo ? <p>{payload.planning_preview.evaluation_memo}</p> : null}
                  </>
                ) : <p className="v2-muted">Waiting for the independent experiment evaluation.</p>}
              </div>
            </section>
          ) : null}

          <div className="v2-rg-workspace">
            <GraphCanvas
              payload={displayPayload}
              selectedNodeId={selectedNodeId}
              onSelectNode={(nodeId) => {
                setSelectedNodeId(nodeId);
                openInspector();
                const selected = displayPayload?.nodes?.find(
                  (node) => node.node_id === nodeId,
                );
                if (
                  activeGraphId === graph.graph_id
                  && thread?.thread_id
                  && !selected?.provisional
                ) {
                  bindGraph(graph.graph_id, nodeId);
                }
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
                      {scoreLabel(selectedNode.innovation_score) || scoreLabel(selectedNode.conservative_score) ? (
                        <>
                          <h4>Current planning evaluation</h4>
                          <p>Innovation {scoreLabel(selectedNode.innovation_score)} · Conservative {scoreLabel(selectedNode.conservative_score)}</p>
                          {recommendationLabel(selectedNode) ? <p>{recommendationLabel(selectedNode)}</p> : null}
                        </>
                      ) : null}
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
                        {selectedNode.provisional ? "Temporary proposal" : experimentStateLabel(selectedNode.state)} · {executionLaneLabel(selectedNode.body.execution_lane)}
                      </p>
                      {selectedNode.active_launch ? (
                        <p className="v2-rg-state">
                          Launch · {launchActivityLabel(selectedNode.active_launch.activity)}
                        </p>
                      ) : null}
                    </>
                  ) : null}
                  {selectedNode.kind === "result" ? (
                    <>
                      <p>{selectedNode.body.summary}</p>
                      <h4>Effect on hypotheses</h4>
                      {hypotheses.some((node) => resultJudgment(selectedNode.node_id, node.node_id) !== "unjudged") ? (
                        <ul>
                          {hypotheses
                            .filter((node) => resultJudgment(selectedNode.node_id, node.node_id) !== "unjudged")
                            .map((node) => (
                              <li key={node.node_id}>
                                {relationLabel(resultJudgment(selectedNode.node_id, node.node_id))}: {node.title}
                              </li>
                            ))}
                        </ul>
                      ) : <p className="v2-muted">Not yet judged against a hypothesis.</p>}
                    </>
                  ) : null}
                  <h4>Sources</h4>
                  <ReferenceList refs={selectedNode.refs} onOpenThread={onOpenThread} onOpenReference={onOpenReference} />
                  <div className="v2-rg-node-actions">
                    {selectedNode.provisional ? (
                      <button
                        type="button"
                        className="v2-primary-btn"
                        onClick={() => materializeProposal(selectedNode.node_id)}
                        disabled={mutating || graph.archived || graph.completed}
                      >
                        <CirclePlus size={13} /> Add this route to the graph
                      </button>
                    ) : (
                      <>
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
                          className="v2-primary-btn"
                          onClick={() => planNextStep(selectedNode.node_id)}
                          disabled={mutating || graph.archived || graph.completed}
                        >
                          <Bot size={13} /> Ask Research to develop the next step
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
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.state === "ready" ? (
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
                    {!selectedNode.provisional && selectedNode.kind === "experiment" && selectedNode.state === "has_results" ? (
                      <button type="button" className="v2-primary-btn" onClick={() => launchExperiment(true)} disabled={mutating || graph.archived}><Play size={13} /> Run replicate</button>
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
                        {["ready", "running", "has_results"].includes(selectedNode.state) ? (
                          <button type="button" className="v2-ghost-btn" disabled={mutating || graph.archived} onClick={() => openModal("result", resultDefaults(selectedNode))}>Record result</button>
                        ) : null}
                      </>
                    ) : null}
                    {!selectedNode.provisional && selectedNode.kind === "result" ? (
                      <>
                        <button
                          type="button"
                          className="v2-primary-btn"
                          onClick={() => planNextStep(selectedNode.node_id)}
                          disabled={mutating || graph.archived || graph.completed}
                        >
                          <Bot size={13} /> Re-evaluate routes from this result
                        </button>
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
                  <Field label="Orchestration"><select value={form.orchestration_mode || "manual"} onChange={(event) => setForm({ ...form, orchestration_mode: event.target.value })}><option value="manual">Manual</option><option value="auto">Automatic</option></select></Field>
                  <label className="v2-rg-checkbox"><input type="checkbox" checked={form.attach !== false} onChange={(event) => setForm({ ...form, attach: event.target.checked })} /> Attach to this thread</label>
                </OptionalDetails>
              </>
            ) : null}
            {modal === "graph" ? (
              <>
                <Field label="Title"><input required value={form.title || ""} onChange={(event) => setForm({ ...form, title: event.target.value })} /></Field>
                <Field label="Research question"><textarea required value={form.question || ""} onChange={(event) => setForm({ ...form, question: event.target.value })} /></Field>
                <Field label="Completion criterion" hint="State what recorded evidence would make this research question sufficiently answered."><textarea required value={form.completion_criterion || ""} onChange={(event) => setForm({ ...form, completion_criterion: event.target.value })} /></Field>
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
                  <Field label="Plan summary" hint="Required only when this proposal is marked ready to run."><textarea required={(form.state || "draft") === "ready"} value={form.plan_summary || ""} onChange={(event) => setForm({ ...form, plan_summary: event.target.value })} /></Field>
                  <Field label="Decision rule" hint="Required only when this proposal is marked ready to run."><textarea required={(form.state || "draft") === "ready"} value={form.decision_rule || ""} onChange={(event) => setForm({ ...form, decision_rule: event.target.value })} /></Field>
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
                  <Field label="Execution lane"><select value={form.execution_lane || "experiment"} onChange={(event) => setForm({ ...form, execution_lane: event.target.value })}><option value="experiment">Experiment</option><option value="research">Research</option><option value="literature_review">Literature review</option></select></Field>
                  <Field label="Estimated compute cost" hint="Optional; leave unspecified rather than inventing an estimate."><select value={form.estimated_compute_cost || ""} onChange={(event) => setForm({ ...form, estimated_compute_cost: event.target.value })}><option value="">Not specified</option><option value="none">None</option><option value="low">Low</option><option value="medium">Medium</option><option value="high">High</option></select></Field>
                  <Field label="Readiness"><select value={form.state || "draft"} onChange={(event) => setForm({ ...form, state: event.target.value })}><option value="draft">Draft</option><option value="ready">Ready to run</option></select></Field>
                  <OptionalSourceFields form={form} setForm={setForm} />
                </OptionalDetails>
              </>
            ) : null}
            {modal === "result" ? (
              <>
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
                      hint="Required before this proposal can be marked ready to run."
                    >
                      <textarea
                        required={(form.state || selectedNode.state) === "ready"}
                        value={form.plan_summary || ""}
                        onChange={(event) => setForm({ ...form, plan_summary: event.target.value })}
                      />
                    </Field>
                    <Field
                      label="Decision rule (optional for a draft)"
                      hint="Required before this proposal can be marked ready to run."
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
                          <option key={state} value={state}>{experimentStateLabel(state)}</option>
                        ))}
                      </select>
                    </Field>
                  </>
                ) : null}
                {selectedNode.kind === "result" ? <Field label="Observed or derived result" hint="Preserve observation, analysis, conditions, and provenance as scientific attributes rather than a global evidence grade."><textarea required value={form.summary || ""} onChange={(event) => setForm({ ...form, summary: event.target.value })} /></Field> : null}
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
            <footer>
              <button type="button" className="v2-ghost-btn" onClick={() => setModal("")}>Cancel</button>
              <button type="submit" className="v2-primary-btn" disabled={mutating}>{modal === "edit" ? "Save confirmed edit" : "Save"}</button>
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
