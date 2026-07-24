import { useCallback, useEffect, useMemo, useState } from "react";
import { Bot, ExternalLink, Pause, Play, RefreshCw, Send } from "lucide-react";

import { apiFetch } from "../useCatMasterThreadRuntime";
import {
  compactNodeLabel,
  layoutResearchTechTree,
} from "../researchTechTree";

const EDGE_LABELS = {
  tested_by: "tested by",
  unlocks: "unlocks",
  produces: "produces",
  derives: "derived from",
  supports: "supports",
  opposes: "opposes",
  inconclusive: "inconclusive",
};

const ACTION_REQUESTABLE_STATUS = new Set(["eligible"]);

function nodeDetailRows(node) {
  if (!node) return [];
  const rows = [
    ["Type", node.kind],
    ["Status", node.status || "unknown"],
  ];
  if (node.kind === "action") {
    rows.push(["Executor", node.executor || "unknown"]);
    rows.push(["Information value", node.information_value || "unknown"]);
    rows.push(["Cost", node.cost || "unknown"]);
    if (node.failure_reason) rows.push(["Failure", node.failure_reason]);
  }
  if (node.kind === "evidence" && node.source) rows.push(["Source", node.source]);
  if (node.kind === "action" && node.rationale) {
    rows.push(["Selection basis", node.rationale]);
  }
  if (Array.isArray(node.reasons) && node.reasons.length) {
    rows.push(["Blocked by", node.reasons.join(", ")]);
  }
  return rows;
}

export default function ResearchTechTreePanel({
  thread,
  isRunning = false,
  onLaunchAction,
  onSetAutopilot,
  onOpenThread,
  onThreadUpdate,
}) {
  const [payload, setPayload] = useState(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState("");
  const [selectedNodeId, setSelectedNodeId] = useState("");
  const [requesting, setRequesting] = useState(false);
  const [requestNotice, setRequestNotice] = useState("");

  const refresh = useCallback(async () => {
    if (!thread?.thread_id) {
      setPayload(null);
      return;
    }
    setLoading(true);
    setError("");
    try {
      const next = await apiFetch(`/api/threads/${encodeURIComponent(thread.thread_id)}/hypothesis-engine`);
      setPayload(next);
      if (next?.automation?.child_thread && onThreadUpdate) {
        onThreadUpdate(next.automation.child_thread);
      }
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setLoading(false);
    }
  }, [onThreadUpdate, thread?.thread_id]);

  useEffect(() => {
    refresh();
    const activeAutomation = payload?.automation?.enabled || payload?.automation?.status === "finishing_current";
    const timer = window.setInterval(refresh, isRunning || activeAutomation ? 3000 : 15000);
    return () => window.clearInterval(timer);
  }, [isRunning, payload?.automation?.enabled, payload?.automation?.status, refresh]);

  const layout = useMemo(
    () => layoutResearchTechTree(payload?.graph),
    [payload?.graph],
  );
  const selectedNode = useMemo(
    () => layout.nodes.find((node) => node.id === selectedNodeId) || layout.nodes[0] || null,
    [layout.nodes, selectedNodeId],
  );
  const controller = payload?.controller || payload?.graph?.controller || {};
  const activePacket = controller?.active_packet || null;
  const revision = payload?.state?.revision || payload?.graph?.revision || 0;
  const automation = payload?.automation || {};

  const launchSelectedAction = useCallback(async (actionId) => {
    if (!onLaunchAction || !payload?.source_thread_id) return;
    setRequesting(true);
    setRequestNotice("");
    setError("");
    try {
      const result = await onLaunchAction({
        sourceThreadId: payload.source_thread_id,
        revision,
        actionId,
      });
      setRequestNotice(`Started ordinary Research thread ${result?.thread?.thread_id || ""}.`);
      window.setTimeout(refresh, 750);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setRequesting(false);
    }
  }, [onLaunchAction, payload?.source_thread_id, refresh, revision]);

  const setAutopilot = useCallback(async (enabled) => {
    if (!onSetAutopilot || !payload?.source_thread_id) return;
    setRequesting(true);
    setRequestNotice("");
    setError("");
    try {
      const result = await onSetAutopilot({
        sourceThreadId: payload.source_thread_id,
        enabled,
      });
      setPayload((current) => current ? {
        ...current,
        automation: result?.automation || current.automation,
      } : current);
      setRequestNotice(
        enabled
          ? "Automatic Research is active. It will launch one ordinary Research thread per eligible check."
          : "Automatic Research will not launch another check; a running check is left intact.",
      );
      window.setTimeout(refresh, 250);
    } catch (err) {
      setError(err.message || String(err));
    } finally {
      setRequesting(false);
    }
  }, [onSetAutopilot, payload?.source_thread_id, refresh]);

  const selectedActionRequestable = (
    selectedNode?.kind === "action"
    && ACTION_REQUESTABLE_STATUS.has(String(selectedNode?.status || ""))
    && controller?.phase === "ready"
    && !isRunning
  );

  return (
    <section className="v2-tab-panel v2-research-tree-panel">
      <div className="v2-panel-toolbar">
        <div>
          <h2>Research campaign map</h2>
          <p className="v2-muted">
            Hypotheses from the proposer, checks from their scientific plan, and judgments from the evidence judge.
          </p>
        </div>
        <div className="v2-research-tree-toolbar-actions">
          {payload?.available && !automation.enabled ? (
            <button type="button" className="v2-primary-btn" onClick={() => setAutopilot(true)} disabled={requesting}>
              <Bot size={14} /> Start automatic Research
            </button>
          ) : null}
          {payload?.available && automation.enabled ? (
            <button type="button" className="v2-ghost-btn" onClick={() => setAutopilot(false)} disabled={requesting}>
              <Pause size={14} /> Stop after current check
            </button>
          ) : null}
          <button type="button" className="v2-ghost-btn" onClick={refresh} disabled={loading || !thread?.thread_id}>
            <RefreshCw size={14} className={loading ? "v2-spin" : ""} />
            Refresh
          </button>
        </div>
      </div>

      {error ? <div className="v2-error">{error}</div> : null}
      {requestNotice ? (
        <div className="v2-research-tree-request-notice">
          <span>{requestNotice}</span>
        </div>
      ) : null}
      {!payload?.available && !error ? (
        <div className="v2-research-tree-empty">
          <strong>No hypothesis campaign yet</strong>
          <p>
            Ask Research to use the hypothesis proposer when a question has competing,
            falsifiable explanations. Ordinary linear work stays in the Research Kernel.
          </p>
          {payload?.engine_path ? <code>{payload.engine_path}</code> : null}
        </div>
      ) : null}

      {payload?.available ? (
        <>
          <div className="v2-research-tree-summary">
            <span><strong>{controller?.phase || "unknown"}</strong> phase</span>
            <span><strong>{controller?.status || "unknown"}</strong> controller</span>
            <span><strong>{automation.status || "off"}</strong> automatic Research</span>
            <span><strong>r{revision}</strong> revision</span>
            <span><strong>{payload.state?.hypotheses?.length || 0}</strong> hypotheses</span>
            <span><strong>{payload.state?.actions?.length || 0}</strong> checks</span>
            <span><strong>{payload.state?.evidence?.length || 0}</strong> judgments</span>
            <code>{payload.engine_path}</code>
          </div>

          {activePacket ? (
            <div className="v2-research-tree-active-packet">
              <div>
                <span>Active scientific packet</span>
                <strong>{activePacket.action_id}</strong>
                <small>
                  {activePacket.delegate_to} · {activePacket.information_value} information · {activePacket.cost} cost
                </small>
              </div>
              <div>
                <span>Scientific task</span>
                <p>{activePacket.task}</p>
                <span>Target hypotheses</span>
                <ul>
                  {(activePacket.hypotheses || []).map((hypothesis) => (
                    <li key={hypothesis.id}>
                      <strong>{hypothesis.id}</strong>: {hypothesis.claim}
                    </li>
                  ))}
                </ul>
                <span>Decision rule</span>
                <p>{activePacket.decision_rule}</p>
              </div>
            </div>
          ) : null}

          {automation.enabled || automation.status === "finishing_current" ? (
            <div className="v2-research-tree-next-action">
              <div>
                <span>Automatic Research worker</span>
                <strong>{automation.status || "ready"}</strong>
                <small>
                  It schedules one ordinary Research thread at a time. Each thread keeps its own
                  Research Kernel and checkpoint while writing judgments to this campaign.
                </small>
              </div>
              {automation.child_thread_id && onOpenThread ? (
                <button type="button" className="v2-ghost-btn" onClick={() => onOpenThread(automation.child_thread_id)}>
                  <ExternalLink size={14} /> Open Research thread
                </button>
              ) : <Play size={16} />}
            </div>
          ) : null}

          <div className="v2-research-tree-workspace">
            <div className="v2-research-tree-canvas" role="region" aria-label="Research hypothesis network">
              <svg viewBox={`0 0 ${layout.width} ${layout.height}`} aria-label="Research campaign graph">
                <defs>
                  <marker id="research-tree-arrow" markerWidth="7" markerHeight="7" refX="6" refY="3.5" orient="auto">
                    <path d="M 0 0 L 7 3.5 L 0 7 z" />
                  </marker>
                </defs>
                {layout.columns.map((column) => (
                  <g key={column.kind}>
                    <text className="v2-research-tree-column-label" x={column.x} y={30}>{column.label}</text>
                    <line className="v2-research-tree-column-rule" x1={column.x} x2={column.x + column.width} y1={44} y2={44} />
                  </g>
                ))}
                <g className="v2-research-tree-edges">
                  {layout.edges.map((edge) => (
                    <path
                      key={edge.id}
                      className={`edge-${edge.kind}`}
                      d={edge.path}
                      markerEnd="url(#research-tree-arrow)"
                    >
                      <title>{EDGE_LABELS[edge.kind] || edge.kind}</title>
                    </path>
                  ))}
                </g>
                <g className="v2-research-tree-nodes">
                  {layout.nodes.map((node) => (
                    <foreignObject key={node.id} x={node.x} y={node.y} width={node.width} height={node.height}>
                      <button
                        type="button"
                        className={`v2-research-tree-node kind-${node.kind} status-${node.status || "unknown"} ${selectedNode?.id === node.id ? "selected" : ""}`}
                        onClick={() => setSelectedNodeId(node.id)}
                        title={node.label}
                      >
                        <span>{node.kind}</span>
                        <strong>{compactNodeLabel(node.label)}</strong>
                        <small>{node.status || "unknown"}</small>
                      </button>
                    </foreignObject>
                  ))}
                </g>
              </svg>
            </div>

            <aside className="v2-research-tree-detail">
              <div className="v2-eyebrow">Selected scientific item</div>
              <h3>{selectedNode?.label || "Nothing selected"}</h3>
              {nodeDetailRows(selectedNode).map(([label, value]) => (
                <div className="v2-research-tree-detail-row" key={label}>
                  <span>{label}</span>
                  {label === "Source" ? <code>{value}</code> : <strong>{value}</strong>}
                </div>
              ))}

              {selectedNode?.kind === "hypothesis" ? (
                <div className="v2-research-tree-task">
                  <span>Rationale</span>
                  <p>{selectedNode.rationale}</p>
                  <span>Predictions</span>
                  <ul>
                    {(selectedNode.predictions || []).map((prediction) => <li key={prediction}>{prediction}</li>)}
                  </ul>
                </div>
              ) : null}

              {selectedNode?.kind === "action" ? (
                <div className="v2-research-tree-task">
                  <span>Scientific task</span>
                  <p>{selectedNode.task}</p>
                  <span>Decision rule</span>
                  <p>{selectedNode.decision_rule}</p>
                  <span>Target hypotheses</span>
                  <p>{(selectedNode.target_hypotheses || []).join(", ")}</p>
                  {selectedNode.prerequisite_action_ids?.length ? (
                    <>
                      <span>Depends on</span>
                      <p>{selectedNode.prerequisite_action_ids.join(", ")}</p>
                    </>
                  ) : null}
                </div>
              ) : null}

              {selectedNode?.kind === "evidence" ? (
                <div className="v2-research-tree-task">
                  <span>Per-hypothesis judgment</span>
                  <ul>
                    {(selectedNode.effects || []).map((effect) => (
                      <li key={effect.hypothesis_id}>
                        <strong>{effect.hypothesis_id}: {effect.verdict}</strong> · {effect.reason}
                      </li>
                    ))}
                  </ul>
                </div>
              ) : null}

              {selectedActionRequestable ? (
                <button
                  type="button"
                  className="v2-primary-btn v2-research-tree-request-action"
                  onClick={() => launchSelectedAction(selectedNode.id.replace(/^action:/, ""))}
                  disabled={requesting}
                >
                  <Send size={14} />
                  {selectedNode.executor === "human"
                    ? "Start Research thread and ask me"
                    : "Start Research thread"}
                </button>
              ) : null}
            </aside>
          </div>

          <div className="v2-research-tree-legend" aria-label="Relationship legend">
            {Object.entries(EDGE_LABELS).map(([kind, label]) => (
              <span key={kind} className={`legend-${kind}`}>{label}</span>
            ))}
          </div>
        </>
      ) : null}
    </section>
  );
}
