import { ArrowUpRight, FileText, ListChecks, Workflow } from "lucide-react";

import { todoSummary } from "../todoPanel.js";
import { displayValue, isInternalStoragePath, userFacingFileTitle } from "../presentation.js";
import ResearchSessionPanel from "./ResearchSessionPanel";
import AsyncSubagentCard from "./AsyncSubagentCard";

function todoStatusClass(status) {
  const value = String(status || "pending").toLowerCase().replace(/[^a-z0-9_-]+/g, "-");
  return value || "pending";
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

export default function PlanPanel({
  groups,
  artifacts,
  onSelectArtifact,
  thread,
  subagents,
  onAsyncSubagentUpdate,
  onAsyncSubagentStop,
  researchActivity,
  researchSessionProps,
}) {
  const rows = Array.isArray(groups) ? groups : [];
  const summary = todoSummary(rows);
  const asyncRows = Array.isArray(subagents) ? subagents : [];
  const pendingRuns = Number(thread?.pending_run_count || 0);
  const activeCount = asyncRows.length + (thread?.active_run_id ? 1 : 0);

  return (
    <aside id="task-context-panel" className="v2-right-context v2-plan-sidebar" aria-label="Task context">
      <header className="v2-task-context-head">
        <div>
          <h2>Task context</h2>
          <div className="v2-context-subtitle">Progress & outputs</div>
        </div>
        {activeCount > 0 ? (
          <small>{activeCount} active</small>
        ) : null}
      </header>
      <div className="v2-task-context-scroll">
        <section className="v2-output-panel" aria-label="Conversation outputs">
          <div className="v2-output-heading"><FileText size={16} /><h3>Outputs</h3><small>{artifacts?.length || 0}</small></div>
          {artifacts?.length ? (
            <ul className="v2-output-list">
              {artifacts.map((artifact) => (
                <li key={artifact.artifact_id || artifact.path}>
                  <button type="button" title={userFacingFileTitle(artifact.title, artifact.path)} onClick={() => onSelectArtifact?.({ type: "artifact", artifact_id: artifact.artifact_id, path: artifact.path, artifact })}>
                    <FileText size={17} />
                    <span><strong>{userFacingFileTitle(artifact.title, artifact.path, "Research output")}</strong><small>{isInternalStoragePath(artifact.path) ? "Open preview" : artifact.path || "Open preview"}</small></span>
                    <ArrowUpRight size={14} />
                  </button>
                </li>
              ))}
            </ul>
          ) : <p className="v2-context-empty">Reports, figures and files from this conversation will appear here.</p>}
        </section>
        <section className="v2-native-runs" aria-label="Agent activity">
          <div className="v2-native-runs-head">
            <Workflow size={16} aria-hidden="true" />
            <div>
              <h3>Activity</h3>
              <small>Runs & background specialists</small>
            </div>
          </div>
          <div className="v2-native-run-summary">
            <span>{thread?.active_run_id ? "Turn active" : "Thread ready"}</span>
            {pendingRuns > 0 ? <strong>{pendingRuns} queued</strong> : <small>No queued runs</small>}
          </div>
          {asyncRows.length ? (
            <div className="v2-native-subagents">
              {asyncRows.map((part) => <AsyncSubagentCard key={part.id} part={part} compact onSelect={onSelectArtifact} />)}
            </div>
          ) : <p className="v2-context-empty">No background specialists running. Completed work remains in the conversation.</p>}

        </section>
      <section className="v2-todo-panel">
        <div className="v2-todo-head">
          <div className="v2-plan-heading">
            <ListChecks size={16} aria-hidden="true" />
            <div>
              <h3>Plan</h3>
            </div>
          </div>
          <small>{summary.total ? `${summary.done}/${summary.total} done` : "0 items"}</small>
        </div>
        <div className="v2-todo-groups">
          {rows.map((group) => (
            <section key={group.source} className="v2-todo-group">
              <div className="v2-todo-source">
                <span title={displayValue(group.source, "Plan")}>{displayValue(group.source, "Plan")}</span>
                <small>{(group.rows || []).filter((item) => ["done", "completed", "complete"].includes(item.status)).length}/{group.rows?.length || 0}</small>
              </div>
              <ol>
                {(group.rows || []).map((todo, index) => (
                  <li key={`${todo.content}-${index}`} className={`status-${todoStatusClass(todo.status)}`}>
                    <span title={displayValue(todo.content, "Plan item")}>{displayValue(todo.content, "Plan item")}</span>
                    <small>{statusLabel(todo.status || "pending")}</small>
                  </li>
                ))}
              </ol>
            </section>
          ))}
          {!rows.length ? (
            <div className="v2-empty compact">A step-by-step plan will appear here when the task needs one.</div>
          ) : null}
        </div>
      </section>
        {researchActivity ? (
          <ResearchSessionPanel
            activity={researchActivity}
            heading="Research progress"
            {...(researchSessionProps || {})}
          />
        ) : null}
      </div>
    </aside>
  );
}
