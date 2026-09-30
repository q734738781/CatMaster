import { CircleAlert, FileText, FlaskConical, Network, Pause, Play, SquareArrowOutUpRight } from "lucide-react";

import { displayValue } from "../presentation.js";
import { ACTIVE_RESEARCH_STATES, researchStateLabel } from "../researchSessions.js";

export default function ResearchSessionPanel({
  activity,
  heading = "Research Session",
  onOpenCurrent,
  onOpenGraph,
  onOpenLatestReport,
  onTogglePause,
  canTogglePause = true,
  busy = false,
}) {
  const current = activity || {};
  const state = String(current.state || "idle").toLowerCase();
  const active = ACTIVE_RESEARCH_STATES.has(state);
  const StatusIcon = current.action_required ? CircleAlert : FlaskConical;
  const status = current.automation_paused && active
    ? `${researchStateLabel(state)} · automation paused after current`
    : researchStateLabel(state);

  return (
    <section className={`v2-research-session-panel state-${state}`} aria-label="Research Session activity">
      <div className="v2-research-session-head">
        <StatusIcon className={active && !current.action_required ? "pulse" : ""} size={17} aria-hidden="true" />
        <div>
          <strong>{heading}</strong>
          <small>{status}</small>
        </div>
        <span>
          {Number(current.execution_count || 0)} research tasks
          {Number(current.decision_round_count || 0) > 0
            ? ` · ${current.decision_round_count} decisions`
            : ""}
        </span>
      </div>
      {current.current_title ? (
        <p className="v2-research-session-current" title={current.current_title}>
          {displayValue(current.current_title, "Research activity")}
        </p>
      ) : null}
      {current.latest_progress ? (
        <p>{displayValue(current.latest_progress, "")}</p>
      ) : null}
      {current.decision_summary ? (
        <p className="v2-muted">Latest decision: {displayValue(current.decision_summary, "")}</p>
      ) : null}
      {current.latest_result_title ? (
        <div className="v2-research-session-result">
          <strong>Latest result</strong>
          <span title={current.latest_result_title}>
            {displayValue(current.latest_result_title, "Recorded result")}
          </span>
          {current.latest_result_summary ? (
            <p>{displayValue(current.latest_result_summary, "")}</p>
          ) : null}
        </div>
      ) : null}
      <div className="v2-research-session-actions">
        {current.active_child_thread_id ? (
          <button type="button" className="v2-ghost-btn" onClick={onOpenCurrent}>
            <SquareArrowOutUpRight size={14} />
            Open current thread
          </button>
        ) : null}
        {current.latest_report_artifact_id || current.latest_report_path ? (
          <button type="button" className="v2-primary-btn v2-report-link" onClick={onOpenLatestReport}>
            <FileText size={14} />
            Open latest report
          </button>
        ) : null}
        <button type="button" className="v2-ghost-btn" onClick={onOpenGraph}>
          <Network size={14} />
          Open Research Graph
        </button>
        {canTogglePause ? (
          <button type="button" className="v2-ghost-btn" disabled={busy} onClick={onTogglePause}
            title="Pause or resume the research session and its automatic continuation. Running child tasks retain their own Stop control.">
            {current.automation_paused ? <Play size={14} /> : <Pause size={14} />}
            {current.automation_paused ? "Resume research" : "Pause research"}
          </button>
        ) : null}
      </div>
    </section>
  );
}
