import { useEffect, useMemo, useState } from "react";
import { ChevronDown, LoaderCircle, Timer } from "lucide-react";

import { formatElapsedSeconds, sortActiveOperations } from "../activeOperations.js";
import { displayValue } from "../presentation.js";

function startedTime(value) {
  const seconds = Number(value || 0);
  if (!(seconds > 0)) return "Start time not recorded";
  return `Started ${new Date(seconds * 1000).toLocaleTimeString([], {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  })}`;
}

function elapsedTime(startedAt, now) {
  const seconds = Number(startedAt || 0);
  if (!(seconds > 0)) return "Elapsed unavailable";
  return `Elapsed ${formatElapsedSeconds((now - (seconds * 1000)) / 1000)}`;
}

export default function ActiveOperationsPanel({ parts, isRunning = false, pendingRuns = 0, onSelect }) {
  const operations = useMemo(() => sortActiveOperations(parts), [parts]);
  const queued = Math.max(0, Number(pendingRuns) || 0);
  const [expanded, setExpanded] = useState(false);
  const [now, setNow] = useState(() => Date.now());

  useEffect(() => {
    if (!operations.length) return undefined;
    setNow(Date.now());
    const timer = window.setInterval(() => setNow(Date.now()), 1000);
    return () => window.clearInterval(timer);
  }, [operations.length]);

  useEffect(() => {
    if (operations.length <= 1) setExpanded(false);
  }, [operations.length]);

  // Tool calls can finish while the same turn is still generating or queued.
  // Keep the run indicator tied to native thread state between operations.
  if (!operations.length && !isRunning && !queued) return null;
  const visible = expanded ? operations : operations.slice(0, 1);
  const summary = [
    operations.length ? `${operations.length} active ${operations.length === 1 ? "operation" : "operations"}` : "",
    queued ? `${queued} queued` : "",
  ].filter(Boolean).join(" · ");
  return (
    <section className="v2-running-now" aria-label="Running now">
      <div className="v2-running-now-head">
        <LoaderCircle className="spin" size={16} aria-hidden="true" />
        <strong>Running now</strong>
        <small>{summary}</small>
        {operations.length > 1 ? (
          <button
            type="button"
            className="v2-running-now-toggle"
            aria-expanded={expanded}
            onClick={() => setExpanded((value) => !value)}
          >
            {expanded ? "Show oldest only" : "Show all"}
            <ChevronDown className={expanded ? "expanded" : ""} size={14} aria-hidden="true" />
          </button>
        ) : null}
      </div>
      {visible.length ? <div className="v2-running-now-list">
        {visible.map((part) => (
          <button
            key={part.id}
            type="button"
            className="v2-running-operation"
            onClick={() => onSelect?.({ type: "activity", part })}
          >
            <span className="v2-running-operation-title">
              <strong>{displayValue(part.title, "Active operation")}</strong>
              <code>Running</code>
            </span>
            {part.summary ? <span className="v2-running-operation-summary">{displayValue(part.summary, "")}</span> : null}
            {part.started_at > 0 ? <span className="v2-running-operation-time">
              <Timer size={13} aria-hidden="true" />
              {startedTime(part.started_at)}
              <span aria-hidden="true">·</span>
              {elapsedTime(part.started_at, now)}
            </span> : null}
          </button>
        ))}
      </div> : null}
    </section>
  );
}
