import { ArrowUpRight, Bot, Compass, FileText, LoaderCircle } from "lucide-react";
import { asyncFollowupLabel } from "../activityPresentation";

export function subagentStatus(status) {
  return { running: "运行中", pending: "等待中", streaming: "运行中", completed: "已完成",
    failed: "失败", interrupted: "已中断", cancelled: "已取消" }[status] || status;
}

export default function AsyncSubagentCard({ part, onSelect, compact = false }) {
  const active = ["running", "pending", "streaming", "created"].includes(part.status);
  const open = (steer = false, showInstructions = false) => onSelect?.({ type: "agent", part, steer, showInstructions });
  const update = String(part.text || "").replace(/^Async specialist run is \w+\.$/, "");
  return <section className={`v2-async-card ${compact ? "compact" : ""}`} aria-label={`${part.title} 子任务`}>
    <button className="v2-async-card-heading" type="button" onClick={() => open()} disabled={!part.detail_ref}>
      <span className="v2-async-avatar"><Bot size={18} /></span>
      <span><strong>{part.title || "Specialist"}</strong><small>后台子任务 · {subagentStatus(part.status)}</small></span>
      {active ? <LoaderCircle className="spin" size={15} /> : <ArrowUpRight size={16} />}
    </button>
    <span className="v2-async-completion-policy">{part.on_completion === "notify" ? "完成后仅通知" : "完成后继续主对话"}</span>
    {part.task_cost ? <span className="v2-async-completion-policy">{active && part.capacity_state === "waiting" ? `等待 ${part.task_cost} 研究槽位 · 将自动继续` : active && part.capacity_state === "waiting_children" ? "等待独立研究分支结果 · 将自动继续" : `任务成本 · ${part.task_cost}`}</span> : null}
    {part.task_description ? <p className="v2-async-brief">{part.task_description}</p> : null}
    {part.task_followup ? <p className="v2-async-brief followup"><strong>{asyncFollowupLabel(part.task_followup_status)}</strong>{part.task_followup}</p> : null}
    {part.progress_summary ? <p className="v2-async-latest">{part.progress_summary}</p> : update ? <p className="v2-async-latest">{update}</p> : <p className="v2-async-latest muted">{active ? "等待下一条进展，可打开查看已记录的过程。" : "打开查看完整工作过程和结果。"}</p>}
    {part.progress_next_step ? <p className="v2-async-brief"><strong>接下来：</strong>{part.progress_next_step}</p> : null}
    {part.fields?.length ? <small className="v2-async-current">{part.fields.map((field) => field.value).join(" · ")}</small> : null}
    <div className="v2-async-card-actions">
      <button type="button" className="v2-ghost-btn compact" onClick={() => open()} disabled={!part.detail_ref}><ArrowUpRight size={14} />查看过程</button>
      <button type="button" className="v2-ghost-btn compact" onClick={() => open(false, true)} disabled={!part.detail_ref}><FileText size={14} />任务说明</button>
      {part.actions?.some((action) => action.id === "update_async_subagent") ? <button type="button" className="v2-ghost-btn compact" onClick={() => open(true)}><Compass size={14} />Steer</button> : null}
    </div>
  </section>;
}
