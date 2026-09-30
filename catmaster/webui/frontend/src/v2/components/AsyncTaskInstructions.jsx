import { FileText } from "lucide-react";
import { asyncFollowupLabel } from "../activityPresentation";

export default function AsyncTaskInstructions({ instructions, error = "", defaultOpen = false }) {
  return <details className="v2-async-instructions" open={defaultOpen || undefined}>
    <summary><FileText size={15} />任务说明<span>最初委派与最新补充指令</span></summary>
    <div className="v2-async-instructions-body">
      {error ? <p role="status">{error}</p> : !instructions ? <p>正在读取任务说明…</p> : <>
        <h3>最初委派</h3>
        <div className="v2-async-instruction-text">{instructions.original || "原始委派说明未保留。"}</div>
        {instructions.followup ? <><h3>{asyncFollowupLabel(instructions.followup_status)}</h3><div className="v2-async-instruction-text">{instructions.followup}</div></> : null}
      </>}
    </div>
  </details>;
}
