import { useCallback, useEffect, useRef, useState } from "react";
import { AssistantRuntimeProvider } from "@assistant-ui/react";
import { Compass, LoaderCircle, Send, Square } from "lucide-react";

import { apiFetch, useCatMasterThreadRuntime } from "../useCatMasterThreadRuntime";
import ThreadMessages from "./ThreadMessages";
import { subagentStatus } from "./AsyncSubagentCard";
import AsyncTaskInstructions from "./AsyncTaskInstructions";

export default function AsyncSubagentActivity({ item, onSelect }) {
  const [thread, setThread] = useState(null);
  const [part, setPart] = useState(item.part);
  const [filter, setFilter] = useState("all");
  const [steering, setSteering] = useState(Boolean(item.steer));
  const [text, setText] = useState("");
  const [busy, setBusy] = useState("");
  const [error, setError] = useState("");
  const [notice, setNotice] = useState("");
  const [instructions, setInstructions] = useState(null);
  const [instructionsError, setInstructionsError] = useState("");
  const inputRef = useRef(null);
  const load = useCallback(async () => {
    const result = await apiFetch(item.part.detail_ref);
    setThread(result.thread);
    if (result.part) setPart(result.part);
    setInstructions(result.instructions || null);
    setInstructionsError(result.instructions_error || "");
  }, [item.part.detail_ref]);
  useEffect(() => { load().catch(setError); }, [load]);
  useEffect(() => { if (steering) inputRef.current?.focus(); }, [steering]);
  const transformMessages = useCallback((messages) => filter === "all" ? messages : messages.map((message) => ({
    ...message, parts: message.parts.filter((part) => filter === "updates"
      ? ["text", "progress"].includes(part.type) && message.role === "assistant"
      : part.type === filter),
  })).filter((message) => message.parts.length), [filter]);
  const onThreadUpdate = useCallback((next) => {
    setThread((current) => ({ ...current, ...next }));
    load().catch(setError);
  }, [load]);
  const runtime = useCatMasterThreadRuntime({ thread, onThreadUpdate, readOnly: true, transformMessages });
  const actions = part?.actions || [];
  const steerAction = actions.find((action) => action.id === "update_async_subagent");
  const stopAction = actions.find((action) => action.id === "stop_async_subagent");

  async function act(action, payload, label) {
    if (!action?.endpoint || busy) return;
    if (action.confirmation && !window.confirm(action.confirmation)) return;
    setBusy(label); setError(""); setNotice("");
    try {
      await apiFetch(action.endpoint, { method: "POST", body: JSON.stringify(payload) });
      setNotice(label === "steer" ? "新指令已提交给这个子任务。" : "已请求中断这个子任务，正在等待后台确认。");
      if (label === "steer") { setSteering(false); setText(""); }
      await load(); await runtime.refreshMessages();
    } catch (reason) { setError(reason); }
    finally { setBusy(""); }
  }

  return <section className="v2-async-inspector" aria-label={`${part.title} 工作过程`}>
    <header className="v2-async-inspector-head">
      <div><div className="v2-eyebrow">子任务工作过程</div><h2>{part.title}</h2><small>{subagentStatus(part.status)}</small></div>
      <div className="v2-async-card-actions">
        {steerAction ? <button type="button" className="v2-ghost-btn" disabled={Boolean(busy)} onClick={() => setSteering(!steering)}><Compass size={16} />Steer · 调整指令</button> : null}
        {stopAction ? <button type="button" className="v2-ghost-btn danger" disabled={Boolean(busy)} onClick={() => act(stopAction, { action: "interrupt", run_id: thread?.active_run_id || "" }, "stop")}><Square size={14} />停止子任务</button> : null}
      </div>
    </header>
    <AsyncTaskInstructions instructions={instructions} error={instructionsError} defaultOpen={Boolean(item.showInstructions)} />
    {steering && steerAction ? <form className="v2-async-steer" onSubmit={(event) => { event.preventDefault(); act(steerAction, { message: text.trim() }, "steer"); }}>
      <label htmlFor="async-steering-text">给这个子任务的新指令</label>
      <textarea ref={inputRef} id="async-steering-text" value={text} onChange={(event) => setText(event.target.value)} rows={3} placeholder="说明需要调整的方向、约束或交付内容" />
      <small>将中断当前执行，并在同一子任务会话中带着已有进展继续。</small>
      <button type="submit" className="v2-primary-btn compact" disabled={Boolean(busy) || !text.trim()}>{busy ? <LoaderCircle size={15} className="spin" /> : <Send size={15} />}发送指令</button>
    </form> : null}
    {error ? <div className="v2-error" role="alert">{error.message || String(error)}</div> : null}
    {notice ? <p className="v2-async-notice" role="status">{notice}</p> : null}
    <nav className="v2-async-filters" aria-label="工作过程筛选">
      {[["all", "全部"], ["updates", "进展与结果"], ["reasoning", "Reasoning"], ["tool", "工具"]].map(([value, label]) => <button key={value} type="button" aria-pressed={filter === value} onClick={() => setFilter(value)}>{label}</button>)}
      <span>内容随任务更新 · 包含嵌套 worker</span>
    </nav>
    <AssistantRuntimeProvider runtime={runtime.runtime}>
      <ThreadMessages threadId={thread?.thread_id || ""} messages={transformMessages(runtime.messages)}
        activityView onSelect={onSelect} loading={!thread || runtime.loading} error={runtime.error}
        hasMore={Boolean(runtime.messagePage?.truncated)} onLoadOlder={runtime.loadOlderMessages} loadingOlder={runtime.loadingOlder} />
    </AssistantRuntimeProvider>
  </section>;
}
