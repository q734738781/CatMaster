import { useCallback, useEffect, useRef, useState } from "react";
import { MessageSquare, RefreshCw, Send, X } from "lucide-react";
import ReactMarkdown from "react-markdown";
import remarkGfm from "remark-gfm";
import { apiFetch } from "../useCatMasterThreadRuntime";
import { subagentStatus } from "./AsyncSubagentCard";

const brief = (task) => {
  const text = String(task?.task_description || "").replace(/\s+/g, " ").trim();
  return text.length > 140 ? `${text.slice(0, 140)}…` : text;
};

export default function ResearchCollaborationPanel({ workspaceName, graphId, selectedNodeId = "", refreshKey = 0, discussionFocus, onOpenThread, onSelectNode }) {
  const [tasks, setTasks] = useState({ tasks: [], next_offset: null });
  const [discussion, setDiscussion] = useState({ messages: [], next_before_seq: null });
  const [taskOffset, setTaskOffset] = useState(0);
  const [beforeSeq, setBeforeSeq] = useState(0);
  const [nodeOnly, setNodeOnly] = useState(false);
  const [topic, setTopic] = useState("");
  const [reply, setReply] = useState(null);
  const [title, setTitle] = useState("");
  const [body, setBody] = useState("");
  const [target, setTarget] = useState("");
  const [error, setError] = useState("");
  const [sending, setSending] = useState(false);
  const [reviewOutcome, setReviewOutcome] = useState("");
  const generation = useRef(0);
  const input = useRef(null);
  const panel = useRef(null);
  useEffect(() => {
    if (!discussionFocus?.id) return;
    setTopic(discussionFocus.id); setNodeOnly(false); setBeforeSeq(0);
    panel.current?.scrollIntoView({ block: "start" });
  }, [discussionFocus]);
  const base = `/api/workspaces/${encodeURIComponent(workspaceName)}/research-graphs/${encodeURIComponent(graphId)}`;
  const refresh = useCallback(async () => {
    const current = ++generation.current;
    try {
      const [nextTasks, nextDiscussion] = await Promise.all([
        apiFetch(`${base}/tasks?offset=${taskOffset}&limit=20`),
        apiFetch(`${base}/discussions?before_seq=${beforeSeq}&node_id=${encodeURIComponent(nodeOnly ? selectedNodeId : "")}&discussion_id=${encodeURIComponent(topic)}`),
      ]);
      if (current !== generation.current) return;
      setTasks(nextTasks); setDiscussion(nextDiscussion); setError("");
    } catch (err) { if (current === generation.current) setError(err.message || String(err)); }
  }, [base, taskOffset, beforeSeq, nodeOnly, selectedNodeId, topic]);
  useEffect(() => { refresh(); return () => { generation.current += 1; }; }, [refresh, refreshKey]);

  async function send(event) {
    event.preventDefault();
    setSending(true); setError("");
    try {
      await apiFetch(`${base}/discussions`, { method: "POST", body: JSON.stringify({
        body, title: reply ? "" : title, reply_to: reply?.message_id || "", target_task_id: target,
        node_id: reply ? "" : nodeOnly ? selectedNodeId : "",
        resolves_message_id: reviewOutcome ? reply.message_id : "",
        review_outcome: reviewOutcome || "addressed",
      }) });
      if (!reply) setTopic("");
      setBody(""); setTitle(""); setReply(null); setTarget(""); setBeforeSeq(0);
      setReviewOutcome("");
      await refresh();
    } catch (err) { setError(err.message || String(err)); }
    finally { setSending(false); }
  }
  function beginReply(message) {
    setReply(message); setTarget(message.author_kind === "agent" ? message.author_thread_id : "");
    setReviewOutcome("");
    input.current?.focus();
  }
  const taskLabel = (id) => {
    if (id === discussion.main_thread_id) return "主研究者";
    const task = tasks.tasks.find((item) => item.task_id === id);
    return task ? `${task.agent_name} · ${brief(task)}` : id;
  };
  const acceptsNotice = (task) => Boolean(task.accepts_discussion);

  if (!discussion.collaboration_enabled && !discussion.messages.length && !error) return null;

  return <section ref={panel} className="v2-research-collaboration" aria-label="研究协作">
    <header><div><h3><MessageSquare size={17} />研究协作</h3><p>Persistent Research 分支可直接交流。主研究者检查讨论，并按需接续原研究者。</p></div>
      <button type="button" className="v2-ghost-btn" onClick={refresh} aria-label="刷新研究协作"><RefreshCw size={15} /></button></header>
    {error ? <p className="v2-error" role="alert">{error}</p> : null}
    <div className="v2-collaboration-columns">
      <section className="v2-research-peers" aria-label="同图研究任务"><h4>同图研究任务</h4>
        {tasks.tasks.length ? tasks.tasks.map((task) => <article key={task.task_id}>
          <div className="v2-peer-heading"><strong>{task.agent_name}</strong><span>{subagentStatus(task.status === "success" ? "completed" : task.status)} · {task.task_cost}</span></div>
          <p className="v2-peer-goal">{brief(task)}</p>
          {task.progress?.summary ? <p>{task.progress.summary}</p> : <p className="v2-muted">尚未发布工作进展。</p>}
          {task.progress?.next_step ? <p><strong>接下来：</strong>{task.progress.next_step}</p> : null}
          <details><summary>完整任务说明</summary><p className="v2-collaboration-brief">{task.instructions?.original || task.task_description}</p>
            {task.instructions?.followup ? <p className="v2-collaboration-brief">补充指令：{task.instructions.followup}</p> : null}</details>
          <div className="v2-collaboration-actions">
            {onOpenThread ? <button type="button" className="v2-link-btn" onClick={() => onOpenThread(task.task_id)}>查看线程</button> : null}
            {task.focus_node_id ? <button type="button" className="v2-link-btn" onClick={() => onSelectNode?.(task.focus_node_id)}>关联节点</button> : null}
            {discussion.collaboration_enabled && acceptsNotice(task) ? <button type="button" className="v2-link-btn" onClick={() => { setReply(null); setReviewOutcome(""); setTarget(task.task_id); input.current?.focus(); }}>向此任务提问</button> : null}
          </div>
        </article>) : <p className="v2-muted">这个图还没有后台研究任务。</p>}
        <div className="v2-collaboration-actions">
          {taskOffset ? <button type="button" className="v2-ghost-btn" onClick={() => setTaskOffset(Math.max(0, taskOffset - 20))}>上一页任务</button> : null}
          {tasks.next_offset != null ? <button type="button" className="v2-ghost-btn" onClick={() => setTaskOffset(tasks.next_offset)}>更多任务</button> : null}
        </div>
      </section>
      <section className="v2-research-discussion" aria-label="研究讨论"><div className="v2-discussion-toolbar"><h4>研究讨论</h4>
        <label><input type="checkbox" checked={nodeOnly} disabled={!selectedNodeId} onChange={(e) => { setNodeOnly(e.target.checked); setBeforeSeq(0); setTopic(""); }} />仅所选节点</label></div>
        <div className="v2-discussion-feed">
          {topic ? <button type="button" className="v2-link-btn" onClick={() => { setTopic(""); setBeforeSeq(0); }}>返回全部话题</button> : null}
          {beforeSeq ? <button type="button" className="v2-link-btn" onClick={() => setBeforeSeq(0)}>返回最新讨论</button> : null}
          {discussion.next_before_seq != null ? <button type="button" className="v2-link-btn" onClick={() => setBeforeSeq(discussion.next_before_seq)}>更早的讨论</button> : null}
          {discussion.messages.length ? discussion.messages.map((message) => <article key={message.message_id}>
            <h5><button type="button" className="v2-discussion-topic" onClick={() => { setTopic(message.discussion_id); setBeforeSeq(0); }}>{message.title}</button>{message.reply_to ? <small> · 回复</small> : null}</h5>
            <div className="v2-discussion-byline"><span title={message.author_thread_id}>{message.author_kind === "user" ? "你" : taskLabel(message.author_thread_id)}</span><time>{new Date(message.created_at * 1000).toLocaleString()}</time></div>
            {message.target_task_id ? <small className="v2-discussion-target">致 {onOpenThread ? <button type="button" className="v2-link-btn" title={taskLabel(message.target_task_id)} onClick={() => onOpenThread(message.target_task_id)}>{taskLabel(message.target_task_id)}</button> : taskLabel(message.target_task_id)}</small> : null}
            <ReactMarkdown remarkPlugins={[remarkGfm]}>{message.body}</ReactMarkdown>
            {message.review_status ? <div className="v2-discussion-review" role="note">
              <strong>{({ pending: "尚待判断", addressed: "已处理", deferred: "暂不接续", follow_up: "已安排后续调查" })[message.review_status]}</strong>
              {message.review_reason ? <p>{message.review_reason}</p> : null}
              {message.review_response_id ? <button type="button" className="v2-link-btn" onClick={() => { setTopic(message.discussion_id); setBeforeSeq(0); }}>查看处理回复</button> : null}
            </div> : null}
            {message.references?.length ? <details><summary>来源与引用</summary>{message.references.map((ref, index) => <p key={index}>{ref}</p>)}</details> : null}
            <div className="v2-collaboration-actions">{discussion.collaboration_enabled ? <button type="button" className="v2-link-btn" onClick={() => beginReply(message)}>回复</button> : null}
              {message.node_id ? <button type="button" className="v2-link-btn" onClick={() => onSelectNode?.(message.node_id)}>关联节点</button> : null}</div>
          </article>) : <p className="v2-muted">暂无讨论。需要共享线索或澄清分工时，在这里留言。</p>}
        </div>
        {discussion.collaboration_enabled ? <form className="v2-discussion-compose" onSubmit={send}>
          {reply ? <div className="v2-discussion-reply">回复：{reply.title}<button type="button" className="v2-link-btn" onClick={() => { setReply(null); setTarget(""); setReviewOutcome(""); }} aria-label="取消回复"><X size={14} /></button></div>
            : <input aria-label="讨论标题" placeholder="讨论标题" required value={title} onChange={(e) => setTitle(e.target.value)} />}
          <select aria-label="留言对象" value={target} onChange={(e) => setTarget(e.target.value)}><option value="">共享讨论 · 不定向通知</option>
            {target && !tasks.tasks.some((task) => task.task_id === target) ? <option value={target}>{target}</option> : null}
            {tasks.tasks.filter(acceptsNotice).map((task) => <option key={task.task_id} value={task.task_id}>{taskLabel(task.task_id)}</option>)}</select>
          <textarea ref={input} aria-label="讨论内容" placeholder="问题、阶段发现、方法提醒或来源链接…" required rows={3} value={body} onChange={(e) => setBody(e.target.value)} />
          {reply ? <select aria-label="处理意见" value={reviewOutcome} onChange={(e) => setReviewOutcome(e.target.value)}>
            <option value="">普通回复</option><option value="addressed">记录为已处理 · 正文说明答案或修正</option><option value="deferred">暂不接续 · 正文说明理由及恢复条件</option>
          </select> : null}
          <div className="v2-discussion-send"><small>留言不唤醒任务。主研究者在已有运行中判断是否需要进一步调查；分支之间仍可直接回复。</small><button type="submit" className="v2-primary-btn" disabled={sending || !body.trim() || (!reply && !title.trim())}><Send size={14} />{sending ? "发送中…" : "留言"}</button></div>
        </form> : <p className="v2-muted">历史讨论可查阅；协作与接续判断仅在 Persistent Research 会话启用。</p>}
      </section>
    </div>
  </section>;
}
