import {
  startTransition,
  useDeferredValue,
  useEffect,
  useRef,
  useState,
} from "react";

function escapePath(value) {
  if (value === null || value === undefined) {
    return "";
  }
  return encodeURIComponent(String(value));
}

async function apiFetch(url, options = {}) {
  const response = await fetch(url, {
    headers: {
      "Content-Type": "application/json",
      ...(options.headers || {}),
    },
    ...options,
  });
  if (!response.ok) {
    throw new Error((await response.text()) || `Request failed: ${response.status}`);
  }
  return response.json();
}

function formatTime(ts) {
  if (!ts) {
    return "";
  }
  try {
    return new Date(ts * 1000).toLocaleTimeString();
  } catch {
    return "";
  }
}

function joinItems(items) {
  return (items || []).filter(Boolean).join(" · ");
}

function StatusPill({ status }) {
  return <span className={`status-pill status-${String(status || "idle").replaceAll("_", "-")}`}>{status || "idle"}</span>;
}

function MetricCard({ label, value, note }) {
  return (
    <div className="metric-card">
      <div className="metric-label">{label}</div>
      <div className="metric-value">{value || "-"}</div>
      {note ? <div className="metric-note">{note}</div> : null}
    </div>
  );
}

function RunCard({ card, active, onSelect }) {
  return (
    <button type="button" className={`run-card ${active ? "active" : ""}`} onClick={() => onSelect(card.run_name)}>
      <div className="run-card-header">
        <div>
          <h3>{card.headline || card.run_name}</h3>
          <p>{joinItems([card.status, card.model_name, card.start_time])}</p>
        </div>
        <span className="run-card-id">{card.run_name}</span>
      </div>
      <p className="run-card-summary">{card.summary || "No summary yet."}</p>
      {(card.next_actions || []).length ? (
        <ul className="run-card-actions">
          {(card.next_actions || []).slice(0, 3).map((item) => (
            <li key={item}>{item}</li>
          ))}
        </ul>
      ) : null}
    </button>
  );
}

function EventFeed({ events }) {
  const containerRef = useRef(null);

  useEffect(() => {
    const node = containerRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [events]);

  return (
    <div ref={containerRef} className="feed-list">
      {(events || []).slice(-120).map((event) => {
        const payload = event.payload || {};
        const title = joinItems([event.category, event.name, payload.tool || payload.model || payload.node]);
        const body =
          payload.text ||
          payload.summary_snippet ||
          payload.reasoning_text ||
          payload.error ||
          payload.text_preview ||
          payload.goal ||
          payload.status ||
          "";
        return (
          <article key={event.seq || `${event.name}-${event.ts}`} className="feed-item">
            <div className="feed-meta">
              <span>{title}</span>
              <span>{formatTime(event.ts)}</span>
            </div>
            <p>{body || "(no body)"}</p>
          </article>
        );
      })}
    </div>
  );
}

function ChatThread({ messages }) {
  const threadRef = useRef(null);

  useEffect(() => {
    const node = threadRef.current;
    if (node) {
      node.scrollTop = node.scrollHeight;
    }
  }, [messages]);

  return (
    <div ref={threadRef} className="chat-thread">
      {(messages || []).map((message, index) => (
        <article key={`${message.role}-${index}`} className={`chat-bubble ${message.role || "assistant"}`}>
          <div className="chat-role">{message.role || "assistant"}</div>
          <p>{message.content || ""}</p>
        </article>
      ))}
    </div>
  );
}

function PromptPanel({ prompt, value, onChange, onSubmit, disabled }) {
  if (!prompt) {
    return null;
  }
  const payload = prompt.payload || {};
  const body = [
    payload.proposal_description,
    payload.report_text,
    payload.guidance,
    Array.isArray(payload.todo) && payload.todo.length
      ? payload.todo.map((item, index) => `${index + 1}. ${item}`).join("\n")
      : "",
    payload.report_path ? `report: ${payload.report_path}` : "",
  ]
    .filter(Boolean)
    .join("\n\n");
  return (
    <section className="prompt-panel">
      <div className="panel-eyebrow">Human Input Required</div>
      <div className="prompt-meta">{joinItems([prompt.kind, payload.run_id, payload.prompt_id || prompt.prompt_id])}</div>
      <pre className="code-card">{body || "(empty prompt payload)"}</pre>
      <textarea
        value={value}
        onChange={(event) => onChange(event.target.value)}
        placeholder="Provide feedback, approval, or revised guidance."
        disabled={disabled}
      />
      <button type="button" onClick={onSubmit} disabled={disabled}>
        Submit Feedback
      </button>
    </section>
  );
}

function ArtifactPanel({ details }) {
  const artifacts = details?.artifacts || [];
  return (
    <div className="artifact-grid">
      {artifacts.slice(0, 24).map((row, index) => (
        <article key={`${row.path || "artifact"}-${index}`} className="artifact-card">
          <div className="panel-eyebrow">{joinItems([row.kind, row.type])}</div>
          <h4>{row.path || "(unknown path)"}</h4>
          <p>{row.description || "No description."}</p>
        </article>
      ))}
    </div>
  );
}

function MonitorTabs({ tab, onChange }) {
  const tabs = [
    ["report", "Report"],
    ["task", "Task State"],
    ["trace", "Trace"],
    ["memory", "Memory"],
  ];
  return (
    <div className="tab-row">
      {tabs.map(([value, label]) => (
        <button
          key={value}
          type="button"
          className={`tab-button ${tab === value ? "active" : ""}`}
          onClick={() => onChange(value)}
        >
          {label}
        </button>
      ))}
    </div>
  );
}

function CodePane({ title, text, helper }) {
  return (
    <section className="stack-compact">
      <div className="panel-title-row">
        <div>
          <div className="panel-eyebrow">{helper || "Details"}</div>
          <h3>{title}</h3>
        </div>
      </div>
      <pre className="code-card tall">{text || "(empty)"}</pre>
    </section>
  );
}

function App({ boot }) {
  const view = boot?.view === "monitor" ? "monitor" : "home";
  const [snapshot, setSnapshot] = useState(null);
  const [details, setDetails] = useState(null);
  const [ctx, setCtx] = useState("");
  const [lane, setLane] = useState("standard");
  const [selectedRun, setSelectedRun] = useState("");
  const [workspaceRoot, setWorkspaceRoot] = useState("");
  const [workspaceName, setWorkspaceName] = useState("");
  const [search, setSearch] = useState("");
  const [statusMessage, setStatusMessage] = useState("");
  const [events, setEvents] = useState([]);
  const [promptResponse, setPromptResponse] = useState("");
  const [monitorTab, setMonitorTab] = useState("report");
  const [streamNonce, setStreamNonce] = useState(0);
  const [form, setForm] = useState({
    prompt: "",
    run_mode: "new_run",
    resume_run_name: "",
    proposal_review: true,
    log_llm: false,
    full_auto_major: false,
    seed_hypotheses: "",
    exploration_policy: "anchored",
    writing_mode: "none",
    output_format: "md",
    target_section: "",
    max_cycles: 6,
    max_literature_queries: 4,
    max_fast_runs: 3,
    max_standard_runs: 2,
    allow_deep_report: false,
  });
  const deferredSearch = useDeferredValue(search);
  const eventSourceRef = useRef(null);
  const latestSeqRef = useRef(0);

  useEffect(() => {
    let cancelled = false;
    const params = new URLSearchParams(window.location.search);
    const nextLane = params.get("lane") || "standard";
    setLane(nextLane);
    (async () => {
      try {
        const data = await apiFetch(`/api/bootstrap?${params.toString()}`);
        if (cancelled) {
          return;
        }
        startTransition(() => {
          setCtx(data.ctx || "");
          setWorkspaceRoot(data.workspace_root || "");
          setSelectedRun(data.selected_run || "");
          setSnapshot(data);
          setStatusMessage(data.status_message || "");
          setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
          latestSeqRef.current = Number(data.runtime?.seq || 0);
        });
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(String(error?.message || error));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, []);

  useEffect(() => {
    if (view !== "monitor" || !ctx || !selectedRun) {
      return;
    }
    let cancelled = false;
    (async () => {
      try {
        const data = await apiFetch(`/api/session/${escapePath(ctx)}/details?run=${escapePath(selectedRun)}`);
        if (!cancelled) {
          startTransition(() => {
            setDetails(data);
          });
        }
      } catch (error) {
        if (!cancelled) {
          setStatusMessage(String(error?.message || error));
        }
      }
    })();
    return () => {
      cancelled = true;
    };
  }, [ctx, selectedRun, view]);

  useEffect(() => {
    if (!ctx) {
      return undefined;
    }
    if (eventSourceRef.current) {
      eventSourceRef.current.close();
    }
    const source = new EventSource(`/api/session/${escapePath(ctx)}/stream?last_seq=${escapePath(latestSeqRef.current)}`);
    eventSourceRef.current = source;

    source.onmessage = (message) => {
      const data = JSON.parse(message.data || "{}");
      const event = data.event || {};
      const runtime = data.runtime || {};
      latestSeqRef.current = Number(runtime.seq || event.seq || latestSeqRef.current || 0);
      if (runtime.run_name && runtime.run_name === selectedRun) {
        startTransition(() => {
          setEvents((prev) => [...prev, event].slice(-120));
          setSnapshot((prev) => {
            if (!prev) {
              return prev;
            }
            return {
              ...prev,
              runtime,
              live_state: runtime.live_state || prev.live_state || {},
              llm: runtime.llm || prev.llm || {},
              graph: runtime.graph || prev.graph || {},
              prompt: runtime.prompt ?? prev.prompt ?? null,
              usage_summary: runtime.usage_totals || prev.usage_summary || {},
              can_submit_prompt: Boolean(runtime.prompt),
              run_status: data.run_status || prev.run_status,
              run_status_text: data.run_status_text || prev.run_status_text,
            };
          });
        });
      }
      if (["RUN_END", "PROMPT_REQUESTED", "PROMPT_RESOLVED"].includes(String(event.name || ""))) {
        refreshSnapshot(selectedRun);
      }
    };

    source.onerror = () => {
      source.close();
      eventSourceRef.current = null;
      window.setTimeout(() => {
        if (ctx) {
          latestSeqRef.current = latestSeqRef.current || 0;
          setStatusMessage((prev) => prev || "Stream disconnected. Reconnecting.");
          setStreamNonce((value) => value + 1);
        }
      }, 1500);
    };

    return () => {
      source.close();
      eventSourceRef.current = null;
    };
  }, [ctx, selectedRun, streamNonce]);

  async function refreshSnapshot(runName = selectedRun) {
    if (!ctx) {
      return;
    }
    const data = await apiFetch(
      `/api/session/${escapePath(ctx)}/snapshot?lane=${escapePath(lane)}&run=${escapePath(runName || "")}`,
    );
    startTransition(() => {
      setSnapshot(data);
      setSelectedRun(data.selected_run || "");
      setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
      latestSeqRef.current = Number(data.runtime?.seq || latestSeqRef.current || 0);
    });
  }

  async function postAndApply(url, payload, { loadDetails = false } = {}) {
    if (!ctx) {
      return;
    }
    const data = await apiFetch(url, {
      method: "POST",
      body: JSON.stringify(payload),
    });
    startTransition(() => {
      setSnapshot(data);
      setStatusMessage(data.status_message || "");
      setWorkspaceRoot(data.workspace_root || workspaceRoot);
      setSelectedRun(data.selected_run || "");
      setEvents(Array.isArray(data.events) ? data.events.slice(-120) : []);
      latestSeqRef.current = Number(data.runtime?.seq || latestSeqRef.current || 0);
      if (data.selected_run) {
        setForm((prev) => ({ ...prev, resume_run_name: data.selected_run }));
      }
    });
    if (loadDetails && data.selected_run) {
      const detailData = await apiFetch(`/api/session/${escapePath(ctx)}/details?run=${escapePath(data.selected_run)}`);
      startTransition(() => {
        setDetails(detailData);
      });
    }
  }

  async function handleWorkspaceRefresh() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/refresh`, {
      root_path: workspaceRoot,
      lane,
    });
  }

  async function handleWorkspaceOpen() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/open`, {
      root_path: workspaceRoot,
      workspace: snapshot?.workspace_name || "",
      lane,
    }, { loadDetails: view === "monitor" });
  }

  async function handleWorkspaceCreate() {
    await postAndApply(`/api/session/${escapePath(ctx)}/workspace/create`, {
      root_path: workspaceRoot,
      workspace: workspaceName,
      lane,
    }, { loadDetails: view === "monitor" });
    setWorkspaceName("");
  }

  async function handleRunSelect(runName) {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/select`, {
      run_name: runName,
      lane,
    }, { loadDetails: view === "monitor" });
  }

  async function handleStartRun() {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/start`, {
      ...form,
      prompt: form.prompt,
      lane,
      resume_run_name: form.resume_run_name || selectedRun,
    }, { loadDetails: view === "monitor" });
    setForm((prev) => ({ ...prev, prompt: "" }));
  }

  async function handleInterrupt() {
    await postAndApply(`/api/session/${escapePath(ctx)}/run/interrupt`, { lane });
  }

  async function handlePromptSubmit() {
    const prompt = snapshot?.prompt;
    if (!prompt) {
      return;
    }
    await postAndApply(`/api/session/${escapePath(ctx)}/prompt/respond`, {
      prompt_id: prompt.prompt_id || prompt.payload?.prompt_id || "",
      text: promptResponse,
      lane,
      run_name: selectedRun,
    }, { loadDetails: view === "monitor" });
    setPromptResponse("");
  }

  const workspaceOptions = snapshot?.workspaces || [];
  const runOptions = snapshot?.runs || [];
  const cards = (snapshot?.cards || []).filter((card) => {
    if (!deferredSearch.trim()) {
      return true;
    }
    return JSON.stringify(card).toLowerCase().includes(deferredSearch.trim().toLowerCase());
  });
  const live = snapshot?.live_state || {};
  const llm = snapshot?.llm || live.llm || {};
  const graph = snapshot?.graph || {};
  const usage = snapshot?.usage_summary || {};
  const reasoningText = llm.reasoning_text || "";
  const visibleEvents = view === "monitor" ? events : [];

  return (
    <main className={`app-shell view-${view}`}>
      <div className="ambient ambient-a" />
      <div className="ambient ambient-b" />
      <header className="hero-bar">
        <div>
          <div className="hero-kicker">CatMaster WebUI</div>
          <h1>{view === "home" ? "Agent cockpit for active work" : "Run monitor for deep execution traces"}</h1>
          <p>
            {view === "home"
              ? "Drive the run from a conversation-first view while keeping live execution, hidden-vs-exposed reasoning, and human interrupts visible."
              : "Track graph nodes, tool calls, exposed reasoning summaries, and final artifacts without polling whole pages."}
          </p>
        </div>
        <nav className="hero-nav">
          <a className={view === "home" ? "active" : ""} href={snapshot?.ctx ? `/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}` : "/"}>
            Home
          </a>
          <a
            className={view === "monitor" ? "active" : ""}
            href={snapshot?.ctx ? `/monitor/?ctx=${escapePath(snapshot.ctx)}&project_space=${escapePath(snapshot.workspace_name || "")}&run=${escapePath(selectedRun)}` : "/monitor/"}
          >
            Monitor
          </a>
        </nav>
      </header>

      <div className="status-strip">
        <StatusPill status={snapshot?.run_status} />
        <span>{snapshot?.run_status_text || "No active run."}</span>
        {statusMessage ? <span className="status-message">{statusMessage}</span> : null}
      </div>

      <div className={`layout ${view}`}>
        <aside className="left-rail glass-card">
          <div className="section-head">
            <div>
              <div className="panel-eyebrow">Workspace</div>
              <h2>Project routing</h2>
            </div>
            <button type="button" className="ghost-button" onClick={handleWorkspaceRefresh}>
              Refresh
            </button>
          </div>

          <div className="control-stack">
            <label>
              <span>Root</span>
              <input value={workspaceRoot} onChange={(event) => setWorkspaceRoot(event.target.value)} placeholder="Project-space root" />
            </label>
            <label>
              <span>Current workspace</span>
              <select
                value={snapshot?.workspace_name || ""}
                onChange={(event) => {
                  const next = event.target.value;
                  startTransition(() => {
                    setSnapshot((prev) => (prev ? { ...prev, workspace_name: next } : prev));
                  });
                }}
              >
                <option value="">(select workspace)</option>
                {workspaceOptions.map((item) => (
                  <option key={item.value} value={item.value}>
                    {item.label}
                  </option>
                ))}
              </select>
            </label>
            <div className="button-row">
              <button type="button" onClick={handleWorkspaceOpen}>
                Open
              </button>
              <button type="button" className="ghost-button" onClick={() => setWorkspaceName(snapshot?.workspace_name || "")}>
                Mirror
              </button>
            </div>
            <label>
              <span>New workspace name</span>
              <input value={workspaceName} onChange={(event) => setWorkspaceName(event.target.value)} placeholder="new workspace" />
            </label>
            <button type="button" onClick={handleWorkspaceCreate}>
              Create Workspace
            </button>
          </div>

          <div className="section-head compact">
            <div>
              <div className="panel-eyebrow">Runs</div>
              <h2>Recent sessions</h2>
            </div>
          </div>
          <input value={search} onChange={(event) => setSearch(event.target.value)} placeholder="Filter by run, status, model, summary" />
          <label>
            <span>Select run</span>
            <select value={selectedRun} onChange={(event) => handleRunSelect(event.target.value)}>
              <option value="">(select run)</option>
              {runOptions.map((item) => (
                <option key={item.value} value={item.value}>
                  {item.label}
                </option>
              ))}
            </select>
          </label>
          <div className="run-list">
            {cards.map((card) => (
              <RunCard key={card.run_name} card={card} active={card.run_name === selectedRun} onSelect={handleRunSelect} />
            ))}
          </div>
        </aside>

        <section className="center-stage">
          <div className="glass-card emphasis-card">
            <div className="section-head">
              <div>
                <div className="panel-eyebrow">Live state</div>
                <h2>{view === "home" ? "Conversation and control" : "Execution stream"}</h2>
              </div>
              <div className="inline-actions">
                <button type="button" className="ghost-button danger" onClick={handleInterrupt}>
                  Interrupt
                </button>
                {view === "monitor" ? (
                  <button type="button" className="ghost-button" onClick={() => refreshSnapshot(selectedRun)}>
                    Refresh snapshot
                  </button>
                ) : null}
              </div>
            </div>

            <div className="metrics-grid">
              <MetricCard label="Phase" value={live.current_phase || snapshot?.run_status} />
              <MetricCard label="Graph node" value={graph.node || live.current_node} />
              <MetricCard label="Task" value={live.current_task_goal || live.current_task_id} />
              <MetricCard label="Tool" value={live.active_toolcall?.tool || "-"} note={live.active_toolcall?.status || ""} />
              <MetricCard label="Output tokens" value={usage.output_tokens || usage.outputTokens || llm.usage?.output_tokens} />
              <MetricCard label="Reasoning tokens" value={usage.reasoning_tokens || llm.usage?.reasoning_tokens || "-"} />
            </div>

            <PromptPanel
              prompt={snapshot?.prompt}
              value={promptResponse}
              onChange={setPromptResponse}
              onSubmit={handlePromptSubmit}
              disabled={!snapshot?.can_submit_prompt}
            />

            {view === "home" ? (
              <div className="home-grid">
                <section className="stack-block">
                  <div className="section-head compact">
                    <div>
                      <div className="panel-eyebrow">Chat</div>
                      <h3>Operator thread</h3>
                    </div>
                  </div>
                  <ChatThread messages={snapshot?.chat_messages || []} />
                  <div className="control-stack">
                    <div className="split-fields">
                      <label>
                        <span>Lane</span>
                        <select value={lane} onChange={(event) => setLane(event.target.value)}>
                          {["fast", "standard", "research", "writing"].map((item) => (
                            <option key={item} value={item}>
                              {item}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Run mode</span>
                        <select
                          value={form.run_mode}
                          onChange={(event) => setForm((prev) => ({ ...prev, run_mode: event.target.value }))}
                        >
                          <option value="new_run">new_run</option>
                          <option value="resume_selected_run">resume_selected_run</option>
                        </select>
                      </label>
                    </div>
                    <textarea
                      value={form.prompt}
                      onChange={(event) => setForm((prev) => ({ ...prev, prompt: event.target.value }))}
                      placeholder="Describe what CatMaster should do next."
                    />
                    <div className="button-row">
                      <button type="button" onClick={handleStartRun}>
                        Start Run
                      </button>
                      <button
                        type="button"
                        className="ghost-button"
                        onClick={() => setForm((prev) => ({ ...prev, resume_run_name: selectedRun }))}
                      >
                        Use selected run for resume
                      </button>
                    </div>
                    <div className="settings-grid">
                      <label>
                        <span>Resume run</span>
                        <select
                          value={form.resume_run_name}
                          onChange={(event) => setForm((prev) => ({ ...prev, resume_run_name: event.target.value }))}
                        >
                          <option value="">(use selected run)</option>
                          {runOptions.map((item) => (
                            <option key={item.value} value={item.value}>
                              {item.label}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Writing mode</span>
                        <select
                          value={form.writing_mode}
                          onChange={(event) => setForm((prev) => ({ ...prev, writing_mode: event.target.value }))}
                        >
                          {["none", "internal_report", "paper_outline", "section_draft", "full_draft"].map((item) => (
                            <option key={item} value={item}>
                              {item}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Output format</span>
                        <select
                          value={form.output_format}
                          onChange={(event) => setForm((prev) => ({ ...prev, output_format: event.target.value }))}
                        >
                          {["md", "tex"].map((item) => (
                            <option key={item} value={item}>
                              {item}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Target section</span>
                        <input
                          value={form.target_section}
                          onChange={(event) => setForm((prev) => ({ ...prev, target_section: event.target.value }))}
                          placeholder="section_draft only"
                        />
                      </label>
                      <label className="toggle-line">
                        <input
                          type="checkbox"
                          checked={form.proposal_review}
                          onChange={(event) => setForm((prev) => ({ ...prev, proposal_review: event.target.checked }))}
                        />
                        <span>Proposal review</span>
                      </label>
                      <label className="toggle-line">
                        <input
                          type="checkbox"
                          checked={form.log_llm}
                          onChange={(event) => setForm((prev) => ({ ...prev, log_llm: event.target.checked }))}
                        />
                        <span>Log LLM</span>
                      </label>
                      <label className="toggle-line">
                        <input
                          type="checkbox"
                          checked={form.full_auto_major}
                          onChange={(event) => setForm((prev) => ({ ...prev, full_auto_major: event.target.checked }))}
                        />
                        <span>Full auto major</span>
                      </label>
                      <label className="toggle-line">
                        <input
                          type="checkbox"
                          checked={form.allow_deep_report}
                          onChange={(event) => setForm((prev) => ({ ...prev, allow_deep_report: event.target.checked }))}
                        />
                        <span>Allow deep report</span>
                      </label>
                    </div>
                    <div className="split-fields">
                      <label>
                        <span>Exploration policy</span>
                        <select
                          value={form.exploration_policy}
                          onChange={(event) => setForm((prev) => ({ ...prev, exploration_policy: event.target.value }))}
                        >
                          {["anchored", "local_expand", "open"].map((item) => (
                            <option key={item} value={item}>
                              {item}
                            </option>
                          ))}
                        </select>
                      </label>
                      <label>
                        <span>Max cycles</span>
                        <input
                          type="number"
                          value={form.max_cycles}
                          onChange={(event) => setForm((prev) => ({ ...prev, max_cycles: Number(event.target.value || 0) }))}
                        />
                      </label>
                    </div>
                    <textarea
                      value={form.seed_hypotheses}
                      onChange={(event) => setForm((prev) => ({ ...prev, seed_hypotheses: event.target.value }))}
                      placeholder="Seed hypotheses, one per line."
                    />
                  </div>
                </section>

                <section className="stack-block">
                  <CodePane title="Plan" helper="Proposal" text={snapshot?.proposal || "No plan yet."} />
                  <CodePane
                    title="Reasoning summary / rationale"
                    helper="Only shown when the model/provider exposes it"
                    text={
                      reasoningText ||
                      "No exposed reasoning summary. This can be expected: many reasoning tokens stay hidden and only token counts are visible unless the provider returns a reasoning summary."
                    }
                  />
                  <CodePane title="Assistant draft" helper="Visible answer tokens" text={llm.text || graph.text_preview || ""} />
                </section>
              </div>
            ) : (
              <div className="monitor-grid">
                <CodePane
                  title="Reasoning summary / rationale"
                  helper="If the provider exposes it"
                  text={
                    reasoningText ||
                    "No exposed reasoning summary for this turn. Hidden reasoning is still possible; in that case only reasoning token usage may be visible."
                  }
                />
                <CodePane title="Assistant stream" helper="Visible output text" text={llm.text || graph.text_preview || ""} />
                <section className="stack-block wide">
                  <div className="section-head compact">
                    <div>
                      <div className="panel-eyebrow">Events</div>
                      <h3>Incremental execution feed</h3>
                    </div>
                  </div>
                  <EventFeed events={visibleEvents} />
                </section>
              </div>
            )}
          </div>
        </section>

        <aside className="right-rail">
          <div className="glass-card">
            <div className="section-head">
              <div>
                <div className="panel-eyebrow">Run details</div>
                <h2>{view === "home" ? "Live internals" : "Artifacts and traces"}</h2>
              </div>
              {view === "monitor" ? (
                <button type="button" className="ghost-button" onClick={() => refreshSnapshot(selectedRun)}>
                  Pull latest
                </button>
              ) : null}
            </div>

            {view === "monitor" ? (
              <>
                <MonitorTabs tab={monitorTab} onChange={setMonitorTab} />
                {monitorTab === "report" ? (
                  <CodePane title="Final report" helper={snapshot?.report_source || "Report source"} text={snapshot?.final_report || ""} />
                ) : null}
                {monitorTab === "task" ? (
                  <CodePane title="Task state" helper="task_state.json" text={details?.task_state || ""} />
                ) : null}
                {monitorTab === "trace" ? (
                  <CodePane
                    title="Trace bundle"
                    helper="event/tool/patch trace"
                    text={[details?.trace_event, details?.trace_tool, details?.trace_patch].filter(Boolean).join("\n\n")}
                  />
                ) : null}
                {monitorTab === "memory" ? (
                  <CodePane title="Memory index" helper="MEMORY/MEMORY.md" text={details?.memory || ""} />
                ) : null}
                <ArtifactPanel details={details} />
              </>
            ) : (
              <>
                <CodePane title="Run context status" helper="Session context" text={snapshot?.entry_context_status || ""} />
                <CodePane title="Memory index" helper="Workspace memory" text={details?.memory || snapshot?.proposal || ""} />
                <ArtifactPanel details={details} />
              </>
            )}
          </div>
        </aside>
      </div>
    </main>
  );
}

export default App;
