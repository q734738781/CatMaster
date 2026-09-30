# DeepAgents integration reference

CatMaster runs native DeepAgents/LangGraph graphs in a local DBOS execution host.
DBOS owns durable turns, queues, cancellation and recovery. LangGraph owns agent
state and checkpoints. CatMaster binds workspaces, routes completion inputs and
projects native events into the WebUI. The scientific Research Graph remains in
the workspace database.

## Execution and storage

### Model-specific harness guidance

Agent prompts combine role instructions, shared runtime guidance, native
capability/skill/memory instructions, and model-specific guidance. The shared
`catmaster.runtime.guidance` bundle composes user-priority and system-usage fragments
through `PromptCatalog` and `PromptRenderer` for all models, including Astra.
Specialist, worker and self-evolution builders include it once per recipient.
`catmaster/runtime/prompts/model_harness.py`
selects a package-owned prompt bundle from the actual resolved model identifier;
YAML labels do not determine the selection. MiMo models select
`catmaster.model.mimo`, which adds a short emphasis on current authority, explicit
bindings, historical completion and checks that can change the next action. No additional YAML field
is required. Other models retain their native harness profiles.

Before constructing a DeepAgent, both specialist and self-evolution builders
register the model bundle using native
`HarnessProfile(system_prompt_suffix=...)` under its exact provider/model key.
It also registers explicit raw child models; inherited children use their own
resolved native profile. Compiled children receive guidance when their graph is
built. Registration replaces the same suffix instead of accumulating it. The
registry is process-global, but its guidance is stable package policy, identical
across workspaces, and never contains task or deployment state. Existing provider
profiles, tool bindings and middleware remain in place. Self-evolution reflection,
proposal, review and investigation intentionally share model guidance, even when
self-evolution is the first entrypoint in the process. Its role-specific permissions
and optional decision tools still determine completion and further action.

DeepAgents 0.7.11 resolves prebuilt models through
`deepagents._models.get_model_identifier/get_model_provider`. CatMaster uses these
same internal helpers because they have no public export and provider aliases can
differ from CatMaster YAML names. Integration tests cover real OpenRouter and
OpenAI-compatible adapters and the final native model-visible messages.
See [native profiles](https://docs.langchain.com/oss/python/deepagents/profiles).

Guidance prioritizes the current user objective, unchanged constraints and reusable
results; it explains when records, skills, delegation and background discussions
serve that objective. It does not impose reasoning quotas or tool playbooks.
Current tasks, corrections and completion notices stay in user-turn inputs.
Plain LangChain proposal checks, title generation and summarization retain their
own prompts outside the DeepAgents composition.

### Durable execution

`catmaster/runtime/execution.py` defines one durable workflow per agent turn.
`catmaster/webui/local_execution.py` builds the existing specialist through
`SpecialistRunner` and projects its native stream. Each deployment runs one
execution host in the WebUI process, with one DBOS queue partition per workspace
and thread. Partition concurrency is one; different threads run concurrently.

| Storage | Purpose |
| --- | --- |
| `<project-space-root>/.catmaster/execution.sqlite` | DBOS workflows, queue, small input and result references |
| `<workspace>/metadata/deepagent_threads.sqlite` | Native LangGraph checkpoints and writes |
| `<workspace>/metadata/deepagent_memory.sqlite` | Native long-term Store, scoped to the workspace project ID |
| `<workspace>/metadata/workspace.sqlite` | Scientific Graph, UI messages/events, artifacts and domain records |

An accepted turn contains the current input reference, workspace/model/permission
binding and scientific context. Graph state never enters DBOS workflow results.
On recovery, checkpoint metadata identifies whether this turn has not started,
is unfinished, or has already completed. Only a new turn appends its HumanMessage.
A completed graph whose workflow has not acknowledged it supplies its saved answer.

Agent tools use the public `DBOSClient` to submit detached work from inside the
agent execution step. The containing workflow delivers completion through native
workflow enqueue with a stable ID. No browser subscription is needed to wake the
parent. SQLite is the supported single-machine deployment backend; running
multiple WebUI processes against one control database is not supported.

Specialists and their native workers use DeepAgents' `DeepAgentState` explicitly,
including raw `SubAgent` specifications and `general-purpose` children. Its
`messages` channel uses native `DeltaChannel`: intervening checkpoints reconstruct
messages from saved writes, with a full snapshot every 50 message-channel updates
under the upstream default. Images remain in the message state; subsequent steps
do not each write another complete copy of that history. Other state channels
retain their own upstream persistence behavior.

LangGraph reads an older full-message checkpoint directly as the starting value
for this channel. Continuing that thread writes native deltas to the same SQLite
database. There is no CatMaster delta reader, new storage format, or startup
conversion. Existing full snapshots remain stored, and enabling delta persistence
does not shrink them. The common builder supplies the schema explicitly because
DeepAgents 0.7.11 defaults its root to `DeepAgentState` but otherwise forwards
`None` to raw subagent construction.

## Codex cache affinity

Codex requests carry a stable `prompt_cache_key` and matching HTTP `session-id`
and `thread-id`. Explicit headers are honored case-insensitively; an omitted
`thread-id` follows the chosen `session-id`.
The default scope uses the logical thread and ancestor subagent namespace,
excluding the model node's per-step UUID. It survives compaction and model
reconstruction for that conversation. Independent threads and delegated tasks
remain separate. Direct model calls without thread metadata use an instance-local
session. Explicit payload keys and session headers take precedence; SDK
`extra_body.prompt_cache_key` overrides the top-level field at serialization.

LangChain OpenAI 1.6.0 forwards supplied keys but does not generate these defaults.
Third-party observations in [Hermes #47126](https://github.com/NousResearch/hermes-agent/issues/47126)
and [IronClaw #7921](https://github.com/nearai/ironclaw/issues/7921) motivate checking
both wire keys and session affinity. Stable identity improves eligibility; it does
not guarantee cache hits after prefix changes, compaction, or backend eviction.
No per-turn server routing token is carried across turns.

Specialists and workers already use provider-native `web_search` with Codex and
OpenAI; other providers use CatMaster's search function. Tool augmentation keeps
one provider-compatible search implementation, preserving an existing native
tool's configuration and replacing a conflicting same-named function. The
general-purpose child inherits its parent's tools. Native search retains normal
web access; it is not added solely as a cache hint. The additional identity header
is motivated by third-party client experiments, not a guaranteed cache-hit rate.

## Provider request capture

Model response callbacks preserve content blocks and their annotations,
additional response fields, and generation metadata. Reasoning extraction keeps
Unicode text and mathematical operators when matching duplicate fragments.
When LangChain supplies a partial `LLMResult` to
[`on_llm_error`](https://reference.langchain.com/python/langchain-core/callbacks/manager/CallbackManagerForLLMRun/on_llm_error),
the callback writes `LLM_RAW_RESPONSE` with `partial: true` and `status: error`
before the error event. This output remains inspectable but does not count as a
completed call. Any reported usage stays in the partial raw record.

Codex OAuth requests can be sampled into the run's `observability.sqlite` to
investigate prompt-cache misses. Capture is disabled by default. Configure the
execution host before constructing the model and its observation callbacks:

```bash
export CATMASTER_CAPTURE_REQUEST_RUN_IDS='exact-run-id'
export CATMASTER_CAPTURE_REQUEST_AGENTS='research,writing'
export CATMASTER_CAPTURE_REQUEST_LIMIT=8
```

Run IDs and agent names are comma-separated exact selections, not wildcard
patterns. A run selection is required. An omitted agent selection includes all
roles in that selected run. The positive limit defaults to eight HTTP attempts
per observation-handler instance, shared by its selected roles; it is a diagnostic
sampling budget, not a model-call limit. Recreating a handler starts a new budget.
Invalid limits disable capture with a warning. Remove the run selection to disable
capture for newly constructed models and handlers. This setting does not enable
capture in an already running task.

`LLM_PROVIDER_REQUEST` records the UTF-8 HTTP body as a string after SDK
serialization, including tools, instructions, input, reasoning/text settings and
any cache fields actually sent. Keeping the body as text preserves key ordering
and whitespace despite the observation store's JSON normalization. There is no
content truncation. These internal records may include the same scientific text,
inline media and encrypted reasoning as the request; they are not a UI summary.
Authentication headers, cookies and URL query parameters are not captured.
Only the `session-id` and `thread-id` request headers are retained as `cache_headers` for affinity
diagnosis. `LLM_PROVIDER_RESPONSE` preserves the terminal SSE event before
LangChain's response projection, including provider cache fields and raw usage.
The observer tees streamed bytes, supports HTTP compression, and accepts Codex
responses without a Content-Type header. It neither pre-reads nor consumes the
stream ahead of the model client.

Join records to `LLM_RAW_REQUEST`, `LLM_CALL_END` and `LLM_ERROR` by
`callback_run_id`. `attempt` distinguishes captured SDK HTTP retries within that
call. A captured request proves an attempted send, not completion or billable
usage. Usage totals still come only from the existing completion records. Capture
does not change payloads or retry requests on recording errors.

The manual writing benchmark selects its generated run before models/callbacks
are constructed and captures up to 1,000 HTTP attempts per handler by default.
Use `--capture-request-limit` to change the diagnostic budget (0 disables it).
`benchmark_result.json` reports captured request and terminal-response counts.

The integration covers the runtime's `invoke`/`ainvoke` and native
`stream_events(version="v3")` paths, including asynchronous events. Plain
`stream`/`astream` calls are not sampled: the pinned LangChain implementation does
not pass their callback manager to the provider stream. The model mixin binds
capture around `_generate_with_cache`/`_agenerate_with_cache` and
`_iter_v2_events`/`_aiter_v2_events`; iterator scopes reset before yielding to the
consumer. HTTPX request hooks run after preparation and before network send;
OpenAI's default clients or caller-supplied clients retain their existing hooks.
See [HTTPX event hooks](https://www.python-httpx.org/advanced/event-hooks/).

## Controls and background tasks

Research exposes `start_async_task`, `check_async_task`, `list_async_tasks`,
`update_async_task`, and `cancel_async_task`. Fresh tasks have independent native
threads. Follow-up targets an existing task and keeps that context. The original
brief and latest follow-up remain in the task descriptor outside compactable
model history. Task listing is paginated; task inspection includes complete briefs
and the saved result. The UI shows a brief excerpt and an entry to read it in full.

Agent `update_async_task` uses `strategy="interrupt"` to retract a premise or correct
a source claim, method, scope or direction the child is currently using. The new
instruction names the replacement and preserves unaffected results. Additional
evidence, later questions and cosmetic wording changes can use `enqueue`, which
remains the omitted-argument default. Enqueue delivers only after the current turn,
including synchronous nested workers, returns. Interrupt resumes with the correction
after the old graph unwinds; it preserves checkpoints and does not cancel remote
scientific jobs. The WebUI's Steer action uses interrupt. Steering transfers the same
task's reservation to its next turn instead of opening another expensive route.

The task card and full instructions distinguish the original brief from the latest
supplementary instruction. Its pending/running/finished status follows the accepted
DBOS run; it is not a model read receipt or evidence that the correction was resolved.
Older briefs without a run association retain their text without an inferred status.

Background specialists receive the explicit delegation brief or follow-up. The
host does not prepend the parent's original request from the shared Research
Graph. A short current-turn binding identifies the graph query target and focus;
it comes from the accepted turn packet, as do the scoped query tools. Workspace
and Graph bindings remain intact, and agents can query relevant
scientific evidence through their ordinary tools. This is context isolation,
not an access-control boundary around shared workspace knowledge.

Other graph-aware entry turns receive that same explicit binding before their
partial scientific context; unbound turns identify that absence explicitly instead
of leaving an earlier turn's binding to be inferred. Saved graph questions and completion criteria are
labeled as records, separately from the current user request. An earlier completed
stage can provide evidence for newly authorized work without selecting a different
graph or automatically changing the recorded completion flag.

The host retains an existing valid graph selection. Unbound scientific entry turns
use the newest unarchived graph by creation time, including completed stages. A
first graph is initialized only when none is available. Inherited task bindings
and accepted turn snapshots stay fixed when other graphs are created or updated.
Workspace graph catalog, creation and switching belong to the UI and host APIs;
Research and its delegates do not receive catalog or graph-creation tools.

Scoped SQL and scientific mutations resolve the accepted turn binding first,
including an explicitly empty binding. Scientific write schemas expose node IDs
and the expected graph revision, but no graph selector. Older direct host callers
may retain an explicit graph ID; it cannot override a supplied turn binding.
Completed records remain usable evidence without implicitly reopening the stage.

`on_completion=resume_parent` creates a semantic HumanMessage turn on the parent.
`on_completion=notify` saves results and updates Activity without invoking the
parent model. Paused automation takes precedence. A child Result event cannot
bypass that choice through the Research Graph watcher. Relevant independent
external evidence can still notify an authorized, unfinished research stage.

Persistent entry and the submitted objective establish continued exploration.
The existing thread `automation_paused` state controls stop/resume. Old Manual
graphs remain paused until explicit Persistent input adopts them; the mode is
not a second user control. Graph recovery checks the session rather than using
Graph mode to choose scientific work. A conversational reply does not require
a graph-wide stopping disposition. For an explicitly declared scientific stall,
closeout middleware requests independent reconsideration through a HumanMessage;
the model chooses the native delegation, and the host does not fabricate an AI
tool call. This uses the documented [middleware state update and jump contract](https://docs.langchain.com/oss/python/langchain/middleware/custom).

Each independent Research branch owns scientific method choice and all H/E/R
cycles needed for its assigned question. Root scope/completion controls remain
root-owned. Historical planning/comparison tool restrictions apply only to their
persisted legacy thread roles, never to a normal root or independent branch.
Meaningful Results are written by their producers during longer investigations;
their milestones include methods and conclusions, and amendments update the
existing message through the normal `message.updated` projection. Session
activity aggregates descendants, including nested background investigations.

When background results are the only remaining dependency, the root returns an
interim reply and ends that conversational turn. The scientific stage stays open
for completion inputs. It does not hold the foreground in a sleep/poll loop.

Named specialists use their existing tools, prompts, skills and worker hierarchy.
Experiment can delegate to its computational workers; Writing can delegate to
writing/plot workers. Native `task` calls and the explicit `general-purpose`
worker retain their existing context isolation.

### Persistent Research discussion and follow-up

Shared discussion and peer task discovery are enabled for the declared
`persistent_research` owner and its descendants in the same Graph. Research
branches keep their `research` entrypoint and native parent relationships.
An independent ordinary Research thread does not opt in by attaching to the same
Graph: its task tools retain child-only discovery, without the shared discussion
tool or discussion middleware. Graph evidence and historical discussion sources
remain readable through ordinary graph SQL.

Within a Persistent Research session, `list_async_tasks(scope="research_graph")`
and `check_async_task` expose peers' complete briefs, explicit progress and saved
results. Execution control still requires the original parent relationship.
`notify_progress` publishes the researcher's own stated work and next step.
`post_research_message` appends a public title/body or reply, with optional node,
references and target task. Research branches exchange messages directly; they
need not route ordinary questions through the main agent. The `research_discussions`
table lives in `workspace.sqlite`, with full text reachable through graph SQL,
ordinary predicates, joins and pagination. No transcript is broadcast in full.

Stable discussion guidance is assembled into the system prompt only for the
receiving research role, with different root and branch authority. Event notices
remain HumanMessages containing the new-message count and query range. Concrete
continuation, interruption and reply operations are documented in the collaboration
skill reference. A discussion notice grants no new authorization; an explicit user
instruction can authorize further work after an earlier completed stage.

At normal model boundaries, a Persistent Research branch receives a small hint
about newly targeted messages. Its main researcher receives hints about new
shared discussion, including questions addressed between branches. When a turn
is closing and discussion has changed since the previous closing check, native
middleware offers one check opportunity. It does not require a reply to every
message, infer importance from keywords, select tasks, or force a tool call.
Native checkpoint cursors record notification opportunities, not proof of reading
or a scientific decision; DeepAgents' incremental message channel is retained.
Dynamic hints are in-turn HumanMessages and do not appear as new user chat turns.

The main researcher compares unresolved questions with actual Methods/Results and
the user's goal. It may answer from existing evidence, decline further work with
a reason, or continue the appropriate original child using `update_async_task`.
Additional questions queue behind valid running work; corrections to its current
premise or instructions use interrupt. Both preserve the child's native context.
Nested workers are reached through their owning branch. The agent checks existing
follow-ups to avoid duplicate investigations. DBOS handles the accepted task and
its existing completion policy. Siblings retain peer reading and discussion, not
authority to start or interrupt each other's execution.

Consequential decisions are saved as discussion replies. The main researcher or
user may set `resolves_message_id` to the same source as `reply_to`, with
`review_outcome` of `addressed`, `deferred`, or `follow_up`. A follow-up reply
references the already accepted task; it does not itself schedule work or claim a
scientific resolution. Decision history remains in the append-only reply bodies.
Discussion records and unresolved markers never block Graph completion.

Posting, replies and decision markers do not wake idle roots or branches. The main
researcher checks discussion on existing user/completion turns; after all work is
idle, new messages wait for the next such turn. There is no discussion-triggered
DBOS workflow or timed LLM review. Stop/pause, completion notification choices and
normal authorization still apply. Discussion neither reopens completed stages nor
authorizes new calculations or experiments. UI events do not advance Graph revision
or scientific wakeup predicates. Formal findings remain Graph Results and relations.
A discussion can be cited using its existing `message` source ID; historical and
explicitly cited cross-graph evidence remains reachable without copying it into chat.

### Independent research tasks and verification cost

`start_async_task(agent="research_specialist", description=..., task_cost="low")`
creates an independent ResearchSpecialist. It owns the hypothesis, proposed method,
result interpretation and next useful check for its assigned question. Its native
Experiment, Literature Review, Writing and general-purpose delegates return
synchronously within that task. They share its admission slot rather than acquiring
another slot while their caller waits. The branch can update scientific records;
only the main research session changes graph-wide scope and completion. Branches
share workspace files, so briefs should assign distinct output paths.

A research branch can start another independent background investigation in the
same deployment pool. Unlike a synchronous delegate, that task owns its own slot.
A parent waiting only for such independent results ends its interim turn and uses
no research slot; the children retain theirs throughout their work. The parent
card shows waiting for branch results, and its final handoff waits for the children
that request continuation. Their completions resume the same parent, which then
reacquires admission for its next active turn. This also works with a one-slot pool.

The deployment's LLM configuration includes:

```yaml
persistent_research:
  agent_pool_size: 16
  agent_task_concurrency:
    low: 8
    medium: 4
    high: 2
  control_concurrency: 4
```

The WebUI reads these limits once on startup from `CATMASTER_LLM_CONFIG` or
`configs/llm.yaml`. They apply across workspaces in that deployment. The total cap
and each tier cap both apply. Foreground/control turns use a separate DBOS queue;
waiting research does not occupy their capacity. Accepted research turns remain
durable DBOS workflows with one active writer per thread.

Cost means the highest expected verification cost of a formed **agent task**.
Literature comparison, evidence analysis and small Python/ML baselines are low;
MLFF exploration and substantial training are medium; DFT campaigns and external
experiments are high, including their cheaper prerequisites. These are adjustable
coarse categories, not estimated hours, job counts or a cumulative resource budget.
Admission is never authorization to calculate or perform an experiment.
Omitting `task_cost` uses medium; evidence-only briefs should explicitly choose low.

An open question can start low. When its method changes, a background specialist
can call `set_research_task_cost` alone after previous work has returned. The native
graph checkpoints an interrupt; the host atomically exchanges its reservation or
queues the new tier, and later resumes that same thread/tool call with native
`Command(resume=...)`. A mixed cost-switch/execution tool batch is returned to the
model without executing either action. There is no interruption of an in-flight
calculation, no release upon remote submission and no change to DPDispatcher's
synchronous wait. Two high tasks awaiting computation still occupy two high slots.

DBOS has one fixed per-partition cap and cannot express unequal tier caps plus a
shared total directly. A small `catmaster_research_admission` table in the existing
execution SQLite database implements that atomic predicate with `BEGIN IMMEDIATE`.
It stores task references and reservations, not graph histories. Eligible waiting
tasks are admitted FIFO; a full high tier does not block low. DBOS durable messages
wake admitted workflows; a native receive timeout repairs missed notifications.
No separate process, execution queue engine, checkpoint format or DBOS private-table
modification is involved. Process recovery preserves active reservations.

`research_pool_status` exposes counts and limits to the main researcher. When high
tasks queue while cheaper capacity is idle, one coalesced state-change input can
ask an authorized Persistent Research session to reconsider its methods and task
distribution. It does not choose or create scientific work, and `notify`, pause and
graph completion take precedence. Slot availability cannot guarantee diversity.

Each Result body contains `methods`, `summary` and `conclusion`. Methods records the
actual evidence/data, comparisons, representations and validation; conclusion
records supported scope, open alternatives and a next discriminating question.
Summary-only corrections preserve both fields. Current readers expose missing old
fields as `missing due to old record`; no older-program forward reader is supported.
These fields stay in existing `body_json`, with normal Graph transactions and source
refs. A unique background message ID is accepted directly, including DBOS IDs
containing colons; explicit `thread_id:message_id` disambiguates repeated IDs.
Independent branches reconcile revision conflicts through ordinary graph reads.

The composer supports `enqueue`, `reject`, `interrupt` and `rollback`. DBOS owns
the queue and cancellation. Steer waits for the cancelled graph to unwind before
accepting a replacement turn. Replace starts at the native checkpoint saved
before the interrupted turn; Stop with rollback creates a native checkpoint from
that same starting state. This cannot undo external tool side effects or cancel
already submitted remote scientific jobs.

DBOS 2.31 marks a workflow cancelled before its preemptible step finishes
unwinding. A per-thread in-process writer mutex covers that teardown interval;
it does not own task state, scheduling or retries. Native approval resumes use
`Command(resume={"decisions": [...]})`. A failed turn with a pending native step
can continue using `input=None`; an older failed message cannot resume a later
turn. New conversational instructions always enter as HumanMessage turns.

## Activity and conversation continuity

Native `astream(version="v2", subgraphs=True)` emits typed messages, updates and
custom events. Root events update the response; nested events retain their worker
source and update child activity. Tool details and transcripts use existing
paginated UI routes. The running indicator follows active/queued turns throughout
gaps between tool calls. Reconnecting a browser replays semantic UI events and
does not launch model work.

Background Activity cards use the same current execution status as task details,
with one card per child thread across foreground turns. Saved cards supply the
latest progress only for that run; historical cards and retained checkpoint
`async_tasks` do not determine current activity. Stopped tasks, including offline
imports, remain inspectable in history without appearing in the active rail. Reading
Activity uses child descriptors and individual cards, without scanning prior
conversation turns or loading child results/checkpoints.

Summarization-internal messages are identified by `lc_source=summarization` and
`lc_internal_call`. They display “正在压缩上下文…” and never become a final reply.
One activity follows the native model-task namespace/step through provider message-ID
changes. A final message or final usage chunk closes that activity before the next
research response, so an interruption afterward does not relabel finished compaction.
The saved summary and cutoff remain native state. `RestoringSummarizationMiddleware`
restores historical JSON-encoded summary messages to the upstream message type.
Raw retained history is not sent in full when a valid summary covers it.
For token-based retention, suffix estimates are calibrated to the same history
count used to trigger compaction. This avoids repeated summary-only rewrites
when reported usage is much larger than LangChain's unscaled tail estimate.
Calibration is local to the call; native AI/tool boundaries and history/media
offloading remain in use. It estimates suffix cost rather than claiming exact
provider tokenization of individual media items.
`agent_runtime.deepagent_context_trigger_token_cap` defaults to 258,000 tokens;
a known smaller model window retains the upstream margin. YAML overrides
`CATMASTER_DEEPAGENT_CONTEXT_TRIGGER_TOKEN_CAP`. Null or nonpositive values use
upstream defaults. Summary input uses `trim_tokens_to_summarize=None`.
The same setting applies to specialists, compiled workers, and explicit
dictionary-based subagents, including `general-purpose`, reasoning and literature
workers. Each raw subagent receives its own summarizer for its selected model;
DeepAgents' documented [subagent middleware](https://docs.langchain.com/oss/python/deepagents/subagents)
does not inherit the root's middleware automatically.

This is a pre-call compaction trigger, not a hard request-token or spending limit.
Its count includes cached context and uses estimates plus previous reported total
usage, including output. New content, especially unmeasured media, may cost more
than its estimate. Compaction itself consumes a model call, and a retained recent
tail may still be large; the setting does not guarantee that every request stays
below 258,000 tokens.

For native media already consumed by the same model, the trigger count uses the
latest actual total usage plus an estimate for subsequent messages and the current
system/tool overhead. This avoids counting a file's base64 transport as prose and
respects high image usage even when Codex's `openai-codex` tracing label differs
from its `openai` response metadata. New unmeasured media and model changes retain
upstream estimation. For text-only histories, same-model reported usage plus new
content provides a lower bound without reducing the upstream estimate.
The trigger threshold, model input, media access, retention
and original checkpoints are unchanged. This compatibility behavior is verified
against the installed counter/streaming source and the documented
[LangGraph message stream](https://docs.langchain.com/oss/python/langgraph/streaming).

Old local SQLite conversations read their existing native checkpoint binding.
There is no automatic full-history import on startup or continuation. Retained
state schemas keep historical filesystem and async-task fields readable; old
async records do not schedule work. Clearing a target checkpoint does not trigger
an import from another store. Intermediate Server-bound conversations require
explicit offline conversion before use; the application has no pickle importer.

Startup reconciles small thread and workflow rows. It does not load all checkpoint
histories. Individual active conversations can still contain large image payloads;
context compaction changes model input but does not delete persisted history.
Back up the control SQLite and workspace metadata with writers stopped or with
SQLite's backup API. Keep workspace files and private configuration with that backup.
An offline database backup can be compressed with an ordinary archive tool and
decompressed before restoration; the active SQLite saver reads an ordinary SQLite
file. `VACUUM` reclaims unused space but does not deduplicate retained snapshots.
Deleting earlier saves removes historical recovery points and can break native
delta reconstruction if required ancestors or writes are removed. The runtime
does not automatically prune checkpoint history.

## Research progression and scientific memory

Both research entrypoints use the same ResearchSpecialist builder/model role and
native specialist hierarchy. Research closes on the requested stage; Persistent
Research continues within that scope. `research_challenger` is an independent
asynchronous task launched with `start_async_task`. It has its own thread and
checkpoint; completion resumes the same root through the existing DBOS bridge.
It inspects sources and can revise scoped scientific graph claims, but cannot
execute experiments, write general files, delegate, or mark the overall goal complete.
Research branches retain full H/E/R ownership.

The root considers independent challenge at consequential narrowing and closeout
decisions: a single method failure generalized to a family, disconnected model
and candidate-selection results, or an assumed unavailable input. The shared
research-direction recovery skill guides evidence-led alternatives and bounded
next checks. Routine complete tasks do not need an extra reviewer; unchanged
evidence does not justify repeated reviews. The model chooses whether to launch
the task; no host rule fabricates a tool call.

`ResearchCloseoutMiddleware` uses installed LangChain node hooks. A declared
scientific stall receives one independent reconsideration before parking. Pending
background work can finish the current turn and resume it on completion. A useful
authorized remedy normally receives one validation; completed delivery, user pause
and concrete execution/authorization boundaries take precedence. The hook does
not infer scientific intent from prose keywords or message roles and does not own
a second scheduler.

Profiles bind the `research_challenger` role. An existing private profile with only
`hypothesis_proposer` retains that model binding through a compatibility fallback.
The old synchronous role is retained solely for historical planning/comparison
paths, not exposed as a current root subagent. The challenger is an internal async
lane, not an additional user-selectable conversation mode.

The workspace Graph stores scoped judgment edges, same-kind `revises` links and
key stopping decisions. Graph recovery only reconciles accepted legacy work and
notifies the existing root of relevant external scientific changes using the DBOS queue. Background children retain their own completion bridge. Notification
cursors live in thread metadata; they do not select scientific work.
Technical edits and late results cannot reopen a delivered stage.

The local integration was checked against installed `create_deep_agent`,
`SubAgent`, middleware hooks and native graph signatures. It uses stable [middleware
hooks](https://docs.langchain.com/oss/python/langchain/middleware/custom) and
[native subagents](https://docs.langchain.com/oss/python/deepagents/subagents);
newer public profile/mode options are not assumed available locally.

## Model calls and delegation

Model request settings come from the selected CatMaster profile. The factory
passes `timeout_s` and an explicitly configured `max_retries` to the provider
integration. Codex OAuth templates leave `max_retries` unset and use the pinned
OpenAI SDK default for transport, rate-limit, and HTTP server errors. A narrow
model middleware handles transient overloads that arrive inside an already
accepted HTTP 200 stream, including the structured `server_is_overloaded` code
and the provider's canonical retry-later or request-ID messages. It is attached
through the DeepAgents `openai-codex` provider profile, so the same behavior
applies to specialists, named workers, declarative subagents, and CatMaster's
explicit `general-purpose` child. The specialist runner only retries a completed
episode when its final report cannot be parsed.

CatMaster supplies one explicit subagent named `general-purpose`, which replaces
the auto-added DeepAgents child for every specialist and named worker. It is a
bounded context worker, not a coordinator: the child completes one self-contained
task brief and cannot delegate further. CatMaster passes the caller's staged
skill roots and backend, installs the same explicit Todo and filesystem contract,
and adds native file access plus nonfatal tool-error handling. The child
does not receive the caller's full specialist prompt or persistent memory.

Fresh subagent tasks remain context-isolated. For an unchanged objective, a
parent forwards the prior handoff's validated parameters, authoritative paths,
completed conclusions, and remaining work; the next worker reuses that compact
block without raw tool history or repeated preflight.

## CatMaster middleware

`SpecialistRunner._build_default_middleware()` installs
`catmaster_nonfatal_tool_errors`.

When a bound tool raises an ordinary exception, this middleware returns a typed error
`ToolMessage` with a compact artifact. The current agent can inspect the error
and decide whether to correct its inputs, choose another method, or report the
blocker. Model request retry remains separate and provider-owned.

LangGraph `GraphBubbleUp` control signals, including `GraphInterrupt`, propagate
unchanged. A capacity switch checkpoints and waits through native
`interrupt`/`Command(resume=...)`; while waiting, no model or tool work continues.
The resumed call returns the granted `task_cost` and `admitted: true`.
It never becomes a recoverable tool error. See the
[LangGraph interrupt contract](https://docs.langchain.com/oss/python/langgraph/interrupts).

The supporting functions are:

- `_nonfatal_tool_error_result` in `catmaster/specialists/runtime.py`
- `tool_error_to_message` in `catmaster/runtime/tool_output_adapter.py`

## Todo and filesystem contract

CatMaster pins DeepAgents 0.7.11 and constructs the model-visible common tools
explicitly at each `create_deep_agent(...)` call site. Every specialist,
compiled worker, explicit `general-purpose` child, Literature Review worker,
and read-only reasoning delegate receives its own `TodoListMiddleware` instance.
`write_todos` replaces the complete current list; callers use
`pending`, `in_progress`, and `completed`, and close or remove pending items
before the final response. The middleware does not add a second planning prompt.

Writable roles receive this backend-bound filesystem set:

```text
ls, read_file, write_file, edit_file, delete, glob, grep, execute
```

Read-only reasoning roles receive only `ls`, `read_file`, `glob`, and `grep`.
At the final model-visible filesystem layer, every role receives the resolved
absolute project `files/` directory as its physical workspace boundary and
shell-location reference. Filesystem function tools use a distinct virtual
namespace whose `/` is that same `files/` directory; the physical absolute path
must not be passed back to those tools. Writable roles are told that `execute`
already starts in the physical directory and should use workspace-relative
paths. A leading `/` in a shell command is a host absolute path, whereas a
leading `/` in a filesystem function tool is the workspace virtual root. Host
files outside the physical directory are not an inspection surface except for
explicit resource mounts. Routed memory uses virtual filesystem paths; bundled
skills and EasySlides additionally expose the run's documented shell mappings.
`write_file` creates a missing file and replaces an existing file in full. Its
tool description tells the agent to read an existing target before an intended
replacement and to prefer `edit_file` for local changes. `delete` recursively
removes an explicit file or directory. CatMaster refuses an empty delete path,
the virtual workspace root `/`, and the routed memory root `/memories`; specific
workspace paths and specific files below `/memories/` remain available to
writable roles. Ordinary authorized file deletion does not require Review.
The virtual `/` is already the workspace's user-visible `files/` directory;
`files/report.md` and `/files/report.md` are accepted aliases for `/report.md`
so a path copied from the UI or supplied by the user is not nested twice.

Self-evolution uses `create_deep_agent(...)` for reflection, proposal, and
review, with ordinary tool selection set to `auto`. Each phase can finish with
a nonempty text response or explicitly submit a decision through the declared
`ReflectionBatch`, `ProposerResult`, or `ReviewerResult` tool. The tool names
and input schemas remain available alongside ordinary inspection/editing tools;
there is no agent-level `ToolStrategy` or provider JSON response constraint.
In the installed LangChain implementation, ToolStrategy forces `any` on every
model call, which becomes OpenRouter `required`; the test3 MiMo/DeepInfra replay
demonstrated repeated calls under that binding.

`_OptionalResultMiddleware` uses the documented `ModelRequest.override`,
tool-call wrapper, and `Command` state update interfaces. A valid decision sets
`structured_response` and ends before the next model call. Invalid arguments
return a normal tool error for correction within the same conversation. Submit
the final transaction alone, after needed inspection/editing results; a batch
that mixes final submission with another call returns an error for that
submission while ordinary calls execute normally. This does not limit ordinary
parallel tool use. Both synchronous and asynchronous invocations follow the
same contract. See [tool state updates](https://docs.langchain.com/oss/python/langchain/tools)
and [custom middleware](https://docs.langchain.com/oss/python/langchain/middleware/custom).

A text ending completes naturally without a reminder or an extra model call.
The receiver preserves text as `TextResult`, including Markdown/JSON code
blocks, without parsing it into a decision. Empty or reasoning-only output is
not a completed response. Reflection text is recorded in the job outcome;
it is not relabeled as `no_change` and does not create a candidate. Proposal
text leaves observations open and retains edited files as an unsubmitted
draft in the job evidence directory, outside immutable revision slots. Review
text is saved with the exact candidate and leaves it unapproved, without
requesting another revision. A requested revision that ends with proposal text
completes the job while preserving the previous candidate revision. The WebUI
shows text completions and their conclusions separately from errors, submitted
decisions, and older jobs without detailed outcomes.

Proposed edits live in files changed through ordinary filesystem tools. There
is no patch payload. Text does not authorize candidate creation or activation;
those actions require an explicit validated decision and the existing review
and activation checks. Full invocation and tool observations remain recorded.

Each phase receives DeepAgents' large-result offloading, automatic
history summarization, and context-overflow recovery. Its summarization prompt
preserves exact run/event handles, inspected query pages, unresolved evidence,
and candidate edits so compaction does not turn a traceable finding into an
unverifiable synopsis. Offloaded working context uses an invocation-local
backend; it is readable during the episode and is not added to an immutable
candidate revision or a persistent LangGraph checkpoint.

Trace and history SQL results above the inline result boundary are written as
complete JSON files in that same readable context backend. Every query gets a
distinct path under `/.self_evolution_context/query_results/`; the model receives
`result_path` and can use the ordinary file tools. Direct Python callers without
a result writer receive an explicit pagination error instead of partial rows.
The complete exported result is also returned as the native tool artifact,
so ordinary raw tool observations retain it after the temporary file is removed.
The model receives the compact content, not another copy of the artifact.
Trace views expose a `handle` column. The ordinary view retains existing event
references; original-record references carry `?view=raw`, so the reader can
return exactly the queried body without a separate field selection. The agent
copies these values unchanged. Old bare references to raw-only records still
read their original body. Field continuation and JSON1 selections remain available.

Query failures use LangChain's native `ToolException` with `handle_tool_error`,
which produces an error `ToolMessage` while allowing the agent to recover.
Descriptions carry the complete SQL column definitions; errors carry only the
operation, reason and relevant recovery guidance. The trace authorizer permits
SQLite's empty-column reads for aggregate queries over the selected run while
retaining the existing column and write boundaries. These behaviors follow
[StructuredTool error handling](https://reference.langchain.com/python/langchain-core/tools/structured/StructuredTool)
and [SQLite's authorizer callback contract](https://www.sqlite.org/c3ref/set_authorizer.html).

Each reflection, proposal, or review invocation writes an independent ordinary
run under `metadata/runs/self_evolution_<id>/`. Its `meta.json` identifies
the phase and source run/thread, plus the learning job/attempt or manually
reviewed candidate when available. Candidate work also records its revision
directory. These observation runs do not create research threads or enqueue
another learning episode, and their usage is not added to the source research
run's summary.

All three phases expose current guidance at `/current/skills/<group>/<name>/`
and workspace guidance at `/current/AGENTS.md`, with a readable index at
`/current/catalog.md`. The complete skill directory includes its referenced
files and shared support directories, including assets without a `SKILL.md`.
Selected workspace revisions replace the corresponding repository directory;
unselected drafts do not replace current guidance. Disabled skills are labeled
and remain readable for comparison, without being enabled by inspection.

The reflector's `query_effective_skills` pages over the same staged selection
and returns `target`, `description`, `enabled`, `selected_version`, and `path`.
The model copies `path` into the ordinary `read_file` tool and can use `ls`,
`glob`, and `grep` for related files. A target without a readable current body
has an empty path and an explanation. UI selection controls remain in the UI
catalog rather than the reflector's result. Selection changes apply when a new
guidance context is prepared; an in-flight catalog continues to describe its
staged files. Proposal and review retain the same candidate revision's context.

Reflection mounts this guidance through the documented
[DeepAgents filesystem routing interface](https://docs.langchain.com/oss/python/deepagents/backends).
Its investigators share the same backend and read tools. Proposer and reviewer
use the same guidance-staging implementation inside the candidate workspace.
Their existing mutation boundary keeps `/current` read-only and allows candidate
edits only in `/proposed/` and the exact candidate memory file. Context inspection
does not migrate or rewrite the workspace selection file.

Request-scoped [LangChain callbacks](https://www.langchain.com/blog/callbacks)
propagate into native investigators and internal summarization calls. The
shared `ObservabilityCallbackHandler` records model requests/responses, reasoning
returned by the provider, tool inputs/outputs, errors, and subagent events in
`observability.sqlite`. `observed_invocation()` supplies the invocation lifecycle,
source links, and per-call `usage_summary.json` updates. The same observation scope
serves thread-title generation and optional legacy monitoring summaries in their
own `thread_title_<id>` and `live_summary_<id>` directories. These calls do not
write into the originating research run or become research resume targets.
The summary includes input, cached input, output and reasoning details; reasoning
is already included in output. Child and summary roles remain distinguishable.
Callback IDs deduplicate repeated notifications, and concurrent callbacks serialize
summary updates. Each new invocation, including a retry or correction, has its own
run directory, so later failures cannot overwrite earlier paid work.

`calls` counts completed responses; `failed_calls`, `pending_calls`, and
`missing_usage_calls` report failed calls, responses still awaited, and terminal
calls without usage respectively. `partial` is true while calls are pending or
known usage is missing. Token totals sum only reported usage. START/END/ERROR
events remain available to investigate an interrupted invocation; a process
termination may leave its last recorded status running. Ordinary workspace
audits discover these databases through the same `metadata/runs/*` scan and
must filter by each call's completion timestamp for daily totals. Accounting
records retain the actual model inputs and outputs through the ordinary raw
callback events; provider HTTP capture retains its usual opt-in behavior.
The existing developer diagnostics run selector, observability snapshot, and
paginated event endpoints can inspect these runs. Public conversation and monitor
projections remain separate from raw debugging details. Historical inputs,
outputs, and usage that were never recorded cannot be reconstructed.

Standalone invocation failures close pending model calls as failed when native
callbacks did not report the terminal event, including cancellation during an
async model call. Completed calls retain their usage. Ordinary specialist runs,
proposal checkpoints, native/background subagents and their internal summary
calls use the runner's existing callbacks. Models invoked inside registered
LangChain tools inherit the request callbacks through
[RunnableConfig propagation](https://reference.langchain.com/python/langchain-core/runnables/config);
adding another callback at those nested call sites would duplicate observations.

Self-evolution's event reader preserves text annotations, unfamiliar content
blocks, readable response metadata and additional response fields. Partial model
responses retain their failure status. Encrypted transport is omitted from the
semantic view; the underlying observability record remains unchanged. Ordinary
event pagination also applies to these fields.

The CLI and compatibility specialist report path retains the authored final
answer even when it contains `Summary`, `Facts`, or `Files` headings. Parsing
extracts metadata without replacing the prose. Newly produced compilation
outputs and diagnostics are appended to the answer.

Local DBOS completion feeds the existing learning queue: native `success` means
execution `done`, not a scientifically verified outcome. Native run IDs are kept
verbatim. Evidence headers use the selected run's execution message and bound
input in `metadata/workspace.sqlite`; model/tool evidence remains in that run's
observability database. Reads are scoped to the original thread and run, so newer
turns and learning notices cannot replace the selected answer. CLI and legacy
run-state evidence retain their existing reader. No checkpoint history is loaded
or duplicated for learning. Checkpoint continuation retains its input episode,
and a resumable error or interruption waits for completion before automatic
learning. The existing worker publishes learning notices to the originating
chat without starting a research turn.

The coordinator persists every durable reflection item before materializing a
candidate. The reflector's target is an attribution anchor; after the proposer
selects the final owner, that resolved target defines the candidate ID, lock,
revision chain, and history scope. Different anchors that resolve to the same
owner therefore converge on one serial revision chain. The default evidence
bundle contains only that target's current open delta. Consolidated observations,
committed revisions, jobs, and exact-version skill-use records remain available
through pageable history queries, and an authorized historical `run_ref` can be
opened lazily without parsing every old trace at agent startup.

`defer` leaves the open delta available for a later real episode; `ignore`
records and consumes a finding that is not durable. A normal new-evidence
revision starts from the currently selected effective version. Only bounded
`needs_revision` repair or an explicit continuation starts from the preceding
draft, so a terminal rejected draft is retained as history but is not silently
copied into the next proposal. Independent reviewer approval follows the
workspace policy directly: in `auto`, an enabled Follow auto target is selected
for the next run without a second human confirmation. Evolution history exposes
only revisions committed through the candidate index, so a reviewer does not
mistake its own still-open transaction for a corrupt historical revision.

Self-evolution explicitly replaces DeepAgents' broad default `general-purpose`
child with a read-only evidence investigator. The child may isolate one bounded
trace, history, source, or candidate inspection and returns concise findings
with exact evidence handles. It cannot edit candidate files or make proposal,
review, canary, or release decisions. The proposer may mutate descendants of
`/proposed/`, may write the exact candidate memory path
`/memories/AGENTS.md`, and cannot delete candidate memory. Reflection and review
have only read tools. The trace SQL views expose callback and parent-callback
IDs so the agents can reconstruct causal branches without flattening modern
multi-agent trajectories.

The shared CatMaster prompt adds only the execution invariants not already
owned by role prompts and tool descriptions: continue multi-step work to a real
completion or blocker, diagnose repeated failures before retrying, keep progress
brief, honor the user's scope, and keep validation proportional to the requested
scientific or technical result.

DeepAgents `execute` reports a command that ran as a successful tool transport
even when the process exit code is nonzero. CatMaster preserves the optional
`artifact.exit_code` in observability records, thread events, and the WebUI tool
card. An unknown exit code is displayed as unknown rather than inferred from
message text.

## Multimodal content

Current-turn attachments use LangChain content blocks when the selected model
supports the media type. CatMaster stores uploaded binaries as workspace
artifacts and keeps raw base64 out of durable WebUI thread messages. DeepAgents
`read_file` image and document results remain multimodal for the next model
call. CatMaster only fills the pinned DeepAgents Word/Excel extension gap in the
backend; provider-specific conversion happens at the model boundary, not
inside a parallel document tool.

See [Multimodal files](deepagents_multimodal.md) for the supported user flow,
persistence rules, and provider behavior.

## Codex OAuth `apply_patch`

Codex OAuth roles receive a LangChain custom tool named `apply_patch`. It accepts
the freeform V4A patch envelope and can add, update, move, or delete several
files in one call.

Execution is restricted to the current project `files/` root. The implementation
uses workspace locking, atomic replacement for individual files, path traversal
checks, symlink checks, and model-visible conflict errors.

The model emits a `custom_tool_call`. DeepAgents executes it through the normal
tool scheduler and returns `custom_tool_call_output` on the next model call.
For LangChain v3 event streaming, CatMaster restores missing scheduler metadata
from the completed provider block at the Codex OAuth model-result boundary. The
original block is retained unchanged for Responses API replay, and recovery is
skipped when LangChain already supplied the tool call.
`/memories` remains a routed DeepAgents store, so persistent memory edits use
`edit_file` rather than this workspace patch tool.

Per-run usage fallback ingestion accepts only `on_chat_model_end` or
`on_llm_end` payloads and counts the last usage-bearing AI message in that
event. LangGraph v3 `values` and `updates` are checkpoint projections and may
contain usage-bearing messages from earlier turns, so they are never treated as
new calls. The legacy multi-mode fallback likewise accepts only its `messages`
projection.

The local execution entrypoint wires the existing usage callback to
`usage_summary.json` and the thread's `usage.updated` events after each completed
model call. A recovered run restores its prior accumulated usage before counting
new calls; historical observability supplies the baseline when no summary exists.

LangChain 1.4.1 can also reconstruct an old hosted-web result as a Responses
`web_search_call` with an empty action object. OpenAI requires an action type on
that item. CatMaster drops only such malformed replay items from the outbound
request while preserving valid hosted-web calls, result text, and checkpoint
history.

The live acceptance script is:

```bash
PYTHONPATH=. \
  conda run -n catmaster-dev python \
  tests/manual/codex_oauth_apply_patch_live.py --workers 3
```

## OpenRouter content conversion

`CatMasterChatOpenRouter._create_message_dicts(...)` converts standard
LangChain media blocks into the shapes accepted by OpenRouter:

- image base64 blocks become `image_url` data URLs;
- file base64 blocks become OpenRouter `file` blocks with `file_data`;
- existing provider-native `image_url` and `file` blocks pass through.

Assistant content is restored before conversion because the installed
LangChain adapter selects only text blocks. Plain reasoning blocks use
OpenRouter's documented
[`reasoning` replay field](https://openrouter.ai/docs/guides/best-practices/reasoning-tokens#preserving-reasoning);
native `reasoning_details` and tool calls retain the upstream conversion.
Unsupported replayed blocks and annotated assistant text blocks are serialized
as complete JSON text before SDK validation. This preserves information that
the SDK would reject or silently remove. The conversion is scoped to CatMaster's
OpenRouter subclass and does not modify stored messages or installed packages.

The relevant functions are in `catmaster/llm/factory.py`:

- `_sanitize_openrouter_message_dicts`
- `_sanitize_openrouter_content`
- `_openrouter_data_url`
- `_attach_openrouter_cache_control`

## Dependency and verification contract

`requirements/pc-conda.yml` is the source of truth for DeepAgents, LangChain,
OpenRouter, and OpenAI package versions.

After changing one of these dependencies, run:

```bash
conda run -n catmaster-dev python -m pytest \
  tests/test_local_execution.py tests/test_local_execution_controls.py tests/test_local_execution_recovery.py \
  tests/test_native_apply_patch.py \
  tests/test_openrouter_message_sanitizer.py \
  tests/test_specialist_runtime.py \
  tests/test_deepagents_074_contract.py \
  tests/test_llm_factory_extra_body.py -q
```

Stage dependency upgrades and their focused and full tests in `catmaster-dev`.
Apply the same exact pins to the main `catmaster` environment only after the dev
checks pass and the environment sync is approved.

Also verify:

- image and PDF handling in a fresh thread;
- replay through the same `thread_id`;
- OpenRouter serialization after a media-bearing message;
- context compaction after media-bearing tool results;
- one Codex OAuth patch call when that provider is enabled.

Notable changes to these contracts belong in the repository
[Changelog](../CHANGELOG.md).


## Instruction context and skill loading

The runtime stages the complete effective skill bundles and exposes the roots
appropriate to each role. Native DeepAgents initially injects skill names,
descriptions and paths, not every skill body. Selected `SKILL.md` files and
supporting references are read through ordinary file tools. Per-invocation context
refresh reloads metadata and instruction memory; it does not eagerly insert all
skill bodies into model messages.

Filesystem tools read skill resources at `/.deepagents/skills/<relative-path>`.
Local shell commands execute the same resource at
`"$CATMASTER_SKILLS_ROOT/<relative-path>"`; the environment is bound to the
effective snapshot of that backend, so stable and canary runs do not share a
mutable symlink. Python bytecode writing is disabled for these shell backends.
Input and output paths remain workspace-relative. Utility scripts can run from
their mounted `scripts/` or `references/` directory without a workspace copy or
source read. Copies are needed only for project-specific implementation changes.
Usage references and `--help` supply ordinary invocation contracts.

Diagnostic helpers print status, actionable violations and report paths; complete
scientific tables remain in the report for ordinary file reads or local queries.
Batch wrappers should preserve that separation rather than putting full metrics
in exception messages. Query/export tools retain their native data-return behavior.

Shared prompts treat skill selection as task-dependent. Agents reuse instructions
already available in their context and read operation-specific references only
when needed. A routine reply is not a startup skill-reading exercise. Scientific
method constraints, writing-quality guidance and actual authorization boundaries
remain binding. Full references and source assets remain available; the host does
not shorten or hide skills to achieve a token quota.

Remote stage layouts link directly to task-specific Markdown references; equivalent
execution variants may share a layout. The task catalog returns those references
for both native and MLFF tasks. Legacy bundled `SKILL.md#task` references resolve
to the corresponding task document when present, without rewriting deployment
configuration or replacing explicit custom references. Before unfamiliar preparation,
agents read that task's internal file and
parameter-placement conventions. The live task spec supplies accepted fields and
defaults; its compact table is normally sufficient, with full schema available
for unclear nested requirements. Internal contracts are retained, not inferred
through failed submissions. Research graph coordination keeps scientific decisions
and shared ownership/continuation boundaries in its entrypoint, with the complete
discussion, correction and cost-change protocol in a short conditional reference.
Citation and venue-template skills route to their
existing topic-specific references and helpers without requiring unrelated figures.
Self-evolution locates plausible owners through the catalog before reading their
bodies. Its reflector distinguishes missing or poorly routed internal guidance
from a failure to follow adequate, discoverable instructions. Proposer and reviewer
preserve internal contracts and verify that an authoritative definition is visible
to the recipient before removing a duplicate. Simplification is judged by correct
completion and total reading, discovery and retry cost; shorter text alone is not
evidence of an improvement. Detailed authoring rules live in
[skills/AGENTS.MD](../skills/AGENTS.MD#progressive-disclosure).
