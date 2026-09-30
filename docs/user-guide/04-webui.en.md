# 4. Working with agents in the WebUI

[Previous](03-llm-configuration.en.md) | [Contents](README.en.md) | [Next](05-agents-and-modules.en.md)

The WebUI keeps conversation, project files, agent activity, human approval, and run observability on one page. You do not need to learn every backend state. Two habits matter most: ask the agent to save important results in the workspace, and inspect what it actually did when it changes a scientific structure or submits remote computation.

## Finding your way around

Without a selected workspace, the page offers workspace selection and a creation form. On first use, name and create a workspace. Neither login nor no-login mode automatically creates `default` or `admin`. Open existing workspaces from the sidebar or a link's `project_space` parameter. A missing workspace opens the selection/creation page without recreating its directory.

The left sidebar holds the workspace selector, **New conversation**, navigation, conversation folders, and a compact file tree. Chat is the main work surface; Research Graph, Files, Monitor, and Skill Evolution open from the same sidebar. Selecting a conversation returns to Chat.

An empty conversation offers literature, calculation, and study-planning starting points. Each fills an editable draft; it does not send a message or start work. The composer keeps the Agent selector and Review/Auto controls next to the draft. Use the header's task-context button to hide or show the right panel on desktop. On narrower screens, navigation and task context open as drawers.

In Research Graph, expand **Research graphs** to browse or switch graphs. The selected graph's full question, completion criterion, and preferences remain available under **Research question & completion criteria**, leaving more room for its nodes and relationships.

## Dividing work into workspaces and threads

The workspace selector is at the top of the left rail. A workspace is a long-lived project containing files, conversations, run records, and project-specific experience. Separate catalyst systems, manuscripts, or ML datasets usually belong in separate workspaces so that file search and project memory do not mix unrelated research.

A workspace can contain many threads. Use a thread for one continuing line of work, such as "CeO2 surface models," "ORR free energies," or "second manuscript revision." Ordinary turns, approval resumes, and checkpoint continuation reuse one local execution thread and its checkpoints; each submission creates a separate native run. CatMaster stores a WebUI projection for navigation and rendering, but does not separately decide run queuing, stopping, or resumption. A new thread begins a separate conversational context, so include the necessary paths and assumptions again. An empty thread displays as `New thread`. Its first ordinary message immediately supplies a local title; when a lightweight title model is configured, that title is refined in the background. Title generation never blocks the research turn, and a manual rename always wins. Use the pencil beside a thread to rename it in place; Enter or the check button saves the title, and Escape or the cancel button leaves it unchanged.

Refreshing a browser tab restores the thread explicitly opened in that tab, including a background execution thread opened from Task Context; it falls back to a visible root conversation only if that thread no longer exists. Once existing messages load, Chat lands on the latest message instead of an older background receipt. A new tab has no tab-scoped selection and still defaults to a user-visible root conversation.

The left rail also contains a compact file tree. Use it to open a structure, report, or log quickly. On desktop, drag the separator on the rail's right edge to change its width; the setting is remembered, and the focused separator also accepts arrow, Home, and End keys. Long filenames remain available through the horizontal scrollbar in both this compact tree and the full Files explorer. The full upload, preview, and download controls are in the Files view.

## Choose the entry that matches the main deliverable

The composer lets you select Research, Persistent Research, Experiment, Writing, Peer Review, or Literature Review. A long-lived thread may switch entries between turns while retaining its conversational checkpoint. The entry cannot change during an active turn because each entry builds the corresponding agent, tools, and workers. Each agent message shows the entry actually used for that turn, so the thread's current selection does not relabel older work.

Graph attachment and Persistent Research orchestration are separate. An ordinary Research thread may automatically attach the workspace's single open graph, focus one of its Experiments, and independently execute work or write back a Result. That does not make it a Research Session root or route it through automatic planning, comparisons, or session steering. When the current entry is Persistent Research, New thread uses the workspace default instead of copying that entry; the standard default is ordinary Research. Switching a Persistent Research root to an ordinary entry between turns retains its graph and focus but ends continuous orchestration; later messages start normal turns in that thread.

Choose Experiment for a bounded structure, calculation, or trajectory task. Use Literature Review for evidence discovery and reading, Writing when source material already exists, and Peer Review for an independent assessment of one PDF. Use Research when an objective crosses several stages but should close on demand. Choose Persistent Research when the same objective should keep advancing through its Research Graph automatically.

The wrong entry may still answer, but it often adds friction. Research is unnecessarily broad for a 3x3x1 supercell. Writing does not have the calculation workers needed to reconsider an adsorption energy. Chapters 3, 5, and 7 give fuller examples.

## Persistent Research sessions appear as folders

A Persistent Research task keeps one stable root in the left rail. Native async specialist activity appears in Task Context. Existing experiment execution threads remain accessible in the Research Session folder, and historical planning/comparison records remain inspectable. The root integrates child results and relevant scientific edits. The Graph remains workspace-owned; the folder does not copy scientific records or outputs.

The folder reports the aggregate research state, not just whether the root thread is generating a message. If the root conversation is idle while a child is planning, using a tool, or waiting on remote computation, the current experiment and latest progress remain visible. `waiting_continue`, `waiting_review`, `operationally_incomplete`, paused, and completed retain distinct labels. Research progress in the right-side Task Context shows the same current phase and `Running now` activity. Open the execution child only when you need its full context.

When a researcher records a meaningful Result, the root receives a `Research milestone` containing its observations, conclusion and recorded methods. Amending a Result updates that same message. Calculation output files do not automatically become Results. The Research Session panel shows the latest scientific result and opens its cited report through **Open latest report**. Activity includes nested research and execution tasks; an idle root does not imply that background work has stopped. `Waiting — research unfinished` means the objective remains open.

The root conversation is also the continuing control point. A message sent there always stays in that thread and is never redirected implicitly to an execution child. While a native run is active, the composer requires an explicit **Steer**, **Queue**, **Replace**, or **Reject if busy** choice and sends the corresponding `interrupt`, `enqueue`, `rollback`, or `reject` strategy directly to local execution. To change a background branch, use Steer on that async specialist in Task Context; Stop likewise targets the displayed native run. Pausing the research session prevents automatic continuation; child tasks and remote jobs retain their own execution controls. Resume reuses existing scientific work and thread relationships.

Search covers both the folder title and its execution children. A child match displays a `root session / child thread` breadcrumb and expands its folder when selected. Active sessions and sessions requiring action open by default; the browser remembers other expansion choices. Refresh, reconnect, and service restart preserve the grouping. An older execution whose parent cannot be established reliably appears under `Related research activity` instead of being assigned to a guessed session.

## How native async specialists return to the root conversation

Task cards show the task's low/medium/high cost. A capacity wait resumes automatically;
it is not an approval or a stopped task. A researcher waiting for its computation
still occupies its slot. Independent ResearchSpecialist branches keep their own
research context while the foreground conversation remains interactive.

Research can use background specialist tools to submit independent Literature Review, Experiment, Writing, or Peer Review branches to their own local execution threads and runs. Each child keeps a card in the conversation. Activity lists active children with their latest update and current tool. Open the process viewer to read specialist and nested-worker messages, with filters for all activity, updates/results, reasoning, and tools. Tool details expose complete paginated input/output and a return button to the child process.

Each worker response updates one message from streaming through completion, including after reconnecting. Multiple messages from the same worker represent successive responses, not additional workers. Completed, interrupted, and failed activity keeps its available text and tool results with the corresponding status.

Returning to the browser tab or reconnecting loads current state before resuming live updates. Activity cards show the latest progress without playing back old tool calls accumulated during the absence. Full process history remains available in the viewer.

Cards and task instructions distinguish the original brief from the latest supplementary instruction. Its waiting, processing, finished, interrupted or failed state follows the corresponding execution turn; it does not establish that the model understood or resolved the issue. Older records without a corresponding turn show their text without an inferred state. The main agent should choose `interrupt` when correcting a premise, source claim, method or scope the child currently uses. Additional evidence, later questions and cosmetic edits can use `enqueue` when current work remains valid. Queued instructions wait until the current turn and its synchronous workers return.

Steer in this viewer sends instructions to the selected child on the same thread; Stop targets the displayed run. Closing the viewer or refreshing does not stop native work. Active children remain visible across parent turns, and completed cards retain their process entry. Historical worker events that were never captured cannot be reconstructed; saved checkpoint messages remain available.

Background tasks normally enqueue their result as a new parent turn for synthesis. The caller can choose `on_completion=notify` to save results and update Activity without invoking the parent. Stop/pause takes precedence. Delivery belongs to the durable workflow and is independent of browser connections.

`Thread ready` therefore means only that the root thread has no active run; async specialists may still be running. Concurrent specialists should be read-only or use disjoint outputs. If they may write the same path, separate the paths or arrange an explicit execution order.

## Give scientific boundaries, not a tool script

Natural-language requests are enough. State the objective, input files, constraints that must survive, the allowed scope, and the artifacts you want to keep. If the method is unsettled, ask the agent to compare choices and explain them. If a project standard already fixes the method, state it directly.

```text
Use Experiment to inspect structures/slab.vasp and build starting structures for CO adsorption.
Preserve the existing Selective Dynamics. Inspect surface coordination, periodic boundaries, and usable
adsorption regions before selecting relevant skills and tools to enumerate deduplicated sites and place CO.

Write candidates, site provenance, and a geometry audit under structures/co_candidates/ and notes/co_sites.md.
If the slab itself is unsuitable, stop and explain the problem instead of continuing. Do not prepare or submit
VASP in this turn.
```

Include units for numerical settings, charge and multiplicity for molecular work, and a seed or reproducibility requirement for stochastic work. Address existing files by workspace-relative paths such as `structures/slab.vasp`, not by private host paths.

## What happens to attachments

Attach accepts images, PDFs, modern Office documents, structures, and other files with the current message. The backend first stores them under `files/attachments/<thread_id>/` and registers them as artifacts. The agent therefore receives a traceable project file rather than browser-only data.

Images can be sent as visual content when the selected model profile supports them. Compact PDF, DOCX, XLSX, and PPTX files can be sent as native file blocks; larger or text-heavy documents are stored first and later opened through bounded `read_file` pages. The agent can render selected PDF pages when visual inspection is required. Audio, video, legacy Office formats, and unsupported media may be stored without being sent to the model. The `multimodal.prepared` event in Monitor records whether an attachment was sent, how it was represented, and any degradation warning.

Attachments are convenient for the current message. Files that will be reused should live at stable project paths such as `literature/corpus/`, `structures/`, or `data/`.

## What Chat reveals about agent work

Chat contains more than final prose. Todo updates from one user task, including new assistant messages created by checkpoint continuation, are consolidated by semantic agent role into one final Plan card per role at the top of the last reply; intermediate snapshots remain in the underlying trace. Reasoning, stage notes, and tool calls follow as a middle activity layer before the final prose and are grouped by a specific subagent lifecycle, so two invocations of the same named worker remain separate activity groups. Short groups open directly; groups with many activities or one substantial reasoning block collapse to the current or latest activity and retain the complete unsummarized trace when expanded. Remote receipts and artifacts remain separate; opening an artifact or complete activity detail uses the large centered preview instead of narrowing Chat or taking over the right rail.

During a long turn, **Running now** stays immediately above the composer. It is a projection of the same persisted active tool records, not a second monitor: the oldest active operation remains first, multiple operations can be expanded, and each item shows its actual start time and locally updated elapsed time. A blocking managed calculation therefore remains visible even when no new model event arrives or its original tool card is deep in the trace. The item leaves this panel only after a terminal tool event. Refresh reconstructs it from the current turn; after an unclean service restart, work owned by that WebUI instance is marked Interrupted instead of remaining falsely Running. CatMaster does not infer an ETA, scheduler phase, or completion percentage.

Research and execution agents can also publish a concise semantic update at a major phase transition, before substantial delegation or a blocking managed calculation, or when evidence materially changes the plan. The latest update stays visible above the Plan card; earlier phase updates remain available in its compact history instead of being buried in the tool trace. These updates are deliberately sparse and never gate execution. They are not periodic heartbeats. Raw reasoning remains expandable below them; independent blocks now carry visible source boundaries instead of being concatenated into one unbroken string.

These parts answer different questions. Progress shows how the agent frames the task. Subagent activity shows which role owns the work. A tool card records the executed action. Its Technical details provide authenticated, character-paged access to the complete stored input and safely redacted output when the compact card is insufficient. An artifact is a reusable result. A remote receipt identifies a submitted job and its recoverable state. Artifact and receipt discovery registers every declared result and pages the UI collection rather than dropping records beyond an inline-card limit.

You do not need to inspect every file read. Expand activity when an important structure or manuscript changes, when the number of candidates differs from expectations, when a tool reports warning or error, when a remote submission has meaningful cost, or when the final answer disagrees with the files on disk.

## Research Graph connects work across threads

The **研究协作 (Research collaboration)** section belongs to Persistent Research and its branches. It shows task goals, current work, full briefs and shared discussion with replies, node filtering and history. Branches can communicate directly. Independent ordinary Research sessions do not receive the discussion tool or closing check; existing Graph evidence and historical discussion remain readable.

During result continuation, synthesis and closeout, the main researcher examines relevant discussion, including unanswered method questions between branches. It can answer from existing evidence, explain why no further work is warranted, or send a concrete follow-up to the original subagent. Additional questions queue behind valid running work; corrections to the child's current premise or instructions use interrupt. Both preserve its context. Nested workers are reached through their owning research branch; parent/child execution authority is unchanged.

Posting never wakes an agent. Messages arriving after the root becomes idle wait for its next user/completion turn. Discussion cards can show an answer, a decision not to continue, or an accepted follow-up, with the recorded reply. Assignment is not scientific resolution or an extra execution button. Unanswered messages do not block Graph completion. Pause settings, task completion notification choices and user authorization still apply; discussion does not reopen completed stages or authorize calculations/experiments.

Research Graph is workspace scientific state. Its catalog shows the question, node counts, runnable frontier, last update, and whether the current thread is attached. The catalog is for user browsing and explicit selection; it is not passed to agents. Attach, Detach, and Switch change the thread's focus without copying or deleting scientific state.

The backend fixes the graph binding when accepting a turn and tells the agent its
query target; background delegation inherits the corresponding binding. Scientific
queries and writes use that graph automatically, without a model-supplied graph ID.
Agents do not receive workspace-wide graph listing, creation or switching tools.
Node choices, scientific methods and revision-conflict handling remain with the
agent. Saved questions and completion criteria are shown separately from the
current request. Attaching a completed stage does not reopen its scientific state.

New graph requires only a research question. Title, completion criterion, explicitly stated decision preferences and seed hypotheses are optional. The default criterion is a defensible answer supported by recorded Results and traceable sources. A seed Hypothesis requires only its claim and may attach its motivating source. You can also submit the question directly in a Persistent Research conversation and let the root develop the investigation.

Research, Persistent Research, Experiment and Literature Review turns retain an
existing valid selection. An unbound thread uses the most recently created
unarchived Graph, including completed stages; editing an older Graph does not make
it the newest. Only a workspace without an unarchived Graph initializes one from
the first request. Later threads reuse it by default. Users can explicitly create
or switch Graphs in the interface. Delegated tasks inherit their accepted binding
instead of creating or selecting a Graph. Writing does not silently select a
Graph, and one-off tasks do not need artificial H/E/R records.

The graph has three scientific node types:

- A Hypothesis shows its concise claim, relative importance, and a relationship summary derived from all related Results; this is not an evidence grade.
- An Experiment proposal shows its objective, plan, decision rule, execution lane, coarse compute cost, and preparation or execution state. The current revision may mark one durable ready internal Experiment as the recommended next action, without attaching a score to the node. The `external` lane instead marks a complete laboratory or collaborator handoff and never enters internal recommendation.
- A Result shows a concise observation or outcome. Literature findings, collaborator results, and historical observations can be recorded without a graph Experiment. Labeled relationships connect a Result to the Hypotheses that it supports, opposes, or does not distinguish.

The canvas supports pan, zoom, fit, a minimap, keyboard access, focus neighborhood, and density limits of 5, 25, or 100 nodes. Node cards keep the complete title and accessible name. Selecting a node opens the full scientific fields and sources in Research Graph's own detail drawer; it does not change the thread's graph focus. Use **Set focus** or **Clear focus** explicitly. The saved focus is visible on the canvas and becomes the starting branch for later turns in that thread.

“Add scientific input” is designed for short entries. A Hypothesis needs one claim, a draft Experiment needs one objective, and an Observation or Result needs one summary; titles, rationale, predictions, links, rankings, interpretation, and sources are optional details. A draft Experiment may remain deliberately incomplete, but it cannot become Ready until it has both a plan and a decision rule. With **External lab / collaborator**, Ready means the handoff is complete: the panel hides Run and Replicate and shows **Record external result** so the returned observation, source, and Hypothesis effects can be entered on the same node. A Hypothesis can develop an experiment proposal, be edited, or open its related evidence. Internal Experiments can be prepared, run, replicated, linked to a dependency, marked blocked, or given a Result. A Result can lead to a user-authored Hypothesis or follow-up Experiment. Its effect on any Hypothesis can be added, replaced, or cleared later without recreating the Result.

Creating a graph through the Research Specialist also attaches it to the current thread automatically. Research, top-level Experiment, Literature Review, and Writing can explicitly move or clear the current thread's focus within its bound graph; child agents do not inherit a new focus automatically. The graph identity and any active launch remain host-bound, so focus cannot redirect an existing launch to another Experiment.

A top-level Experiment or Literature Review turn receives read-only graph query and Result writeback only when it is graph-bound. An Experiment-produced Result requires an actual Experiment focus and fails without changing the graph when that focus is absent. After an ad hoc calculation yields a reusable scientific Result, top-level Experiment can explicitly create and focus one graph Experiment before recording it. A Literature Review finding may instead enter as a sourced standalone Result without inventing a retrospective Experiment. Ordinary one-off work may still finish without a graph node. Authorization, input preparation, source or model acquisition, compatible recovery, build or scheduler diagnostics, and platform-feasibility work may finish without inventing a Result. An Experiment is blocked only when its scientific decision rule genuinely cannot be completed. The host adds the actual thread and run sources, while Writing can change its navigation focus but remains unable to edit scientific graph content.

Running an Experiment atomically claims one launch, then creates an ordinary child thread bound to the graph and focus node. Repeated clicks on the same active launch are deduplicated. A completed Experiment can start an explicit replicate. Preparation and scientifically equivalent recovery remain inside the same Experiment. A completed scientific observation creates a Result; clarification or provenance correction for the same run, dataset, conditions, and observation updates that Result in place with revision checks and preserves its relations and existing sources. A new run, condition, dataset, or independent observation creates a new Result. An error or stop produces only a retryable operationally incomplete state, not a scientific conclusion. A real writeback completes only the exact launch bound to that turn, so later Writing, questions, or attached analyses cannot rewrite an old launch's run identity. When remote submission status is uncertain, recovery checks the existing thread, run, and receipt before any new submission.

The root ResearchSpecialist owns the overall objective and synthesis. Independent Research branches own hypothesis development, scientific method selection, execution delegation, interpretation and further H/E/R cycles within their assigned objectives and existing authority. Producers record meaningful Results as they become ready. Domain specialists retain their narrower responsibilities. The optional `hypothesis_proposer` supplies independent interpretation or stopping review. An Experiment with only an objective can be saved as Draft.

When Persistent Research would stop because of a scientific stall while the user’s request remains unmet, it performs one independent reconsideration. A concrete, authorized remedy normally receives one bounded validation. Literature and recommendation tasks authorize source checks and synthesis, never unrequested calculations or laboratory work. Completion of the requested stage ends automatic work. Open premises, reconsideration findings, and resumption conditions remain in the graph.

**Pause research** in the Research Session panel uses the same thread control as Chat Stop: it stops the root turn and prevents automatic continuation. Running child tasks retain their own Stop controls, and remote jobs retain their existing execution controls. **Resume research** submits continuation of the existing objective to the same root, reusing ongoing work and completed evidence; a direct user instruction can also resume the session. Background notifications cannot clear a pause. Ordinary Research does not become persistent merely by attaching a Graph. The composer's Auto/Review setting governs tool approvals separately.

Completed marks the user’s requested stage as finished. New sources, late Results, and scientific revisions remain writable without reopening it. The user can explicitly reopen the goal. Archived graphs are read-only until Restore.

Graph nodes contain short scientific statements only. Papers, detailed notes, structures, logs, reports, artifacts, and receipts remain in their existing stores and connect through Sources. A moved or deleted source appears as "Source unavailable"; its reference is not silently removed. Graph actions do not grant protected execution. Computation still follows specialist ownership, managed execution, and the ordinary approval cards.

After a Writing thread explicitly attaches the graph, each turn receives the same partial focus context and the Writing coordinator can query the complete bound graph read-only. It locates section-relevant Results, opposing or inconclusive judgments, and their Sources before opening the original note, artifact, run, thread message, DOI, or URL. A Result summary is navigation, not a substitute for the source, and Writing cannot edit the graph. Unattached Writing threads behave as before, and Writing never guesses among several graphs by title.

Updates from other threads arrive through the durable graph event stream. If the graph changes before you submit an edit, the server rejects the overwrite and shows a readable conflict message. Refresh, review the new content, and submit again. A human can revision-safely edit a Result or delete one with a required audit reason. Agent retraction is narrower: it is limited to an unjudged, same-run accidental Result with no dependent scientific node. The overview separates recent mutation history from scientific nodes, and older mutation records remain available through its stable event-ID paging.

## Auto and Review support different working styles

Auto lets an agent proceed within its current permissions and works well for reading, analysis, and trusted project workflows. Review pauses before `remote_submission` and `remote_submission_batch`, then presents an approval card in the message. Local `write_file`, `edit_file`, and Codex OAuth `apply_patch` operations do not open approval cards.

Review is useful when a thread may submit real remote computation. The card supports four actions:

- Approve executes the proposed action.
- Reject declines it and can include a reason.
- Respond gives the agent feedback so it can rework the action.
- Edit action changes the action JSON and is intended for users who understand the tool schema.

Review is not a global approval gate. Reading, search, analysis, and local file edits still run automatically. Its purpose is to put actual remote compute submission at a clear human checkpoint. Resume an interruption through the card rather than sending an unrelated normal message.

`write_file`, `edit_file`, Codex OAuth `apply_patch`, and domain tools such as `supercell` and `build_slab` all write directly into the workspace. Give those operations an explicit destination, inspect input and output paths on the tool card, and review the artifact in Files. Review protects remote submission; it is not a transaction lock around workspace changes.

## Steer a running task without scripting every move

The composer separates agent/permission settings, active-run stop controls, and message submission into aligned rows. Stop offers keep-progress or discard-turn choices. Message strategies have icons and remain accessible on narrow screens. Ctrl+Enter uses the same strategy as the send button. The idle submit button is Send. While an agent is running, the composer explicitly offers Steer, Queue, Replace, and Reject if busy. Steer uses local execution's `interrupt` strategy to interrupt the current run and accept the new message from its saved checkpoint. Queue preserves the current run and waits behind it. Replace rolls back this turn's checkpoint effects before running the new message. Reject if busy declines the submission while a run remains active. An intentional Steer/Stop preserves partial output and marks the original message as interrupted. If a provider call or tool cannot cancel immediately, the UI can report only that the interrupt was accepted; it cannot claim that the model has already read the instruction.

If the new request changes the objective entirely, waiting for a safe stop and opening another thread is often clearer. New attachments are disabled during a run, so wait or stop before adding another file.

Stop targets the current local execution run directly. Keep progress uses `interrupt` and retains the saved checkpoint; Discard this turn uses `rollback` and removes this turn's checkpoint effects. Neither action cancels jobs already submitted to Slurm or a remote shell. Those jobs require receipt-aware scheduler handling.

## Files holds the deliverables

Files provides Browse, Preview, and Uploads. It can preview text, Markdown, JSON, images, PDF, CSV/TSV, common crystal and molecular formats, trajectories, volume grids, and selected OUTCAR vibration content.

The agent filesystem is already rooted at this Files tree. A destination such as `reports/result.md` and the UI-style `files/reports/result.md` therefore refer to the same file. File links returned in chat, including `sandbox:/files/...` links, open that file in the large overlay preview. When a successful final response names existing workspace files in an explicit `## Files` section, Chat also registers them as directly openable artifact cards so the final report is not hidden in the file tree.

An agent can also place a workspace image directly in its answer, for example `![Tensile curve](sandbox:/files/figures/tensile.png "Stress-strain curve")`. The WebUI displays it as a responsive figure card at that point in the message, with its caption and **View larger** action kept in a quiet footer; clicking the image opens the same large preview. The message stores only the workspace path, not a base64 copy. This is intended for structure renders, key curves, and mechanism diagrams; other files continue to use ordinary links or artifact cards.

Crystal, slab, defect, adsorbate, and ordinary molecule previews use MatterViz. Choose **Open Structure Workbench** for a full-viewport editor with base-atom selection, coordinate and cell editing, measurements, constraints, undo/redo, supercell and symmetry previews, slab/defect/adsorption candidate galleries, and explicit Save As. Display copies are view-only; use Make supercell before creating a real single defect. Large structures keep the complete source model for selection and saving while the canvas switches to a bounded representation.

XYZ, extxyz, and other coordinate files open in MatterViz's 3D view. Preview and trajectory playback do not require valid chemical valence or assign bond orders and charges. Displayed connections are geometric estimates; atoms remain viewable when connectivity cannot be inferred. SDF/MOL connection tables remain authoritative for the lazy Ketcher 2D editor; explicit chemical edits and conformer generation retain their corresponding checks. Saving a molecule as XYZ loses bonds, aromaticity, bond order, charge, and stereo; saving as SMILES loses the current 3D coordinates. The Workbench blocks that save until the warning is acknowledged. Periodic constraints round-trip through POSCAR/VASP and ASE `.traj`; formats that cannot express them receive the same explicit warning.

Trajectories are read-only and report their real frame count. Scrub or play them, inspect scalar properties, and extract one frame before editing. CUBE, CHGCAR, LOCPOT, ELFCAR, and XSF open as volume artifacts with structure overlay, positive/negative isosurfaces, and slices. JSmol remains the compatibility path for OUTCAR vibration and formats that the primary renderer cannot open; it is not a second editable state. VESTA renders can still appear as image artifacts.

After an agent reports completion, check that the main deliverables exist at the promised paths. For structures, inspect candidates and the audit. For calculations, inspect the stage, status, stdout/stderr, and analysis. For literature, inspect the candidate and evidence tables plus the reference library. For writing, retain editable source files rather than only a compiled PDF.

Uploading a file with the same name overwrites it. Directory deletion is recursive and permanent. Keep important originals backed up outside the workspace. The ordinary file tree shows user deliverables and working files; internal metadata, tool-result offloads, and transient extracts remain available to diagnostics but are not presented as deliverables.

## Monitor helps determine whether the process is healthy

Monitor summarizes models, agents, tools, tasks, tokens, cost, and machine time. Token totals are updated after each completed LLM call, including input, output, cache, and reasoning tokens when the provider supplies them; a call still in progress has no final usage yet. Checkpoint continuation may replay old messages to the stream, but those historical usage records are excluded from the new run's totals. Expand **Token details by model** to compare configured model labels by uncached input, cached input, cache writes, output, total tokens, and completed calls. Overview is useful for status and scale. Live shows the active stage, tools, todo list, subagents, and recent logs. Events can be filtered by thread, run, agent, tool, category, and channel. Raw and Details are for deeper diagnosis.

If an agent appears stuck, check whether a remote tool or subagent is still active. If a result is incomplete, look for tool errors, document warnings, or multimodal degradation. If cost is unexpected, inspect model calls, tokens, and machine time. Monitor is a diagnostic surface, not a report that must be copied into every deliverable.

The current UI has no historical run selector. Overview may summarize the current or most recent run for a workspace and lane. For precise remote tracking, correlate thread ID, run ID, artifact, and receipt.

## The right rail holds task context; artifacts use a large preview

The Chat right rail is one Task Context stream, not an embedded file viewer. Outputs lists files explicitly reported by the completed conversation, including workspace report links and localized output-file lists; each opens in the shared preview. Activity shows current runs and background specialists, followed by Plan and, when available, Research progress with a prominent latest-report button. Research Graph remains a full independent page. Plan follows the latest `write_todos` state: a completed root reply does not complete unfinished plan items or independently running specialists. Background completion notifications are separate, expandable notices rather than messages attributed to you. Reloading reconnects to the stream from the message snapshot's cursor. On desktop the left separator changes the rail width, and the header button hides or shows the panel. On a narrow screen the Task Context button opens it as a drawer and shows the active or attention count.

Clicking an inline image, workspace link, artifact card, or activity that needs full inspection opens a separate preview above the workspace. The surrounding page dims and the centered window uses almost the full viewport while reusing the image, Markdown, table, PDF, structure, and text renderers. Close it with the top-right button, Escape, or the backdrop; keyboard focus returns to the entry that opened it. The preview is presentation state only, so closing it does not change background work or Research Graph state.

Completed specialist-task cards use the returned scientific Markdown directly: the card shows its content title, central conclusion, and a short section outline. Open details shows the expanded activity outline and its technical reference.

```text
I am reviewing notes/slab_audit.md. Reinspect the third termination with its structure,
explain the cutoff used for CN=1, and compare its top and side views with termination 1.
Analyze first. Do not delete or overwrite any candidate.
```

## Skill Evolution preserves repeated project methods

In both login mode and trusted no-login mode, every terminal run with a user task enters one Skill Evolution semantic reflection unless the workspace mode is `off`. Login deployments use the authenticated username and isolated user root; no-login uses the fixed local `admin` actor and direct local workspace scope. The model receives a compact deterministic run index, then reopens only the needed semantic events through host-bound read-only queries; complete trajectories are not copied into prompts. It distinguishes no durable change, failure to follow adequate guidance, `defer` while later real use is needed, `ignore` when a finding should not become guidance, and evidence that durable behavior should change. CatMaster does not use regular expressions, embeddings, fixed recurrence counts, or replay scores to decide this. The final resolved owner defines one revision chain. Ordinary new evidence starts from the current effective version, and rejected draft files are not inherited implicitly. Processing history pages through every reflection's change, rationale, anchor and resolved targets, evidence handles, and terminal result. Failed results remain unchanged and are never automatically retried; their cards expose an authorized retry for the selected job or failed item.

Reviewer approval is the semantic decision for an exact immutable revision. In workspace `auto` mode, an approved, valid revision is selected for the next run when its target is enabled and follows automatic updates; there is no routine human review, canary, or promotion queue. `needs_revision` starts bounded automatic proposer-reviewer repair, and every attempted revision remains inspectable. Only a real authorization, safety choice, or unresolved user preference pauses activation and asks a concrete question in the originating chat. Automatic activation and `no_change` also return concise, non-blocking chat updates.

For exact post-hoc control, open **Skill Evolution** and click **Manage skills**. The dialog shows the complete paginated effective-skill catalog, persistent workspace mode (`off`, `observe`, or `auto`), enabled state, selected exact version, `auto_head`, latest draft, immutable versions, evidence references, and ordered automatic/chat/GUI history. Selecting a version pins it; returning to Follow auto resumes automatic selection. Disabling a skill removes it from next-run staging without deleting its files or history. These controls affect subsequent runs, not the current run.

## Resuming interrupted or older work

Return to the same workspace and thread, then ask the agent to reread the authoritative artifacts. State what must be retained, where the previous work stopped, whether recomputation is forbidden, and the new stopping point.

```text
Continue the CO adsorption screen. Reread notes/co_sites.md, structures/co_candidates/,
and calculations/mlff_screen/output/. Verify existing candidates, failures, and ranking evidence.

Do not regenerate or resubmit completed structures. Decide which candidates deserve VASP,
explain why, and list the shared settings that still require my confirmation. Stop before VASP stage approval.
```

After a remote error, inspect the receipt and old job state before any retry. Chapter 8 provides recovery prompts and Chapter 11 contains diagnostic commands.

When the latest turn fails inside a resumable LangGraph step, its error card
shows **Continue from checkpoint**. This resumes the same durable thread with no
new user message. Completed checkpointed steps remain available and only the
failed graph tail runs again. It does not continue from the middle of a partial
LLM token stream. If the error card instead shows **Review and try again**, the
failure has no safe graph continuation and the composer remains the recovery
path. An older failure card becomes inactive as soon as the thread advances.

## Important current limitations

The WebUI does not yet delete or branch threads, and it cannot replay an arbitrary historical checkpoint or select a historical run. It can natively continue only the latest resumable failed graph tail. Research Graph can manage scientific branches across threads, but it is not a thread history or rollback control. Files overwrites same-name uploads and has no recycle bin. Approval interruptions must resume through their message cards. Stop does not cancel remote jobs. Skill Evolution changes affect the next run rather than hot-reloading the current run.

Use versioned file names or external backup, divide independent objectives or incompatible project scopes into separate threads, and manage remote jobs through receipts. These practices cover the current UI gaps without pretending the agent can provide controls that do not exist.

Graph node details show judgment scope/reasons and scientific revision links. Research decisions and open premises shows consequential stopping decisions, independent reconsideration, the recommended validation and actual Results, and resumption conditions. The running Plan belongs to the conversation; long-term scientific knowledge remains in the Graph.
