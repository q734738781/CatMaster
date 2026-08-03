# 4. Working with agents in the WebUI

[Previous](03-llm-configuration.en.md) | [Contents](README.en.md) | [Next](05-agents-and-modules.en.md)

The WebUI keeps conversation, project files, agent activity, human approval, and run observability on one page. You do not need to learn every backend state. Two habits matter most: ask the agent to save important results in the workspace, and inspect what it actually did when it changes a scientific structure or submits remote computation.

## Dividing work into workspaces and threads

The workspace selector is at the top of the left rail. A workspace is a long-lived project containing files, conversations, run records, and project-specific experience. Separate catalyst systems, manuscripts, or ML datasets usually belong in separate workspaces so that file search and project memory do not mix unrelated research.

A workspace can contain many threads. Use a thread for one continuing line of work, such as "CeO2 surface models," "ORR free energies," or "second manuscript revision." Returning to the same thread preserves checkpoint continuity. A new thread begins a separate conversational context, so include the necessary paths and assumptions again.

The left rail also contains a compact file tree. Use it to open a structure, report, or log quickly. The full upload, preview, and download controls are in the Files view.

## Choose the entry that matches the main deliverable

The composer lets you select Research, Experiment, Writing, Peer Review, or Literature Review. A long-lived thread may switch entries between turns while retaining its conversational checkpoint. The entry cannot change during an active turn because each entry builds a different agent with different tools and workers. Each agent message shows the entry actually used for that turn, so the thread's current selection does not relabel older work.

Choose Experiment for a bounded structure, calculation, or trajectory task. Use Literature Review for evidence discovery and reading, Writing when source material already exists, and Peer Review for an independent assessment of one PDF. Use Research when the objective genuinely crosses several stages and requires decisions about their order.

The wrong entry may still answer, but it often adds friction. Research is unnecessarily broad for a 3x3x1 supercell. Writing does not have the calculation workers needed to reconsider an adsorption energy. Chapters 3, 5, and 7 give fuller examples.

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

Images can be sent as visual content when the selected model profile supports them. PDF, DOCX, XLSX, and PPTX are parsed through bounded document readers. The agent can render selected PDF pages when visual inspection is required. Audio, video, legacy Office formats, and oversized media may be stored without being sent to the model. The `multimodal.prepared` event in Monitor records whether an attachment was sent, how it was represented, and any degradation warning.

Attachments are convenient for the current message. Files that will be reused should live at stable project paths such as `literature/corpus/`, `structures/`, or `data/`.

## What Chat reveals about agent work

Chat contains more than final prose. Progress cards show the current reasoning or stage note. Activity groups tool calls, subagents, and remote receipts. Individual tool cards can be expanded to inspect input, status, and returned summaries. Files created by the agent appear as artifact cards and open in the right-side inspector.

These parts answer different questions. Progress shows how the agent frames the task. Subagent activity shows which role owns the work. A tool card records the executed action. An artifact is a reusable result. A remote receipt identifies a submitted job and its recoverable state.

You do not need to inspect every file read. Expand activity when an important structure or manuscript changes, when the number of candidates differs from expectations, when a tool reports warning or error, when a remote submission has meaningful cost, or when the final answer disagrees with the files on disk.

## Research Graph connects work across threads

Research Graph is workspace scientific state, not a child of the current thread. Its catalog shows the question, node counts, runnable frontier, manual or automatic mode, last update, and whether the current thread is attached. The interface may preselect the only active graph. When several exist, you must choose one. Attach, Detach, and Switch change the thread's focus without copying or deleting scientific state.

New graph requires only a research question. Title, completion criterion, orchestration, and seed hypotheses are under optional setup; when the criterion is empty, CatMaster uses the visible default of a defensible answer supported by recorded Results and traceable sources. A seed Hypothesis requires only its claim and may attach its motivating paper, note, or other source immediately. With no seed, click “Ask Research to propose starting routes”; you do not need to initialize the graph in Chat first.

The graph has three scientific node types:

- A Hypothesis shows its concise claim, relative importance, and a relationship summary derived from all related Results; this is not an evidence grade.
- An Experiment proposal shows its objective, plan, decision rule, execution lane, coarse compute cost, and preparation or execution state. A temporary planning candidate may also show its current innovation and conservative evaluation; those values are not part of the durable Experiment.
- A Result shows a concise observation or outcome. Literature findings, collaborator results, and historical observations can be recorded without a graph Experiment. Labeled relationships connect a Result to the Hypotheses that it supports, opposes, or does not distinguish.

The canvas supports pan, zoom, fit, a minimap, keyboard access, focus neighborhood, and density limits of 5, 25, or 100 nodes. Node cards keep the complete title and accessible name. Selecting a node opens the full scientific fields and sources in the inspector, so truncation never hides the only copy of the content.

“Add scientific input” is designed for short entries. A Hypothesis needs one claim, a draft Experiment needs one objective, and an Observation or Result needs one summary; titles, rationale, predictions, links, rankings, interpretation, and sources are optional details. A draft Experiment may remain deliberately incomplete, but it cannot become Ready or run until it has both a plan and a decision rule. A Hypothesis can develop an experiment proposal, be edited, or open its related evidence. An Experiment can be prepared, run, replicated, linked to a dependency, marked blocked, or given a Result. A Result can lead to a user-authored Hypothesis or follow-up Experiment. Its effect on any Hypothesis can be added, replaced, or cleared later without recreating the Result.

Creating a graph through the Research Specialist attaches it to the current
thread automatically. At the start of each turn, the host fixes the actual
entry, graph, focus, and matching unfinished launch. Changing the UI focus while
the turn is running cannot retarget its tools, and interrupt resume keeps the
interrupted turn's binding. A top-level Experiment or Literature Review turn
receives read-only graph query and Result writeback only when that turn is graph
bound. An Experiment focus creates a linked Result; without an Experiment focus,
the agent may record a standalone Result. Only a real Experiment focus can be
marked with a concrete blocker. Unbound turns, ordinary questions, input
preparation, waits, and repeated explanations do not write the graph. Internal
Research delegates return evidence to the Research specialist instead. The host
adds the actual thread and run sources, while Writing remains read-only and
follows those sources to the original evidence.

Running an Experiment atomically claims one launch, then creates an ordinary child thread bound to the graph and focus node. Repeated clicks on the same active launch are deduplicated. A completed Experiment can start an explicit replicate. If an ordinary turn ends without a Result or blocker, the launch remains unfinished and the UI shows “Waiting to continue”; the user can continue in the same child thread. An error or stop produces only a retryable operationally incomplete state, not a scientific conclusion. A real writeback completes only the exact launch bound to that turn, so later Writing, questions, or attached analyses cannot rewrite an old launch's run identity. When remote submission status is uncertain, recovery checks the existing thread, run, and receipt before any new submission.

Research planning first gives `hypothesis_proposer` a partial focus snippet and read-only SQL access to the complete bound graph. The proposer may also search the web and local corpus and read or acquire selected sources. For a Result-focused turn, it compares the Result with existing predictions and Results before proposing a distinct new Hypothesis. It may conclude that no new Hypothesis or route is needed. Staging publishes sourced temporary Hypothesis and Experiment branches but does not materialize or launch them. Branch count follows the scientifically distinct alternatives supported by the current evidence. A temporary Experiment may remain a draft with only an objective and becomes runnable only after it has a usable plan and decision rule.

The evaluator compares every candidate Experiment in the current revision, including the complete runnable frontier, and assigns an innovation score and a conservative score. These values and their two recommendations exist only in the current preview. Candidate branches appear as translucent nodes, and the inspector shows both scores and recommendation choices. The planning run does not clutter the ordinary thread list. In Manual mode, selecting a temporary node atomically adds only its required route. Unselected branches are replaced by the next planning pass rather than entering the durable scientific graph.

Automatic orchestration plans after every graph change and then runs at most one real Experiment. This is an execution-concurrency limit, not a limit on parallel Hypotheses. It uses the current revision's conservative recommendation. A recommended temporary Experiment is materialized in a separate transition; an existing ready Experiment can launch directly. No-change, a missing or invalid evaluation, an empty recommendation, or a concurrent graph edit leaves the graph waiting instead of falling back to the first runnable Experiment. Once recorded Results satisfy the completion criterion, the graph becomes Completed and automatic advancement stops. Switching back to Manual prevents later automatic launches but does not cancel the current thread or remote job.

Completed is a stop marker, not a lock: adding or changing scientific content reopens the graph, while attaching another source alone does not. Archived graphs are read-only until explicitly restored. This keeps historical graphs inspectable without silently accepting edits.

Graph nodes contain short scientific statements only. Papers, detailed notes, structures, logs, reports, artifacts, and receipts remain in their existing stores and connect through Sources. A moved or deleted source appears as "Source unavailable"; its reference is not silently removed. Graph actions do not grant protected execution. Computation still follows specialist ownership, managed execution, and the ordinary approval cards.

After a Writing thread explicitly attaches the graph, each turn receives the same partial focus context and the Writing coordinator can query the complete bound graph read-only. It locates section-relevant Results, opposing or inconclusive judgments, and their Sources before opening the original note, artifact, run, thread message, DOI, or URL. A Result summary is navigation, not a substitute for the source, and Writing cannot edit the graph. Unattached Writing threads behave as before, and Writing never guesses among several graphs by title.

Updates from other threads arrive through the durable graph event stream. If the graph changes before you submit an edit, the server rejects the overwrite and shows a readable conflict message. Refresh, review the new content, and submit again.

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

The idle submit button is Send. While an agent is running, a text-only message becomes Steer. Steering does not forcibly interrupt an active tool. After the current tool completes and its checkpoint is written, the queued message continues from the same thread state. If the model reaches a final response without another tool call, the queued message starts after that turn finishes. Use it for constraints discovered during the run, such as preserving an original file, analyzing only the first 20 ps, or keeping both terminations.

If the new request changes the objective entirely, waiting for a safe stop and opening another thread is often clearer. New attachments are disabled during a run, so wait or stop before adding another file.

Stop asks the local agent turn to end at a stream boundary, with repeated requests escalating toward emergency cancellation. It does not cancel jobs already submitted to Slurm or a remote shell. Those jobs require receipt-aware scheduler handling.

## Files holds the deliverables

Files provides Browse, Preview, and Uploads. It can preview text, Markdown, JSON, images, PDF, CSV/TSV, common crystal and molecular formats, trajectories, volume grids, and selected OUTCAR vibration content.

Crystal, slab, defect, adsorbate, and ordinary molecule previews use MatterViz. Choose **Open Structure Workbench** for a full-viewport editor with base-atom selection, coordinate and cell editing, measurements, constraints, undo/redo, supercell and symmetry previews, slab/defect/adsorption candidate galleries, and explicit Save As. Display copies are view-only; use Make supercell before creating a real single defect. Large structures keep the complete source model for selection and saving while the canvas switches to a bounded representation.

Molecule files open a lazy Ketcher 2D editor and a MatterViz 3D conformer view. SDF or MOL is the connection-table authority. Saving a molecule as XYZ loses bonds, aromaticity, bond order, charge, and stereo; saving as SMILES loses the current 3D coordinates. The Workbench blocks that save until the warning is acknowledged. Periodic constraints round-trip through POSCAR/VASP and ASE `.traj`; formats that cannot express them receive the same explicit warning.

Trajectories are read-only and report their real frame count. Scrub or play them, inspect scalar properties, and extract one frame before editing. CUBE, CHGCAR, LOCPOT, ELFCAR, and XSF open as volume artifacts with structure overlay, positive/negative isosurfaces, and slices. JSmol remains the compatibility path for OUTCAR vibration and formats that the primary renderer cannot open; it is not a second editable state. VESTA renders can still appear as image artifacts.

After an agent reports completion, check that the main deliverables exist at the promised paths. For structures, inspect candidates and the audit. For calculations, inspect the stage, status, stdout/stderr, and analysis. For literature, inspect the candidate and evidence tables plus the reference library. For writing, retain editable source files rather than only a compiled PDF.

Uploading a file with the same name overwrites it. Directory deletion is recursive and permanent. Keep important originals backed up outside the workspace. The ordinary file tree shows user deliverables and working files; internal metadata, tool-result offloads, and transient extracts remain available to diagnostics but are not presented as deliverables.

## Monitor helps determine whether the process is healthy

Monitor summarizes models, agents, tools, tasks, tokens, cost, and machine time. Token totals are updated after each completed LLM call, including input, output, cache, and reasoning tokens when the provider supplies them; a call still in progress has no final usage yet. Overview is useful for status and scale. Live shows the active stage, tools, todo list, subagents, and recent logs. Events can be filtered by thread, run, agent, tool, category, and channel. Raw and Details are for deeper diagnosis.

If an agent appears stuck, check whether a remote tool or subagent is still active. If a result is incomplete, look for tool errors, document warnings, or multimodal degradation. If cost is unexpected, inspect model calls, tokens, and machine time. Monitor is a diagnostic surface, not a report that must be copied into every deliverable.

The current UI has no historical run selector. Overview may summarize the current or most recent run for a workspace and lane. For precise remote tracking, correlate thread ID, run ID, artifact, and receipt.

## The inspector supports side-by-side review

Clicking an artifact or file opens it in tabs on the right while Chat remains visible. This is useful for comparing a structure, report, table, or log while asking a follow-up. The Todo tab is a read-only canonical projection of the current turn's latest plans. During a run it follows `write_todos` updates; when the assistant message completes, the server push replaces it with the terminal projection and removes unfinished child-agent scratch plans. Message pagination and refresh preserve that state. It is not a project-management form the user must maintain.

Completed specialist-task cards use the returned scientific Markdown directly: the card shows its content title, central conclusion, and a short section outline. Open details shows the expanded activity outline and its technical reference.

```text
I am reviewing notes/slab_audit.md. Reinspect the third termination with its structure,
explain the cutoff used for CN=1, and compare its top and side views with termination 1.
Analyze first. Do not delete or overwrite any candidate.
```

## Skill Evolution preserves repeated project methods

In login mode, every terminal run with a user task enters one Skill Evolution semantic reflection. The model reads the complete recorded trajectory and result, then distinguishes no durable change, failure to follow an adequate existing skill, and evidence that long-term behavior should change. CatMaster does not use regular expressions, embeddings, or a fixed recurrence count to make that decision. One explicit durable correction may be sufficient; repeated wording is not sufficient by itself. Product/schema defects and detailed scientific facts are kept out of skills, and an existing owner skill is preferred over a duplicate.

Candidate cards lead with behavioral changes, evidence episodes and sources, applicability boundaries, static validation, reviewer counterexamples, concerns, and human checks rather than raw JSON. Use status filters and Load more for the newest-first candidate and observation lists. The exact reviewed diff is revision-bound and appears under Technical details. AI review is advisory. The lifecycle uses only `pending`, `review`, `revision`, `canary`, `stable`, `rejected`, and `inactive`. A user may request a new immutable revision or reject it; a skill must then pass an explicitly scoped canary with successful actual use before Promote stable becomes available. Starting a canary only changes the exact-version pointer for the selected thread or run; it does not start another conversation or model call. Changed targets return the candidate to revision instead of being overwritten. Canary failures stop only that pointer, and stable revisions can be quarantined, retired, or rolled back.

## Resuming interrupted or older work

Return to the same workspace and thread, then ask the agent to reread the authoritative artifacts. State what must be retained, where the previous work stopped, whether recomputation is forbidden, and the new stopping point.

```text
Continue the CO adsorption screen. Reread notes/co_sites.md, structures/co_candidates/,
and calculations/mlff_screen/output/. Verify existing candidates, failures, and ranking evidence.

Do not regenerate or resubmit completed structures. Decide which candidates deserve VASP,
explain why, and list the shared settings that still require my confirmation. Stop before VASP stage approval.
```

After a remote error, inspect the receipt and old job state before any retry. Chapter 8 provides recovery prompts and Chapter 11 contains diagnostic commands.

## Important current limitations

The WebUI does not yet rename, delete, branch, or retry threads, and it has no historical run selector. Research Graph can manage scientific branches across threads, but it is not a thread history or rollback control. Files overwrites same-name uploads and has no recycle bin. Approval interruptions must resume through their message cards. Stop does not cancel remote jobs. Skill Evolution appears only in login mode and affects the next run.

Use versioned file names or external backup, divide independent objectives or incompatible project scopes into separate threads, and manage remote jobs through receipts. These practices cover the current UI gaps without pretending the agent can provide controls that do not exist.
