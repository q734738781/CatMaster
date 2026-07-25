# 4. Working with agents in the WebUI

[Previous](03-llm-configuration.en.md) | [Contents](README.en.md) | [Next](05-agents-and-modules.en.md)

The WebUI keeps conversation, project files, agent activity, human approval, and run observability on one page. You do not need to learn every backend state. Two habits matter most: ask the agent to save important results in the workspace, and inspect what it actually did when it changes a scientific structure or submits remote computation.

## Dividing work into workspaces and threads

The workspace selector is at the top of the left rail. A workspace is a long-lived project containing files, conversations, run records, and project-specific experience. Separate catalyst systems, manuscripts, or ML datasets usually belong in separate workspaces so that file search and project memory do not mix unrelated research.

A workspace can contain many threads. Use a thread for one continuing line of work, such as "CeO2 surface models," "ORR free energies," or "second manuscript revision." Returning to the same thread preserves checkpoint continuity. A new thread begins a separate conversational context, so include the necessary paths and assumptions again.

The left rail also contains a compact file tree. Use it to open a structure, report, or log quickly. The full upload, preview, and download controls are in the Files view.

## Choose the entry that matches the main deliverable

The composer lets you select Research, Experiment, Writing, Peer Review, or Literature Review before a run begins. The entry cannot be changed while the agent is running because each one builds a different agent with different tools and workers.

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

## Research Map shows a live verification network

Research Map is the operational view of a branch-aware campaign in the current thread. It appears as its own top-level tab. Research creates a campaign only when competing explanations, shared evidence, dependent checks, or meaningful cost and wait tradeoffs justify the extra structure. An empty view is normal for a linear task.

The graph has three node types:

- Hypotheses show each claim, rationale, predictions, derivation, and current `open`, `supported`, `rejected`, or `contested` status.
- Verification actions show the executor, scientific question, bounded task, decision rule, dependencies, information value, coarse cost, and controller status.
- Evidence judgments show the result summary, scientific source, and one `supports`, `opposes`, or `inconclusive` effect for every target hypothesis.

Click a node to inspect its scientific content. The active packet panel shows the current executor, task, target hypotheses, and decision rule. Execution attempts, resource accounting, and remote receipts remain in the ordinary Activity and artifact views.

Select an eligible action and use "Start Research thread." The server checks the displayed revision, reserves that action, creates a new Research thread, and submits one ordinary Research turn. It does not steer or reuse the source chat thread. The child keeps the complete Research behavior, specialist delegation, streaming, and Auto/Review permission mode. Its DeepAgent checkpoint and lightweight Research Kernel are isolated under the child thread id; only controller calls use the source campaign id. The reservation is rejected if the Map is stale or the source Research thread is still running, stopping, or waiting for review, so two Research turns cannot compete for the same campaign.

"Start automatic Research" enables a persistent asynchronous worker for the source campaign. The worker selects ranked non-human actions one at a time and creates the same kind of ordinary Research child thread. It waits while a child is running or interrupted, then reads the updated campaign before launching another. "Stop after current check" prevents the next launch without cancelling the current child. A human-owned action is never answered by the worker; start its thread manually and provide the requested evidence there. The original single-thread Research submit path is unchanged and can still run without Research Map.

Map selection and automatic scheduling are not protected-tool approval. A selected experiment still follows the Experiment worker's managed execution path and uses the ordinary approval card when required. Cost affects ranking only and does not replace that approval system.

The same evidence judgment can affect several hypotheses, and competing hypotheses can share one verification action. When evidence reveals a missing explanation, Research asks the hypothesis proposer for a separate revision before adding a derived hypothesis or follow-up action. This is why the view is a network rather than a literal tree. `Supported`, `rejected`, and `contested` summarize the recorded verdicts; they are not posterior probabilities. Check the cited paper, run, or artifact before treating a branch as a scientific conclusion.

## Auto and Review support different working styles

Auto lets an agent proceed within its current permissions and works well for reading, analysis, and trusted project workflows. Review pauses before `write_file`, `edit_file`, `remote_submission`, and `remote_submission_batch`, then presents an approval card in the message.

Review is a good default for a new project, important source files, or real remote computation. The card supports four actions:

- Approve executes the proposed action.
- Reject declines it and can include a reason.
- Respond gives the agent feedback so it can rework the action.
- Edit action changes the action JSON and is intended for users who understand the tool schema.

Review is not a global approval gate. Reading, search, and some analyses still run automatically. Its purpose is to put protected file edits and remote compute at a clear human checkpoint. Resume an interruption through the card rather than sending an unrelated normal message.

In that statement, file mutation means the currently protected `write_file` and `edit_file` calls. Domain tools such as `supercell` and `build_slab` generate their declared outputs inside the tool call and do not receive a card solely because a file is created. Give those operations an explicit destination, inspect input and output paths on the tool card, and review the artifact in Files. Review protects named calls; it is not a transaction lock around every workspace change.

## Steer a running task without scripting every move

The idle submit button is Send. While an agent is running, a text-only message becomes Steer. Steering does not forcibly interrupt an active tool. It becomes the next instruction at a safe boundary. Use it for constraints discovered during the run, such as preserving an original file, analyzing only the first 20 ps, or keeping both terminations.

If the new request changes the objective entirely, waiting for a safe stop and opening another thread is often clearer. New attachments are disabled during a run, so wait or stop before adding another file.

Stop asks the local agent turn to end at a stream boundary, with repeated requests escalating toward emergency cancellation. It does not cancel jobs already submitted to Slurm or a remote shell. Those jobs require receipt-aware scheduler handling.

## Files holds the deliverables

Files provides Browse, Preview, and Uploads. It can preview text, Markdown, JSON, images, PDF, CSV/TSV, common crystal and molecular formats, trajectories, and selected OUTCAR vibration content. JSmol handles many structure and trajectory previews, while VESTA renders can appear as image artifacts.

After an agent reports completion, check that the main deliverables exist at the promised paths. For structures, inspect candidates and the audit. For calculations, inspect the stage, status, stdout/stderr, and analysis. For literature, inspect the candidate and evidence tables plus the reference library. For writing, retain editable source files rather than only a compiled PDF.

Uploading a file with the same name overwrites it. Directory deletion is recursive and permanent. Keep important originals backed up outside the workspace. The file tree also exposes `metadata/`, which is system state rather than an ordinary project directory. Do not move, rename, or delete it casually.

## Monitor helps determine whether the process is healthy

Monitor summarizes models, agents, tools, tasks, tokens, cost, and machine time. Token totals are updated after each completed LLM call, including input, output, cache, and reasoning tokens when the provider supplies them; a call still in progress has no final usage yet. Overview is useful for status and scale. Live shows the active stage, tools, todo list, subagents, and recent logs. Events can be filtered by thread, run, agent, tool, category, and channel. Raw and Details are for deeper diagnosis.

If an agent appears stuck, check whether a remote tool or subagent is still active. If a result is incomplete, look for tool errors, document warnings, or multimodal degradation. If cost is unexpected, inspect model calls, tokens, and machine time. Monitor is a diagnostic surface, not a report that must be copied into every deliverable.

The current UI has no historical run selector. Overview may summarize the current or most recent run for a workspace and lane. For precise remote tracking, correlate thread ID, run ID, artifact, and receipt.

## The inspector supports side-by-side review

Clicking an artifact or file opens it in tabs on the right while Chat remains visible. This is useful for comparing a structure, report, table, or log while asking a follow-up. The Todo tab is a read-only projection of the current turn's plan, not a project-management form the user must maintain.

```text
I am reviewing notes/slab_audit.md. Reinspect the third termination with its structure,
explain the cutoff used for CN=1, and compare its top and side views with termination 1.
Analyze first. Do not delete or overwrite any candidate.
```

## Skill Evolution preserves repeated project methods

In login mode, a completed run can produce a workspace-scoped improvement candidate. Repeated directory conventions, units, structure audits, or report formats can become a memory or skill candidate. The default `observe` mode never activates one automatically. Users review candidates in Skill Evolution before choosing Promote or Reject.

Good candidates are stable, project-specific, and supported by repeated evidence. A temporary file name, a one-off network failure, or an unverified scientific guess should remain in the task record instead. Promoted content takes effect on the next run. A changed target produces a conflict instead of being overwritten, and a harmful promotion can be rolled back while the target still matches.

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

The WebUI does not yet rename, delete, branch, or retry threads, and it has no historical run selector. A Research Map can branch scientific hypotheses inside one thread, but it is not a thread-history branch or rollback control. Files overwrites same-name uploads and has no recycle bin. Approval interruptions must resume through their message cards. Stop does not cancel remote jobs. Skill Evolution appears only in login mode and affects the next run.

Use versioned file names or external backup, divide independent objectives or incompatible project scopes into separate threads, and manage remote jobs through receipts. These practices cover the current UI gaps without pretending the agent can provide controls that do not exist.
