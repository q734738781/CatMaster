# 11. Prompt library and troubleshooting

[Previous](10-deployment-operations.en.md) | [Contents](README.en.md)

The first half of this chapter contains prompts that can be adapted directly. The second half diagnoses installation, model, file, literature, and remote-task problems by symptom. These prompts are not rigid forms. Keep the scientific boundary and delivery requirements that matter, remove irrelevant clauses, and let the agent choose its skills and tools.

## Adapting a reference prompt

The agent needs the real research question more than a tool sequence. Paths, constraints that must survive, allowed computation, and intended artifacts reduce ambiguity. When the method is unsettled, ask for options and a pause at consequential decisions. When the project already fixes the method, state it instead of asking the agent to rediscover it.

Autonomy should not turn one request into an endless project. Authorize the technical judgment needed for the objective and state whether this turn should stop at candidate structures, input review, remote approval, or result analysis.

## Research prompt

```text
Use Research to investigate <research question>. Read <existing directories or files> first and separate
established facts, working hypotheses, and genuine evidence gaps. Decide whether Literature Review,
Experiment, Writing, or Peer Review is needed, but advance one bounded stage at a time and inspect its artifacts.

The deliverable for this turn is <evidence map, plan, or stage synthesis>. Do not run a calculation just because
literature evidence is missing. If new computation is warranted, explain which hypotheses it distinguishes,
what inputs it requires, and its cost, then wait for approval.
```

Use Research for an open question such as reversible structural change under redox cycling. Use Experiment directly when the task is simply to build one slab.

## Slab and adsorption prompts

```text
Use Experiment to build the <Miller index> surface from <bulk structure path> for <downstream purpose>.
Ask Materials to choose suitable slab, termination, and visual-inspection skills.

Require <thickness or layer count> and <vacuum>. Explain slab symmetry, lateral expansion, and fixed layers.
Preserve Selective Dynamics. If constraints should change, present options before editing.
Inspect top and bottom surfaces, stoichiometry, coordination, short contacts, and isolated atoms.
Save every reasonable termination, views, and an audit. Do not submit computation. Pause on polarity,
stoichiometry, or termination choices.
```

For adsorption, add:

```text
Build adsorption candidates for <adsorbate> on the accepted slab. Define adsorbate conformation and anchor,
then enumerate and deduplicate chemically meaningful sites and orientations. Record site provenance,
starting distance, coverage, and inherited constraints. Reject collisions and obvious periodic interactions.

If the set is large, you may propose MLFF single-point or relaxation screening after querying the current backend.
Wait for approval before execution. Deliver a candidate ledger, views, rejection rationale, and recommended DFT set.
```

## VASP or other remote-compute prompt

```text
Use Experiment to review <stage path> for <calculation type>.
Ask the owning worker to inspect the structure, constraints, input files, scientific settings, and expected outputs,
then query the current remote task, resource, and full schema. Do not guess overrides from an old prompt.

Stop with a precise blocker if preparation or deployment is incomplete. If all checks pass, show task, work_dir,
task count, resource, critical parameters, and cleanup behavior in Review, then wait for approval.
After transfer, inspect status, stdout/stderr, program convergence, and scientific output rather than relying on
scheduler completion alone.
```

For a batch, name the first-level stage directory and common settings. For MLFF, require backend, model, device, and dtype and label the result as model prediction.

## Dynamics and restart prompt

```text
Use Experiment to continue <existing MD directory>. Ask Dynamics to audit the last valid step, structure,
velocities, integrator or thermostat state, random state, time axis, and restart files.
Do not overwrite the original or call a last-frame run continuous when restart evidence is incomplete.

Build a separate continuation stage and document how segments join and which settings remain identical.
Query the current remote task and wait for approval. After transfer, check temperature, energy, volume,
trajectory continuity, short contacts, and restart usability before MSD, RDF, or diffusion analysis.
```

## Dataset and MACE prompt

```text
Use Experiment and ML to build a MACE dataset from <VASP result tree>.
Separate converged, unconverged, and incomplete runs. Check units, reference energies, element coverage,
duplicates, outliers, and mixed calculation settings. Fix a seed and retain a split manifest.

Deliver extxyz, split files, and a data audit. Prepare training only after the audit passes.
Do not submit mace_train until I approve data scope, foundation model, replay or E0 settings, and GPU cost.
```

## Molecular, xTB, and ORCA prompt

```text
Use Experiment on <SMILES or structure path> with total charge <charge>, multiplicity <multiplicity>,
and <solvent and target property>. Ask ORCA/xTB to choose a conformer, CREST/xTB screening,
deduplication, and ORCA strategy while retaining structures, relative energies, and exclusion reasons.

Deliver a reviewable ensemble and ORCA stage plan first. Explain which frequency, thermochemistry, TS/IRC,
TDDFT, or NMR steps are relevant instead of running all of them by default. Wait for approval before remote work.
```

## Literature Review prompt

```text
Use Literature Review on <topic> within <years, systems, document types, and exclusions>.
Design and save the search strategy, collect a sufficiently broad candidate set, and deduplicate DOI, title,
preprint, and journal versions.

Separate discovered records from abstract, full-text, and SI evidence. Read core papers around <specific question>
and extract system, conditions, method, result, limitation, and disagreement into a claim-evidence table.
Save candidates, full-text availability, unavailable records, and the final bibliography. Do not infer precise
parameters from titles or abstracts.
```

For one-paper close reading, require paper order, figure placement, page or source anchors, and explicitly say that a summary-only result is not acceptable.

## Writing and review prompts

```text
Use Writing to draft or revise <section or document> from <evidence files, data, figures, and bibliography>
for <reader or venue>. Inspect what argument the evidence can support, then select relevant writing skills.

Every value, unit, figure, and citation must trace to the supplied material. Preserve <terms or claim boundary>
and do not add <new results, unverified citations, or causal claims>. Write connected prose to <output path>
and list evidence gaps and author decisions separately.
```

```text
Use Peer Review on <canonical PDF> for <venue and article type>.
Ask reviewers to assess novelty, method, evidence, figures, reporting, and reproducibility independently.
Major comments must cite pages or figures. Preserve complete reports, then create an editor synthesis separating
consensus, disagreement, required revision, and optional improvement. Do not edit the manuscript or write an
author response in this turn.
```

## Continuation and recovery prompts

```text
Continue the original thread. Reread <key reports, directories, receipts, and logs> and treat current files
as authoritative rather than accepting old chat completion claims. Report confirmed work, incomplete items,
active remote jobs, and decisions still waiting for me.

Preserve successful results and forbid duplicate generation or computation. Resume from <specific stage>
and stop at <new boundary>.
```

After an ambiguous remote error:

```text
The previous remote call failed or disconnected. Do not resubmit or clean anything.
Read the receipt and confirm remote_context_id, submission_hash, task, original stage, and known job state.
Use scheduler or DPDispatcher evidence to determine whether the job is running, complete but not downloaded,
or terminated. Retrieve finished output and failure logs first. Discuss resubmission only after proving the old
job can no longer write or consume resources.
```

---

## The WebUI does not open

Run in the foreground and read the first real traceback:

```bash
CATMASTER_PROJECT_SPACE_ROOT="$HOME/catmaster_projects" \
./start_webui.sh --foreground --host 127.0.0.1 --port 7991
```

In another terminal:

```bash
./start_webui.sh --status
tail -n 100 .runtime/webui.log
ss -ltnp | grep 7991
```

`conda is not available` means the shell lacks conda initialization or `CATMASTER_CONDA_ENV` is wrong. `Address already in use` means another process owns the port. Repair project-root ownership instead of running as root. A JSmol download error affects OUTCAR vibration and fallback previews, not the primary MatterViz Workbench.

Always use an explicit host and port while diagnosing because the launcher and direct Python CLI have different embedded defaults.

## The profile parses but conversation fails

First parse offline:

```bash
python -c 'from catmaster.llm.config import LLMProfile; p=LLMProfile.from_env_or_file(); print(sorted(p.models)); print(p.agents)'
```

If parsing fails, check YAML indentation, role labels, and provider fields. If it succeeds, check in order:

1. The key is exported into the process that starts the WebUI.
2. Model ID and base URL belong to the selected provider.
3. Reasoning and provider options use that provider's schema.
4. The model supports tool calling and the current tool schema.
5. The first provider 4xx, 5xx, or timeout message.

When the model writes prose but never calls a tool, inspect Chat and Monitor before making the prompt more forceful. Verify the Entry, worker delegation, model tool support, and whether the schema reached the provider.

If a long task stops early, inspect the real tool error, `max_tool_calls`, recursion, context, and scope before raising every limit.

## The attachment was saved but not read

Images require visual capability in the profile. PDF, DOCX, XLSX, and PPTX use bounded parsing. Legacy `.doc`, `.xls`, `.ppt`, and unknown formats are generally stored without parsing. Inspect `multimodal.prepared` for `sent_to_model`, `sent_as`, and warnings.

Common limits are 64 MiB per Composer file, 512 MiB for backend storage, 32 MiB for current-turn media inline, 50 MiB and 60,000 parsed characters for PDF or Office input, and 20 PDF pages or slides by default. Ask the agent to read selected pages or split large documents.

## A structure, PDF, or table does not preview

For a blank primary structure view, open the browser console and network panel, confirm that `chunk-MatterVizHost.js` and its local assets return 200, and read the human error shown by the renderer boundary. Use **Source** to determine whether the file itself is malformed. Large structures deliberately show a bounded canvas notice while keeping the full atom count in Properties and the paged coordinate table.

If only molecule 2D editing fails, check the lazy `chunk-KetcherEditor.js`; the 3D conformer and source remain available. If volume loading fails, inspect the worker request and use Cancel before retrying another grid. For OUTCAR vibration or an explicit JSmol fallback, inspect the pinned JSmol cache and format. Text, directory, and tree previews have documented size or count limits, so a missing preview does not mean the file is absent. Open the original PDF when fonts or layout look suspicious.

Files overwrites same-name uploads. Restore from an external backup if necessary. After accidental `metadata/` deletion, stop writes and recover a consistent backup. Re-uploading `files/` cannot restore checkpoints.

## Literature Review finds a title but no full text

This is not necessarily a search failure or a reason to keep pursuing full text. Use an abstract or substantive search summary when it supports the required statement and explain its boundary. Only when a key decision depends on missing detail should the agent call `acquire_literature_source`. The tool tries direct legal OA sources, then one internal ScanSci/CloakBrowser pass on the DOI landing page, accepts only a structurally valid and identity-matched PDF, and otherwise saves one static page snapshot. After failure, continue with other sources rather than retrying publisher pages or mirrors.

Record abstract-only evidence as abstract. Resolve metadata conflicts from DOI, publisher, and paper records while preserving version differences. For missing local-corpus hits, inspect the ingest manifest, parse status, and document limits.

## A remote task is absent from the catalog

Administrators should check:

1. `machines.yaml`, `resources.yaml`, `tasks.yaml`, and `mlff_backends.yaml` exist.
2. Active filenames do not contain `template`.
3. YAML parses and duplicate active files do not override the same key.
4. Task and backend are enabled.
5. Resource audience includes the worker.
6. Machine SSH, remote root, queue, and `source_list` are valid.

Do not replace a missing managed task with local scientific-engine execution. Complete configuration and a smoke case first.

## A remote task cannot connect or start

Test from the same noninteractive environment:

```bash
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'hostname; python3 --version'
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'test -w <REMOTE_ROOT>'
ssh -o BatchMode=yes -i <SSH_KEY> <USER>@<HOST> 'command -v sbatch; command -v squeue; command -v scancel'
```

`command not found` or exit 127 usually points to machine `env_setup`, resource `source_list`, or a scientific binary path. Do not modify the scientific stage to hide an environment failure.

## A remote call disconnected and job state is unknown

Do not resubmit. Find the receipt and `submission_hash`. After confirming a DPDispatcher record, choose only the command needed:

```bash
dpdisp submission <submission_hash> --download-finished-task
dpdisp submission <submission_hash> --download-terminated-log
dpdisp submission <submission_hash> --reset-fail-count
dpdisp submission <submission_hash> --clean
```

These are not a fixed sequence. Download completed output and failure logs first. Use fail-count reset or cleanup only after understanding the consequence. An empty `submission_hash` normally means no recoverable DPDispatcher record exists.

Scheduler completion proves only that scheduling ended. Missing outputs require backward-file, permission, and program-log checks. A successful `status.json` with failed scientific convergence is a scientific failure. Preserve successful children in a partially failed batch.

## A remote job continues after Stop

This is expected. Stop cancels the local agent turn, not Slurm or a remote shell. Use receipt job data with the appropriate scheduler and retain cancellation evidence. Do not delete the local stage or receipt first.

## Login, workspace, or thread history appears missing

Verify `CATMASTER_PROJECT_SPACE_ROOT` and the username. Login deployments use `users/<username>/`, while no-login mode uses `admin/`. A legacy `.catmaster` root needs migration to `files/` and `metadata/`.

Restoring only `files/` does not restore threads. `metadata/`, DeepAgent SQLite, and the authentication database must belong to the same consistent backup.

## Skill Evolution did not create or release a candidate

This is usually a boundary, not a queue failure. A successful run creates no
observation unless it has a named verified outcome. Ordinary experience needs
recurrence across runs and threads plus a counterexample; an explicit durable
correction may proceed without recurrence but still needs static validation,
independent advice, and human review. Check the observation status and route in Skill Evolution
before asking the developer worker to process raw jobs. Tool/schema issues and
scientific notes are intentionally routed away from skills.

`pending` and `revision` cannot be promoted. Open the exact revision, read the
evidence, counterexamples, applicability boundaries, static validation, and reviewer
concerns, then choose Request revision or Reject. A skill in `review` must first start a canary on an explicit
thread or run. Promote stable appears only after that exact revision has a
successful actual-use record with no failure or false activation. Starting the
canary does not create a conversation or model call; it only binds the exact
version to the selected scope.

If a canary disappears, inspect its candidate card: a failed or falsely
activated exact revision automatically loses only its canary pointer; the
stable revision remains unchanged. A builtin or target hash change returns the
candidate to `revision` rather than silently shadowing the newer skill. For a developer,
copy the diagnostics reference from the card instead of pasting raw event JSON.

## Current UI limitations

- There is no historical run selector or UI for thread rename, delete, branch, or retry.
- Interrupted runs must resume through the message approval card.
- Monitor overview may describe the current or most recent run for the workspace and lane.
- Files overwrites same-name uploads and deletes recursively without a recycle bin.
- Stop does not cancel remote jobs.
- Skill Evolution appears only in login mode and affects the next run.

Use versioned files, external backup, clear thread boundaries, and receipt-based remote management instead of assuming the agent can supply controls that the UI does not implement.

## Deployment acceptance

A deployment ready for users should prove that accounts are isolated; all five entries are selectable; the base model can converse and call tools; attachments, artifacts, Files, Review, and Monitor work; one local structure task can delegate to a worker and write a file; literature claims match the actual search and browser surface; every enabled remote engine passes one minimal real case with status, logs, and receipt; a simulated interruption can be recovered without duplicate compute; and projects, auth, private config, and secrets can be restored from backup.

The purpose is not to install every optional tool. It is to make the capabilities shown to users match the deployment and to retain enough evidence to diagnose failure.
