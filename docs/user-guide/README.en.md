# CatMaster user manual

English | [中文](README.zh.md)

This manual is for people who use or operate CatMaster. It follows the current
DeepAgent specialist runtime and WebUI v2. It covers local installation, model
configuration, project spaces, the five task entrypoints, computational
modules, remote machines, files and runs, skill evolution, deployment, and
troubleshooting.

Last verified: 2026-07-20.

To get the system running, start with [Quick start](01-quickstart.en.md). To
connect a cluster, complete the local check first and then read [Remote machines
and execution](08-remote-execution.en.md).

## Manual map

| Chapter | What it covers |
|---|---|
| [1. Quick start](01-quickstart.en.md) | Create the environment, configure one model, and start the WebUI safely |
| [2. Concepts and project spaces](02-concepts.en.md) | Control plane, workspace, thread, run, artifact, and directory boundaries |
| [3. LLM and runtime configuration](03-llm-configuration.en.md) | Providers, role models, API keys, reasoning options, and output policy |
| [4. WebUI guide](04-webui.en.md) | Accounts, workspaces, threads, attachments, approvals, Monitor, and Files |
| [5. Agents and modules](05-agents-and-modules.en.md) | Choosing Research, Experiment, Writing, Peer Review, or Literature Review |
| [6. Computational workflows](06-computational-workflows.en.md) | Structures, DFT, MD, MLFF, molecular calculations, and result checks |
| [7. Literature, writing, and review](07-literature-writing-review.en.md) | Search, local corpora, manuscript work, polishing, and multi-model review |
| [8. Remote machines and execution](08-remote-execution.en.md) | SSH, Slurm, DPDispatcher, resource cards, stages, receipts, and recovery |
| [9. Tools, skills, and evolution](09-tools-skills-evolution.en.md) | Tool permissions, skills, project improvements, approval, and rollback |
| [10. Deployment, operations, and security](10-deployment-operations.en.md) | Server deployment, SSH tunnels, backup, upgrades, and external programs |
| [11. Reference and troubleshooting](11-reference-troubleshooting.en.md) | Environment variables, task matrices, limits, diagnosis, and acceptance checks |

## Three operating rules

1. Put user and agent work under the project space's `files/` directory.
   `metadata/` holds threads, checkpoints, run records, and internal indexes. Do
   not manage it as an ordinary file directory.
2. VASP, CP2K, LAMMPS, ORCA, xTB, CREST, and managed MLFF tasks use registered
   remote execution paths. CatMaster does not silently fall back to local
   execution when remote configuration is missing.
3. A finished process is not automatically a trustworthy result. Check the
   structure, parameters, convergence, logs, physical plausibility, and returned
   files.

## Conventions

- Commands are run from the repository root unless stated otherwise.
- Examples use port `7991` and explicitly bind to `127.0.0.1`. Do not rely on
  the launcher's implicit address.
- A `project path` is a real host path. A `workspace path` is relative to the
  `files/` root visible to the agent.
- Replace every `<LIKE_THIS>` placeholder. Example tokens, hosts, users, and
  paths are not usable credentials.
- Queue names, core counts, GPU counts, and executable paths in templates are
  starting points that must be adapted to the site.

## Scope

CatMaster organizes work, prepares and checks files, invokes registered tools,
retains execution evidence, and submits to configured machines. It does not
provide commercial software licenses, cluster accounts, network authorization,
potential-file rights, institutional subscriptions, or scientific judgment.
