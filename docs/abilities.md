# CatMaster abilities (current)

This document summarizes what the codebase can do right now, based on the current implementation and demos.



## Runtime and orchestration

- Task-based orchestrator (`catmaster/agents/orchestrator.py`) that:
  - generates a JSON plan (`todo` + `plan_description`), supports iterative plan review (`yes` to approve),
  - runs each task via `TaskStepper` + tool calls,
  - summarizes each task and updates a structured whiteboard via UPSERT/DEPRECATE ops.

- Whiteboard memory + context packs:
  - Whiteboard sections: Goal, Key Facts, Key Files, Constraints, Open Questions, Journal.
  - Task summarizer proposes whiteboard ops; ops are validated/applied and diffs are traced.
  - Context packs include whiteboard excerpts + key files + artifact-log slice + workspace policy.

- HITL (human-in-the-loop) loop for `needs_intervention`:
  - on intervention, generates an interrupted report, prompts for free-form feedback, and replans only remaining work.
  - pending tasks are marked `skipped_deprecated`, new tasks are appended, and the run resumes after plan approval.
  - HITL bundles stored under `run_dir/hitl/hitl_###/` (report, feedback, packed feedback, revised plan, ops).

- Run tracking & auditability:
  - per-run directory under `.catmaster/runs/<run_id>` with `meta.json`, `task_state.json`, `observations/`, `toolcalls/`, `llm.jsonl`.
  - unified traces: `event_trace.jsonl`, `tool_trace.jsonl`, `patch_trace.jsonl`.
  - reports: `workspace/reports/FINAL_REPORT.md`, `workspace/reports/WHITEBOARD.md`, `workspace/reports/latest_run` symlink.

- Tool execution:
  - `ToolExecutor` validates inputs with Pydantic schemas, rejects extra fields.



## UI

- Reporter abstraction with three modes: `rich`, `plain`, `off` (`catmaster/ui`).
- Rich live dashboard shows tasks, current step, tool usage, whiteboard counts, and an event feed; optional debug panel.
- Scrollable plan review prompt and final-summary viewer (rich mode), with plain console fallbacks.
- HITL report viewer + feedback prompt (rich + plain).



## Core capabilities (tools)



### Geometry + input preparation

- **create_molecule_from_smiles**: RDKit + ASE 3D conformer generation, optimization, XYZ/POSCAR output.

- **relax_prepare**: MPRelaxSet-based VASP inputs with `calc_type` presets, k-product mesh, D3/DFT+U toggles, INCAR overrides; supports batch directories.

- **build_slab**: slab construction for all terminations of a Miller index; thickness/vacuum, symmetry slabs, orthogonalization, LLL reduction, supercell expansion; batch mode supported.

- **fix_atoms_by_layers** / **fix_atoms_by_height**: selective dynamics by layer count or z-ranges; batch mode supported.

- **supercell**: replicate a bulk/surface structure and write POSCAR.

- **enumerate_adsorption_sites**: Pymatgen ASF site list to JSON (ontop/bridge/hollow).

- **place_adsorbate**: place a molecule on a selected site, preserving slab selective dynamics.

- **generate_batch_adsorption_structures**: batch adsorbate placement for single or multiple slabs; emits JSON manifest.

- **make_neb_geometry/make_neb_incar**: NEB initial structures and INCAR generation; supports CI-NEB, IDPP, and image count.



### Execution (DPDispatcher)

<<<<<<< ours
<<<<<<< ours
- **vasp_execute / vasp_execute_batch**: submit VASP jobs (single or batch) via DPDispatcher; uses `configs/dpdispatcher/*` and task defaults in `tasks.yaml`.
=======
- **vasp_execute / vasp_execute_batch**: submit VASP jobs (single or batch) via DPDispatcher; uses `configs/dpdispatcher/*` and tool input defaults for resources/machine.
>>>>>>> theirs
=======
- **vasp_execute / vasp_execute_batch**: submit VASP jobs (single or batch) via DPDispatcher; uses `configs/dpdispatcher/*` and router defaults.
>>>>>>> theirs

- **mace_relax / mace_relax_batch**: submit MACE relaxations (single or batch) via DPDispatcher; outputs relaxed structure, trajectory, log, `summary.json`.

- Local runners exist for VASP/MACE in `catmaster/tools/execution/*_jobs.py` (used by DPDispatcher task templates).



### Retrieval

- **mp_search_materials**: Materials Project search with rich criteria (chemsys, energy above hull, band gap, sites, density, etc.) and CSV output.

- **mp_download_structure**: download structures by mp-id to POSCAR/CIF/JSON.



### Utilities

- **workspace_* file ops**: list/read/write/mkdir/copy/delete/grep/head/tail/move within workspace root.

- **python_exec**: Python calculations and longtail-tool self-implementation. VERY powerful tool for latest LLMs. Use with caution. Use SAFE models only. 

- Special Notice: Mainstream llms, has the report that it may execute DANGEROUS operations in their CLI products (e.g. rm -rf unexpected folder for codex-cli, gemini-cli etc.), however, we never met this behavior in our early stage development via python exec tool. If you are concerned about this, it is better to use a dedicate sub-system (e.g. WSL) for agent execution.

- **write_note**: append a memory note to the observation log.



## Demos (checked against current files)




- `demos/demo_llm_o2_vasp.py`:

  - LLM-driven O2-in-a-box VASP workflow.



- `demos/demo_llm_o2_spin_compare.py`:

  - Singlet vs triplet O2 comparison; writes results markdown.



- `demos/demo_llm_fe_surface.py`:

  - Fe surface energies for (100)/(110)/(111).



- `demos/demo_llm_fe_comprehension.py`:

  - Fe surface energies + CO adsorption site screening.



- `demos/demo_llm_fecuni_111_CO.py`:

  - Fe/Cu/Ni (111) CO adsorption energy comparison.



- `demos/demo_llm_Alloy_HER.py`:

  - Pt–Ni–Cu alloy HER screening with MACE + DFT validation.


- `demos/show_tool_descriptions.py`:

  - Print tool schemas and descriptions.



## Config and environment notes

- Workspace root: `CATMASTER_WORKSPACE` (defaults to cwd).

- DPDispatcher config lookup:

  - `$CATMASTER_DP_CONFIG` / `$CATMASTER_DP_TASKS`

  - `~/.catmaster/dpdispatcher.yaml` / `~/.catmaster/dpdispatcher.d/*`

  - `configs/dpdispatcher/*` directly in the repo

- Materials Project tools require `MP_API_KEY`.
