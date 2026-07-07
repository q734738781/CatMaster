---
name: uma-screening-and-relaxation
description: Use this skill for FairChem UMA single-point screening or relaxation of materials, slabs, adsorbates, catalysts, MOFs, and related periodic structures through managed remote UMA tasks.
license: project-local
compatibility: local
allowed-tools: "ls read_file write_file edit_file execute get_avail_remote_task remote_submission remote_submission_batch"
---

# uma-screening-and-relaxation

Use this skill when a materials workflow needs FairChem UMA inference as a rapid screening or pre-relaxation layer. UMA is separate from the MACE environment and uses the managed remote task names `uma_sp_dir` and `uma_relax_dir`.

## Required workflow
1. Read `remote-stage-layouts` for `uma_sp_dir` or `uma_relax_dir`.
2. Build one clean stage directory with `input/` containing the candidate structures.
3. Choose `uma_task` explicitly when the scientific domain is known.
4. Submit one prepared stage with `remote_submission` and put method-critical settings in `template_overrides`; use `remote_submission_batch` only for multiple independent UMA stage directories.
5. Report `remote_context_id`, `submission_hash`, `receipt_rel`, and the downloaded `output/batch_summary.json`.

## Task choice
- Use `uma_task=omat` for inorganic bulk, defects, surfaces, and generic periodic materials when no catalyst-specific UMA task is intended.
- Use `uma_task=oc20`, `oc22`, or `oc25` for catalyst/adsorbate systems when that dataset task matches the problem. Do not let `auto` decide catalyst semantics.
- Use `uma_task=odac` for MOF/direct-air-capture structures and `uma_task=omc` for molecular crystals.
- Use `uma_task=auto` only for quick molecule-vs-periodic routing in exploratory batches.

## Guardrails
- Keep UMA and MACE as separate managed paths. Do not submit UMA work through `mace_gpu` or MACE task names.
- Do not use `omol` to compute OC20/OC25 adsorption reference energies or catalyst-side adsorption energies.
- Keep `charge=0` and `spin=0` for non-`omol` UMA tasks.
- For `uma_task=omol`, set `charge` and `spin` explicitly through `template_overrides` and verify the downloaded `summary.json`; do not rely on the default `spin=0`.
- Keep `relax_cell=false` unless the task is explicitly `omat` and the user wants cell relaxation.
- Treat UMA outputs as screening evidence unless the user explicitly accepts ML-potential-level conclusions.
- Do not edit copied `task_script/` files or use `sitecustomize.py` to force UMA arguments. If the requested `uma_task`, `charge`, or `spin` cannot be expressed through `template_overrides` or metadata, report the template gap.

## Stage examples
Single-point screening:

```text
stage/
  input/
    bulk_a.vasp
    slab_b.vasp
```

Submit with `task_name="uma_sp_dir"` and `template_overrides` such as:

```json
{"uma_task": "omat", "model": "uma-s-1p2", "device": "auto"}
```

Relaxation:

```json
{"uma_task": "omat", "fmax": 0.02, "steps": 500, "relax_cell": false}
```

For mixed task batches, write `params/uma_metadata.json` and pass `template_overrides={"metadata_path": "params/uma_metadata.json"}`.
