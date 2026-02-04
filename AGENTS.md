# Agents Notes

- Remote GPU task scripts must be staged under `task_script/` in the DPDispatcher workdir.
- `mace_relax_dir` runs `python task_script/mace_jobs.py ...` and forwards that script via DPDispatcher.
- The canonical MACE script source is `catmaster/remote/gpu/mace_jobs.py`.
