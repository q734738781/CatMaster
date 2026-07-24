# Remote Execution Smoke Tests

These tests submit real DPDispatcher jobs and are opt-in.

For deployment checks on a remote machine, prefer the CLI wrapper first:

```bash
python scripts/remote_execution_smoke.py --list
python scripts/remote_execution_smoke.py --suite core --check-interval 30
python scripts/remote_execution_smoke.py --suite all --check-interval 60
python scripts/remote_execution_smoke.py --suite uma --uma-check-interval 60
python scripts/remote_execution_smoke.py --suite mlff_si512 --check-interval 30
```

The CLI writes a JSON report under `/tmp/catmaster_remote_execution_smoke`
by default and exercises the current agent-visible `remote_submission` path.
The `mlff_si512` suite uses one deterministic perturbed 512-atom diamond-Si
structure for every enabled backend and runs SP, bounded relaxation, and short
MD through the same task contracts. The `mlff_operations` suite also includes a
fixed-image NEB smoke for every enabled backend.

Run the whole group explicitly:

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution -s
```

Full command with the recommended defaults spelled out:

```bash
env MPLCONFIGDIR=/tmp/catmaster_mplconfig \
XDG_CACHE_HOME=/tmp/catmaster_xdg_cache \
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 \
CATMASTER_REMOTE_CHECK_INTERVAL=30 \
CATMASTER_REMOTE_VASP_CHECK_INTERVAL=60 \
CATMASTER_REMOTE_MACE_MODEL=mh-1 \
CATMASTER_REMOTE_MACE_HEAD=omat_pbe \
CATMASTER_REMOTE_MACE_DTYPE=float32 \
CATMASTER_REMOTE_VASP_TASK=vasp_execute \
CATMASTER_REMOTE_LAMMPS_CHECK_INTERVAL=30 \
CATMASTER_REMOTE_UMA_MODEL=uma-s-1p2 \
CATMASTER_REMOTE_UMA_TASK=omat \
CATMASTER_REMOTE_UMA_RELAX_STEPS=5 \
pytest tests/remote_execution -s -vv
```

Run individual cases when isolating a failure:

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_mace_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_vasp_prepare_then_remote_submission_o2_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_cp2k_prepare_then_remote_submission_o2_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_cp2k_prepare_then_remote_submission_o2_geo_opt_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_cp2k_aimd_prepare_then_remote_submission_o2_short_nvt_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_lammps_lj_prepare_then_remote_submission_o2_minimize_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_lammps_lj_prepare_then_remote_submission_o2_short_nvt_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_omol_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_periodic_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_omol_relax_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_periodic_relax_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_omol_ts_nh3_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 CATMASTER_RUN_REMOTE_UMA_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_uma_omol_general_vib_remote -s -vv
```

Prerequisites:

- `configs/dpdispatcher/machines.yaml` contains reachable machines.
- `configs/dpdispatcher/resources.yaml` binds `mace_gpu` and `vasp_cpu`.
- The GPU remote environment can run MACE single-point jobs.
- The UMA remote environment can run FairChem UMA single-point jobs when
  `CATMASTER_RUN_REMOTE_UMA_TESTS=1` is set. It must use `uma_gpu`, a
  separate UMA/FairChem environment, a valid Hugging Face token or prewarmed
  cache, and should not reuse the MACE environment.
- Local pymatgen can locate VASP pseudopotentials, because the VASP test starts
  from `vasp_prepare` and requires POTCAR generation before remote dispatch.
- The CPU remote environment can run VASP through `catmaster/remote/cpu/vasp_boot.py`.
- The CPU remote environment can run CP2K through `cp2k.psmp` for full-suite
  deployment checks.
- The CPU remote environment can expose a LAMMPS executable in `PATH`; the
  LAMMPS boot script auto-detects common names such as `lmp_mpi`, `lmp`,
  KOKKOS/GPU variants, and can be overridden with `CATMASTER_LAMMPS_BIN`.
  It may enable KOKKOS/GPU-package acceleration when visible and falls back to
  CPU execution if acceleration fails.

Useful overrides:

```bash
CATMASTER_REMOTE_CHECK_INTERVAL=30
CATMASTER_REMOTE_MACE_MODEL=mh-1
CATMASTER_REMOTE_MACE_HEAD=omat_pbe
CATMASTER_REMOTE_MACE_DTYPE=float32
CATMASTER_REMOTE_VASP_TASK=vasp_execute
CATMASTER_REMOTE_CP2K_CHECK_INTERVAL=60
CATMASTER_REMOTE_LAMMPS_CHECK_INTERVAL=30
CATMASTER_REMOTE_UMA_CHECK_INTERVAL=60
CATMASTER_REMOTE_UMA_MODEL=uma-s-1p2
CATMASTER_REMOTE_UMA_TASK=omat
CATMASTER_REMOTE_UMA_DEVICE=auto
CATMASTER_REMOTE_UMA_MOL_SPIN=1
CATMASTER_REMOTE_UMA_RELAX_FMAX=0.05
CATMASTER_REMOTE_UMA_RELAX_STEPS=5
```

Expected coverage:

- The MACE test stages one O2 POSCAR under an `mlff_sp` stage with backend `mace`,
  calls `remote_submission` through `ToolRegistry.as_langchain_tools(...)`,
  and checks `status.json`, `batch_summary.json`, finite single-point energy,
  `summary.json`, and `sp.vasp`.
- The VASP test stages the O2 POSCAR as `O2.vasp`, invokes `vasp_prepare`
  through `ToolRegistry.as_langchain_tools(...)` with `preset=static` and
  `regime=gas`, asserts `INCAR/KPOINTS/POSCAR/POTCAR` were generated, copies
  then dispatches the prepared folder with `remote_submission` and checks
  `status.json` plus at least one VASP output file (`OUTCAR`, `OSZICAR`, or
  `vasprun.xml`).
- The CP2K tests prepare O2 single-point, fixed-cell geometry optimization, and
  short NVT AIMD stages, dispatch each through `remote_submission` with
  `task_name=cp2k_execute`, and check `status.json`, `cp2k_summary.json`, and
  native CP2K output presence.
- The LAMMPS tests validate an explicit LJ force-field card, prepare
  minimization and short NVT stages, dispatch them with
  `task_name=lammps_execute`, and check `status.json`, `lammps_summary.json`,
  LAMMPS log output, and trajectory output where applicable.
- The UMA tests are opt-in on top of the global remote gate. They stage H2O
  for `mlff_sp`/`mlff_relax` with backend `fairchem_uma`, `audience=orca_xtb_worker`, and
  `uma_task=omol`, and an O2 periodic VASP structure for
  `audience=materials_worker` with `uma_task=omat` by default. They check
  `status.json`, `batch_summary.json`, finite energy or final energy, max
  force for relaxations, per-item `summary.json`, and output structures.
- The UMA TS test stages planar NH3 as extxyz with component constraints, runs
  `mlff_ts` through the real `uma_gpu` DPDispatcher binding, and requires
  constrained finite-difference RS-pRFO convergence, exactly one significant
  imaginary mode, zero fixed-component drift, retained `ts.extxyz`, and the
  compact Hessian/frequency/mode artifacts.
- The UMA VIB test stages non-TS H2O as extxyz with component constraints, runs
  the general `mlff_vib` task through `uma_gpu`, and requires an exact seven-DOF
  constrained spectrum in `vibrations.npz`, `frequencies.csv`, and one
  `modes.extxyz`, with no ASE displacement-cache JSON files.

Failure triage:

- A failure before submission while writing `POTCAR` means pymatgen cannot find
  the configured pseudopotential library, usually `PMG_VASP_PSP_DIR` or the
  pymatgen config path.
- Missing machine/resource keys means local DPDispatcher config is incomplete.
- SSH, queue, or upload/download errors point to local DPDispatcher or remote
  access configuration.
- A transient remote `tar ... Resource temporarily unavailable` during download
  can be retried by DPDispatcher; persistent repeats indicate remote filesystem
  or result-download pressure.
- Non-zero `status.json.returncode` with stderr/stdout tails means the remote
  environment launched but MACE/VASP itself failed.
- UMA authentication, network, or offline-cache failures should be triaged in
  the `uma_gpu.source_list` script: check `HF_TOKEN`, `HF_HOME/HF_HUB_CACHE`,
  model prewarm, and that `HF_HUB_OFFLINE=1` is used only after the cache is
  populated.
- CP2K `status.json.returncode=2` before launch usually means `SLURM_NTASKS`
  was unavailable or `job.inp` was missing in the submitted stage.
- LAMMPS acceleration failures should be visible in `lammps_stdout.out`; by
  default the boot script retries the same input on CPU.

All smoke tests invoke managed preparation and/or generic `remote_submission`
through the LangChain tool wrapper, matching the current agent-visible call path
while still diagnosing local DPDispatcher configuration, pymatgen POTCAR setup,
and remote MACE/VASP/CP2K/LAMMPS environments.
