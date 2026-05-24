# Remote Execution Smoke Tests

These tests submit real DPDispatcher jobs and are opt-in.

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
pytest tests/remote_execution -s -vv
```

Run only one side when isolating a failure:

```bash
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_agent_tool_mace_sp_remote -s -vv
CATMASTER_RUN_REMOTE_EXECUTION_TESTS=1 pytest tests/remote_execution/test_dpdispatcher_remote_smoke.py::test_vasp_prepare_then_remote_submission_o2_sp_remote -s -vv
```

Prerequisites:

- `configs/dpdispatcher/machines.yaml` contains reachable machines.
- `configs/dpdispatcher/resources.yaml` binds `mace_gpu` and `vasp_cpu`.
- The GPU remote environment can run MACE single-point jobs.
- Local pymatgen can locate VASP pseudopotentials, because the VASP test starts
  from `vasp_prepare` and requires POTCAR generation before remote dispatch.
- The CPU remote environment can run VASP through `catmaster/remote/cpu/vasp_boot.py`.

Useful overrides:

```bash
CATMASTER_REMOTE_CHECK_INTERVAL=30
CATMASTER_REMOTE_MACE_MODEL=mh-1
CATMASTER_REMOTE_MACE_HEAD=omat_pbe
CATMASTER_REMOTE_MACE_DTYPE=float32
CATMASTER_REMOTE_VASP_TASK=vasp_execute
```

Expected coverage:

- The MACE test stages one O2 POSCAR under a low-level `mace_sp_dir` stage,
  calls `remote_submission` through `ToolRegistry.as_langchain_tools(...)`,
  and checks `status.json`, `batch_summary.json`, finite single-point energy,
  `summary.json`, and `sp.vasp`.
- The VASP test stages the O2 POSCAR as `O2.vasp`, invokes `vasp_prepare`
  through `ToolRegistry.as_langchain_tools(...)` with `preset=static` and
  `regime=gas`, asserts `INCAR/KPOINTS/POSCAR/POTCAR` were generated, copies
  then dispatches the prepared folder with `remote_submission` and checks
  `status.json` plus at least one VASP output file (`OUTCAR`, `OSZICAR`, or
  `vasprun.xml`).

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

Both smoke tests invoke the generic `remote_submission` tool through the
LangChain tool wrapper, matching the current agent-visible call path while still
diagnosing local DPDispatcher configuration, pymatgen POTCAR setup, and remote
MACE/VASP environments.
