# CatMaster Demo Scripts (DPDispatcher)

Remote targets
- **gpu_server**: DPDispatcher Shell backend on the GPU host; used for MACE relax.
- **cpu_hpc**: DPDispatcher Slurm backend on the HPC cluster; used for VASP jobs.

Demos
- `demo_dpdispatcher_mace_o2.py` — MACE relaxation of O2 on gpu_server (POSCAR from tests/assets/O2_in_the_box). Dry-run unless `--run`.
- `demo_dpdispatcher_vasp_o2.py` — VASP relaxation of O2 on cpu_hpc (INCAR/KPOINTS/POSCAR/POTCAR from tests/assets/O2_in_the_box). Dry-run unless `--run`.
- `demo_dpdispatcher_remote_submit.py` — minimal hostname/date submission smoke test.

Usage
```bash
# MACE on GPU (dry-run)
python demos/demo_dpdispatcher_mace_o2.py
# Actually submit
python demos/demo_dpdispatcher_mace_o2.py --run

# VASP on HPC (dry-run)
python demos/demo_dpdispatcher_vasp_o2.py
# Actually submit
python demos/demo_dpdispatcher_vasp_o2.py --run
```

Prereqs: `~/.catmaster/dpdispatcher.yaml` (or split JSONs) defines machines/resources; password-less SSH is assumed.
