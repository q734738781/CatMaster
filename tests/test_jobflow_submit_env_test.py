#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Environment check:
- submit one job to cpu-worker and one to gpu-worker
- pre_run writes env.json, pip.json, nvidia.csv remotely (ASCII-only)
- poll JobStore until done, then fetch files and print summary
"""
from __future__ import annotations
import json
import subprocess
import time
from pathlib import Path

from jobflow import Flow
from jobflow_remote import get_jobstore, submit_flow
from jobflow_remote.config.base import ExecutionConfig
from jobflow_remote.utils.examples import add

CPU_WORKER = "cpu-worker"
GPU_WORKER = "gpu-worker"
POLL_INTERVAL = 2.0
TIMEOUT_SEC = 600

def wait_output(js, uuid: str, timeout: int = TIMEOUT_SEC, interval: float = POLL_INTERVAL):
    start = time.time()
    while True:
        try:
            out = js.get_output(uuid)
            if out is not None:
                return out
        except Exception:
            pass
        if time.time() - start > timeout:
            raise TimeoutError(f"Job {uuid} did not finish in {timeout} s")
        time.sleep(interval)

_PRE_RUN_TEMPLATE = r"""
# basic info (ASCII only)
date +"%F %T %Z" > ENV_CAPTURE.log
hostname >> ENV_CAPTURE.log

# write env.json via inline Python
python - <<'PY'
import os, sys, json, platform, subprocess
def run(cmd):
    try:
        out = subprocess.run(cmd, check=True, text=True, capture_output=True)
        return out.stdout.strip()
    except Exception as e:
        return "ERROR: %s" % e
info = {
    "worker_label": "__WORKER_LABEL__",
    "hostname": platform.node(),
    "platform": platform.platform(),
    "python_executable": sys.executable,
    "python_version": sys.version,
    "pip_version": run([sys.executable, "-m", "pip", "--version"]),
    "conda_prefix": os.environ.get("CONDA_PREFIX", ""),
    "cuda_visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES", ""),
    "which_python": run(["which", "python"]),
}
with open("env.json", "w", encoding="utf-8") as f:
    json.dump(info, f, indent=2, ensure_ascii=False)
PY

# dump pip list
python -m pip list --format=json > pip.json 2>/dev/null || python -m pip freeze > pip_freeze.txt || true

# optional: GPU summary
if command -v nvidia-smi >/dev/null 2>&1; then
  nvidia-smi --query-gpu=index,name,driver_version,memory.total --format=csv,noheader > nvidia.csv || true
fi
"""

def make_exec_config(worker_label: str) -> ExecutionConfig:
    wl = worker_label.replace("\\", "\\\\").replace('"', r'\"')
    pre_run = _PRE_RUN_TEMPLATE.replace("__WORKER_LABEL__", wl)
    return ExecutionConfig(pre_run=pre_run)

def submit_one(worker: str, a: int, b: int):
    j = add(a, b)
    flow = Flow([j])
    db_ids = submit_flow(flow, worker=worker, exec_config=make_exec_config(worker))
    job_db_id = db_ids[0] if isinstance(db_ids, (list, tuple)) else db_ids
    return job_db_id, j.uuid

def list_remote_files(job_db_id: int) -> str:
    # best-effort listing for debugging
    try:
        res = subprocess.run(
            ["jf", "job", "files", "ls", str(job_db_id)],
            check=True, text=True, capture_output=True
        )
        return res.stdout.strip()
    except Exception:
        return ""

def fetch_files(db_id, dest_dir: Path):
    dest_dir.mkdir(parents=True, exist_ok=True)
    job_db_id = db_id[0] if isinstance(db_id, (list, tuple)) else db_id
    job_db_id = str(job_db_id)

    print(f"[fetch] remote job {job_db_id} -> {dest_dir}")
    listing = list_remote_files(job_db_id)
    if listing:
        print(f"[fetch] remote list:\n{listing}")

    candidates = ["env.json", "pip.json", "nvidia.csv", "ENV_CAPTURE.log"]
    for fname in candidates:
        try:
            subprocess.run(
                ["jf", "job", "files", "get", job_db_id, fname, "--path", str(dest_dir)],
                check=True, text=True, capture_output=True
            )
            print(f"[fetch] ok: {fname}")
        except subprocess.CalledProcessError as e:
            msg = (e.stdout or "") + (e.stderr or "")
            if "No such file" in msg or "not found" in msg:
                print(f"[fetch] skip missing: {fname}")
                continue
            raise

def pretty_print(dest_dir: Path, label: str):
    env_path = dest_dir / "env.json"
    pip_path = dest_dir / "pip.json"
    nvsmi_path = dest_dir / "nvidia.csv"

    if env_path.exists():
        info = json.loads(env_path.read_text(encoding="utf-8"))
        pyver = info.get("python_version", "").splitlines()[0]
        print(f"[{label}] host={info.get('hostname','?')}  python={pyver}")
        print(f"[{label}] pip={info.get('pip_version','?')}")
        print(f"[{label}] CONDA_PREFIX={info.get('conda_prefix','')}")
        print(f"[{label}] CUDA_VISIBLE_DEVICES={info.get('cuda_visible_devices','')}")
    else:
        print(f"[{label}] env.json not found")

    if pip_path.exists():
        try:
            pkgs = json.loads(pip_path.read_text(encoding="utf-8"))
            names = {p.get("name","").lower(): p.get("version","") for p in pkgs}
            for key in ["jobflow", "jobflow-remote", "pymatgen", "ase"]:
                if key in names:
                    print(f"[{label}] {key}=={names[key]}")
        except Exception:
            print(f"[{label}] failed to read pip.json")

    if nvsmi_path.exists():
        print(f"[{label}] nvidia.csv:")
        print(nvsmi_path.read_text(encoding="utf-8").strip() or "(empty)")
    else:
        print(f"[{label}] nvidia.csv missing (no GPU or nvidia-smi not present)")

def main():
    js = get_jobstore()
    js.connect()

    print("Submitting CPU env job...")
    cpu_dbid, cpu_uuid = submit_one(CPU_WORKER, 1, 2)

    print("Submitting GPU env job...")
    gpu_dbid, gpu_uuid = submit_one(GPU_WORKER, 10, 20)

    wait_output(js, cpu_uuid)
    wait_output(js, gpu_uuid)

    base = Path("./env_dumps")
    fetch_files(cpu_dbid, base / "cpu")
    fetch_files(gpu_dbid, base / "gpu")

    print("\n== Environment summary ==")
    pretty_print(base / "cpu", "CPU")
    pretty_print(base / "gpu", "GPU")

if __name__ == "__main__":
    main()
