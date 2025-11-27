#!/usr/bin/env python3
# -*- coding: utf-8 -*-
from __future__ import annotations

import os, sys, time, yaml, importlib, types, re, subprocess
from pathlib import Path
import paramiko
from paramiko.proxy import ProxyCommand
from paramiko.config import SSHConfig

from jobflow import Flow
from jobflow_remote import submit_flow, get_jobstore
from jobflow_remote.config.base import ExecutionConfig

# ===== 可按需调整 =====
PROJECT_NAME = "catmaster"
CPU_WORKER   = "cpu-worker"
LOCAL_ASSETS = Path("./assets/O2_in_the_box")
REMOTE_TMP_BASE = "/public/home/chenhh/tmp/jfr_uploads"
DOWNLOAD_DIR = Path("./test_download")
POLL, TIMEOUT = 2.0, 36000

# ===== 解析 worker -> host =====
def load_worker_host(project: str, worker: str) -> str:
    cfgp = Path.home() / ".jfremote" / f"{project}.yaml"
    data = yaml.safe_load(cfgp.read_text())
    return data["workers"][worker]["host"]

# ===== SSH 连接与 SFTP 上传 =====
def connect_via_ssh_config(host_alias: str) -> paramiko.SSHClient:
    ssh_cfg_path = os.path.expanduser("~/.ssh/config")
    cfg = SSHConfig()
    looked = {}
    if os.path.exists(ssh_cfg_path):
        with open(ssh_cfg_path) as f:
            cfg.parse(f)
        looked = cfg.lookup(host_alias)
    hostname = looked.get("hostname", host_alias)
    username = looked.get("user")
    port     = int(looked.get("port", 22))
    keyfiles = looked.get("identityfile", [])
    proxycmd = looked.get("proxycommand")
    sock = ProxyCommand(proxycmd) if proxycmd else None

    cli = paramiko.SSHClient()
    cli.load_system_host_keys()
    cli.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    cli.connect(
        hostname=hostname, port=port, username=username,
        key_filename=keyfiles or None, sock=sock,
        allow_agent=True, look_for_keys=True,
        banner_timeout=60, auth_timeout=60, timeout=60,
    )
    return cli

def sftp_mkdirs(sftp: paramiko.SFTPClient, remote_dir: str) -> None:
    cur = "/"
    for p in remote_dir.strip("/").split("/"):
        cur = os.path.join(cur, p)
        try:
            sftp.stat(cur)
        except IOError:
            sftp.mkdir(cur, mode=0o755)

def sftp_upload_tree(host_alias: str, local_dir: Path, remote_dir: str) -> None:
    cli = connect_via_ssh_config(host_alias)
    try:
        sftp = cli.open_sftp()
        sftp_mkdirs(sftp, remote_dir)
        for root, dirs, files in os.walk(local_dir):
            rel = os.path.relpath(root, local_dir)
            rdir = remote_dir if rel == "." else os.path.join(remote_dir, rel)
            sftp_mkdirs(sftp, rdir)
            for d in dirs:
                sftp_mkdirs(sftp, os.path.join(rdir, d))
            for f in files:
                lp = os.path.join(root, f)
                rp = os.path.join(rdir, f)
                sftp.put(lp, rp)
        sftp.close()
    finally:
        cli.close()

# ===== 远端作业函数:由于jobflow-remote必须要求使用import能import上的job，不能放在main中，因此这里用注入模块的方式=====
# 对于实际开发，可以用rsync+pythonpath的方式同步脚本部署
MODULE_NAME = "jfr_inline_vasp"
MODULE_CODE = r'''
from jobflow import job
from pathlib import Path
import os, json, subprocess, re

@job
def run_vasp():
    vasp = os.environ.get("VASP_STD_BIN")
    if not vasp:
        raise RuntimeError("VASP_STD_BIN is not set")
    np = os.environ.get("SLURM_NTASKS") or "1"
    cmd = ["mpirun","-n",str(np),vasp]

    with open("vasp_stdout.txt","wb") as f:
        subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT)

    summary = {"parsed": False}
    try:
        txt = Path("OUTCAR").read_text(errors="ignore")
        m = list(re.finditer(r"free\\s+energy\\s+TOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if not m:
            m = list(re.finditer(r"\\bTOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if m:
            summary["final_energy_eV"] = float(m[-1].group(1)); summary["parsed"] = True
    except Exception as e:
        summary["error"] = str(e)

    return {"summary": summary}
'''

def ensure_inline_module_in_memory(name: str, code: str):
    if name in sys.modules:
        del sys.modules[name]
    mod = types.ModuleType(name)
    mod.__file__ = f"<inline:{name}>"
    mod.__package__ = ""
    sys.modules[name] = mod
    exec(code, mod.__dict__)
    return importlib.import_module(name)

def make_exec_config_copyin(remote_tmp_dir: str) -> ExecutionConfig:
    pre_run = f"""
set -euo pipefail
test -d "{remote_tmp_dir}" || {{ echo "missing upload dir: {remote_tmp_dir}" >&2; exit 2; }}
cp -a "{remote_tmp_dir}"/. .
rm -rf "{remote_tmp_dir}"
cat > {MODULE_NAME}.py <<'PY'
{MODULE_CODE}
PY
export PYTHONPATH="$PWD${{PYTHONPATH+::$PYTHONPATH}}"
"""
    return ExecutionConfig(pre_run=pre_run)

# ===== 轮询：失败即退出 =====
_DONE = {"COMPLETED","FINISHED","SUCCESS","DONE"}
_FAIL = {"FAILED","ERROR","CANCELLED","STOPPED","REJECTED","TIMEOUT","REMOTE_ERROR"}

def _query_state_via_cli(job_uuid: str):
    try:
        res = subprocess.run(["jf","job","info",job_uuid], check=True, text=True, capture_output=True)
        text = res.stdout
    except Exception as e:
        return None, f"[cli-error] {e}"
    m = re.search(r"state\s*=\s*'([^']+)'", text, re.IGNORECASE)
    return (m.group(1).strip() if m else None), text

def wait_for_output_or_fail(js, job_uuid: str, timeout: int = TIMEOUT, poll: float = POLL):
    t0 = time.time()
    while True:
        try:
            out = js.get_output(job_uuid, load=True)
            if out is not None:
                return out
        except Exception:
            pass
        state, raw = _query_state_via_cli(job_uuid)
        if state:
            up = state.upper()
            print(f"[state] {state}")
            if up in _FAIL:
                raise RuntimeError(f"作业 {job_uuid} 失败，state={state}。\n------ jf job info ------\n{raw}\n--------------------------")
            if up in _DONE:
                try:
                    out = js.get_output(job_uuid, load=True)
                except Exception:
                    out = None
                return out
        if time.time() - t0 > timeout:
            raise TimeoutError(f"等待作业 {job_uuid} 超过 {timeout} 秒未完成")
        time.sleep(poll)

# ===== 下载运行目录 =====
def download_run_dir(job_db_id: int, dest: Path) -> None:
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(["jf","job","todir",str(job_db_id),"--path",str(dest)],
                       check=True, text=True, capture_output=True)
        return
    except Exception:
        pass
    try:
        ls = subprocess.run(["jf","job","files","ls",str(job_db_id)],
                            check=True, text=True, capture_output=True).stdout.splitlines()
        names = []
        for line in ls:
            line = line.strip()
            if not line or line[0] in "│╭╰": continue
            tok = line.split()[-1]
            if "/" not in tok and tok not in (".",".."): names.append(tok)
        for fn in sorted(set(names)):
            try:
                subprocess.run(["jf","job","files","get",str(job_db_id),fn,"--path",str(dest)],
                               check=True, text=True, capture_output=True)
            except Exception:
                pass
    except Exception:
        pass

def main():
    assert LOCAL_ASSETS.is_dir(), f"missing assets dir: {LOCAL_ASSETS}"
    host_alias = load_worker_host(PROJECT_NAME, CPU_WORKER)

    # 在内存中注册模块，构造作业对象，不在本地落盘任何文件
    mod = ensure_inline_module_in_memory(MODULE_NAME, MODULE_CODE)
    job = mod.run_vasp()

    # 按作业 UUID 建远端临时目录并上传
    remote_tmp_dir = os.path.join(REMOTE_TMP_BASE, job.uuid)
    print(f"[CPU] uploading to {host_alias}:{remote_tmp_dir} ...")
    sftp_upload_tree(host_alias, LOCAL_ASSETS, remote_tmp_dir)
    print("[CPU] upload done.")

    flow = Flow([job])
    exec_cfg = make_exec_config_copyin(remote_tmp_dir)
    db_ids = submit_flow(flow, worker=CPU_WORKER, exec_config=exec_cfg)
    job_db_id = db_ids[0] if isinstance(db_ids,(list,tuple)) else db_ids
    print(f"[CPU] submitted db_id={job_db_id}, uuid={job.uuid}")

    js = get_jobstore(); js.connect()
    out = wait_for_output_or_fail(js, job.uuid)
    print("[CPU] summary:", out.get("summary"))

    dest = DOWNLOAD_DIR / f"vasp_cpu_{job_db_id}"
    download_run_dir(job_db_id, dest)
    print(f"[CPU] downloaded to {dest.resolve()}")

if __name__ == "__main__":
    main()
