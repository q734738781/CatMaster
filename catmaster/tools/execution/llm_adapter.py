#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LLM execution adapter tool, encapsulating jobflow-remote execution flow to provide a unified interface for LLM Agents.
"""
from __future__ import annotations

import os
import re
import time
import subprocess
import hashlib
import uuid
from pathlib import Path
from typing import Dict, Any, Optional, Tuple, List
from pydantic import BaseModel, Field

from jobflow import Flow, job
from jobflow_remote import submit_flow, get_jobstore
from jobflow_remote.config.base import ExecutionConfig

from catmaster.tools.ssh import load_worker_host, sftp_upload_tree, sftp_download_tree


class JobExecutionRequest(BaseModel):
    """Job execution request model"""
    project: str = Field(..., description="Jobflow-remote project name")
    worker: str = Field(..., description="Worker name (cpu-worker/gpu-worker)")
    job_func: str = Field(..., description="Full path to job function to execute, e.g., 'catmaster.tools.execution.vasp_jobs:run_vasp'")
    job_kwargs: Dict[str, Any] = Field(default_factory=dict, description="Parameters to pass to job function")
    input_dir: Optional[str] = Field(None, description="Input file directory, will be uploaded to remote if provided")
    remote_tmp_base: str = Field(..., description="Remote temporary directory root path")
    extra_pre_run: Optional[str] = Field(None, description="Extra pre_run script")
    wait: bool = Field(True, description="Whether to wait for job completion")
    timeout_s: int = Field(36000, description="Wait timeout in seconds")
    poll_s: float = Field(5.0, description="Polling interval in seconds")
    download_results: bool = Field(False, description="Whether to download run directory to local")
    download_dir: Optional[str] = Field(None, description="Download directory path")


class JobExecutionResponse(BaseModel):
    """Job execution response model"""
    project: str
    worker: str
    job_db_id: int
    job_uuid: str
    remote_staging: Optional[str] = None
    output: Optional[Dict[str, Any]] = None
    state: Optional[str] = None
    download_path: Optional[str] = None
    error: Optional[str] = None


def _make_token(base: str) -> str:
    """Generate unique token"""
    seed = f"{base}|{time.time_ns()}|{uuid.uuid4().hex}|{os.getpid()}"
    return hashlib.md5(seed.encode("utf-8")).hexdigest()[:24]


def _query_state_via_cli(job_uuid: str) -> Tuple[Optional[str], str]:
    """Query job state via CLI"""
    try:
        res = subprocess.run(
            ["jf", "job", "info", job_uuid],
            check=False, text=True, capture_output=True,
            timeout=30
        )
        text = res.stdout + "\n" + res.stderr
        
        if res.returncode != 0:
            return None, f"[cli-error code={res.returncode}] {text}"
    except subprocess.TimeoutExpired:
        return None, "[cli-timeout] jf job info took too long"
    except Exception as e:
        return None, f"[cli-exception] {e}"
    
    # Try to parse state
    m = re.search(r"state\s*=\s*['\"]([^'\"]+)['\"]", text, re.IGNORECASE)
    if m:
        return m.group(1).strip(), text
    
    # Alternative: look for "State:" line
    for line in text.split('\n'):
        if 'state' in line.lower() and ':' in line:
            parts = line.split(':', 1)
            if len(parts) == 2:
                state_val = parts[1].strip().strip('\'"')
                if state_val:
                    return state_val, text
    
    return None, text


def _download_run_dir(job_db_id: int, dest: Path) -> None:
    """Download run directory"""
    dest.mkdir(parents=True, exist_ok=True)
    try:
        subprocess.run(
            ["jf", "job", "todir", str(job_db_id), "--path", str(dest)],
            check=True, text=True, capture_output=True
        )
        return
    except Exception:
        pass
    
    # Fallback method: download file by file
    try:
        ls = subprocess.run(
            ["jf", "job", "files", "ls", str(job_db_id)],
            check=True, text=True, capture_output=True
        ).stdout.splitlines()
        
        names = []
        for line in ls:
            line = line.strip()
            if not line or line[0] in "│╭╰": 
                continue
            tok = line.split()[-1]
            if "/" not in tok and tok not in (".", ".."):
                names.append(tok)
        
        for fn in sorted(set(names)):
            try:
                subprocess.run(
                    ["jf", "job", "files", "get", str(job_db_id), fn, "--path", str(dest)],
                    check=True, text=True, capture_output=True
                )
            except Exception:
                pass
    except Exception:
        pass


def execute_job(request: JobExecutionRequest) -> JobExecutionResponse:
    """
    Unified job execution interface, encapsulating the complete jobflow-remote execution flow.
    
    Supports:
    1. Optional input file upload
    2. Dynamic job function import
    3. Job submission and monitoring
    4. Optional result download
    """
    response = JobExecutionResponse(
        project=request.project,
        worker=request.worker,
        job_db_id=-1,
        job_uuid=""
    )
    
    try:
        # 1. Dynamically import job function
        print(f"\n      >> execute_job() called")
        print(f"         project: {request.project}")
        print(f"         worker: {request.worker}")
        print(f"         job_func: {request.job_func}")
        print(f"         job_kwargs: {request.job_kwargs}")
        print(f"         input_dir: {request.input_dir}")
        
        module_path, func_name = request.job_func.rsplit(":", 1)
        print(f"         → Importing: {module_path}.{func_name}")
        
        module = __import__(module_path, fromlist=[func_name])
        job_func = getattr(module, func_name)
        
        print(f"         [OK] Function imported: {job_func}")
        print(f"            Is @job decorated: {hasattr(job_func, 'original')}")
        
        # 2. If input directory provided, upload to remote
        remote_staging = None
        pre_run_parts = []
        
        if request.input_dir:
            input_path = Path(request.input_dir).resolve()
            if not input_path.exists():
                raise FileNotFoundError(f"Input directory not found: {input_path}")
            
            host_alias = load_worker_host(request.project, request.worker)
            token = _make_token(str(input_path))
            remote_staging = os.path.join(request.remote_tmp_base, f"stg_{token}")
            
            sftp_upload_tree(host_alias, input_path, remote_staging)
            response.remote_staging = remote_staging
            
            # Add copy and cleanup pre_run
            pre_run_parts.append(f"""
test -d "{remote_staging}" || {{ echo "missing upload dir: {remote_staging}" >&2; exit 2; }}
cp -a "{remote_staging}"/. .
rm -rf "{remote_staging}"
""")
        
        if request.extra_pre_run:
            pre_run_parts.append(request.extra_pre_run)
        
        # 3. Create job and flow
        job_instance = job_func(**request.job_kwargs)
        flow = Flow([job_instance])
        
        # 4. Build execution config
        exec_cfg = ExecutionConfig(
            pre_run="\n".join(pre_run_parts) if pre_run_parts else None
        )
        
        # 5. Submit job
        print(f"\n      >> Submitting job to {request.worker}")
        print(f"         Job function: {request.job_func}")
        print(f"         Job kwargs: {request.job_kwargs}")
        print(f"         Pre-run script length: {len('\\n'.join(pre_run_parts)) if pre_run_parts else 0} chars")
        
        db_ids = submit_flow(flow, worker=request.worker, exec_config=exec_cfg)
        job_db_id = db_ids[0] if isinstance(db_ids, (list, tuple)) else db_ids
        
        response.job_db_id = job_db_id
        response.job_uuid = job_instance.uuid
        
        print(f"         [OK] Job submitted: ID={job_db_id}, UUID={job_instance.uuid}")
        
        if not request.wait:
            print(f"         → Not waiting for completion (wait=False)")
            return response
        
        # 6. Wait for job completion
        print(f"         → Waiting for job completion (timeout={request.timeout_s}s, poll={request.poll_s}s)")
        js = get_jobstore()
        js.connect()
        
        _DONE = {"COMPLETED", "FINISHED", "SUCCESS", "DONE"}
        _FAIL = {"FAILED", "ERROR", "CANCELLED", "STOPPED", "REJECTED", "TIMEOUT", "REMOTE_ERROR"}
        
        t0 = time.time()
        poll_count = 0
        while True:
            poll_count += 1
            elapsed = time.time() - t0
            
            # Try to get output
            try:
                out = js.get_output(job_instance.uuid, load=True)
                if out is not None:
                    print(f"         [OK] Got output after {elapsed:.1f}s ({poll_count} polls)")
                    print(f"            Output keys: {list(out.keys()) if isinstance(out, dict) else type(out)}")
                    response.output = out
                    response.state = "COMPLETED"
                    break
            except Exception as e:
                if poll_count == 1:
                    print(f"         → Output not ready yet (will poll...)")
            
            # Check state via CLI
            state, raw = _query_state_via_cli(job_instance.uuid)
            if state:
                response.state = state
                up = state.upper()
                
                if poll_count % 5 == 0 or up in _DONE or up in _FAIL:
                    print(f"         [Poll {poll_count}, t={elapsed:.1f}s] State: {state}")
                
                if up in _FAIL:
                    print(f"         [FAIL] Job failed with state: {state}")
                    print(f"         CLI output:\n{raw}")
                    raise RuntimeError(f"Job {job_instance.uuid} failed with state={state}")
                
                if up in _DONE:
                    print(f"         [OK] Job completed with state: {state}")
                    try:
                        out = js.get_output(job_instance.uuid, load=True)
                        if out:
                            print(f"            Output: {list(out.keys()) if isinstance(out, dict) else type(out)}")
                        response.output = out
                    except Exception as e:
                        print(f"            [WARN]  Could not load output: {e}")
                    break
            
            if time.time() - t0 > request.timeout_s:
                print(f"         [FAIL] Timeout after {elapsed:.1f}s")
                raise TimeoutError(f"Job {job_instance.uuid} timed out after {request.timeout_s}s")
            
            time.sleep(request.poll_s)
        
        # 7. Optional: download results
        if request.download_results and request.download_dir:
            # Download directly to specified directory without worker/job_id subdirectory
            dest = Path(request.download_dir)
            dest.mkdir(parents=True, exist_ok=True)
            _download_run_dir(job_db_id, dest)
            response.download_path = str(dest.resolve())
    
    except Exception as e:
        response.error = str(e)
    
    return response


def execute_vasp(
    input_dir: str,
    project: str = "catmaster",
    worker: str = "cpu-worker",
    remote_tmp_base: str = "/public/home/chenhh/tmp/jfr_uploads",
    wait: bool = True,
    timeout_s: int = 36000,
    download_results: bool = True,
    download_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to execute VASP calculation
    """
    request = JobExecutionRequest(
        project=project,
        worker=worker,
        job_func="catmaster.tools.execution.vasp_jobs:run_vasp",
        job_kwargs={},
        input_dir=input_dir,
        remote_tmp_base=remote_tmp_base,
        wait=wait,
        timeout_s=timeout_s,
        download_results=download_results,
        download_dir=download_dir or "./vasp_results",
    )
    
    response = execute_job(request)
    return response.model_dump()


def execute_mace(
    input_structure: str,
    work_dir: str = ".",
    project: str = "catmaster",
    worker: str = "gpu-worker",
    remote_tmp_base: str = "/ssd/chenhh/tmp/jfr_uploads",
    fmax: float = 0.05,
    steps: int = 500,
    model: str = "medium-mpa-0",
    device: str = "auto",
    wait: bool = True,
    timeout_s: int = 7200,
    download_results: bool = True,
    download_dir: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Convenience function to execute MACE relaxation using the standard catmaster module
    """
    # Prepare input directory
    input_dir = Path(work_dir).resolve()
    
    # Use the proper mace_jobs module (requires catmaster to be synced to remote)
    request = JobExecutionRequest(
        project=project,
        worker=worker,
        job_func="catmaster.tools.execution.mace_jobs:run_mace",
        job_kwargs={
            "structure_file": Path(input_structure).name,
            "fmax": fmax,
            "steps": steps,
            "model": model,
            "device": device,
        },
        input_dir=str(input_dir),
        remote_tmp_base=remote_tmp_base,
        wait=wait,
        timeout_s=timeout_s,
        download_results=download_results,
        download_dir=download_dir or "./mace_results",
    )
    
    response = execute_job(request)
    return response.model_dump()


__all__ = [
    "JobExecutionRequest",
    "JobExecutionResponse", 
    "execute_job",
    "execute_vasp",
    "execute_mace",
]
