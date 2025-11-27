from __future__ import annotations

import os
import re
import json
import subprocess
from pathlib import Path
from typing import Dict, Optional

from jobflow import job


@job
def run_vasp(vasp_command: Optional[str] = None, env: Optional[Dict[str, str]] = None) -> Dict[str, object]:
    """
    Run VASP in the current working directory and emit a minimal summary.
    - Uses SLURM/MPICH mpirun if available (via SLURM_NTASKS), else runs the binary directly.
    - Writes stdout to vasp_stdout.txt
    - Parses final energy from OUTCAR (best-effort).
    """
    vasp = vasp_command or os.environ.get("VASP_STD_BIN")
    if not vasp:
        raise RuntimeError("VASP_STD_BIN is not set and vasp_command not provided.")
    np = os.environ.get("SLURM_NTASKS") or "1"
    if os.environ.get("SLURM_JOB_ID") or os.environ.get("PMI_RANK") or os.environ.get("OMPI_COMM_WORLD_SIZE"):
        cmd = ["mpirun", "-n", str(np), vasp]
    else:
        cmd = [vasp]
    env_all = dict(os.environ)
    if env:
        env_all.update(env)
    with open("vasp_stdout.txt", "wb") as f:
        subprocess.run(cmd, check=False, stdout=f, stderr=subprocess.STDOUT, env=env_all)

    summary = {"parsed": False}
    try:
        txt = Path("OUTCAR").read_text(errors="ignore")
        m = list(re.finditer(r"free\\s+energy\\s+TOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if not m:
            m = list(re.finditer(r"\\bTOTEN\\s*=\\s*([\\-\\d\\.Ee\\+]+)", txt))
        if m:
            summary["final_energy_eV"] = float(m[-1].group(1))
            summary["parsed"] = True
    except Exception as e:
        summary["error"] = str(e)

    try:
        Path("summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    except Exception:
        pass
    return {"summary": summary}


