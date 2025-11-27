from __future__ import annotations

from typing import Any, Dict
import time
import logging
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output

logger = logging.getLogger(__name__)


class MaceRelaxInput(BaseModel):
    structure_file: str = Field(..., description="Input structure file relative to work_dir (POSCAR/CIF/etc.).")
    work_dir: str = Field(".", description="Working directory.")
    device: str = Field("auto", description="Device: auto|cpu|cuda or cuda:0 etc.")
    use_d3: bool = Field(True, description="Use D3 correction in MACE relaxations.")
    fmax: float = Field(0.05, gt=0, description="Force threshold for relaxation.")
    maxsteps: int = Field(500, ge=1, description="Max steps for relaxation.")
    optimizer: str = Field("FIRE", description="ASE optimizer.")
    model: str = Field("medium-mpa-0", description="MACE model name.")
    mode: str = Field("molecular", description="Constraint mode.")
    constraint_entities: str = Field("auto", description="Entities to constrain.")
    bond_rt_scale: float = Field(0.10, description="Hooke bond scale.")
    hooke_k: float = Field(15.0, description="Hooke spring constant.")
    amp_mode: str = Field("auto", description="AMP mode.")
    amp_dtype: str = Field("fp16", description="AMP dtype.")
    tf32: bool = Field(False, description="Enable TF32 on CUDA.")


def mace_relax(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit MACE relaxation job to gpu-worker via jobflow-remote.
    
    Pydantic Args Schema: MaceRelaxInput
    Returns: dict with job info and results if wait=True
    """
    from pathlib import Path
    import shutil
    from .llm_adapter import JobExecutionRequest, execute_job
    
    # Extract parameters
    structure_file = payload["structure_file"]
    work_dir = payload.get("work_dir", ".")
    device = payload.get("device", "auto")
    use_d3 = payload.get("use_d3", True)
    fmax = payload.get("fmax", 0.05)
    maxsteps = payload.get("maxsteps", 500)
    optimizer = payload.get("optimizer", "FIRE")
    model = payload.get("model", "medium-mpa-0")
    mode = payload.get("mode", "molecular")
    
    # Create work directory
    work_path = Path(work_dir)
    work_path.mkdir(parents=True, exist_ok=True)
    
    # Copy structure file to work directory
    structure_path = Path(structure_file)
    input_filename = structure_path.name
    input_file = work_path / input_filename
    
    # Create job request
    request = JobExecutionRequest(
        project="catmaster",
        worker="gpu-worker",
        job_func="catmaster.tools.execution.mace_jobs:run_mace",
        job_kwargs={
            "structure_file": input_filename,  # Use original filename
            "fmax": fmax,
            "steps": maxsteps,
            "model": model,
            "device": device,
        },
        input_dir=str(work_path),
        remote_tmp_base="/ssd/chenhh/tmp/jfr_uploads",  # SSD for GPU worker
        wait=True,
        timeout_s=7200,
        download_results=True,
        download_dir=str(work_path / "mace_results"),
    )
    
    # Execute job
    response = execute_job(request)
    result = response.model_dump()
    
    # Process result for compatibility
    if result.get("output") and result["output"].get("summary"):
        summary = result["output"]["summary"]
        download_path = Path(result.get("download_path", work_path / "mace_results"))
        
        # Look for relaxed structure in download path
        relaxed_vasp = None
        if download_path.exists():
            # Check for CONTCAR directly in download path
            contcar = download_path / "CONTCAR"
            if contcar.exists():
                relaxed_vasp = str(contcar)
                # Also create a standardized name for downstream compatibility
                relaxed_std = download_path.parent / "relaxed.vasp"
                try:
                    shutil.copy(contcar, relaxed_std)
                    relaxed_vasp = str(relaxed_std)
                except Exception as e:
                    logger.warning(f"Could not copy CONTCAR to relaxed.vasp: {e}")
        
        # Return standardized output
        return create_tool_output(
            tool_name="mace_relax",
            success=True,
            data={
                "converged": summary.get("converged", False),
                "final_energy": summary.get("energy_eV", None),
                "relaxed_structure": relaxed_vasp if relaxed_vasp else None,
                "model": model,
                "optimizer": optimizer,
                "fmax": fmax,
            },
            execution_time=result.get("execution_time")
        )
    else:
        # Return raw result if format is different
        return result




# ===== VASP execution via jobflow-remote (CPU side) =====

from typing import Optional

class VaspExecuteInput(BaseModel):
    """Input parameters for VASP execution via jobflow-remote."""
    input_dir: str = Field(..., description="Directory containing VASP inputs (INCAR/KPOINTS/POSCAR/POTCAR)")
    project: str = Field("catmaster", description="jobflow-remote project name")
    worker: str = Field("cpu-worker", description="jobflow-remote worker name")
    remote_tmp_base: str = Field("/public/home/chenhh/tmp/jfr_uploads", description="Remote staging directory")
    vasp_command: Optional[str] = Field(None, description="VASP executable path")
    env: Optional[Dict[str, str]] = Field(None, description="Environment variables")
    wait: bool = Field(True, description="Wait for completion")
    timeout_s: int = Field(36000, description="Timeout in seconds")
    poll_s: float = Field(5.0, description="Poll interval in seconds")
    download_results: bool = Field(True, description="Download results after completion")
    download_dir: Optional[str] = Field(None, description="Download directory")


def vasp_execute(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Submit a VASP job to CPU worker via jobflow-remote.
    
    Pydantic Args Schema: VaspExecuteInput
    Returns: dict with job results
    """
    from .llm_adapter import execute_vasp as _execute_vasp
    
    # Call the existing implementation
    result = _execute_vasp(
        input_dir=payload["input_dir"],
        project=payload.get("project", "catmaster"),
        worker=payload.get("worker", "cpu-worker"),
        remote_tmp_base=payload.get("remote_tmp_base", "/public/home/chenhh/tmp/jfr_uploads"),
        vasp_command=payload.get("vasp_command"),
        env=payload.get("env"),
        wait=payload.get("wait", True),
        timeout_s=payload.get("timeout_s", 36000),
        download_results=payload.get("download_results", True),
        download_dir=payload.get("download_dir")
    )
    
    # Return standardized output
    return create_tool_output(
        tool_name="vasp_execute",
        success=True,
        data=result
    )
