from __future__ import annotations

from typing import Any, Dict
from pydantic import BaseModel, Field

from catmaster.tools.base import create_tool_output, resolve_workspace_path
from pathlib import Path


class VaspSummarizeInput(BaseModel):
    work_dir: str = Field(..., description="Directory containing VASP outputs (must include vasprun.xml).")


def vasp_summarize(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Extract structured summary from a VASP run directory.

    Pydantic Args Schema: VaspSummarizeInput
    Returns: dict with energies, convergence flags, final structure (if converged), and metadata
    """
    from .vasp_summarizer import summarize_vasp
    
    work_dir = str(resolve_workspace_path(payload["work_dir"]))
    
    try:
        summary = summarize_vasp(Path(work_dir))
        return create_tool_output(
            tool_name="vasp_summarize",
            success=True,
            data=summary
        )
    except Exception as e:
        return create_tool_output(
            tool_name="vasp_summarize",
            success=False,
            error=str(e)
        )
