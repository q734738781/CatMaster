from __future__ import annotations

from typing import Dict, Any
import os
from pydantic import BaseModel, Field
from langchain_experimental.utilities.python import PythonREPL

from catmaster.tools.base import create_tool_output, workspace_root


class PythonExecInput(BaseModel):
    """
    Execute inline Python code for expression calculation, result post analysis, or figure drawing when other tools provided can not meet the requirement.
    ALWAYS use `print(...)` to output the final result; only printed output will be returned.
    Basic packages for material science are provided (e.g. numpy, matplotlib, pymatgen, ase, etc.), these packages are encouraged to be used to enhance analysis robustness.
    The working directory is set to the workspace root.
    """

    code: str = Field(..., description="Python code to execute. ")


_REPL = PythonREPL()


def python_exec(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = PythonExecInput(**payload)
    try:
        cwd = os.getcwd()
        os.chdir(workspace_root())
        try:
            result = _REPL.run(params.code)
        finally:
            os.chdir(cwd)
        return create_tool_output("python_exec", True, data={"run_result": result})
    except Exception as exc:
        return create_tool_output("python_exec", False, error=str(exc))


__all__ = ["python_exec", "PythonExecInput"]
