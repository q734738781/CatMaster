from __future__ import annotations

from typing import Dict, Any
from pydantic import BaseModel, Field
from langchain_experimental.utilities.python import PythonREPL

from catmaster.tools.base import create_tool_output


class PythonExecInput(BaseModel):
    """
    Execute inline Python code for calculations or figure drawing.
    ALWAYS use `print(...)` to output the final result; only printed output will be returned.
    Basic packages for material science are provided (e.g. numpy, matplotlib, pymatgen, ase, etc.)
    """

    code: str = Field(..., description="Python code to execute. ")


_REPL = PythonREPL()


def python_exec(payload: Dict[str, Any]) -> Dict[str, Any]:
    params = PythonExecInput(**payload)
    try:
        result = _REPL.run(params.code)
        return create_tool_output("python_exec", True, data={"run_result": result})
    except Exception as exc:
        return create_tool_output("python_exec", False, error=str(exc))


__all__ = ["python_exec", "PythonExecInput"]
