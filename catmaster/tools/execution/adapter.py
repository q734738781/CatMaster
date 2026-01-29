from __future__ import annotations

"""LLM-facing adapter that hides resource selection and submission mechanics."""

from pathlib import Path
from typing import Any, Dict

from catmaster.tools.execution.mace_dispatch import mace_relax
from catmaster.tools.execution.vasp_dispatch import vasp_execute


class DPDispatcherAdapter:
    def __init__(self) -> None:
        pass

    def submit(self, task_name: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        if task_name == "mace_relax":
            return self._submit_mace(payload)
        if task_name == "vasp_execute":
            return self._submit_vasp(payload)
        raise ValueError(f"Unsupported task {task_name}")

    def _submit_mace(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return mace_relax(payload)

    def _submit_vasp(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        return vasp_execute(payload)


__all__ = ["DPDispatcherAdapter"]
