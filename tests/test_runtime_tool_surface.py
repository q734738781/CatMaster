from __future__ import annotations

from dataclasses import dataclass

import pytest

pytest.importorskip("langchain_core")

from catmaster.runtime.run_context import RunContext
from catmaster.runtime.tool_surface import build_runtime_tool_surface


@dataclass
class _Tool:
    name: str


class _DummyInputModel:
    __doc__ = "Dummy local tool input model."


class _Registry:
    def as_langchain_tools(self, *, run_dir=None, workspace=None):
        _ = (run_dir, workspace)
        return [
            _Tool("bash_exec"),
            _Tool("read_file"),
            _Tool("list_directory_with_sizes"),
            _Tool("get_file_info"),
            _Tool("create_molecule_from_smiles"),
            _Tool("apply_aider_edits"),
        ]

    def get_tool_info(self, name: str):
        _ = name
        return {"input_model": _DummyInputModel}


class _MCPRuntime:
    def role_filtered_tools(self, *, role: str):
        if role in {"proposal", "director"}:
            return [
                _Tool("search_files"),
                _Tool("read_text_file"),
                _Tool("read_file"),
                _Tool("list_directory_with_sizes"),
                _Tool("get_file_info"),
            ]
        if role == "task_runner":
            return [
                _Tool("search_files"),
                _Tool("read_text_file"),
                _Tool("read_file"),
                _Tool("list_directory_with_sizes"),
                _Tool("get_file_info"),
                _Tool("write_file"),
                _Tool("edit_file"),
            ]
        return []

    def render_capability_guide(self, *, mode: str = "full") -> str:
        if mode == "short":
            return "Filesystem read/discovery: search_files, read_text_file"
        return "Filesystem tools:\n- read/discovery: search_files, read_text_file\n- write/edit: write_file, edit_file"


def test_build_runtime_tool_surface_role_split(tmp_path) -> None:
    run_ctx = RunContext.create(
        workspace=tmp_path,
        model_name="dummy-model",
    )
    surface = build_runtime_tool_surface(
        registry=_Registry(),
        run_context=run_ctx,
        run_dir=run_ctx.run_dir,
        mcp_fs_runtime=_MCPRuntime(),
        task_runner_denylist=set(),
    )

    proposal_names = [tool.name for tool in surface.proposal_tools]
    director_names = [tool.name for tool in surface.director_tools]
    task_names = [tool.name for tool in surface.task_tools]

    assert "bash_exec" in proposal_names
    assert "search_files" in proposal_names
    assert proposal_names == director_names
    assert "read_file" not in proposal_names
    assert "list_directory_with_sizes" not in proposal_names
    assert "get_file_info" not in proposal_names
    assert "apply_aider_edits" in proposal_names

    assert "apply_aider_edits" in task_names
    assert "create_molecule_from_smiles" in task_names
    assert "write_file" in task_names
    assert "read_file" not in task_names
    assert "list_directory_with_sizes" not in task_names
    assert "get_file_info" not in task_names

    assert "Task runner standard capabilities" in surface.task_runner_capability_guide_full
    assert "Filesystem tools" in surface.task_runner_capability_guide_full
    assert "Task runner capabilities (short)" in surface.task_runner_capability_guide_short
