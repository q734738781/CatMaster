from __future__ import annotations

from pathlib import Path

from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.misc.export_builtin_tool_source import export_builtin_tool_source


def test_export_builtin_tool_source_vasp_prepare(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        content, artifact = export_builtin_tool_source(
            {
                "tool_name": "vasp_prepare",
                "output_root": "notes/tool_refs/vasp_prepare",
                "overwrite": True,
            }
        )

        out_root = workspace_root() / "notes" / "tool_refs" / "vasp_prepare"
        function_path = out_root / "function.py"
        dependencies_path = out_root / "dependencies.py"

        assert function_path.exists()
        assert dependencies_path.exists()
        assert artifact["data"]["function_path"] == "notes/tool_refs/vasp_prepare/function.py"
        assert artifact["data"]["dependencies_path"] == "notes/tool_refs/vasp_prepare/dependencies.py"
        assert "Exported builtin tool source for vasp_prepare." in content

        function_text = function_path.read_text(encoding="utf-8")
        dependencies_text = dependencies_path.read_text(encoding="utf-8")

        assert "class VaspPrepareInput" in function_text
        assert "def _validate_single_structure_input(" in function_text
        assert "def vasp_prepare(" in function_text
        assert "from dependencies import (" in function_text
        assert "from catmaster.tools.base import" not in function_text
        assert "from .adsorbate_tool import" not in function_text

        assert "def resolve_workspace_path(" in dependencies_text
        assert "def workspace_relpath(" in dependencies_text
        assert "class StructWriter" in dependencies_text
        assert "def propagate_adsorbate_metadata(" in dependencies_text
