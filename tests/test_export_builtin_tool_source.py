import importlib
from pathlib import Path

import pytest

from catmaster.tools.base import workspace_root, workspace_scope
from catmaster.tools.misc.export_builtin_tool_source import export_builtin_tool_source
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError


@pytest.fixture
def source_tree(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> dict[str, bytes]:
    package = tmp_path / "source" / "catmaster"
    files = {
        "__init__.py": b"from . import shared\n",
        "shared.py": b"VALUE = 1\n",
        "example/__init__.py": b"from .helper import helper\n",
        "example/main.py": (
            b"import catmaster.shared\nfrom .helper import helper\n"
            b"from catmaster.example import nested\nimport missing_third_party\n"
            b"import importlib\nimportlib.import_module('catmaster.example.dynamic')\n"
            b"def later():\n    from . import nested\n"
            b"raise RuntimeError('source must never be executed')\n"
        ),
        "example/helper.py": b"from . import main\nfrom ..shared import VALUE\ndef helper(): return VALUE\n",
        "example/nested.py": b"from .helper import helper\n",
        "example/dynamic.py": b"# Explicitly discoverable even without a static import.\n",
        "example/data.json": b'{"value": 1}\n',
    }
    for relative, content in files.items():
        path = package / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(content)
    implementation = importlib.import_module("catmaster.tools.misc.export_builtin_tool_source")
    monkeypatch.setattr(implementation, "_PACKAGE_ROOT", package)
    return files


@pytest.mark.parametrize("tool_name", ["vasp_prepare", "remote_submission"])
def test_export_preserves_original_module_and_reachable_imports(tmp_path: Path, tool_name: str) -> None:
    repo = Path(__file__).resolve().parents[1]
    with workspace_scope(tmp_path):
        content, artifact = export_builtin_tool_source({"tool_name": tool_name, "output_root": "refs"})
        data = artifact["data"]
        exported = workspace_root() / data["source_path"]
        original = repo / exported.relative_to(workspace_root() / "refs")
        assert exported.read_bytes() == original.read_bytes()
        assert data["source_path"] in content
        assert "catmaster.tools.base" in data["referenced_modules"]
        assert set(data["referenced_modules"]) <= data["exported_modules"].keys()
        for relative in data["exported_modules"].values():
            path = workspace_root() / relative
            assert path.read_bytes() == (repo / path.relative_to(workspace_root() / "refs")).read_bytes()
        _, dependency = export_builtin_tool_source({"module_name": "catmaster.tools.base", "output_root": "refs"})
        assert (workspace_root() / dependency["data"]["source_path"]).read_bytes() == (repo / "catmaster/tools/base.py").read_bytes()


def test_package_submodule_import_and_nested_dependency_navigation(tmp_path: Path) -> None:
    from catmaster.tools.misc.export_builtin_tool_source import _referenced_modules
    refs = _referenced_modules("catmaster.runtime.example", Path("example.py"),
                               "from catmaster.runtime import observation_events\nfrom . import tool_runtime\n")
    assert "catmaster.runtime.observation_events" in refs
    assert "catmaster.runtime.tool_runtime" in refs
    with workspace_scope(tmp_path):
        content, result = export_builtin_tool_source({"module_name": "catmaster.runtime.observation_events", "output_root": "refs"})
        assert (workspace_root() / result["data"]["source_path"]).is_file()
        assert "observation_events.py" in content
        for module in ["os", "catmaster...secrets", "../configs"]:
            with pytest.raises(CatMasterToolExecutionError):
                export_builtin_tool_source({"module_name": module, "output_root": "refs"})


def test_recursive_export_keeps_separate_files_packages_and_cycles(tmp_path: Path, source_tree: dict[str, bytes]) -> None:
    with workspace_scope(tmp_path / "workspace"):
        content, result = export_builtin_tool_source({"module_name": "catmaster.example.main", "output_root": "refs"})
        root = workspace_root() / "refs" / "catmaster"
        expected = {name for name in source_tree if name.endswith(".py") and name != "example/dynamic.py"}
        assert {str(path.relative_to(root)) for path in root.rglob("*") if path.is_file()} == expected
        for name in expected:
            assert (root / name).read_bytes() == source_tree[name]
        assert len(result["data"]["exported_modules"]) == len(expected)
        assert "refs/catmaster" in content
        assert "refs/catmaster/example/main.py" in content


def test_single_module_mode_and_explicit_dynamic_module(tmp_path: Path, source_tree: dict[str, bytes]) -> None:
    with workspace_scope(tmp_path / "workspace"):
        _, result = export_builtin_tool_source({
            "module_name": "catmaster.example.main", "output_root": "refs", "include_dependencies": False,
        })
        assert set(result["data"]["exported_modules"]) == {"catmaster.example.main"}
        assert "catmaster.example.helper" in result["data"]["referenced_modules"]
        export_builtin_tool_source({"module_name": "catmaster.example.dynamic", "output_root": "refs"})
        assert (workspace_root() / "refs/catmaster/example/dynamic.py").read_bytes() == source_tree["example/dynamic.py"]


def test_overlapping_export_reuses_files_and_preserves_edits(tmp_path: Path, source_tree: dict[str, bytes]) -> None:
    with workspace_scope(tmp_path / "workspace"):
        request = {"module_name": "catmaster.example.main", "output_root": "refs"}
        export_builtin_tool_source(request)
        root = workspace_root() / "refs/catmaster"
        helper = root / "example/helper.py"
        timestamp = helper.stat().st_mtime_ns
        export_builtin_tool_source(request)
        assert helper.stat().st_mtime_ns == timestamp
        helper.write_text("# workspace edit\n")
        (root / "shared.py").unlink()
        with pytest.raises(CatMasterToolExecutionError, match="Exported source differs"):
            export_builtin_tool_source(request)
        assert not (root / "shared.py").exists()
        assert helper.read_text() == "# workspace edit\n"
        export_builtin_tool_source({**request, "overwrite": True})
        assert helper.read_bytes() == source_tree["example/helper.py"]
        assert (root / "shared.py").is_file()


def test_registered_surface_exports_dependencies_and_visible_paths(tmp_path: Path, source_tree: dict[str, bytes]) -> None:
    from catmaster.tools.registry import get_tool_registry

    registry = get_tool_registry()
    schema = next(tool for tool in registry.as_openai_tools() if tool["name"] == "export_builtin_tool_source")["parameters"]
    tool = next(tool for tool in registry.as_langchain_tools() if tool.name == "export_builtin_tool_source")
    langchain_schema = tool.args_schema
    if hasattr(langchain_schema, "model_json_schema"):
        langchain_schema = langchain_schema.model_json_schema()
    for parameters in (schema, langchain_schema):
        field = parameters["properties"]["include_dependencies"]
        assert field["type"] == "boolean"
        assert field["default"] is True
        assert "anyOf" not in field
        assert "include_dependencies" not in parameters.get("required", [])
    with workspace_scope(tmp_path / "workspace"):
        message = tool.invoke({
            "name": tool.name, "id": "source_export_test", "type": "tool_call",
            "args": {"module_name": "catmaster.example.main", "output_root": "refs"},
        })
        assert "refs/catmaster/example/main.py" in message.content
        assert "source_root=refs/catmaster" in message.content
        assert (workspace_root() / "refs/catmaster/example/helper.py").is_file()
