from __future__ import annotations

import ast
import importlib.util
import inspect
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_PACKAGE_ROOT = Path(__file__).resolve().parents[2]


class ExportBuiltinToolSourceInput(BaseModel):
    """[implementation/read] Export original tool or CatMaster module source for inspection.

    By default, export the module and recursively referenced CatMaster modules,
    including parent package initializers, as separate files with original imports.
    Sources are reading references, not a standalone executable tool copy:
    dynamic imports, non-Python assets and third-party packages are not collected.
    Use module_name for an additional known module, or include_dependencies=false
    to inspect only one file.
    """

    model_config = ConfigDict(extra="forbid")

    tool_name: str = Field("", description="Registered tool name; leave empty when using module_name.")
    module_name: str = Field("", description="CatMaster module to inspect, e.g. catmaster.runtime.observation_events; leave empty with tool_name.")
    output_root: str = Field(..., description="Workspace-relative directory for original source files, preserving module paths.")
    include_dependencies: bool = Field(True, description="Export recursively referenced local modules and parent package __init__.py files in one call. False exports only the selected module.")
    overwrite: bool = Field(False, description="Replace existing exported files that differ from the original source. Identical files are reused without rewriting.")

    @model_validator(mode="after")
    def _one_source(self) -> "ExportBuiltinToolSourceInput":
        if bool(self.tool_name.strip()) == bool(self.module_name.strip()):
            raise ValueError("Provide exactly one of tool_name or module_name")
        return self


def _module_file(name: str) -> Path:
    parts = name.split(".")
    if parts[0] != "catmaster" or not all(part.isidentifier() for part in parts):
        raise ValueError("Only CatMaster source modules may be exported")
    path = _PACKAGE_ROOT.joinpath(*parts[1:])
    for candidate in (path.with_suffix(".py"), path / "__init__.py"):
        if candidate.is_file():
            return candidate
    raise ValueError(f"CatMaster source module not found: {name}")


def _referenced_modules(module: str, path: Path, source: str | bytes) -> list[str]:
    """Find source navigation targets without importing or rewriting code."""
    package = module if path.name == "__init__.py" else module.rsplit(".", 1)[0]
    found: set[str] = set()

    def add(name: str) -> None:
        if name != "catmaster" and not name.startswith("catmaster."):
            return
        try:
            _module_file(name)
        except ValueError:
            return  # A from-import may name a symbol rather than a module.
        if name != module:
            found.add(name)

    for node in ast.walk(ast.parse(source)):
        if isinstance(node, ast.Import):
            for alias in node.names:
                add(alias.name)
        elif isinstance(node, ast.ImportFrom):
            name = node.module or ""
            if node.level:
                name = importlib.util.resolve_name("." * node.level + name, package)
            add(name)
            for alias in node.names:
                add(name + "." + alias.name)
    return sorted(found)


def _collect_sources(module: str, include_dependencies: bool) -> dict[str, tuple[Path, bytes, list[str]]]:
    """Walk static local imports without loading modules or transforming source."""
    sources: dict[str, tuple[Path, bytes, list[str]]] = {}
    pending = [module]
    while pending:
        name = pending.pop()
        if name in sources:
            continue
        path = _module_file(name)
        source = path.read_bytes()
        references = _referenced_modules(name, path, source)
        sources[name] = (path, source, references)
        if include_dependencies:
            pending.extend(references)
            # Imports also traverse package initializers, which may re-export
            # helpers or import further modules. Keep their source reachable.
            parts = name.split(".")
            for length in range(1, len(parts)):
                parent = ".".join(parts[:length])
                if _PACKAGE_ROOT.joinpath(*parts[1:length], "__init__.py").is_file():
                    pending.append(parent)
    return sources


def export_builtin_tool_source(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[implementation/read] Export original tool or CatMaster module source for inspection."""
    tool = "export_builtin_tool_source"
    try:
        params = ExportBuiltinToolSourceInput(**payload)
        module = params.module_name.strip()
        canonical_name = ""
        if not module:
            from catmaster.tools.registry import get_tool_registry

            registry = get_tool_registry()
            canonical_name = registry._canonical_tool_name(params.tool_name.strip())
            if canonical_name not in registry.tools:
                raise ValueError(f"Unknown registered tool: {params.tool_name}")
            func = inspect.unwrap(registry.get_tool_function(canonical_name))
            module = func.__module__
        sources = _collect_sources(module, params.include_dependencies)
        root = resolve_workspace_path(params.output_root, must_exist=False)
        exported_modules: dict[str, str] = {}
        writes: list[tuple[Path, bytes]] = []
        # Check existing files before writing any new ones, so an overlapping
        # export can reuse references without replacing workspace edits.
        for name, (path, source, _) in sorted(sources.items()):
            target = root / "catmaster" / path.relative_to(_PACKAGE_ROOT)
            exported_modules[name] = workspace_relpath(target)
            if target.exists():
                if target.read_bytes() == source:
                    continue
                if not params.overwrite:
                    raise FileExistsError(
                        f"Exported source differs: {workspace_relpath(target)}; "
                        "choose another output_root or use overwrite=true to replace it"
                    )
            writes.append((target, source))
        for target, source in writes:
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(source)
        data = {
            "tool_name": canonical_name,
            "module_name": module,
            "source_path": exported_modules[module],
            "referenced_modules": sources[module][2],
            "exported_modules": exported_modules,
        }
        content = (
            f"Source available for {canonical_name or module}: {len(sources)} original module files.\n"
            f"source_path={data['source_path']}\n"
            f"source_root={workspace_relpath(root / 'catmaster')}\n"
            "Read or search the source tree directly; paths and imports are preserved.\n"
            "This is a reading reference, not a standalone program. Dynamic imports, "
            "non-Python assets and third-party packages are not collected."
        )
        if not params.include_dependencies:
            content += "\nOnly the selected module was exported; use include_dependencies=true to collect static local imports."
        return content, {"tool_name": tool, "data": data}
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool,
            public_message=f"{tool} failed: {exc}",
            artifact={"tool_name": tool, "data": {}},
            error_code="export_builtin_tool_source_failed",
        ) from exc


__all__ = ["ExportBuiltinToolSourceInput", "export_builtin_tool_source"]
