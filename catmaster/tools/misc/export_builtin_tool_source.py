from __future__ import annotations

import ast
import importlib.util
import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from pydantic import BaseModel, Field

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

_CATMASTER_PREFIX = "catmaster."


class ExportBuiltinToolSourceInput(BaseModel):
    """Export a builtin CatMaster tool's canonical source into workspace-readable reference files."""

    tool_name: str = Field(..., description="Registered builtin tool name.")
    output_root: str = Field(
        ...,
        description="Workspace-relative output directory that will receive function.py and dependencies.py.",
    )
    overwrite: bool = Field(
        False,
        description="When true, overwrite existing function.py and dependencies.py under output_root.",
    )


@dataclass(frozen=True)
class _ImportRef:
    alias: str
    module: str
    symbol: str | None
    kind: str  # 'symbol' or 'module'


@dataclass(frozen=True)
class _SymbolKey:
    module: str
    symbol: str


@dataclass
class _ModuleInfo:
    module: str
    path: Path
    source: str
    tree: ast.Module
    defs: dict[str, ast.AST]
    imports_by_alias: dict[str, _ImportRef]
    import_nodes: list[ast.stmt]
    source_order: list[tuple[int, str, ast.AST]]
    refs_cache: dict[str, "_SymbolRefs"]


@dataclass(frozen=True)
class _SymbolRefs:
    same_module: frozenset[str]
    direct_internal_imports: tuple[_ImportRef, ...]
    module_attr_internal_imports: tuple[tuple[_ImportRef, str], ...]


class _ReferenceCollector(ast.NodeVisitor):
    def __init__(self, module_aliases: set[str]) -> None:
        self.names: set[str] = set()
        self.module_attr_refs: set[tuple[str, str]] = set()
        self._module_aliases = module_aliases

    def visit_Name(self, node: ast.Name) -> None:
        self.names.add(node.id)

    def visit_Attribute(self, node: ast.Attribute) -> None:
        if isinstance(node.value, ast.Name) and node.value.id in self._module_aliases:
            self.module_attr_refs.add((node.value.id, node.attr))
        self.generic_visit(node)


class _SymbolRewrite(ast.NodeTransformer):
    def __init__(
        self,
        *,
        name_rewrite: dict[str, str],
        attr_rewrite: dict[tuple[str, str], str],
    ) -> None:
        self._name_rewrite = name_rewrite
        self._attr_rewrite = attr_rewrite

    def visit_Name(self, node: ast.Name) -> ast.AST:
        if isinstance(node.ctx, ast.Load) and node.id in self._name_rewrite:
            return ast.copy_location(ast.Name(id=self._name_rewrite[node.id], ctx=node.ctx), node)
        return node

    def visit_Attribute(self, node: ast.Attribute) -> ast.AST:
        node = self.generic_visit(node)
        if isinstance(node, ast.Attribute) and isinstance(node.value, ast.Name):
            replacement = self._attr_rewrite.get((node.value.id, node.attr))
            if replacement:
                return ast.copy_location(ast.Name(id=replacement, ctx=ast.Load()), node)
        return node


_MODULE_CACHE: dict[str, _ModuleInfo] = {}


def _tool_error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _module_name_for_file(path: Path) -> str:
    parts = list(path.resolve().parts)
    try:
        idx = parts.index("catmaster")
    except ValueError as exc:
        raise ValueError(f"Could not derive module name for {path}") from exc
    rel_parts = parts[idx:]
    if rel_parts[-1] == "__init__.py":
        rel_parts = rel_parts[:-1]
    else:
        rel_parts[-1] = rel_parts[-1][:-3]
    return ".".join(rel_parts)


def _resolve_module_file(module_name: str) -> Path:
    spec = importlib.util.find_spec(module_name)
    if spec is None or spec.origin is None:
        raise ValueError(f"Could not resolve module {module_name}")
    return Path(spec.origin).resolve()


def _resolve_imported_module(current_module: str, node: ast.ImportFrom) -> str:
    if node.level:
        current_path = _resolve_module_file(current_module)
        package = current_module if current_path.name == "__init__.py" else current_module.rsplit(".", 1)[0]
        base = "." * int(node.level)
        tail = node.module or ""
        return importlib.util.resolve_name(f"{base}{tail}", package)
    if not node.module:
        raise ValueError(f"Unsupported import without module in {current_module}")
    return node.module


def _top_level_defined_names(node: ast.AST) -> list[str]:
    if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
        return [node.name]
    if isinstance(node, ast.Assign):
        names: list[str] = []
        for target in node.targets:
            if isinstance(target, ast.Name):
                names.append(target.id)
        return names
    if isinstance(node, ast.AnnAssign) and isinstance(node.target, ast.Name):
        return [node.target.id]
    return []


def _load_module_info(module_name: str) -> _ModuleInfo:
    cached = _MODULE_CACHE.get(module_name)
    if cached is not None:
        return cached

    path = _resolve_module_file(module_name)
    source = path.read_text(encoding="utf-8")
    tree = ast.parse(source, filename=str(path))
    defs: dict[str, ast.AST] = {}
    imports_by_alias: dict[str, _ImportRef] = {}
    import_nodes: list[ast.stmt] = []
    source_order: list[tuple[int, str, ast.AST]] = []

    for node in tree.body:
        if isinstance(node, (ast.Import, ast.ImportFrom)):
            import_nodes.append(node)
            if isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name.startswith(_CATMASTER_PREFIX):
                        local_alias = alias.asname or alias.name
                        imports_by_alias[local_alias] = _ImportRef(
                            alias=local_alias,
                            module=alias.name,
                            symbol=None,
                            kind="module",
                        )
            else:
                imported_module = _resolve_imported_module(module_name, node)
                if imported_module.startswith(_CATMASTER_PREFIX):
                    for alias in node.names:
                        local_alias = alias.asname or alias.name
                        imports_by_alias[local_alias] = _ImportRef(
                            alias=local_alias,
                            module=imported_module,
                            symbol=alias.name,
                            kind="symbol",
                        )

        for name in _top_level_defined_names(node):
            defs[name] = node
            source_order.append((getattr(node, "lineno", 0), name, node))

    info = _ModuleInfo(
        module=module_name,
        path=path,
        source=source,
        tree=tree,
        defs=defs,
        imports_by_alias=imports_by_alias,
        import_nodes=import_nodes,
        source_order=sorted(source_order, key=lambda item: (item[0], item[1])),
        refs_cache={},
    )
    _MODULE_CACHE[module_name] = info
    return info


def _refs_for_symbol(info: _ModuleInfo, symbol: str) -> _SymbolRefs:
    cached = info.refs_cache.get(symbol)
    if cached is not None:
        return cached
    node = info.defs.get(symbol)
    if node is None:
        raise ValueError(f"Unknown symbol {symbol} in {info.module}")
    collector = _ReferenceCollector({alias for alias, ref in info.imports_by_alias.items() if ref.kind == "module"})
    collector.visit(node)

    same_module = frozenset(name for name in collector.names if name in info.defs and name != symbol)
    direct_internal_imports: list[_ImportRef] = []
    for name in collector.names:
        ref = info.imports_by_alias.get(name)
        if ref and ref.kind == "symbol":
            direct_internal_imports.append(ref)
    module_attr_internal_imports: list[tuple[_ImportRef, str]] = []
    for alias, attr in collector.module_attr_refs:
        ref = info.imports_by_alias.get(alias)
        if ref and ref.kind == "module":
            module_attr_internal_imports.append((ref, attr))
    result = _SymbolRefs(
        same_module=same_module,
        direct_internal_imports=tuple(direct_internal_imports),
        module_attr_internal_imports=tuple(module_attr_internal_imports),
    )
    info.refs_cache[symbol] = result
    return result


def _local_symbol_closure(info: _ModuleInfo, seed_symbols: set[str]) -> set[str]:
    selected: set[str] = set()
    stack = [symbol for symbol in seed_symbols if symbol in info.defs]
    while stack:
        symbol = stack.pop()
        if symbol in selected:
            continue
        if symbol not in info.defs:
            continue
        selected.add(symbol)
        refs = _refs_for_symbol(info, symbol)
        for dep in refs.same_module:
            if dep not in selected:
                stack.append(dep)
    return selected


def _resolve_internal_symbol(module_name: str, symbol: str) -> _SymbolKey:
    target = _load_module_info(module_name)
    if symbol not in target.defs:
        raise ValueError(f"Could not resolve internal symbol {module_name}:{symbol}")
    return _SymbolKey(module=module_name, symbol=symbol)


def _selected_external_refs(info: _ModuleInfo, selected_symbols: set[str]) -> set[_SymbolKey]:
    refs: set[_SymbolKey] = set()
    for symbol in selected_symbols:
        symbol_refs = _refs_for_symbol(info, symbol)
        for ref in symbol_refs.direct_internal_imports:
            if ref.symbol:
                refs.add(_resolve_internal_symbol(ref.module, ref.symbol))
        for ref, attr in symbol_refs.module_attr_internal_imports:
            refs.add(_resolve_internal_symbol(ref.module, attr))
    return refs


def _stable_export_names(keys: list[_SymbolKey]) -> dict[_SymbolKey, str]:
    grouped: dict[str, list[_SymbolKey]] = {}
    for key in keys:
        grouped.setdefault(key.symbol, []).append(key)
    out: dict[_SymbolKey, str] = {}
    used: set[str] = set()
    for symbol, group in grouped.items():
        if len(group) == 1 and symbol not in used:
            out[group[0]] = symbol
            used.add(symbol)
            continue
        for key in sorted(group, key=lambda item: item.module):
            module_parts = key.module.split(".")
            prefix_parts = list(reversed(module_parts[1:]))
            export_name = ""
            for idx in range(1, len(prefix_parts) + 1):
                prefix = "_".join(reversed(prefix_parts[:idx]))
                candidate = f"{prefix}__{key.symbol}"
                if candidate not in used:
                    export_name = candidate
                    break
            if not export_name:
                suffix = 2
                candidate = f"{module_parts[-1]}__{key.symbol}"
                export_name = candidate
                while export_name in used:
                    export_name = f"{candidate}_{suffix}"
                    suffix += 1
            out[key] = export_name
            used.add(export_name)
    return out


def _non_catmaster_import_lines(info: _ModuleInfo) -> list[str]:
    lines: list[str] = []
    seen: set[str] = set()
    for node in info.import_nodes:
        if isinstance(node, ast.Import):
            kept = [alias for alias in node.names if not alias.name.startswith(_CATMASTER_PREFIX)]
            if not kept:
                continue
            text = ast.unparse(ast.Import(names=kept)).strip()
        else:
            module_name = _resolve_imported_module(info.module, node)
            if module_name.startswith(_CATMASTER_PREFIX):
                continue
            text = ast.unparse(node).strip()
        if text and text not in seen:
            lines.append(text)
            seen.add(text)
    return lines


def _ordered_nodes(info: _ModuleInfo, selected_symbols: set[str]) -> list[tuple[str, ast.AST]]:
    seen_nodes: set[int] = set()
    ordered: list[tuple[str, ast.AST]] = []
    for _, symbol, node in info.source_order:
        if symbol not in selected_symbols:
            continue
        node_id = id(node)
        if node_id in seen_nodes:
            continue
        seen_nodes.add(node_id)
        ordered.append((symbol, node))
    return ordered


def _render_node(
    node: ast.AST,
    *,
    name_rewrite: dict[str, str],
    attr_rewrite: dict[tuple[str, str], str],
) -> str:
    rewritten = _SymbolRewrite(name_rewrite=name_rewrite, attr_rewrite=attr_rewrite).visit(ast.fix_missing_locations(ast.parse(ast.unparse(node))).body[0])
    return ast.unparse(ast.fix_missing_locations(rewritten)).strip()


def _render_function_file(
    *,
    info: _ModuleInfo,
    selected_symbols: set[str],
    dependency_imports: list[tuple[str, str]],
) -> str:
    lines: list[str] = ["from __future__ import annotations", ""]
    import_lines = _non_catmaster_import_lines(info)
    if import_lines:
        lines.extend(import_lines)
        lines.append("")
    if dependency_imports:
        lines.append("from dependencies import (")
        for export_name, local_name in dependency_imports:
            if export_name == local_name:
                lines.append(f"    {export_name},")
            else:
                lines.append(f"    {export_name} as {local_name},")
        lines.append(")")
        lines.append("")

    local_attr_rewrite: dict[tuple[str, str], str] = {}
    for _, local_name in dependency_imports:
        if "__" in local_name:
            alias, attr = local_name.split("__", 1)
            local_attr_rewrite[(alias, attr)] = local_name

    for _, node in _ordered_nodes(info, selected_symbols):
        lines.append(
            _render_node(
                node,
                name_rewrite={},
                attr_rewrite=local_attr_rewrite,
            )
        )
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _render_dependencies_file(
    *,
    module_order: list[str],
    selected_by_module: dict[str, set[str]],
    export_names: dict[_SymbolKey, str],
) -> str:
    lines: list[str] = ["from __future__ import annotations", ""]
    seen_imports: set[str] = set()
    for module_name in module_order:
        info = _load_module_info(module_name)
        for import_line in _non_catmaster_import_lines(info):
            if import_line not in seen_imports:
                lines.append(import_line)
                seen_imports.add(import_line)
    if seen_imports:
        lines.append("")

    for module_name in module_order:
        info = _load_module_info(module_name)
        selected = selected_by_module.get(module_name) or set()
        if not selected:
            continue
        lines.append(f"# --- from {module_name} ({workspace_relpath(info.path)})")
        lines.append("")

        name_rewrite = {
            symbol: export_names[_SymbolKey(module=module_name, symbol=symbol)]
            for symbol in selected
            if export_names[_SymbolKey(module=module_name, symbol=symbol)] != symbol
        }
        attr_rewrite: dict[tuple[str, str], str] = {}
        external_name_rewrite: dict[str, str] = {}
        for symbol in selected:
            refs = _refs_for_symbol(info, symbol)
            for ref in refs.direct_internal_imports:
                if ref.symbol:
                    key = _SymbolKey(module=ref.module, symbol=ref.symbol)
                    external_name_rewrite[ref.alias] = export_names[key]
            for ref, attr in refs.module_attr_internal_imports:
                key = _SymbolKey(module=ref.module, symbol=attr)
                attr_rewrite[(ref.alias, attr)] = export_names[key]

        merged_name_rewrite = {**external_name_rewrite, **name_rewrite}
        for _, node in _ordered_nodes(info, selected):
            lines.append(
                _render_node(
                    node,
                    name_rewrite=merged_name_rewrite,
                    attr_rewrite=attr_rewrite,
                )
            )
            lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def export_builtin_tool_source(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "export_builtin_tool_source"
    try:
        params = ExportBuiltinToolSourceInput(**payload)
        output_root = resolve_workspace_path(params.output_root, must_exist=False)
        function_path = output_root / "function.py"
        dependencies_path = output_root / "dependencies.py"
        if not params.overwrite and (function_path.exists() or dependencies_path.exists()):
            _tool_error(
                tool_name,
                f"{tool_name} refused to overwrite existing files under {workspace_relpath(output_root)}.",
                data={"output_root_rel": workspace_relpath(output_root)},
                error_code="output_exists",
            )
        output_root.mkdir(parents=True, exist_ok=True)

        from catmaster.tools.registry import get_tool_registry

        registry = get_tool_registry()
        canonical_name = registry._canonical_tool_name(params.tool_name)
        if canonical_name not in registry.tools:
            _tool_error(
                tool_name,
                f"{tool_name} could not find registered tool {params.tool_name!r}.",
                data={"requested_tool_name": params.tool_name},
                error_code="unknown_tool",
            )

        func = registry.get_tool_function(canonical_name)
        input_model = registry.get_tool_info(canonical_name)["input_model"]
        func_file = Path(inspect.getsourcefile(func) or "").resolve()
        if not func_file.exists():
            _tool_error(
                tool_name,
                f"{tool_name} could not resolve source file for {canonical_name}.",
                data={"tool_name": canonical_name},
                error_code="missing_source_file",
            )
        root_module = _module_name_for_file(func_file)
        root_info = _load_module_info(root_module)
        root_symbols = _local_symbol_closure(root_info, {func.__name__, input_model.__name__})

        selected_by_module: dict[str, set[str]] = {}
        module_order: list[str] = []
        visited: set[_SymbolKey] = set()

        def _collect_dependency(key: _SymbolKey) -> None:
            if key in visited or key.module == root_module:
                return
            visited.add(key)
            info = _load_module_info(key.module)
            local_selected = _local_symbol_closure(info, {key.symbol})
            current = selected_by_module.setdefault(key.module, set())
            new_symbols = local_selected - current
            current.update(local_selected)
            for symbol in new_symbols:
                refs = _refs_for_symbol(info, symbol)
                for ref in refs.direct_internal_imports:
                    if ref.symbol:
                        _collect_dependency(_resolve_internal_symbol(ref.module, ref.symbol))
                for ref, attr in refs.module_attr_internal_imports:
                    _collect_dependency(_resolve_internal_symbol(ref.module, attr))
            if key.module not in module_order:
                module_order.append(key.module)

        for dependency in sorted(_selected_external_refs(root_info, root_symbols), key=lambda item: (item.module, item.symbol)):
            _collect_dependency(dependency)

        dependency_keys = [
            _SymbolKey(module=module_name, symbol=symbol)
            for module_name in module_order
            for symbol in sorted(selected_by_module.get(module_name) or set())
        ]
        export_names = _stable_export_names(dependency_keys)

        dependency_imports: list[tuple[str, str]] = []
        seen_local_imports: set[str] = set()
        for symbol in sorted(root_symbols):
            refs = _refs_for_symbol(root_info, symbol)
            for ref in refs.direct_internal_imports:
                if not ref.symbol:
                    continue
                key = _resolve_internal_symbol(ref.module, ref.symbol)
                local_name = ref.alias
                if local_name in seen_local_imports:
                    continue
                dependency_imports.append((export_names[key], local_name))
                seen_local_imports.add(local_name)
            for ref, attr in refs.module_attr_internal_imports:
                key = _resolve_internal_symbol(ref.module, attr)
                local_name = f"{ref.alias}__{attr}"
                if local_name in seen_local_imports:
                    continue
                dependency_imports.append((export_names[key], local_name))
                seen_local_imports.add(local_name)

        function_text = _render_function_file(
            info=root_info,
            selected_symbols=root_symbols,
            dependency_imports=dependency_imports,
        )
        dependencies_text = _render_dependencies_file(
            module_order=module_order,
            selected_by_module=selected_by_module,
            export_names=export_names,
        )
        function_path.write_text(function_text, encoding="utf-8")
        dependencies_path.write_text(dependencies_text, encoding="utf-8")

        data = {
            "tool_name": canonical_name,
            "output_root_rel": workspace_relpath(output_root),
            "function_path": workspace_relpath(function_path),
            "dependencies_path": workspace_relpath(dependencies_path),
            "root_source_module": root_module,
            "root_source_path": workspace_relpath(func_file),
            "dependency_modules": module_order,
        }
        content = (
            f"Exported builtin tool source for {canonical_name}.\n"
            f"function.py: {data['function_path']}\n"
            f"dependencies.py: {data['dependencies_path']}\n"
            f"dependency_modules: {len(module_order)}"
        )
        return content, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            tool_name,
            f"{tool_name} failed: {exc}",
            data={"requested_tool_name": payload.get("tool_name"), "output_root": payload.get("output_root")},
            error_code="export_builtin_tool_source_failed",
        )


__all__ = ["ExportBuiltinToolSourceInput", "export_builtin_tool_source"]
