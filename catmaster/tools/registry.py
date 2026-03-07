"""
Tool registry that maps tool names to their functions and Pydantic input models.
"""
from __future__ import annotations

from pathlib import Path
from typing import Awaitable, Dict, Any, Callable, Optional
from uuid import uuid4
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from langchain.tools import ToolRuntime

from catmaster.runtime.tool_output_adapter import adapt_tool_return
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import workspace_root, workspace_scope


class ToolRegistry:
    """Simple tool registry mapping names to functions and their input models."""

    def __init__(self, register_all_tools: bool = True):
        self.tools = {}
        if register_all_tools:
            self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available tools"""
        
        # Geometry/Input tools
        from catmaster.tools.geometry_inputs import (
            create_molecule_from_smiles,
            vasp_relax_prepare,
            vasp_sp_prepare,
            build_slab,
            fix_atoms_by_layers,
            fix_atoms_by_height,
            supercell,
            enumerate_adsorption_sites,
            place_adsorbate,
            generate_batch_adsorption_structures,
            make_neb_geometry,
            make_neb_incar,
        )
        from catmaster.tools.geometry_inputs import (
            MoleculeFromSmilesInput,
            VaspRelaxPrepareInput,
            VaspSPPrepareInput,
            SlabBuildInput,
            FixAtomsByLayersInput,
            FixAtomsByHeightInput,
            SupercellInput,
            EnumerateAdsorptionSitesInput,
            PlaceAdsorbateInput,
            GenerateBatchAdsorptionStructuresInput,
            MakeNebGeometryInput,
            MakeNebIncarInput,
        )
        
        # Execution tools  
        from catmaster.tools.execution import mace_relax_batch, mace_sp_batch, vasp_execute_batch
        from catmaster.tools.execution import MaceRelaxBatchInput, MaceSPBatchInput, VaspExecuteBatchInput
        from catmaster.tools.analysis import (
            analyze_images,
            render_structure_views,
            AnalyzeImagesInput,
            RenderStructureViewsInput,
        )
        from catmaster.runtime.literature import (
            run_literature_research,
            RunLiteratureResearchInput,
        )

        # File management tools
        from catmaster.tools.misc.bash_exec import bash_exec, BashExecInput

        # Retrieval tools
        from catmaster.tools.retrieval.matdb import (
            mp_search_materials,
            mp_download_structure,
            MPSearchMaterialsInput,
            MPDownloadStructureInput,
        )

        # Memory patch
        from catmaster.tools.misc.memory_patch_apply import (
            apply_aider_edits,
            ApplyAiderEditsInput,
        )
        from catmaster.tools.misc.memory import (
            write_note,
            MemoryNoteInput,
        )
        
        # Register each tool with its Pydantic schema
        self.register_tool("create_molecule_from_smiles", create_molecule_from_smiles, MoleculeFromSmilesInput)
        self.register_tool("mace_relax_batch", mace_relax_batch, MaceRelaxBatchInput)
        self.register_tool("mace_sp_batch", mace_sp_batch, MaceSPBatchInput)
        self.register_tool("vasp_relax_prepare", vasp_relax_prepare, VaspRelaxPrepareInput)
        self.register_tool("vasp_sp_prepare", vasp_sp_prepare, VaspSPPrepareInput)
        self.register_tool("build_slab", build_slab, SlabBuildInput)
        self.register_tool("fix_atoms_by_layers", fix_atoms_by_layers, FixAtomsByLayersInput)
        self.register_tool("fix_atoms_by_height", fix_atoms_by_height, FixAtomsByHeightInput)
        self.register_tool("supercell", supercell, SupercellInput)
        self.register_tool("enumerate_adsorption_sites", enumerate_adsorption_sites, EnumerateAdsorptionSitesInput)
        self.register_tool("place_adsorbate", place_adsorbate, PlaceAdsorbateInput)
        self.register_tool("generate_batch_adsorption_structures", generate_batch_adsorption_structures, GenerateBatchAdsorptionStructuresInput)
        self.register_tool("make_neb_geometry", make_neb_geometry, MakeNebGeometryInput)
        self.register_tool("make_neb_incar", make_neb_incar, MakeNebIncarInput)
        self.register_tool("vasp_execute_batch", vasp_execute_batch, VaspExecuteBatchInput)
        self.register_tool("mp_search_materials", mp_search_materials, MPSearchMaterialsInput)
        self.register_tool("mp_download_structure", mp_download_structure, MPDownloadStructureInput)
        self.register_tool("render_structure_views", render_structure_views, RenderStructureViewsInput)
        self.register_tool("analyze_images", analyze_images, AnalyzeImagesInput)
        self.register_tool("run_literature_research", run_literature_research, RunLiteratureResearchInput)
        self.register_tool("bash_exec", bash_exec, BashExecInput)
        self.register_tool("apply_aider_edits", apply_aider_edits, ApplyAiderEditsInput)
        self.register_tool("write_note", write_note, MemoryNoteInput)
    
    def register_tool(
        self, 
        name: str, 
        func: Callable | None,
        input_model: type[BaseModel],
        *,
        coroutine: Callable[..., Awaitable[Any]] | None = None,
    ):
        """Register a tool with sync/async callables and its input model."""
        if func is None and coroutine is None:
            raise ValueError(f"Tool {name!r} must provide at least one callable.")
        self.tools[name] = {
            "function": func,
            "coroutine": coroutine,
            "input_model": input_model,
            "parameters": input_model.model_json_schema()
        }

    def as_openai_tools(
        self,
        *,
        allowlist: list[str] | None = None,
        strict: bool = False,
    ) -> list[dict]:
        tools: list[dict] = []
        names = allowlist if allowlist is not None else list(self.tools.keys())
        for name in names:
            info = self.tools.get(name)
            if not info:
                continue
            model = info["input_model"]
            description = (model.__doc__ or f"Input for {name}").strip()
            schema = info.get("parameters") or model.model_json_schema()
            tools.append({
                "type": "function",
                "name": name,
                "description": description,
                "parameters": sanitize_json_schema(schema),
                "strict": strict,
            })
        return tools
    
    def get_tool_info(self, name: str) -> Dict[str, Any]:
        """Get tool information by name."""
        return self.tools.get(name, {})
    
    def get_tool_function(self, name: str) -> Callable:
        """Get tool function by name."""
        tool_info = self.tools.get(name)
        if tool_info and tool_info.get("function") is not None:
            return tool_info["function"]
        if tool_info and tool_info.get("coroutine") is not None:
            raise ValueError(f"Tool {name} is async-only and has no sync function.")
        raise ValueError(f"Unknown tool: {name}")
    
    def list_tools(self) -> Dict[str, Dict[str, Any]]:
        """List all registered tools with their schemas."""
        return {
            name: {
                "parameters": info["parameters"]
            }
            for name, info in self.tools.items()
        }
    
    def get_tool_descriptions_for_llm(self, allowlist: list[str] | None = None) -> str:
        """Get tool descriptions formatted for LLM consumption."""
        descriptions = []
        names = allowlist if allowlist is not None else list(self.tools.keys())
        for name in names:
            info = self.tools.get(name)
            if not info:
                continue
            model = info["input_model"]
            doc = model.__doc__ or f"Input for {name}"
            params = []
            for field_name, field_info in model.model_fields.items():
                desc = field_info.description or "No description"
                params.append(f"  - {field_name}: {desc}")

            descriptions.append(f"{name} : {doc}\n" + "\n".join(params))

        return "\n\n".join(descriptions)

    def get_short_tool_descriptions_for_llm(self, allowlist: list[str] | None = None) -> str:
        """Get short tool descriptions (name + docstring only) for LLM planning."""
        descriptions = []
        names = allowlist if allowlist is not None else list(self.tools.keys())
        for name in names:
            info = self.tools.get(name)
            if not info:
                continue
            model = info["input_model"]
            doc = model.__doc__ or f"Input for {name}"
            descriptions.append(f"{name} : {doc}")

        return "\n\n".join(descriptions)

    def as_langchain_tools(
        self,
        *,
        allowlist: Optional[list[str]] = None,
        run_dir: Optional[str] = None,
        workspace: Optional[str] = None,
    ) -> list[StructuredTool]:
        """Convert registered tools to LangChain StructuredTool instances.

        CatMaster tools accept ``payload: dict`` and must return:
        - ``(content, artifact)`` (native), or
        - ``ToolMessage`` (advanced)
        The wrapper maps LangChain keyword arguments (unpacked from the
        Pydantic args_schema) back into the ``payload`` dict the tool expects
        and post-processes tool returns to ``(content, artifact)``.
        """
        tools: list[StructuredTool] = []
        names = allowlist if allowlist is not None else list(self.tools.keys())
        for name in names:
            info = self.tools.get(name)
            if not info:
                continue
            tools.append(_make_langchain_tool(
                name=name,
                func=info["function"],
                coroutine=info.get("coroutine"),
                input_model=info["input_model"],
                run_dir=run_dir,
                workspace=workspace,
            ))
        return tools


def _make_langchain_tool(
    name: str,
    func: Callable | None,
    input_model: type[BaseModel],
    coroutine: Callable[..., Awaitable[Any]] | None = None,
    run_dir: Optional[str] = None,
    workspace: Optional[str] = None,
) -> StructuredTool:
    """Wrap CatMaster ``func/coroutine(payload)`` as a LangChain StructuredTool."""
    if func is None and coroutine is None:
        raise ValueError(f"Tool {name!r} requires func or coroutine.")

    resolved_workspace = workspace
    resolved_run_dir = (run_dir or "").strip()

    def _runtime_scope(runtime: ToolRuntime | None) -> tuple[str, str]:
        toolcall_key = str(getattr(runtime, "tool_call_id", "") or "").strip()
        if not toolcall_key:
            toolcall_key = f"{name}_{uuid4().hex[:8]}"

        runtime_run_dir = resolved_run_dir
        runtime_context = getattr(runtime, "context", None)
        if not runtime_run_dir and isinstance(runtime_context, dict):
            runtime_run_dir = str(runtime_context.get("run_dir") or "").strip()
        return toolcall_key, runtime_run_dir

    def _workspace_files_root() -> Path:
        if resolved_workspace:
            return workspace_root(resolved_workspace)
        return workspace_root()

    def _wrapper(runtime: ToolRuntime | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if func is None:
            raise NotImplementedError(f"Tool {name} does not support sync invocation.")
        toolcall_key, runtime_run_dir = _runtime_scope(runtime)

        with toolcall_context(toolcall_key, run_dir=runtime_run_dir):
            if resolved_workspace:
                with workspace_scope(resolved_workspace):
                    result = func(kwargs)
            else:
                result = func(kwargs)

        return adapt_tool_return(
            tool_name=name,
            raw_result=result,
            tool_args=kwargs,
            workspace_files_root=_workspace_files_root(),
        )

    async def _awrapper(runtime: ToolRuntime | None = None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
        if coroutine is None:
            raise NotImplementedError(f"Tool {name} does not support async invocation.")
        toolcall_key, runtime_run_dir = _runtime_scope(runtime)

        with toolcall_context(toolcall_key, run_dir=runtime_run_dir):
            if resolved_workspace:
                with workspace_scope(resolved_workspace):
                    result = await coroutine(kwargs)
            else:
                result = await coroutine(kwargs)

        return adapt_tool_return(
            tool_name=name,
            raw_result=result,
            tool_args=kwargs,
            workspace_files_root=_workspace_files_root(),
        )

    if func is not None:
        _wrapper.__name__ = name
    if coroutine is not None:
        _awrapper.__name__ = f"{name}_async"
    description = (input_model.__doc__ or f"Input for {name}").strip()
    args_schema = sanitize_json_schema(input_model.model_json_schema())

    return StructuredTool.from_function(
        func=_wrapper if func is not None else None,
        coroutine=_awrapper if coroutine is not None else None,
        name=name,
        description=description,
        args_schema=args_schema,
        infer_schema=False,
        response_format="content_and_artifact",
    )


def sanitize_json_schema(schema: dict) -> dict:
    if isinstance(schema, list):
        return [sanitize_json_schema(item) for item in schema]
    if not isinstance(schema, dict):
        return schema

    cleaned: dict = {}
    for key, value in schema.items():
        if isinstance(value, dict):
            cleaned[key] = sanitize_json_schema(value)
        elif isinstance(value, list):
            cleaned[key] = [sanitize_json_schema(item) for item in value]
        else:
            cleaned[key] = value

    if "properties" in cleaned and isinstance(cleaned["properties"], dict):
        cleaned["properties"] = {
            prop: sanitize_json_schema(prop_schema)
            for prop, prop_schema in cleaned["properties"].items()
        }
    for key in ("anyOf", "allOf", "oneOf"):
        if key in cleaned and isinstance(cleaned[key], list):
            cleaned[key] = [sanitize_json_schema(item) for item in cleaned[key]]
    if "items" in cleaned:
        cleaned["items"] = sanitize_json_schema(cleaned["items"])
    if "prefixItems" in cleaned and isinstance(cleaned["prefixItems"], list):
        cleaned["prefixItems"] = [sanitize_json_schema(item) for item in cleaned["prefixItems"]]

    schema_type = cleaned.get("type")
    if schema_type == "object" or (isinstance(schema_type, list) and "object" in schema_type):
        cleaned.setdefault("additionalProperties", False)

    return cleaned


# Singleton instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the singleton tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
