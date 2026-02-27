"""
Tool registry that maps tool names to their functions and Pydantic input models.
"""
from __future__ import annotations

import json
from typing import Dict, Any, Callable, Optional
from pydantic import BaseModel
from langchain_core.tools import StructuredTool
from catmaster.runtime.tool_result_normalizer import normalize_tool_result


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

        # File management tools
        from catmaster.tools.misc.bash_exec import bash_exec, BashExecInput

        # Retrieval tools
        from catmaster.tools.retrieval.matdb import (
            mp_search_materials,
            mp_download_structure,
            MPSearchMaterialsInput,
            MPDownloadStructureInput,
        )

        # Memory/notes
        from catmaster.tools.misc import memory
        from catmaster.tools.misc.memory_patch_apply import (
            memory_apply_aider_edits,
            MemoryApplyAiderEditsInput,
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
        self.register_tool("bash_exec", bash_exec, BashExecInput)
        self.register_tool("write_note", memory.write_note, memory.MemoryNoteInput)
        self.register_tool("memory_apply_aider_edits", memory_apply_aider_edits, MemoryApplyAiderEditsInput)
    
    def register_tool(
        self, 
        name: str, 
        func: Callable,
        input_model: type[BaseModel],
    ):
        """Register a tool with its function and input model."""
        self.tools[name] = {
            "function": func,
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
        if tool_info:
            return tool_info["function"]
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
    ) -> list[StructuredTool]:
        """Convert registered tools to LangChain StructuredTool instances.

        Each CatMaster tool has signature ``func(payload: dict) -> dict``.
        The wrapper maps LangChain keyword arguments (unpacked from the
        Pydantic args_schema) back into the ``payload`` dict the tool expects.
        Tool output dicts are JSON-serialised so LangChain receives a string
        (required by ToolMessage).
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
                input_model=info["input_model"],
            ))
        return tools


def _make_langchain_tool(
    name: str,
    func: Callable,
    input_model: type[BaseModel],
) -> StructuredTool:
    """Wrap a CatMaster ``func(payload) -> dict`` tool as a LangChain StructuredTool."""

    def _wrapper(**kwargs: Any) -> str:
        try:
            result = func(kwargs)
        except Exception as exc:
            normalized = normalize_tool_result(
                {
                    "status": "failed",
                    "tool_name": name,
                    "data": {},
                    "error": f"{type(exc).__name__}: {exc}",
                },
                tool_name=name,
                is_control_tool=False,
            )
            return json.dumps(normalized, ensure_ascii=False)

        normalized = normalize_tool_result(
            result,
            tool_name=name,
            is_control_tool=False,
        )
        return json.dumps(normalized, ensure_ascii=False)

    _wrapper.__name__ = name
    description = (input_model.__doc__ or f"Input for {name}").strip()

    return StructuredTool.from_function(
        func=_wrapper,
        name=name,
        description=description,
        args_schema=input_model,
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
