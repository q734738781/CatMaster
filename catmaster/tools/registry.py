"""
Tool registry that maps tool names to their functions and Pydantic input models.
"""
from __future__ import annotations

from typing import Dict, Any, Callable
from pydantic import BaseModel


class ToolRegistry:
    """Simple tool registry mapping names to functions and their input models."""
    
    def __init__(self):
        self.tools = {}
        self._register_all_tools()
    
    def _register_all_tools(self):
        """Register all available tools"""
        
        # Geometry/Input tools
        from catmaster.tools.geometry_inputs import (
            create_molecule_from_smiles,
            relax_prepare,
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
            RelaxPrepareInput,
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
        from catmaster.tools.execution import mace_relax, vasp_execute, mace_relax_batch, vasp_execute_batch
        from catmaster.tools.execution import MaceRelaxInput, VaspExecuteInput, MaceRelaxBatchInput, VaspExecuteBatchInput

        # File management tools
        from catmaster.tools.misc import file_manager
        from catmaster.tools.misc.python_repl import python_exec, PythonExecInput

        # Retrieval tools
        from catmaster.tools.retrieval.matdb import (
            mp_search_materials,
            mp_download_structure,
            MPSearchMaterialsInput,
            MPDownloadStructureInput,
        )

        # Memory/notes
        from catmaster.tools.misc import memory
        
        # Register each tool with its Pydantic schema
        self.register_tool("create_molecule_from_smiles", create_molecule_from_smiles, MoleculeFromSmilesInput)
        self.register_tool("mace_relax", mace_relax, MaceRelaxInput)
        self.register_tool("mace_relax_batch", mace_relax_batch, MaceRelaxBatchInput)
        self.register_tool("relax_prepare", relax_prepare, RelaxPrepareInput)
        self.register_tool("build_slab", build_slab, SlabBuildInput)
        self.register_tool("fix_atoms_by_layers", fix_atoms_by_layers, FixAtomsByLayersInput)
        self.register_tool("fix_atoms_by_height", fix_atoms_by_height, FixAtomsByHeightInput)
        self.register_tool("supercell", supercell, SupercellInput)
        self.register_tool("enumerate_adsorption_sites", enumerate_adsorption_sites, EnumerateAdsorptionSitesInput)
        self.register_tool("place_adsorbate", place_adsorbate, PlaceAdsorbateInput)
        self.register_tool("generate_batch_adsorption_structures", generate_batch_adsorption_structures, GenerateBatchAdsorptionStructuresInput)
        self.register_tool("make_neb_geometry", make_neb_geometry, MakeNebGeometryInput)
        self.register_tool("make_neb_incar", make_neb_incar, MakeNebIncarInput)
        self.register_tool("vasp_execute", vasp_execute, VaspExecuteInput)
        self.register_tool("vasp_execute_batch", vasp_execute_batch, VaspExecuteBatchInput)
        self.register_tool("mp_search_materials", mp_search_materials, MPSearchMaterialsInput)
        self.register_tool("mp_download_structure", mp_download_structure, MPDownloadStructureInput)
        self.register_tool("workspace_list_files", file_manager.workspace_list_files, file_manager.WorkspaceListFilesInput)
        self.register_tool("workspace_read_file", file_manager.workspace_read_file, file_manager.WorkspaceReadFileInput)
        self.register_tool("workspace_write_file", file_manager.workspace_write_file, file_manager.WorkspaceWriteFileInput)
        self.register_tool("workspace_mkdir", file_manager.workspace_mkdir, file_manager.WorkspaceMkdirInput)
        self.register_tool("workspace_copy_files", file_manager.workspace_copy_files, file_manager.WorkspaceCopyFilesInput)
        self.register_tool("workspace_delete", file_manager.workspace_delete, file_manager.WorkspaceDeleteInput)
        self.register_tool("workspace_grep", file_manager.workspace_grep, file_manager.WorkspaceGrepInput)
        self.register_tool("workspace_head", file_manager.workspace_head, file_manager.WorkspaceHeadInput)
        self.register_tool("workspace_tail", file_manager.workspace_tail, file_manager.WorkspaceTailInput)
        self.register_tool("workspace_move_files", file_manager.workspace_move_files, file_manager.WorkspaceMoveFilesInput)
        self.register_tool("python_exec", python_exec, PythonExecInput)
        self.register_tool("write_note", memory.write_note, memory.MemoryNoteInput)
    
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
    
    def get_tool_descriptions_for_llm(self) -> str:
        """Get tool descriptions formatted for LLM consumption."""
        descriptions = []
        for name, info in self.tools.items():
            model = info["input_model"]
            doc = model.__doc__ or f"Input for {name}"
            params = []
            for field_name, field_info in model.model_fields.items():
                desc = field_info.description or "No description"
                params.append(f"  - {field_name}: {desc}")

            descriptions.append(f"{name} : {doc}\n" + "\n".join(params))

        return "\n\n".join(descriptions)

    def get_short_tool_descriptions_for_llm(self) -> str:
        """Get short tool descriptions (name + docstring only) for LLM planning."""
        descriptions = []
        for name, info in self.tools.items():
            model = info["input_model"]
            doc = model.__doc__ or f"Input for {name}"
            descriptions.append(f"{name} : {doc}")

        return "\n\n".join(descriptions)


# Singleton instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the singleton tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry
