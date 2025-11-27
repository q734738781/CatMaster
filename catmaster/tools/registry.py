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
        from catmaster.tools.geometry_inputs import create_molecule, vasp_prepare
        from catmaster.tools.geometry_inputs import MoleculeCreateInput, VaspPrepareInput
        
        # Execution tools  
        from catmaster.tools.execution import mace_relax, vasp_execute
        from catmaster.tools.execution import MaceRelaxInput, VaspExecuteInput
        
        # Analysis tools
        from catmaster.tools.analysis import vasp_summarize, VaspSummarizeInput
        
        # Register each tool with its Pydantic schema
        self.register_tool("create_molecule", create_molecule, MoleculeCreateInput)
        self.register_tool("mace_relax", mace_relax, MaceRelaxInput)
        self.register_tool("vasp_prepare", vasp_prepare, VaspPrepareInput)
        self.register_tool("vasp_execute", vasp_execute, VaspExecuteInput)
        self.register_tool("vasp_summarize", vasp_summarize, VaspSummarizeInput)
    
    def register_tool(
        self, 
        name: str, 
        func: Callable,
        input_model: type[BaseModel],
        device: str = "local"
    ):
        """Register a tool with its function and input model."""
        self.tools[name] = {
            "function": func,
            "input_model": input_model,
            "device": device,
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
                "device": info["device"],
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
            
            descriptions.append(f"{name} ({info['device']}): {doc}\n" + "\n".join(params))
        
        return "\n\n".join(descriptions)


# Singleton instance
_registry = None

def get_tool_registry() -> ToolRegistry:
    """Get the singleton tool registry instance."""
    global _registry
    if _registry is None:
        _registry = ToolRegistry()
    return _registry