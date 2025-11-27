#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Execution Agent: Responsible for actually executing computational tasks.
"""
from __future__ import annotations

import json
import time
import logging
from typing import Dict, List, Any, Optional
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

# Import implemented tools
from catmaster.tools.execution.llm_adapter import execute_vasp, execute_mace, execute_job
from catmaster.tools.geometry_inputs import vasp_prepare
from catmaster.tools.analysis import vasp_summarize


class ExecutionResult(BaseModel):
    """Execution result model"""
    task_name: str
    success: bool
    output: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    execution_time: float = 0.0
    device_used: Optional[str] = None


class ExecutionAgent:
    """
    Task Execution Agent: Responsible for calling specific computational tools to execute tasks.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        
        # Get tool registry
        from catmaster.tools.registry import get_tool_registry
        self.tool_registry = get_tool_registry()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a computational task execution expert. Your responsibilities are:

1. Execute computational tools with provided parameters
2. Handle file paths and data dependencies
3. Monitor execution progress
4. Handle execution errors appropriately"""),
            ("human", "{input}")
        ])
        
        # Working directory - use current directory (workflow should set this)
        self.work_dir = Path(".")
        self.work_dir = self.work_dir.resolve()  # Get absolute path
    
    def execute_task(self, task: Dict[str, Any]) -> ExecutionResult:
        """Execute a single task"""
        start_time = time.time()
        task_name = task["name"]
        method = task["method"]
        params = task.get("params", {})
        device = task.get("device", "local")
        
        logger.info(f"ExecutionAgent.execute_task()")
        logger.info(f"  Task: {task_name}")
        logger.info(f"  Method: {method}")
        logger.info(f"  Device: {device}")
        logger.info(f"  Params: {params}")
        
        try:
            # Use tool registry to call the appropriate function
            logger.info(f"  Looking up tool: {method}")
            
            if method in self.tool_registry.tools:
                # Direct tool execution
                tool_func = self.tool_registry.get_tool_function(method)
                logger.info(f"  Calling {method} from registry")
                
                # Prepare payload for tool
                if method == "create_molecule":
                    # Ensure output_path is set and points to workspace
                    if "output_path" not in params:
                        params["output_path"] = str(self.work_dir / f"{params.get('molecule', 'molecule')}_initial.vasp")
                    else:
                        # Ensure path is in workspace directory
                        output_path = Path(params["output_path"])
                        if not output_path.is_absolute():
                            # Make it relative to workspace
                            params["output_path"] = str(self.work_dir / output_path)
                    output = tool_func(params)
                    
                elif method == "mace_relax":
                    # Call mace_relax tool directly - it handles work_dir properly
                    output = tool_func(params)
                    
                elif method == "vasp_prepare":
                    # Use prepare_vasp wrapper
                    output = self._prepare_vasp(**params)
                    
                elif method == "vasp_execute":
                    # Use execute_vasp wrapper
                    output = self._execute_vasp(**params)
                    
                elif method == "vasp_summarize":
                    # vasp_summarize expects a dictionary with work_dir key
                    output = tool_func(params)
                    
                else:
                    # Generic tool execution
                    output = tool_func(params)
            else:
                raise ValueError(f"Unknown method: {method}. Available: {list(self.tool_registry.tools.keys())}")
            
            execution_time = time.time() - start_time
            
            logger.info(f"  Method completed in {execution_time:.2f}s")
            logger.info(f"  Output type: {type(output).__name__}")
            
            # Handle standardized output format
            if isinstance(output, dict) and "status" in output:
                logger.info(f"  Status: {output.get('status')}")
                
                # Track files
                if "input_files" in output:
                    logger.info(f"  Input files: {len(output['input_files'])}")
                    for file_info in output["input_files"]:
                        if isinstance(file_info, dict):
                            logger.info(f"    - {file_info.get('path', file_info)} ({file_info.get('type', 'unknown')})")
                
                if "output_files" in output:
                    logger.info(f"  Output files: {len(output['output_files'])}")
                    for file_info in output["output_files"]:
                        if isinstance(file_info, dict):
                            exists = file_info.get('exists', False)
                            status = "✓" if exists else "✗"
                            logger.info(f"    {status} {file_info.get('path', file_info)} ({file_info.get('type', 'unknown')})")
                
                if "warnings" in output and output["warnings"]:
                    logger.warning(f"  Warnings:")
                    for warning in output["warnings"]:
                        logger.warning(f"    - {warning}")
                
                # Extract data for backward compatibility
                task_output = output.get("data", output)
                success = output.get("status") == "success"
            else:
                # Legacy output format
                if isinstance(output, dict):
                    logger.info(f"  Output keys: {list(output.keys())}")
                task_output = output
                success = True
            
            return ExecutionResult(
                task_name=task_name,
                success=success,
                output=task_output,
                execution_time=execution_time,
                device_used=device
            )
            
        except Exception as e:
            execution_time = time.time() - start_time
            logger.error(f"  Method failed after {execution_time:.2f}s")
            logger.error(f"  Error: {e}")
            import traceback
            logger.error(traceback.format_exc())
            
            return ExecutionResult(
                task_name=task_name,
                success=False,
                error=str(e),
                execution_time=execution_time,
                device_used=device
            )
    
    def _execute_mace(self, structure_file: str = None, 
                     fmax: float = 0.05, 
                     steps: int = 500, **kwargs) -> Dict[str, Any]:
        """Execute MACE optimization"""
        # Use provided structure file or find latest
        if structure_file:
            structure_path = Path(structure_file)
            if not structure_path.is_absolute():
                structure_path = self.work_dir / structure_path
            if not structure_path.exists():
                raise FileNotFoundError(f"Structure file not found: {structure_path}")
        else:
            structure_path = self._find_latest_structure()
        
        # Create MACE working directory
        mace_dir = self.work_dir / "mace_opt"
        mace_dir.mkdir(exist_ok=True)
        
        # Copy structure file
        import shutil
        dest = mace_dir / "POSCAR"
        shutil.copy(structure_path, dest)
        
        # Also copy meta file if exists
        # Handle different extensions (.vasp, .cif, etc)
        meta_file = structure_path.with_suffix('.meta.json')
        if meta_file.exists():
            shutil.copy(meta_file, mace_dir / "POSCAR.meta.json")
        
        # Execute MACE optimization
        result = execute_mace(
            input_structure=str(dest),
            work_dir=str(mace_dir),
            project="catmaster",
            worker="gpu-worker",
            remote_tmp_base="/ssd/chenhh/tmp/jfr_uploads",
            fmax=fmax,
            steps=steps,
            wait=True,
            download_results=True,
            download_dir=str(self.work_dir / "mace_results")
        )
        
        # Process results
        if result.get("output") and result["output"].get("summary"):
            summary = result["output"]["summary"]
            
            # Find downloaded CONTCAR
            download_path = result.get("download_path")
            if download_path:
                contcar = Path(download_path) / "CONTCAR"
                if contcar.exists():
                    # Copy to working directory
                    optimized_file = self.work_dir / "O2_mace_optimized.vasp"
                    shutil.copy(contcar, optimized_file)
                    result["relaxed_vasp"] = str(optimized_file)
                    
                    # Read optimized bond length
                    from ase.io import read
                    atoms = read(str(contcar))
                    if len(atoms) == 2:
                        bond_length = atoms.get_distance(0, 1)
                        result["optimized_bond_length"] = bond_length
        
        return result
    
    def _prepare_vasp(self, calc_type: str = "gas", k_product: int = 1,
                     user_incar_settings: Dict[str, Any] = None, 
                     input_path: str = None, 
                     output_root: str = None, **kwargs) -> Dict[str, Any]:
        """Prepare VASP input files"""
        logger.info(f"_prepare_vasp() called")
        logger.info(f"  calc_type: {calc_type}")
        logger.info(f"  k_product: {k_product}")
        logger.info(f"  input_path: {input_path}")
        logger.info(f"  user_incar_settings: {user_incar_settings}")
        logger.info(f"  kwargs: {kwargs}")
        
        if not input_path:
            logger.info(f"  Finding latest structure...")
            input_path = str(self._find_latest_structure())
            logger.info(f"  Found: {input_path}")
        
        # VASP output directory - use output_root if provided
        if output_root:
            vasp_input_dir = Path(output_root)
        else:
            vasp_input_dir = self.work_dir / "vasp_inputs"
            logger.info(f"  Output dir: {vasp_input_dir}")
        
        # Prepare VASP input
        payload = {
            "input_path": input_path,
            "output_root": str(vasp_input_dir),
            "calc_type": calc_type,
            "k_product": k_product,
            "user_incar_settings": user_incar_settings or {}
        }
        
        logger.info(f"  Calling vasp_prepare() with payload:")
        import json
        logger.info(json.dumps(payload, indent=10))
        
        result = vasp_prepare(payload)
        
        logger.info(f"  vasp_prepare() returned: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        return result
    
    def _execute_vasp(self, input_dir: str = None, 
                     **kwargs) -> Dict[str, Any]:
        """Execute VASP calculation"""
        logger.info(f"_execute_vasp() called")
        logger.info(f"  input_dir: {input_dir}")
        logger.info(f"  kwargs: {kwargs}")
        
        # Handle placeholder or find latest VASP input
        if not input_dir or "<" in input_dir:
            logger.info(f"  Finding latest VASP input directory...")
            # First check in the default vasp_inputs directory
            vasp_inputs_dir = self.work_dir / "vasp_inputs"
            if vasp_inputs_dir.exists():
                vasp_dirs = list(vasp_inputs_dir.glob("*"))
                if vasp_dirs:
                    # Get most recent directory
                    vasp_dirs.sort(key=lambda p: p.stat().st_mtime, reverse=True)
                    input_dir = str(vasp_dirs[0])
                    logger.info(f"  Found {len(vasp_dirs)} VASP input directories")
                    logger.info(f"  Using most recent: {input_dir}")
                else:
                    raise ValueError("No VASP input directories found in vasp_inputs directory")
            else:
                raise ValueError("No VASP input directories found - vasp_inputs directory does not exist")
        else:
            # Valid input_dir provided, use it as is
            logger.info(f"  Using provided input_dir: {input_dir}")
        
        # Check what files are in the input directory
        input_path = Path(input_dir)
        if input_path.exists():
            files = list(input_path.glob("*"))
            logger.info(f"  Input directory contains {len(files)} files:")
            for f in sorted(files):
                logger.info(f"    - {f.name} ({f.stat().st_size} bytes)")
        else:
            logger.error(f"  Input directory does not exist: {input_dir}")
        
        # Get download_dir from kwargs if provided
        download_dir = kwargs.get("download_dir", str(self.work_dir / "vasp_results"))
        
        # Execute VASP using the standard catmaster module approach
        logger.info(f"  Calling execute_vasp()...")
        result = execute_vasp(
            input_dir=input_dir,
            project=kwargs.get("project", "catmaster"),
            worker=kwargs.get("worker", "cpu-worker"),
            remote_tmp_base=kwargs.get("remote_tmp_base", "/public/home/chenhh/tmp/jfr_uploads"),
            wait=kwargs.get("wait", True),
            timeout_s=kwargs.get("timeout_s", 36000),  # Default to 10 hours
            download_results=kwargs.get("download_results", True),
            download_dir=download_dir
        )
        
        logger.info(f"  execute_vasp() returned")
        logger.info(f"    Keys: {list(result.keys()) if isinstance(result, dict) else type(result)}")
        
        # Analyze results
        if result.get("download_path"):
            try:
                summary = vasp_summarize(result["download_path"])
                result["vasp_summary"] = summary
                
                # Extract key information
                if summary.get("converged"):
                    result["final_energy_eV"] = summary.get("final_energy", None)
                    
                    # If there's an optimized structure, read bond length
                    if summary.get("final_structure"):
                        from ase.io import read
                        atoms = read(summary["final_structure"])
                        if len(atoms) == 2:
                            result["final_bond_length"] = atoms.get_distance(0, 1)
            except Exception as e:
                result["analysis_error"] = str(e)
        
        return result
    
    def _find_latest_structure(self) -> Path:
        """Find the latest structure file"""
        logger.info(f"_find_latest_structure() called")
        logger.info(f"  work_dir: {self.work_dir}")
        logger.info(f"  work_dir exists: {self.work_dir.exists()}")
        
        # Supported structure file extensions
        structure_extensions = ['.vasp', '.cif', '.xyz', '.POSCAR', '.CONTCAR']
        
        # Priority: optimized > initial (check multiple extensions)
        logger.info(f"  Checking priority candidates:")
        for base_name in ["O2_mace_optimized", "O2_initial", "o2_init"]:
            for ext in structure_extensions:
                candidate = self.work_dir / f"{base_name}{ext}"
            exists = candidate.exists()
            logger.info(f"    {candidate.name}: {exists}")
            if exists:
                logger.info(f"  Found: {candidate}")
                return candidate
        
        # Find any structure file with supported extensions
        logger.info(f"  Searching for any structure files...")
        all_structure_files = []
        for ext in structure_extensions:
            files = list(self.work_dir.glob(f"*{ext}"))
            all_structure_files.extend(files)
            logger.info(f"  Found {len(files)} {ext} files")
        
        if all_structure_files:
            # Sort by modification time, newest first
            all_structure_files.sort(key=lambda p: p.stat().st_mtime, reverse=True)
            selected = all_structure_files[0]
            logger.info(f"  Using most recent: {selected}")
            return selected
        
        logger.error(f"  No structure file found!")
        raise FileNotFoundError(f"No structure file found in workspace: {self.work_dir}")
