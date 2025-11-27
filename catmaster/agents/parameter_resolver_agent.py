#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Parameter Resolver Agent: Responsible for resolving parameters dynamically based on context and previous results.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, Any, List, Optional
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from langchain_core.output_parsers import PydanticOutputParser
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class ParameterResolution(BaseModel):
    """Parameter resolution result"""
    task_name: str = Field(..., description="Name of the task being resolved")
    resolved_params: Dict[str, Any] = Field(..., description="All resolved parameters as key-value pairs")
    warnings: List[str] = Field(default_factory=list, description="Any warnings during resolution")


class ParameterResolverAgent:
    """
    Parameter Resolver Agent: Resolves task parameters based on context and previous results.
    
    Converts high-level parameter descriptions from workflow planning into concrete values
    that can be passed to tool functions.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        
        # Get tool registry for parameter schemas
        from catmaster.tools.registry import get_tool_registry
        self.tool_registry = get_tool_registry()
        
        # Create output parser
        self.output_parser = PydanticOutputParser(pydantic_object=ParameterResolution)
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a parameter resolution expert. Convert high-level parameter descriptions into concrete values.

CRITICAL RULES:
1. Replace ALL descriptive strings with actual values
2. For file paths, use actual paths from previous outputs or available files
3. For numeric parameters, use appropriate values based on the calculation type
4. For dictionaries, provide complete key-value pairs

{format_instructions}"""),
            ("human", """Task: {task_name}
Method: {method}

PARAMETER REQUIREMENTS (from tool):
{param_requirements}

PARAMETER DESCRIPTIONS (from plan):
{param_descriptions}

PREVIOUS TASK OUTPUTS:
{previous_results}

AVAILABLE FILES IN {working_dir}:
{available_files}

Resolve all parameters to concrete values.""")
        ])
    
    def resolve_parameters(
        self, 
        task: Dict[str, Any],
        previous_results: Dict[str, Any],
        working_dir: Path
    ) -> ParameterResolution:
        """
        Resolve parameters for a task based on context.
        
        Args:
            task: Current task with name, method, and parameter descriptions
            previous_results: Results from all previous tasks
            working_dir: Current working directory
            
        Returns:
            ParameterResolution with resolved parameters
        """
        task_name = task["name"]
        method = task["method"]
        param_descriptions = task.get("params", {})
        
        # Get tool parameter schema
        tool_info = self.tool_registry.get_tool_info(method)
        if not tool_info:
            raise ValueError(f"Unknown tool method: {method}")
        
        param_requirements = tool_info.get("parameters", {})
        
        # Scan available files
        available_files = self._scan_available_files(working_dir)
        
        # Format previous results
        formatted_results = self._format_previous_results(previous_results)
        
        # Get format instructions
        format_instructions = self.output_parser.get_format_instructions()
        
        # Build prompt
        messages = self.prompt.format_messages(
            task_name=task_name,
            method=method,
            param_requirements=json.dumps(param_requirements, indent=2),
            param_descriptions=json.dumps(param_descriptions, indent=2),
            previous_results=formatted_results,
            working_dir=str(working_dir),
            available_files=json.dumps(available_files, indent=2),
            format_instructions=format_instructions
        )
        
        # Call LLM
        logger.info("Calling LLM to resolve parameters...")
        response = self.llm.invoke(messages)
        
        # Parse response
        try:
            result = self.output_parser.parse(response.content)
            logger.info("Successfully parsed LLM response")
            return result
        except Exception as e:
            logger.error(f"Failed to parse LLM response: {str(e)}")
            raise ValueError(f"Failed to resolve parameters: {str(e)}")
    
    def _scan_available_files(self, working_dir: Path) -> List[str]:
        """Scan working directory for available files."""
        files = []
        if working_dir.exists():
            for path in working_dir.rglob("*"):
                if path.is_file():
                    files.append(str(path.relative_to(working_dir)))
        return sorted(files)
    
    def _format_previous_results(self, previous_results: Dict[str, Any]) -> str:
        """Format previous task results for LLM consumption."""
        formatted = []
        
        for task_name, result in previous_results.items():
            if isinstance(result, dict):
                summary = {
                    "task": task_name,
                    "status": result.get("status", "unknown"),
                    "data": result.get("data", {})
                }
                formatted.append(summary)
        
        return json.dumps(formatted, indent=2)