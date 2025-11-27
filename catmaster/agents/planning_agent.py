#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Planning Agent: Responsible for decomposing user computational requirements into specific task steps.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class ComputationPlan(BaseModel):
    """Computation plan model"""
    objective: str = Field(..., description="Computation objective description")
    steps: List[Dict[str, Any]] = Field(..., description="List of computation steps")
    resources: Dict[str, str] = Field(default_factory=dict, description="Required resources")
    constraints: Dict[str, Any] = Field(default_factory=dict, description="Constraints")
    estimated_time: Optional[str] = Field(None, description="Estimated time")


class PlanningAgent:
    """
    Planning Agent: Receives user's computational requirements and creates detailed computation plans.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        
        # Get tool registry with schemas
        from catmaster.tools.registry import get_tool_registry
        self.tool_registry = get_tool_registry()
        self.available_tools = self.tool_registry.get_tool_descriptions_for_llm()
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a professional computational chemistry planning expert. Your task is to create high-level strategic plans for computational workflows.

Available Tools:
{tools}

Your role is to:
1. Analyze the user's computational objectives
2. Break down the work into logical steps
3. Select appropriate tools for each step
4. Describe data dependencies between steps (NOT specific parameters)
5. Assign appropriate computational devices

Rules:
- create_molecule: Creates initial molecular structures (local)
- mace_relax: ML-based geometry pre-optimization (gpu-worker)  
- vasp_prepare: Generates VASP input files from structures (local)
- vasp_execute: Runs VASP DFT calculations (cpu-worker)
- vasp_summarize: Extracts results from VASP calculations (local)

IMPORTANT: Do NOT specify concrete parameters like file paths or numeric values.
Instead, describe what each step needs in plain language.

Output your plan as JSON with this structure:
{{
  "objective": "Clear description of the goal",
  "steps": [
    {{
      "name": "descriptive_step_name",
      "description": "What this step accomplishes",
      "method": "tool_method_name",
      "params": {{
        // Use descriptive strings, not concrete values
        // GOOD: "structure_file": "relaxed structure from previous step"
        // BAD: "structure_file": "${{step1.outputs[0].path}}"
        // BAD: "structure_file": "work/structure.vasp"
      }},
      "dependencies": ["list", "of", "previous", "step", "names"],
      "device": "local|gpu-worker|cpu-worker"
    }}
  ],
  "data_flow": "Description of how data flows between steps",
  "rationale": "Why you chose this approach"
}}"""),
            ("human", "{input}")
        ])
    
    def plan(self, user_request: str, context: Optional[Dict[str, Any]] = None) -> ComputationPlan:
        """
        Create computation plan based on user request
        
        Args:
            user_request: User's computation request description
            context: Additional context information
            
        Returns:
            ComputationPlan: Detailed computation plan
        """
        # Build prompt with tool descriptions
        messages = self.prompt.format_messages(
            input=user_request,
            tools=self.available_tools
        )
        
        print(f"\n[PLAN] Calling LLM to generate plan...")
        print(f"       Model: {self.llm.model_name}")
        
        # Call LLM to generate plan
        response = self.llm.invoke(messages)
        
        print(f"[PLAN] LLM responded ({len(response.content)} chars)")
        
        # Parse response into structured plan
        plan = self._parse_plan(response.content, user_request)
        
        return plan
    
    def _parse_plan(self, llm_response: str, original_request: str) -> ComputationPlan:
        """Parse LLM response into structured plan"""
        import json
        import re
        
        # Try to extract JSON from LLM response
        # LLM might wrap JSON in markdown code blocks
        json_match = re.search(r'```(?:json)?\s*(\{.*?\})\s*```', llm_response, re.DOTALL)
        if json_match:
            json_str = json_match.group(1)
        else:
            # Try to find raw JSON
            json_match = re.search(r'\{.*"steps".*\}', llm_response, re.DOTALL)
            if json_match:
                json_str = json_match.group(0)
            else:
                json_str = llm_response
        
        try:
            plan_dict = json.loads(json_str)
            print(f"[PLAN] Successfully parsed JSON plan")
            print(f"       Objective: {plan_dict.get('objective', 'N/A')}")
            print(f"       Steps: {len(plan_dict.get('steps', []))}")
            if 'rationale' in plan_dict:
                print(f"       Rationale: {plan_dict['rationale'][:100]}...")
            
            # Build ComputationPlan from parsed JSON
            return ComputationPlan(
                objective=plan_dict.get("objective", original_request),
                steps=plan_dict.get("steps", []),
                resources=plan_dict.get("resources", {}),
                constraints=plan_dict.get("constraints", {}),
                estimated_time=plan_dict.get("estimated_time")
            )
            
        except json.JSONDecodeError as e:
            print(f"[WARN] Failed to parse LLM response as JSON: {e}")
            print(f"[WARN] LLM Response: {llm_response[:500]}...")
            
            # Fallback: Create a simple default plan for common cases
            if "O2" in original_request or "oxygen" in original_request.lower():
                print(f"[PLAN] Using fallback plan for O2 molecule")
                steps = [
                    {
                        "name": "prepare_structure",
                        "description": "Prepare O2 molecule initial structure",
                        "method": "create_molecule",
                        "params": {
                            "molecule": "O2",
                            "bond_length": 1.21,
                            "box_size": [15, 15, 15]
                        },
                        "device": "local"
                    },
                    {
                        "name": "vasp_prepare",
                        "description": "Prepare VASP input files",
                        "method": "vasp_prepare",
                        "params": {
                            "calc_type": "gas",
                            "k_product": 1
                        },
                        "device": "local"
                    },
                    {
                        "name": "vasp_calculate", 
                        "description": "Run VASP calculation",
                        "method": "vasp_execute",
                        "params": {},
                        "device": "cpu-worker"
                    }
                ]
            else:
                # Generic fallback
                steps = [{
                    "name": "unknown",
                    "description": "Could not parse plan from LLM",
                    "method": "manual",
                    "params": {},
                    "device": "local"
                }]
            
            return ComputationPlan(
                objective=original_request,
                steps=steps,
                resources={},
                constraints={}
            )
