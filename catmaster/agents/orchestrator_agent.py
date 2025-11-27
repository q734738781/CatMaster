#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Task Orchestrator Agent: Responsible for coordinating and scheduling computational task execution.
"""
from __future__ import annotations

import json
import re
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from catmaster.agents.planning_agent import ComputationPlan


class TaskStatus(BaseModel):
    """Task status model"""
    task_name: str
    status: str  # pending, running, completed, failed
    result: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    start_time: Optional[str] = None
    end_time: Optional[str] = None


class OrchestrationState(BaseModel):
    """Orchestration state model"""
    plan: ComputationPlan
    current_step: int = 0
    task_statuses: List[TaskStatus] = Field(default_factory=list)
    intermediate_results: Dict[str, Any] = Field(default_factory=dict)
    final_output: Optional[Dict[str, Any]] = None


class OrchestratorAgent:
    """
    Task Orchestrator Agent: Responsible for coordinating task execution according to the plan.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", """You are a task orchestration expert. Your responsibilities are:

1. Determine task execution order based on the computation plan
2. Manage data dependencies between tasks
3. Handle exceptions during task execution
4. Optimize resource usage and avoid conflicts

Always maintain high efficiency and reliability in task execution."""),
            ("human", "{input}")
        ])
        self.state: Optional[OrchestrationState] = None
    
    def initialize(self, plan: ComputationPlan) -> OrchestrationState:
        """Initialize orchestration state"""
        self.state = OrchestrationState(
            plan=plan,
            task_statuses=[
                TaskStatus(task_name=step["name"], status="pending")
                for step in plan.steps
            ]
        )
        return self.state
    
    def get_next_task(self) -> Optional[Dict[str, Any]]:
        """Get the next task to execute"""
        if not self.state or self.state.current_step >= len(self.state.plan.steps):
            return None
        
        current_task = self.state.plan.steps[self.state.current_step]
        
        # Check dependencies
        if self._check_dependencies(current_task):
            return current_task
        
        return None
    
    def update_task_status(
        self, 
        task_name: str, 
        status: str, 
        result: Optional[Dict[str, Any]] = None,
        error: Optional[str] = None
    ) -> None:
        """Update task status"""
        if not self.state:
            return
        
        for task_status in self.state.task_statuses:
            if task_status.task_name == task_name:
                task_status.status = status
                task_status.result = result
                task_status.error = error
                
                if result:
                    self.state.intermediate_results[task_name] = result
                
                if status == "completed":
                    self.state.current_step += 1
                
                break
    
    def handle_task_failure(self, task_name: str, error: str) -> Dict[str, Any]:
        """Handle task failure situations"""
        # Use LLM to decide how to handle failure
        messages = self.prompt.format_messages(
            input=f"""Task {task_name} execution failed with error: {error}
Current plan: {json.dumps(self.state.plan.dict() if self.state else {}, ensure_ascii=False)}
Please suggest how to handle this failure: retry, skip, or adjust the plan?"""
        )
        
        response = self.llm.invoke(messages)
        
        # Simplified handling: directly return retry suggestion
        return {
            "action": "retry",
            "max_retries": 3,
            "adjustments": {}
        }
    
    def _check_dependencies(self, task: Dict[str, Any]) -> bool:
        """Check if task dependencies are satisfied"""
        # Simplified implementation: execute in sequence
        if not self.state:
            return False
        
        task_index = next(
            (i for i, step in enumerate(self.state.plan.steps) 
             if step["name"] == task["name"]),
            -1
        )
        
        # Check if all previous tasks are completed
        for i in range(task_index):
            task_status = self.state.task_statuses[i]
            if task_status.status != "completed":
                return False
        
        return True
    
    def prepare_task_input(self, task: Dict[str, Any]) -> Dict[str, Any]:
        """Prepare task input - now only does minimal processing since parameter resolution is handled separately"""
        if not self.state:
            return task
        
        # Simply return a copy of the task
        # The ParameterResolverAgent now handles all parameter resolution
        return task.copy()
    
    def _make_relative_path(self, path_str: str) -> str:
        """Convert absolute path to relative if possible"""
        from pathlib import Path
        path = Path(path_str)
        if path.is_absolute():
            try:
                return str(path.relative_to(Path.cwd()))
            except ValueError:
                return str(path)
        return str(path)
    
    def finalize(self) -> Dict[str, Any]:
        """Finalize orchestration, generate final output"""
        if not self.state:
            return {"error": "No orchestration state"}
        
        # Collect all results
        final_output = {
            "objective": self.state.plan.objective,
            "completed_tasks": [
                ts.task_name for ts in self.state.task_statuses 
                if ts.status == "completed"
            ],
            "results": self.state.intermediate_results,
            "success": all(
                ts.status == "completed" 
                for ts in self.state.task_statuses
            )
        }
        
        self.state.final_output = final_output
        return final_output
