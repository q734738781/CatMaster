#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LangGraph workflow: Orchestrate collaboration between four agents.
"""
from __future__ import annotations

import json
import logging
from typing import Dict, List, Any, Optional, TypedDict, Annotated
from datetime import datetime
from enum import Enum

from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END
from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)

from catmaster.agents.planning_agent import PlanningAgent, ComputationPlan
from catmaster.agents.orchestrator_agent import OrchestratorAgent
from catmaster.agents.parameter_resolver_agent import ParameterResolverAgent, ParameterResolution
from catmaster.agents.execution_agent import ExecutionAgent, ExecutionResult
from catmaster.agents.summary_agent import SummaryAgent, ComputationSummary


class WorkflowState(TypedDict):
    """Workflow state definition"""
    # Input
    user_request: str
    
    # Planning phase
    computation_plan: Optional[ComputationPlan]
    
    # Orchestration phase
    current_task_index: int
    current_task: Optional[Dict[str, Any]]
    task_results: Dict[str, Any]
    
    # Parameter resolution phase
    parameter_resolution: Optional[ParameterResolution]
    
    # Execution phase
    execution_result: Optional[ExecutionResult]
    retry_count: int
    max_retries: int
    
    # Summary phase
    final_summary: Optional[ComputationSummary]
    
    # Control flow
    should_continue: bool
    error: Optional[str]
    
    # File tracking
    file_tracker: Dict[str, List[str]]


class CatMasterWorkflow:
    """
    CatMaster workflow: Coordinate four agents to complete computational tasks.
    """
    
    def __init__(self, llm: Optional[ChatOpenAI] = None):
        self.llm = llm or ChatOpenAI(temperature=0, model="gpt-4")
        
        # Initialize Agents
        self.planning_agent = PlanningAgent(self.llm)
        self.orchestrator_agent = OrchestratorAgent(self.llm)
        self.parameter_resolver_agent = ParameterResolverAgent(self.llm)
        self.execution_agent = ExecutionAgent(self.llm)
        self.summary_agent = SummaryAgent(self.llm)
        
        # Build workflow
        self.workflow = self._build_workflow()
    
    def _build_workflow(self):
        """Build LangGraph workflow"""
        workflow = StateGraph(WorkflowState)
        
        # Add nodes
        workflow.add_node("plan", self._planning_node)
        workflow.add_node("orchestrate", self._orchestration_node)
        workflow.add_node("resolve_params", self._parameter_resolution_node)
        workflow.add_node("execute", self._execution_node)
        workflow.add_node("summarize", self._summary_node)
        workflow.add_node("error_handler", self._error_handler_node)
        
        # Define edges and conditions
        workflow.set_entry_point("plan")
        
        # Planning -> Orchestration
        workflow.add_edge("plan", "orchestrate")
        
        # Orchestration -> Parameter Resolution or Summary
        workflow.add_conditional_edges(
            "orchestrate",
            self._should_resolve_or_summarize,
            {
                "resolve": "resolve_params",
                "summarize": "summarize",
                "error": "error_handler"
            }
        )
        
        # Parameter Resolution -> Execution
        workflow.add_edge("resolve_params", "execute")
        
        # Execution -> Orchestration (continue to next task)
        workflow.add_conditional_edges(
            "execute",
            self._should_continue_or_retry,
            {
                "continue": "orchestrate",
                "retry": "resolve_params",  # Retry through parameter resolution
                "error": "error_handler"
            }
        )
        
        # Summary -> End
        workflow.add_edge("summarize", END)
        
        # Error handling -> Summary or End
        workflow.add_conditional_edges(
            "error_handler",
            self._should_summarize_after_error,
            {
                "summarize": "summarize",
                "end": END
            }
        )
        
        return workflow.compile()
    
    def _planning_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Planning node: Create computation plan"""
        logger.info("="*60)
        logger.info("[PLANNING NODE]")
        logger.info("="*60)
        try:
            user_request = state["user_request"]
            logger.info(f"User Request: {user_request}")
            
            plan = self.planning_agent.plan(user_request)
            
            logger.info(f"Generated Plan:")
            logger.info(f"  Objective: {plan.objective}")
            logger.info(f"  Total Steps: {len(plan.steps)}")
            for i, step in enumerate(plan.steps, 1):
                logger.info(f"    {i}. {step['name']} ({step['method']}) on {step['device']}")
            
            # Initialize orchestration state
            self.orchestrator_agent.initialize(plan)
            
            return {
                "computation_plan": plan,
                "current_task_index": 0,
                "task_results": {},
                "should_continue": True,
                "retry_count": 0,
                "max_retries": 3,
                "file_tracker": {}
            }
        except Exception as e:
            logger.error(f"Planning failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Planning failed: {str(e)}",
                "should_continue": False
            }
    
    def _orchestration_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Orchestration node: Get next task"""
        logger.info("="*60)
        logger.info("[ORCHESTRATE] ORCHESTRATION NODE")
        logger.info("="*60)
        logger.info(f"Current task index: {state.get('current_task_index', 0)}")
        
        try:
            # Get next task
            next_task = self.orchestrator_agent.get_next_task()
            
            if next_task:
                logger.info(f"Next Task: {next_task['name']}")
                logger.info(f"  Method: {next_task['method']}")
                logger.info(f"  Device: {next_task['device']}")
                
                # Prepare task input
                prepared_task = self.orchestrator_agent.prepare_task_input(next_task)
                
                logger.info(f"Task Parameters:")
                import json
                logger.info(json.dumps(prepared_task.get('params', {}), indent=2))
                
                return {
                    "current_task": prepared_task,
                    "should_continue": True,
                    "retry_count": 0  # Reset retry count
                }
            else:
                # No more tasks, prepare for summary
                logger.info("All tasks completed, preparing summary...")
                return {
                    "current_task": None,
                    "should_continue": False
                }
        except Exception as e:
            logger.error(f"Orchestration failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Orchestration failed: {str(e)}",
                "should_continue": False
            }
    
    def _parameter_resolution_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Parameter resolution node: Resolve task parameters based on context"""
        logger.info("="*60)
        logger.info("[RESOLVE PARAMS] PARAMETER RESOLUTION NODE")
        logger.info("="*60)
        
        try:
            current_task = state["current_task"]
            if not current_task:
                logger.error("No task to resolve parameters for")
                return {"error": "No task to resolve parameters"}
            
            logger.info(f"Task: {current_task['name']}")
            logger.info(f"Method: {current_task['method']}")
            logger.info(f"Parameter Descriptions:")
            for key, desc in current_task.get('params', {}).items():
                logger.info(f"  - {key}: {desc}")
            
            # Resolve parameters using context
            from pathlib import Path
            resolution = self.parameter_resolver_agent.resolve_parameters(
                task=current_task,
                previous_results=state["task_results"],
                working_dir=Path.cwd()
            )
            
            logger.info(f"Parameter Resolution:")
            logger.info(f"  Resolved Parameters:")
            for key, value in resolution.resolved_params.items():
                logger.info(f"    - {key}: {value}")
            
            if resolution.warnings:
                logger.warning(f"  Warnings:")
                for warning in resolution.warnings:
                    logger.warning(f"    - {warning}")
            
            # Check if we have parameters
            if not resolution.resolved_params:
                return {
                    "error": f"Failed to resolve parameters: {'; '.join(resolution.warnings)}",
                    "should_continue": False
                }
            
            # Update task with resolved parameters
            current_task["params"] = resolution.resolved_params
            
            return {
                "current_task": current_task,
                "parameter_resolution": resolution
            }
            
        except Exception as e:
            logger.error(f"Parameter resolution failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Parameter resolution failed: {str(e)}",
                "should_continue": False
            }
    
    def _execution_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Execution node: Execute current task"""
        logger.info("="*60)
        logger.info("[EXECUTE]  EXECUTION NODE")
        logger.info("="*60)
        
        try:
            current_task = state["current_task"]
            if not current_task:
                logger.error("No task to execute")
                return {"error": "No task to execute"}
            
            logger.info(f"Task: {current_task['name']}")
            logger.info(f"Method: {current_task['method']}")
            logger.info(f"Device: {current_task['device']}")
            logger.info(f"Input Parameters:")
            import json
            logger.info(json.dumps(current_task.get('params', {}), indent=2, ensure_ascii=False))
            
            # Execute task
            logger.info(f"Executing task...")
            result = self.execution_agent.execute_task(current_task)
            
            logger.info(f"Execution Result:")
            logger.info(f"  Success: {result.success}")
            logger.info(f"  Execution time: {result.execution_time:.2f}s")
            if result.error:
                logger.error(f"  Error: {result.error}")
            if result.output:
                logger.info(f"  Output keys: {list(result.output.keys()) if isinstance(result.output, dict) else 'N/A'}")
            
            # Update orchestrator state
            if result.success:
                logger.info(f"Task {current_task['name']} completed successfully")
                
                self.orchestrator_agent.update_task_status(
                    current_task["name"],
                    "completed",
                    result.output
                )
                
                # Save results
                task_results = state.get("task_results", {})
                task_results[current_task["name"]] = result.output or {}
                task_results[current_task["name"]]["execution_time"] = result.execution_time
                task_results[current_task["name"]]["device_used"] = result.device_used
                
                # Update file tracker
                file_tracker = state.get("file_tracker", {})
                if isinstance(result.output, dict):
                    # Check for standardized output format
                    if "output_files" in result.output:
                        output_paths = []
                        for file_info in result.output.get("output_files", []):
                            if isinstance(file_info, dict):
                                output_paths.append(file_info.get("path", str(file_info)))
                            else:
                                output_paths.append(str(file_info))
                        file_tracker[current_task["name"]] = output_paths
                
                return {
                    "execution_result": result,
                    "task_results": task_results,
                    "file_tracker": file_tracker,
                    "current_task_index": state["current_task_index"] + 1
                }
            else:
                # Task failed
                logger.error(f"Task {current_task['name']} failed: {result.error}")
                
                self.orchestrator_agent.update_task_status(
                    current_task["name"],
                    "failed",
                    error=result.error
                )
                
                return {
                    "execution_result": result,
                    "retry_count": state["retry_count"] + 1
                }
                
        except Exception as e:
            logger.error(f"Execution exception: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Execution failed: {str(e)}",
                "retry_count": state["retry_count"] + 1
            }
    
    def _summary_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Summary node: Generate final report"""
        logger.info("="*60)
        logger.info("[SUMMARY] SUMMARY NODE")
        logger.info("="*60)
        
        try:
            plan = state.get("computation_plan")
            task_results = state.get("task_results", {})
            
            logger.info(f"Collected {len(task_results)} task results")
            
            if plan:
                # Get final result from orchestration
                orchestration_result = self.orchestrator_agent.finalize()
                logger.info(f"Orchestration finalized: {orchestration_result.get('success', False)}")
                
                # Generate summary
                logger.info("Generating summary report...")
                summary = self.summary_agent.summarize(
                    objective=plan.objective,
                    results=task_results,
                    execution_info={
                        "total_tasks": len(plan.steps),
                        "completed_tasks": len(task_results),
                        "workflow_start": state.get("workflow_start"),
                        "workflow_end": datetime.now().isoformat()
                    }
                )
                
                logger.info(f"Summary generated with {len(summary.key_results)} key results")
                
                return {
                    "final_summary": summary,
                    "should_continue": False
                }
            else:
                # No plan, create error summary
                logger.warning("No plan found, creating error summary")
                summary = ComputationSummary(
                    title="Computation Not Completed",
                    objective=state.get("user_request", "Unknown"),
                    key_results={},
                    conclusions=["Computation workflow did not complete normally"],
                    recommendations=["Please check input and retry"]
                )
                
                return {
                    "final_summary": summary,
                    "should_continue": False
                }
                
        except Exception as e:
            logger.error(f"Summary generation failed: {e}")
            import traceback
            logger.error(traceback.format_exc())
            return {
                "error": f"Summary generation failed: {str(e)}",
                "should_continue": False
            }
    
    def _error_handler_node(self, state: WorkflowState) -> Dict[str, Any]:
        """Error handler node"""
        logger.info("="*60)
        logger.info("[ERROR] ERROR HANDLER NODE")
        logger.info("="*60)
        
        error = state.get("error", "Unknown error")
        execution_result = state.get("execution_result")
        current_task = state.get("current_task")
        
        logger.error(f"Error message: {error}")
        logger.error(f"Current task: {current_task['name'] if current_task else 'None'}")
        logger.error(f"Retry count: {state.get('retry_count', 0)}/{state.get('max_retries', 3)}")
        
        if execution_result:
            logger.error(f"Execution Result Details:")
            logger.error(f"  Task: {execution_result.task_name}")
            logger.error(f"  Success: {execution_result.success}")
            logger.error(f"  Error: {execution_result.error}")
            logger.error(f"  Execution time: {execution_result.execution_time:.2f}s")
        
        # Record error
        task_results = state.get("task_results", {})
        task_results["_workflow_error"] = {
            "error": error,
            "current_task": current_task["name"] if current_task else None,
            "retry_count": state.get("retry_count", 0),
            "timestamp": datetime.now().isoformat()
        }
        
        logger.warning("Workflow will attempt to generate partial summary")
        
        return {
            "task_results": task_results,
            "should_continue": False
        }
    
    def _should_resolve_or_summarize(self, state: WorkflowState) -> str:
        """Decide whether to resolve parameters for task or summarize"""
        has_error = state.get("error")
        has_task = state.get("current_task") is not None
        should_continue = state.get("should_continue")
        
        logger.info("[DECISION] Decision: Orchestrate → ?")
        logger.info(f"  Has error: {has_error}")
        logger.info(f"  Has current task: {has_task}")
        logger.info(f"  Should continue: {should_continue}")
        
        if has_error:
            logger.info(f"  Going to: error")
            return "error"
        elif has_task and should_continue:
            logger.info(f"  Going to: resolve")
            return "resolve"
        else:
            logger.info(f"  Going to: summarize")
            return "summarize"
    
    def _should_continue_or_retry(self, state: WorkflowState) -> str:
        """Decide whether to continue, retry, or handle error"""
        result = state.get("execution_result")
        
        logger.info("[DECISION] Decision: Execute → ?")
        
        if not result:
            logger.info("  No execution result")
            logger.info("  Going to: error")
            return "error"
        
        logger.info(f"  Task: {result.task_name}")
        logger.info(f"  Success: {result.success}")
        
        if result.success:
            logger.info(f"  Going to: continue (orchestrate)")
            return "continue"
        else:
            # Check retry count
            retry_count = state.get("retry_count", 0)
            max_retries = state.get("max_retries", 3)
            
            logger.info(f"  Failed, retry count: {retry_count}/{max_retries}")
            
            if retry_count < max_retries:
                logger.info(f"  Going to: retry (execute again)")
                logger.info(f"[RETRY] Task {state['current_task']['name']} failed, retrying ({retry_count}/{max_retries})...")
                return "retry"
            else:
                logger.info(f"  Going to: error (max retries exceeded)")
                return "error"
    
    def _should_summarize_after_error(self, state: WorkflowState) -> str:
        """Decide whether to generate summary after error"""
        has_results = bool(state.get("task_results"))
        
        logger.info("[DECISION] Decision: Error → ?")
        logger.info(f"  Has task results: {has_results}")
        
        # If there are partial results, still generate summary
        if has_results:
            logger.info(f"  Going to: summarize (with partial results)")
            return "summarize"
        else:
            logger.info(f"  Going to: end (no results to summarize)")
            return "end"
    
    def run(self, user_request: str) -> ComputationSummary:
        """
        Run complete workflow
        
        Args:
            user_request: User's computation request
            
        Returns:
            ComputationSummary: Computation summary
        """
        print("\n" + "="*30)
        print("[START] STARTING CATMASTER WORKFLOW")
        print("="*30)
        print(f"Request: {user_request}")
        print()
        
        # Initialize state
        initial_state = {
            "user_request": user_request,
            "workflow_start": datetime.now().isoformat(),
            "computation_plan": None,
            "current_task_index": 0,
            "current_task": None,
            "task_results": {},
            "execution_result": None,
            "retry_count": 0,
            "max_retries": 3,
            "final_summary": None,
            "should_continue": True,
            "error": None
        }
        
        # Run workflow
        print(" Invoking LangGraph workflow...\n")
        final_state = self.workflow.invoke(initial_state, {"recursion_limit": 100})
        
        print("\n" + "="*30)
        print("[END] WORKFLOW COMPLETED")
        print("="*30)
        print(f"Final state keys: {list(final_state.keys())}")
        print(f"Error: {final_state.get('error')}")
        print(f"Has summary: {final_state.get('final_summary') is not None}")
        print()
        
        # Return summary
        return final_state.get("final_summary")


def create_workflow(llm: Optional[ChatOpenAI] = None) -> CatMasterWorkflow:
    """Create CatMaster workflow instance"""
    return CatMasterWorkflow(llm)
