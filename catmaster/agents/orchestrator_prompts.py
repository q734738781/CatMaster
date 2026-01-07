from __future__ import annotations

from langchain_core.prompts import ChatPromptTemplate


def build_plan_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """
You are an expert computational workflow planner.

Available tools:
{tools}

Create:
- A ToDo list (long-term), each item a short clause that can be completed with the listed tools, which will be used as the reference plan skeleton for the next step decision.
- A NextStep (single actionable intent) that can be attempted immediately.
- A reasoning field explaining briefly why this plan/next step was chosen.

Rules:
- Use only the available tools; do NOT invent capabilities.
- Always express any file or directory paths as relative paths; they will be resolved relative to workspace root.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema: {{"todo": [...], "next_step": "...", "reasoning": "..."}}"""),
        ("human", "{user_request}")
    ])


def build_plan_repair_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """
You are an expert computational workflow planner.

Available tools:
{tools}

Your previous response was not valid JSON. Re-emit a valid JSON object only.

Rules:
- Use only the available tools; do NOT invent capabilities.
- Always express any file or directory paths as relative paths; they will be resolved relative to workspace root.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema: {{"todo": [...], "next_step": "...", "reasoning": "..."}}"""),
        ("human", "User request: {user_request}\nParse error: {error}\nInvalid response: {raw}")
    ])


def build_plan_feedback_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """
You are an expert computational workflow planner.

Available tools:
{tools}

Revise the plan based on human feedback.

Inputs:
- Original user request
- Current plan JSON
- Human feedback
- Feedback history (oldest first)

Rules:
- Use only the available tools; do NOT invent capabilities.
- If feedback is unclear or conflicts with tool limits, make the smallest safe change and note the constraint in reasoning.
- Always express any file or directory paths as relative paths; they will be resolved relative to workspace root.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema: {{"todo": [...], "next_step": "...", "reasoning": "..."}}"""),
        ("human", "User request: {user_request}\nCurrent plan: {plan_json}\nHuman feedback: {feedback}\nFeedback history: {feedback_history}")
    ])


def build_step_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """
You are an execution controller. Decide ONE tool call or finish_project.
Context:
- Available tools: {tools}
- Reference Plan Skeleton: {todo}
- Observations so far: {observations}
- Last result: {last_result}

Rules:
- Choose at most one tool per turn.
- Return exactly one JSON object; no code fences or extra text.
- JSON schema (all lowercase keys):
{{"action": "call"|"finish_project",
    "method": "tool_name"|null,
    "params": {{...}},
    "next_step": "a concrete, testable intent that can be acted on in the next turn (either a tool call you plan to attempt, or a condition to check via a specific tool)",
    "note": "optional short self-note for memory",
    "reasoning": "brief rationale for this decision"}}
- If action='call', method MUST exactly match one tool name in available tools. Otherwise set action='finish_project' and explain why.
- If you ensure that to finish the project, set action="finish_project" and method=null in that single object.
- Prefer creating/using subfolders under the workspace for each step; reuse paths returned by previous tool calls instead of guessing.
- Always provide file or directory paths as relative paths; they will be resolved relative to workspace root.
- If a needed file might not exist, first list or create it with the appropriate tool.
- Treat user instruction as a suggestion. If observations contradict it or a better action is available, you CAN revise the plan by choosing a different tool call and writing an updated next_step.
- The controller may skip a turn if your JSON is invalid; in that case, you must output a valid JSON decision next turn.
"""),
        ("human", "Suggested next step (may be revised): {instruction}")
    ])


def build_summary_prompt() -> ChatPromptTemplate:
    return ChatPromptTemplate.from_messages([
        ("system", """You are a scientific workflow assistant. Summarize the run from observations and respond the results for the user.
Include key numerical results present in observations.
Mention where outputs are stored (use relative paths if provided). Keep the summary concise and informative."""),
        ("human", "User request: {user_request}\nObservations: {observations}")
    ])


__all__ = [
    "build_plan_prompt",
    "build_plan_repair_prompt",
    "build_plan_feedback_prompt",
    "build_step_prompt",
    "build_summary_prompt",
]
