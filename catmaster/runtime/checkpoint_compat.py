"""Retain native historical channels when changing the execution host.

These are upstream state definitions, not a second checkpoint format. An old
filesystem or async-task value remains readable even when its middleware no
longer owns execution. It must never be interpreted as a request to relaunch.
"""
from langchain.agents.middleware import AgentMiddleware
from deepagents.middleware.async_subagents import AsyncSubAgentState
from deepagents.middleware.filesystem import FilesystemState


class RetainedState(FilesystemState, AsyncSubAgentState):
    pass


class RetainedCheckpointState(AgentMiddleware):
    state_schema = RetainedState
