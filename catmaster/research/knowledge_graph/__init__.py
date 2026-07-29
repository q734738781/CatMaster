"""Workspace-scoped Research Knowledge Graph.

The graph is the durable scientific situation shared by research threads.
Long-form notes, artifacts, run receipts, and conversation history remain in
their owning stores and are connected through typed references.
"""

from .context import ResearchGraphContextBuilder
from .models import (
    EdgeRelation,
    ExperimentBody,
    ExperimentState,
    HypothesisBody,
    NodeKind,
    OrchestrationMode,
    RefKind,
    ResultBody,
)
from .service import ResearchGraphService
from .store import ResearchGraphConflict, ResearchGraphStore

__all__ = [
    "EdgeRelation",
    "ExperimentBody",
    "ExperimentState",
    "HypothesisBody",
    "NodeKind",
    "OrchestrationMode",
    "RefKind",
    "ResearchGraphConflict",
    "ResearchGraphContextBuilder",
    "ResearchGraphService",
    "ResearchGraphStore",
    "ResultBody",
]
