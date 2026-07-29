"""Workspace-scoped persistence shared by independent CatMaster subsystems."""

from .workspace_db import (
    connect_workspace_db,
    ensure_workspace_ui_events,
    workspace_database_path,
)

__all__ = [
    "connect_workspace_db",
    "ensure_workspace_ui_events",
    "workspace_database_path",
]
