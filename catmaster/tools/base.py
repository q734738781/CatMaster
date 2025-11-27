#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Base utilities for tools to create standardized outputs.
"""
from __future__ import annotations

from typing import Dict, List, Any, Optional
import time


def create_tool_output(
    tool_name: str,
    success: bool = True,
    data: Dict[str, Any] = None,
    error: str = None,
    warnings: List[str] = None,
    execution_time: float = None
) -> Dict[str, Any]:
    """
    Create standardized tool output dictionary.
    
    Args:
        tool_name: Name of the tool
        success: Whether execution succeeded
        data: Tool-specific output data
        error: Error message if failed
        warnings: List of warning messages
        execution_time: Execution time in seconds
        
    Returns:
        Standardized output dictionary
    """
    return {
        "status": "success" if success else "failed",
        "tool_name": tool_name,
        "data": data or {},
        "warnings": warnings or [],
        "error": error,
        "execution_time": execution_time
    }