from __future__ import annotations

"""Retrieval tool entrypoints and deprecated stubs."""

from typing import Any, Dict, List, Optional

from pydantic import BaseModel, Field


class LitSearchInput(BaseModel):
    """Args schema for lit_search."""

    query: str = Field(..., description="Search query text.")
    site: Optional[str] = Field(None, description="Optional site restriction, e.g. scholar.google.com")
    max_results: int = Field(10, ge=1, le=50, description="Max number of results to return.")
    chromedriver_path: Optional[str] = Field(None, description="Path to ChromeDriver (if not using Selenium Manager).")
    headless: bool = Field(True, description="Run headless Chrome.")
    timeout: int = Field(20, ge=5, le=120, description="Page load timeout (seconds).")


def lit_search(
    query: str,
    site: Optional[str] = None,
    max_results: int = 10,
    chromedriver_path: Optional[str] = None,
    headless: bool = True,
    timeout: int = 20,
) -> List[Dict[str, Any]]:
    _ = (query, site, max_results, chromedriver_path, headless, timeout)
    raise NotImplementedError("lit_search is deprecated and needs new implementation.")


class LitCaptureInput(BaseModel):
    """Args schema for lit_capture."""

    target_url: str = Field(..., description="URL to capture.")
    wait_seconds: int = Field(5, ge=0, le=60, description="Wait time after page load before capture (seconds).")
    chromedriver_path: Optional[str] = Field(None, description="Path to ChromeDriver (if not using Selenium Manager).")
    headless: bool = Field(True, description="Run headless Chrome.")
    timeout: int = Field(20, ge=5, le=120, description="Page load timeout (seconds).")


def lit_capture(
    target_url: str,
    wait_seconds: int = 5,
    chromedriver_path: Optional[str] = None,
    headless: bool = True,
    timeout: int = 20,
) -> Dict[str, Any]:
    _ = (target_url, wait_seconds, chromedriver_path, headless, timeout)
    raise NotImplementedError("lit_capture is deprecated and needs new implementation.")


class MatdbQueryInput(BaseModel):
    """Args schema for matdb_query."""

    criteria: Dict[str, Any] = Field(..., description="Search criteria. Supports keys: material_ids|formula|chemsys|elements.")
    properties: Optional[List[str]] = Field(None, description="Requested fields from materials DB.")
    structures_dir: str = Field("structures", description="Directory to write downloaded structures (CIF).")
    api_key: Optional[str] = Field(None, description="API key for provider (if needed).")


def matdb_query(
    criteria: Dict[str, Any],
    properties: Optional[List[str]] = None,
    structures_dir: str = "structures",
    api_key: Optional[str] = None,
) -> Dict[str, Any]:
    _ = (criteria, properties, structures_dir, api_key)
    raise NotImplementedError("matdb_query is deprecated and needs new implementation.")


__all__ = [
    "LitSearchInput",
    "lit_search",
    "LitCaptureInput",
    "lit_capture",
    "MatdbQueryInput",
    "matdb_query",
]
