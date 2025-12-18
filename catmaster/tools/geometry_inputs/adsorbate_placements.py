"""
Minimal placeholder implementation for adsorbate placement generation.
Generates no candidates but preserves interface without external deps.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from catmaster.tools.base import resolve_workspace_path, workspace_relpath


def generate_placements(config: Mapping[str, object]) -> Dict[str, object]:
    outdir = resolve_workspace_path(str(config.get("outdir", "adsorbate_candidates")))
    outdir.mkdir(parents=True, exist_ok=True)
    return {
        "schema_version": "v2",
        "output_dir_rel": workspace_relpath(outdir),
        "num_candidates": 0,
        "candidate_paths_rel": [],
        "note": "adsorbate_placements placeholder: no candidates generated.",
    }


__all__ = ["generate_placements"]
