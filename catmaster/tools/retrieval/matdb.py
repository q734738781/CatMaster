"""
Materials Project retrieval tools exposed to LLM agents.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional
from pymatgen.core.structure import Structure
from mp_api.client import MPRester
from pydantic import BaseModel, Field
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class MPSearchMaterialsInput(BaseModel):
    """
    Search Materials Project with flexible criteria and write results to CSV.

    Common criteria keys (pass via `criteria`):
      - material_ids (list[str] or str)
      - elements (list[str] or str)
      - chemsys (str)
      - formula (str)
      - is_stable (bool)
      - energy_above_hull ([min, max])
      - formation_energy ([min, max])
      - band_gap ([min, max])
      - crystal_system (str)
      - spacegroup_number / spacegroup_symbol
      - num_sites ([min, max])
      - density ([min, max])
      - volume ([min, max])
      - num_elements ([min, max])
    Common fields for output rows (pass via `fields`):
      material_id, formula_pretty (or formula), energy_above_hull, formation_energy_per_atom (or formation_energy),
      band_gap, crystal_system, spacegroup_number, spacegroup_symbol, nsites (or num_sites), density, volume.
    """

    criteria: Optional[Dict[str, Any]] = Field(
        None,
        description="Direct summary.search filters (keys listed above). Range values may be [min, max].",
    )
    fields: List[str] = Field(
        default_factory=lambda: [
            "material_id",
            "formula_pretty",
            "energy_above_hull",
            "formation_energy_per_atom",
            "band_gap",
            "nsites",
            "volume",
            "density",
        ],
        description="Fields to include in CSV rows; supports common aliases.",
    )
    limit: int = Field(50, ge=1, description="Maximum number of hits to return (default 50).")
    output_csv: str = Field(..., description="Workspace-relative CSV path to write search results.")


class MPDownloadStructureInput(BaseModel):
    """Download one or more structures from Materials Project into the workspace."""

    mp_ids: List[str] = Field(..., description="Materials Project IDs, e.g., ['mp-149', 'mp-13'].")
    fmt: str = Field("poscar", pattern="^(poscar|cif|json)$", description="Output format: poscar|cif|json.")
    output_dir: str = Field("retrieval/mp", description="Workspace-relative directory to save the structure.")


def _mpr(*, monty_decode: bool = True, use_document_model: bool = True) -> MPRester:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable is not set.")
    return MPRester(api_key, monty_decode=monty_decode, use_document_model=use_document_model)

_FIELD_ALIASES = {
    "formula": "formula_pretty",
    "formation_energy": "formation_energy_per_atom",
    "num_sites": "nsites",
    "spacegroup_number": "symmetry.number",
    "spacegroup_symbol": "symmetry.symbol",
    "crystal_system": "symmetry.crystal_system",
    "spacegroup.number": "symmetry.number",
    "spacegroup.symbol": "symmetry.symbol",
}

_CRITERIA_KEY_ALIASES = {
    "formation_energy_per_atom": "formation_energy",
    "nsites": "num_sites",
    "e_above_hull": "energy_above_hull",
    "mp_id": "material_ids",
    "material_id": "material_ids",
}

_RANGE_KEYS = {
    "band_gap",
    "energy_above_hull",
    "formation_energy",
    "density",
    "volume",
    "num_sites",
    "num_elements",
}

_KNOWN_CRITERIA = {
    "material_ids",
    "elements",
    "chemsys",
    "formula",
    "is_stable",
    "energy_above_hull",
    "formation_energy",
    "band_gap",
    "crystal_system",
    "spacegroup_number",
    "spacegroup_symbol",
    "num_sites",
    "density",
    "volume",
    "num_elements",
}


def _coerce_range_tuple(value: Any) -> Any:
    if isinstance(value, list) and len(value) == 2:
        return (value[0], value[1])
    return value


def _normalize_criteria(criteria: Dict[str, Any]) -> Dict[str, Any]:
    normalized: Dict[str, Any] = {}
    for key, value in criteria.items():
        mapped = _CRITERIA_KEY_ALIASES.get(key, key)
        if mapped in {"elements", "exclude_elements"} and isinstance(value, str):
            value = [value]
        if mapped in normalized and mapped != key:
            raise ValueError(f"Conflicting criteria keys: {key} and {mapped}")
        if mapped == "material_ids" and isinstance(value, str):
            value = [value]
        if mapped in _RANGE_KEYS:
            value = _coerce_range_tuple(value)
        normalized[mapped] = value
    unknown = sorted([k for k in normalized if k not in _KNOWN_CRITERIA])
    if unknown:
        raise ValueError(f"Unknown criteria keys: {', '.join(unknown)}")
    return normalized


def _normalize_fields(fields: List[str]) -> List[tuple[str, str]]:
    normalized: List[tuple[str, str]] = []
    for field in fields:
        mapped = _FIELD_ALIASES.get(field, field)
        normalized.append((field, mapped))
    return normalized


def _get_doc_value(doc: Any, path: str) -> Any:
    current = doc
    for part in path.split("."):
        if current is None:
            return None
        if isinstance(current, dict):
            current = current.get(part)
        else:
            current = getattr(current, part, None)
    return current


def _serialize_csv_value(value: Any) -> Any:
    if value is None:
        return ""
    if isinstance(value, (str, int, float, bool)):
        return value
    return json.dumps(value, ensure_ascii=False)


def mp_search_materials(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Search Materials Project with flexible criteria and write the result table to CSV.
    """
    params = MPSearchMaterialsInput(**payload)
    try:
        criteria = _normalize_criteria(params.criteria or {})
    except Exception as exc:
        return create_tool_output("mp_search_materials", success=False, error=str(exc))
    warnings: List[str] = []

    if not criteria:
        return create_tool_output("mp_search_materials", success=False, error="Provide criteria.")

    field_pairs = _normalize_fields(params.fields)
    if not field_pairs:
        return create_tool_output("mp_search_materials", success=False, error="fields must not be empty.")
    request_fields = sorted({mapped.split(".")[0] for _, mapped in field_pairs})

    out_path = resolve_workspace_path(params.output_csv)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    preview_rows: List[Dict[str, Any]] = []
    written = 0
    total: Optional[int] = None

    try:
        with _mpr(monty_decode=False, use_document_model=False) as client:
            try:
                total = client.materials.summary.count(criteria)
            except Exception as exc:
                warnings.append(f"count failed: {exc}")

            chunk_size = min(max(params.limit, 1), 1000)
            num_chunks = (params.limit + chunk_size - 1) // chunk_size

            docs = client.materials.summary.search(
                **criteria,
                fields=request_fields,
                all_fields=False,
                chunk_size=chunk_size,
                num_chunks=num_chunks,
            )

            with out_path.open("w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=[h for h, _ in field_pairs])
                writer.writeheader()
                for doc in docs:
                    row = {}
                    for header, mapped in field_pairs:
                        row[header] = _serialize_csv_value(_get_doc_value(doc, mapped))
                    writer.writerow(row)
                    if len(preview_rows) < 5:
                        preview_rows.append(row)
                    written += 1
                    if written >= params.limit:
                        break
    except Exception as exc:
        return create_tool_output("mp_search_materials", success=False, error=str(exc))

    if isinstance(total, int):
        returned = written
        truncated = total > written
        count_value: Optional[int] = total
    else:
        returned = written
        truncated = written >= params.limit
        count_value = None

    return create_tool_output(
        "mp_search_materials",
        success=True,
        data={
            "count": count_value,
            "returned": returned,
            "truncated": truncated,
            "criteria": criteria,
            "fields": [h for h, _ in field_pairs],
            "output_csv_rel": workspace_relpath(out_path),
            "preview_rows": preview_rows,
        },
        warnings=warnings,
    )


def mp_download_structure(payload: Dict[str, object]) -> Dict[str, object]:
    """
    Download one or more structures from Materials Project and write them under the workspace. Downloaded structures are conventional cells.
    Args:
        mp_ids: Materials Project IDs, e.g., ["mp-149", "mp-13"].
        fmt: Output format: poscar|cif|pymatgen_json.
        output_dir: Directory to write the structure.
    Returns:
        Paths to the written structures. 
    """
    params = MPDownloadStructureInput(**payload)

    out_dir = resolve_workspace_path(params.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    fmt = params.fmt.lower()
    ext = {"poscar": "vasp", "cif": "cif", "pymatgen_json": "json"}.get(fmt, fmt)
    results: List[Dict[str, Any]] = []
    errors: List[Dict[str, str]] = []

    try:
        with _mpr() as client:
            for mp_id in params.mp_ids:
                try:
                    structure = client.get_structure_by_material_id(mp_id)
                    if isinstance(structure, dict):
                        structure = Structure.from_dict(structure)
                    c_structure = SpacegroupAnalyzer(structure).get_conventional_standard_structure()
                except Exception as exc:  # pragma: no cover - remote call
                    errors.append({"mp_id": mp_id, "error": str(exc)})
                    continue

                out_path = out_dir / f"{mp_id}.{ext}"
                if fmt == "json":
                    out_path.write_text(json.dumps(c_structure.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                else:
                    c_structure.to(fmt=fmt, filename=str(out_path))

                results.append(
                    {
                        "mp_id": mp_id,
                        "structure_rel": workspace_relpath(out_path),
                        "metadata": {
                            "formula": c_structure.composition.reduced_formula,
                            "natoms": len(structure),
                        },
                    }
                )
    except Exception as exc:
        return create_tool_output("mp_download_structure", success=False, error=str(exc))

    if errors:
        return create_tool_output(
            "mp_download_structure",
            success=False,
            data={
                "format": fmt,
                "output_dir_rel": workspace_relpath(out_dir),
                "results": results,
                "errors": errors,
            },
            error="One or more downloads failed.",
        )

    return create_tool_output(
        "mp_download_structure",
        success=True,
        data={
            "format": fmt,
            "output_dir_rel": workspace_relpath(out_dir),
            "results": results,
        },
    )


__all__ = [
    "MPSearchMaterialsInput",
    "MPDownloadStructureInput",
    "mp_search_materials",
    "mp_download_structure",
]
