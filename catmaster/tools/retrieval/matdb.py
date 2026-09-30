"""
Materials Project retrieval tools exposed to LLM agents.
"""

from __future__ import annotations

import csv
import json
import os
from typing import Any, Dict, List, Optional, Literal
from pymatgen.core.structure import Structure
from pydantic import BaseModel, Field
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath


class MPSearchMaterialsInput(BaseModel):
    """[material/discovery] Search Materials Project with flexible criteria and write a CSV candidate table."""

    criteria: Dict = Field(
        default_factory=dict,
        description="Non-empty summary.search filters as a dict. Range values may be [min, max].",
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
        description="Non-empty fields to include in CSV rows; omit for the default selection. Supports common aliases.",
    )
    limit: int = Field(50, ge=1, description="Maximum number of hits per page (default 50).")
    page: int = Field(1, ge=1, description="One-based provider page; keep limit, criteria and sort_field fixed when continuing. Provider updates can change results between calls.")
    sort_field: str = Field("material_id", description="Native MP sort field; prefix '-' for descending. Stable material_id order is the default.")
    output_csv: str = Field(..., description="Workspace-relative CSV path to write search results.")


class MPDownloadStructureInput(BaseModel):
    """[material/discovery] Download one or more structures from Materials Project into the workspace."""

    mp_ids: List[str] = Field(..., description="Materials Project IDs, e.g., ['mp-149', 'mp-13'].")
    fmt: str = Field("poscar", pattern="^(poscar|cif|json)$", description="Output format: poscar|cif|json.")
    output_dir: str = Field("retrieval/mp", description="Workspace-relative directory to save the structure.")
    cell: Literal["as_returned", "primitive", "conventional"] = Field("as_returned", description="as_returned preserves the provider structure; primitive/conventional explicitly standardize it locally.")
    symprec: float = Field(0.01, gt=0, description="Symmetry tolerance in angstrom when standardizing the cell.")
    angle_tolerance: float = Field(5.0, gt=0, description="Angular symmetry tolerance in degrees when standardizing the cell.")
    overwrite: bool = Field(False, description="Allow replacing an existing structure file; otherwise report that item as failed and continue.")


def _mpr(*, monty_decode: bool = True, use_document_model: bool = True) -> Any:
    api_key = os.environ.get("MP_API_KEY")
    if not api_key:
        raise RuntimeError("MP_API_KEY environment variable is not set.")
    try:
        from mp_api.client import MPRester
    except Exception as exc:
        raise RuntimeError(
            "Materials Project client could not be imported. Check mp-api/emmet-core compatibility in the active environment."
        ) from exc
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

def _success(
    tool_name: str,
    *,
    content: str,
    data: dict[str, Any],
    warnings: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return content, artifact


def _fail(
    tool_name: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    error_code: str = "",
) -> None:
    details: list[str] = [str(message).strip()]
    if isinstance(data, dict):
        for key in ("output_csv_rel", "output_dir_rel", "requested", "downloaded", "count", "returned"):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            details.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(details),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


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
    controls = {"fields", "all_fields", "num_chunks", "chunk_size", "_page", "_sort_fields"}.intersection(normalized)
    if controls:
        raise ValueError(f"Use top-level fields/limit/page/sort_field for result controls, not criteria: {', '.join(sorted(controls))}")
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


def mp_search_materials(payload: Dict[str, object]) -> tuple[str, dict[str, Any]]:
    """
    [material/discovery] Search Materials Project with flexible criteria and write the result table to CSV.
    """
    limit_source = "agent_selected" if "limit" in payload else "visible_default"
    params = MPSearchMaterialsInput(**payload)
    try:
        criteria = _normalize_criteria(params.criteria)
    except Exception as exc:
        _fail(
            "mp_search_materials",
            message=f"Invalid criteria: {exc}",
            error_code="invalid_criteria",
        )
    warnings: List[str] = []

    if not criteria:
        _fail(
            "mp_search_materials",
            message="Provide criteria.",
            error_code="missing_criteria",
        )

    field_pairs = _normalize_fields(params.fields)
    if not field_pairs:
        _fail(
            "mp_search_materials",
            message="fields must not be empty.",
            error_code="empty_fields",
        )
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

            docs = client.materials.summary.search(
                **criteria,
                fields=request_fields,
                all_fields=False,
                chunk_size=params.limit,
                num_chunks=1,
                _page=params.page,
                _sort_fields=params.sort_field,
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
        _fail(
            "mp_search_materials",
            message=f"Materials Project search failed: {exc}",
            data={"criteria": criteria},
            error_code="search_failed",
        )

    if isinstance(total, int):
        returned = written
        truncated = total > (params.page - 1) * params.limit + written
        count_value: Optional[int] = total
    else:
        returned = written
        truncated = written >= params.limit
        count_value = None

    if params.page == 1 and isinstance(total, int) and total <= written:
        source_completeness = "complete"
    elif limit_source == "agent_selected":
        source_completeness = "explicit_subset"
    elif total is None:
        source_completeness = "provider_bounded_unknown"
    else:
        source_completeness = "visible_default_subset"

    data = {
        "count": count_value,
        "page": params.page,
        "next_page": params.page + 1 if truncated else None,
        "returned": returned,
        "truncated": truncated,
        "requested_limit": params.limit,
        "limit_source": limit_source,
        "source_completeness": source_completeness,
        "criteria": criteria,
        "fields": [h for h, _ in field_pairs],
        "output_csv_rel": workspace_relpath(out_path),
        "preview_rows": preview_rows,
    }
    preview_line = ""
    if preview_rows:
        preview_line = f"\npreview_row_0={json.dumps(preview_rows[0], ensure_ascii=False)}"
    content = (
        "mp_search_materials completed.\n"
        f"returned={returned} total_count={count_value} truncated={truncated}\n"
        f"page={params.page} next_page={data['next_page']}\n"
        f"requested_limit={params.limit} limit_source={limit_source} "
        f"source_completeness={source_completeness}\n"
        f"output_csv_rel={data['output_csv_rel']}\n"
        f"preview_rows={len(preview_rows)}"
        f"{preview_line}"
    )
    return _success("mp_search_materials", content=content, data=data, warnings=warnings)


def mp_download_structure(payload: Dict[str, object]) -> tuple[str, dict[str, Any]]:
    """
    [material/discovery] Download one or more structures from Materials Project and write them under the workspace.
    Cell standardization is explicit; the default preserves the provider structure.
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
    warnings: List[str] = []

    try:
        with _mpr() as client:
            for mp_id in params.mp_ids:
                try:
                    structure = client.get_structure_by_material_id(mp_id)
                    if isinstance(structure, dict):
                        structure = Structure.from_dict(structure)
                    if params.cell == "as_returned":
                        c_structure = structure
                    else:
                        analyzer = SpacegroupAnalyzer(structure, symprec=params.symprec, angle_tolerance=params.angle_tolerance)
                        c_structure = analyzer.get_primitive_standard_structure() if params.cell == "primitive" else analyzer.get_conventional_standard_structure()
                    out_path = out_dir / f"{mp_id}.{ext}"
                    if out_path.exists() and not params.overwrite:
                        raise ValueError(f"Output exists: {workspace_relpath(out_path)}; use another directory or overwrite=true")
                    if fmt == "json":
                        out_path.write_text(json.dumps(c_structure.as_dict(), ensure_ascii=False, indent=2), encoding="utf-8")
                    else:
                        c_structure.to(fmt=fmt, filename=str(out_path))
                except Exception as exc:  # pragma: no cover - remote call
                    errors.append({"mp_id": mp_id, "error": str(exc)})
                    continue

                results.append(
                    {
                        "mp_id": mp_id,
                        "structure_rel": workspace_relpath(out_path),
                        "metadata": {
                            "formula": c_structure.composition.reduced_formula,
                            "natoms": len(c_structure),
                            "cell": params.cell,
                        },
                    }
                )
    except Exception as exc:
        _fail(
            "mp_download_structure",
            message=f"mp_download_structure failed: {exc}",
            data={"output_dir_rel": workspace_relpath(out_dir), "format": fmt},
            error_code="download_failed",
        )

    if errors:
        warnings.append(f"Partial failure: {len(errors)} of {len(params.mp_ids)} mp_ids failed.")

    summary_path = out_dir / "mp_download_structure_summary.json"
    summary_rel = ""
    try:
        summary_path.write_text(
            json.dumps(
                {
                    "format": fmt,
                    "output_dir_rel": workspace_relpath(out_dir),
                    "requested": len(params.mp_ids),
                    "downloaded": len(results),
                    "results": results,
                    "errors": errors,
                },
                ensure_ascii=False,
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )
        summary_rel = workspace_relpath(summary_path)
    except Exception as exc:
        warnings.append(f"Failed to persist MP download summary: {type(exc).__name__}: {exc}")

    data = {
        "format": fmt,
        "output_dir_rel": workspace_relpath(out_dir),
        "requested": len(params.mp_ids),
        "downloaded": len(results),
        "summary_json_rel": summary_rel,
        **compact_list_for_artifact(
            results,
            count_key="results_count",
            inline_key="results",
            preview_key="results_preview",
            truncated_key="results_truncated",
            full_rel_key="results_full_rel",
            full_rel=summary_rel or None,
        ),
        **compact_list_for_artifact(
            errors,
            count_key="errors_count",
            inline_key="errors",
            preview_key="errors_preview",
            truncated_key="errors_truncated",
            full_rel_key="errors_full_rel",
            full_rel=summary_rel or None,
        ),
    }
    if errors and not results:
        _fail(
            "mp_download_structure",
            message="Failed to download structures for all requested mp_ids.",
            data=data,
            error_code="all_downloads_failed",
        )

    downloaded_ids_all = [str(item.get("mp_id") or "") for item in results if isinstance(item, dict)]
    downloaded_paths_all = [str(item.get("structure_rel") or "") for item in results if isinstance(item, dict)]
    failed_ids_all = [str(item.get("mp_id") or "") for item in errors if isinstance(item, dict)]
    downloaded_ids = downloaded_ids_all[:3]
    downloaded_paths = downloaded_paths_all[:3]
    failed_ids = failed_ids_all[:3]

    content = (
        "mp_download_structure completed.\n"
        f"requested={data['requested']} downloaded={data['downloaded']} errors={len(errors)}\n"
        f"output_dir_rel={data['output_dir_rel']}\n"
        f"downloaded_examples={downloaded_ids}\n"
        f"downloaded_path_examples={downloaded_paths}\n"
        f"failed_examples={failed_ids}"
    )
    if summary_rel:
        content += f"\nsummary_json_rel={summary_rel}"
    return _success("mp_download_structure", content=content, data=data, warnings=warnings)


__all__ = [
    "MPSearchMaterialsInput",
    "MPDownloadStructureInput",
    "mp_search_materials",
    "mp_download_structure",
]
