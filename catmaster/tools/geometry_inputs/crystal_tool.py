from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure
from pymatgen.io.vasp import Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class SupercellInput(BaseModel):
    """[structure/modeling] Create supercells from structure file(s) while preserving selective dynamics in POSCAR/VASP outputs.
    Provide exactly one of structure_file or structure_dir. When structure_dir is used, output_dir is required.
    Batch outputs are written to output_dir/<structure_id>.vasp and a summary JSON is written to
    output_dir/batch_supercell.json.

    The JSON format is:
    {
      "results": [
        {"input_rel": "a/CO.vasp", "structure_id": "a__CO", "output_rel": "out/a__CO.vasp", "natoms": 2}
      ]
    }

    """

    structure_file: Optional[str] = Field(
        None, description="Bulk structure file (POSCAR/CIF/etc.), workspace-relative."
    )
    structure_dir: Optional[str] = Field(
        None, description="Directory containing structure files for batch supercell generation."
    )
    supercell: List[int] = Field(..., min_length=3, max_length=3, description="Supercell replication [a,b,c].")
    output_path: Optional[str] = Field(
        None, description="Output structure path for single structure (workspace-relative, e.g., bulk_supercell.vasp)."
    )
    output_dir: Optional[str] = Field(
        None,
        description=(
            "Output directory for batch mode. Outputs are written as <output_dir>/<structure_id>.vasp where "
            "structure_id encodes the relative input path (without suffix) using '__'."
        ),
    )


def _error_message(message: str, *, data: Dict[str, Any] | None = None) -> str:
    lines = [str(message).strip()]
    if isinstance(data, dict):
        for key in ("structure_dir_rel", "input_rel", "output_rel", "output_dir_rel"):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}={value}")
    return "\n".join(lines)


def _collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"POSCAR", "CONTCAR"}:
            files.append(path)
            continue
        if path.suffix.lower() in {".vasp", ".poscar", ".cif"}:
            files.append(path)
    return sorted(files, key=lambda p: str(p))


def _structure_id_from(rel_path: Path) -> str:
    return "__".join(rel_path.with_suffix("").parts)


def _write_structure(path: Path, structure: Structure) -> None:
    if path.suffix.lower() in {".vasp", ".poscar", ""}:
        Poscar(structure).write_file(str(path))
        return
    structure.to(filename=str(path))


def supercell(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[structure/modeling] Create supercells for one structure or a directory batch."""
    try:
        params = SupercellInput(**payload)
        if (params.structure_file is None) == (params.structure_dir is None):
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message("Provide exactly one of structure_file or structure_dir."),
                    artifact={"tool_name": "supercell", "data": {}},
                    error_code="invalid_input_mode",
                )

        matrix = np.diag([int(x) for x in params.supercell])

        if params.structure_dir is not None:
            if params.output_dir is None:
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message("output_dir is required when structure_dir is provided."),
                    artifact={"tool_name": "supercell", "data": {}},
                    error_code="missing_output_dir",
                )
            structure_root = resolve_workspace_path(params.structure_dir, must_exist=True)
            if not structure_root.is_dir():
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message(
                        f"structure_dir is not a directory: {structure_root}",
                        data={"structure_dir_rel": workspace_relpath(structure_root)},
                    ),
                    artifact={
                        "tool_name": "supercell",
                        "data": {"structure_dir_rel": workspace_relpath(structure_root)},
                    },
                    error_code="invalid_structure_dir",
                )
            output_root = resolve_workspace_path(params.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            structures = _collect_structure_files(structure_root)
            if not structures:
                raise CatMasterToolExecutionError(
                    tool_name="supercell",
                    public_message=_error_message(
                        "No structure files found in structure_dir.",
                        data={"structure_dir_rel": workspace_relpath(structure_root)},
                    ),
                    artifact={
                        "tool_name": "supercell",
                        "data": {"structure_dir_rel": workspace_relpath(structure_root)},
                    },
                    error_code="no_structures",
                )

            results = []
            errors = []
            for structure_path in structures:
                rel_path = structure_path.relative_to(structure_root)
                structure_id = _structure_id_from(rel_path)
                out_path = output_root / f"{structure_id}.vasp"
                try:
                    structure = Structure.from_file(structure_path)
                    structure.make_supercell(matrix)
                    out_path.parent.mkdir(parents=True, exist_ok=True)
                    _write_structure(out_path, structure)
                    results.append(
                        {
                            "input_rel": str(rel_path),
                            "structure_id": structure_id,
                            "output_rel": workspace_relpath(out_path),
                            "natoms": len(structure),
                        }
                    )
                except Exception as exc:
                    errors.append({"input_rel": str(rel_path), "error": str(exc)})

            batch_json = output_root / "batch_supercell.json"
            try:
                batch_json.write_text(
                    json.dumps({"results": results, "errors": errors}, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception:
                pass

            data: dict[str, Any] = {
                "structure_dir_rel": workspace_relpath(structure_root),
                "output_dir_rel": workspace_relpath(output_root),
                "supercell": params.supercell,
                "structures_found": len(structures),
                "structures_processed": len(results),
                "batch_json_rel": workspace_relpath(batch_json) if batch_json.exists() else None,
                "errors_count": len(errors),
            }
            if errors:
                data["errors"] = errors
            first_output = results[0]["output_rel"] if results else ""
            lines = [
                "supercell completed.",
                f"structures_processed={len(results)} structures_found={len(structures)} errors_count={len(errors)}",
                f"output_dir_rel={data['output_dir_rel']}",
            ]
            if data["batch_json_rel"]:
                lines.append(f"batch_json_rel={data['batch_json_rel']}")
            if first_output:
                lines.append(f"first_output_rel={first_output}")
            content = "\n".join(lines)
            return content, {"tool_name": "supercell", "data": data}

        if params.output_path is None:
            raise CatMasterToolExecutionError(
                tool_name="supercell",
                public_message=_error_message("output_path is required when structure_file is provided."),
                artifact={"tool_name": "supercell", "data": {}},
                error_code="missing_output_path",
            )
        structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        structure = Structure.from_file(structure_path)
        structure.make_supercell(matrix)
        _write_structure(out_path, structure)

        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_rel": workspace_relpath(out_path),
            "supercell": params.supercell,
            "natoms": len(structure),
        }
        content = (
            "supercell completed.\n"
            f"input_rel={data['input_rel']} output_rel={data['output_rel']} natoms={data['natoms']}"
        )
        return content, {"tool_name": "supercell", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name="supercell",
            public_message=_error_message(f"supercell failed: {exc}"),
            artifact={"tool_name": "supercell", "data": {}},
            error_code="supercell_failed",
        )


__all__ = ["SupercellInput", "supercell"]
