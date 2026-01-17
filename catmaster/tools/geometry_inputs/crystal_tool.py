from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Any, List, Optional

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.core import Structure

from catmaster.tools.base import resolve_workspace_path, workspace_relpath, create_tool_output


class SupercellInput(BaseModel):
    """Create supercells from structure file(s). THIS WILL LOSE ALL SELECTIVE DYNAMICS INFORMATION.
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


def supercell(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = SupercellInput(**payload)
        if (params.structure_file is None) == (params.structure_dir is None):
            return create_tool_output(
                "supercell",
                success=False,
                error="Provide exactly one of structure_file or structure_dir.",
            )

        matrix = np.diag([int(x) for x in params.supercell])

        if params.structure_dir is not None:
            if params.output_dir is None:
                return create_tool_output(
                    "supercell",
                    success=False,
                    error="output_dir is required when structure_dir is provided.",
                )
            structure_root = resolve_workspace_path(params.structure_dir, must_exist=True)
            if not structure_root.is_dir():
                return create_tool_output(
                    "supercell",
                    success=False,
                    error=f"structure_dir is not a directory: {structure_root}",
                )
            output_root = resolve_workspace_path(params.output_dir)
            output_root.mkdir(parents=True, exist_ok=True)
            structures = _collect_structure_files(structure_root)
            if not structures:
                return create_tool_output(
                    "supercell",
                    success=False,
                    error="No structure files found in structure_dir.",
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
                    structure.to(fmt="poscar", filename=str(out_path))
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

            data = {
                "structure_dir_rel": workspace_relpath(structure_root),
                "output_dir_rel": workspace_relpath(output_root),
                "supercell": params.supercell,
                "structures_found": len(structures),
                "structures_processed": len(results),
                "batch_json_rel": workspace_relpath(batch_json) if batch_json.exists() else None,
                "errors_count": len(errors),
            }
            return create_tool_output("supercell", success=True, data=data)

        if params.output_path is None:
            return create_tool_output(
                "supercell",
                success=False,
                error="output_path is required when structure_file is provided.",
            )
        structure_path = resolve_workspace_path(params.structure_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        structure = Structure.from_file(structure_path)
        structure.make_supercell(matrix)
        structure.to(fmt="poscar", filename=str(out_path))

        data = {
            "input_rel": workspace_relpath(structure_path),
            "output_rel": workspace_relpath(out_path),
            "supercell": params.supercell,
            "natoms": len(structure),
        }
        return create_tool_output("supercell", success=True, data=data)
    except Exception as exc:
        return create_tool_output("supercell", success=False, error=str(exc))


__all__ = ["SupercellInput", "supercell"]
