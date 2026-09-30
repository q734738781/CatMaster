from __future__ import annotations

from .batch_paths import batch_names

import json
import re
from pathlib import Path
from typing import Any

from ase import Atoms
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field, field_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_records_for_artifact, resolve_workspace_path, workspace_relpath

from .native_stage import require_fresh_stage

_SUPPORTED_EXTS = {".xyz", ".mol", ".sdf", ".mol2", ".pdb"}
_BLOCK_START_RE = re.compile(r"(?im)^\s*%(geom|neb)\b")


def _tool_error(tool_name: str, message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message=str(message).strip(),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _read_molecule(path: Path) -> Atoms:
    try:
        atoms = ase_read(str(path))
    except Exception as exc:
        raise ValueError(f"Failed to read molecular structure {workspace_relpath(path)}: {exc}") from exc
    if any(bool(value) for value in atoms.get_pbc()):
        raise ValueError(
            f"ORCA molecular preparation does not accept active periodic boundaries: {workspace_relpath(path)}. "
            "Construct an explicit finite cluster first."
        )
    return atoms


def _discover_molecules(path: Path) -> list[Path]:
    if path.is_file():
        if path.suffix.lower() not in _SUPPORTED_EXTS:
            raise ValueError(f"Unsupported molecular structure format: {workspace_relpath(path)}")
        return [path]
    return sorted(
        candidate
        for candidate in path.rglob("*")
        if candidate.is_file() and candidate.suffix.lower() in _SUPPORTED_EXTS
    )


def _validate_simple_keywords(value: list[str]) -> list[str]:
    normalized: list[str] = []
    for raw in value:
        keyword = str(raw or "").strip()
        if not keyword or any(character.isspace() for character in keyword):
            raise ValueError(f"Every simple_keywords item must be one non-empty native token: {raw!r}")
        if keyword.startswith(("!", "%", "*", "#")):
            raise ValueError(f"simple_keywords tokens must not include an ORCA line prefix: {keyword!r}")
        normalized.append(keyword)
    return normalized


def _validate_native_blocks(value: list[str]) -> list[str]:
    blocks = [str(block) for block in value]
    if any(not block.strip() for block in blocks):
        raise ValueError("input_blocks cannot contain an empty block")
    counts: dict[str, int] = {}
    for match in _BLOCK_START_RE.finditer("\n".join(blocks)):
        name = match.group(1).lower()
        counts[name] = counts.get(name, 0) + 1
    duplicated = sorted(name for name, count in counts.items() if count > 1)
    if duplicated:
        raise ValueError("Duplicate native ORCA block ownership is ambiguous: " + ", ".join(f"%{name}" for name in duplicated))
    return blocks


class OrcaPrepareInput(BaseModel):
    """[orca/prepare] Prepare molecular ORCA stages from native simple keywords, blocks, and an explicit electronic state."""

    model_config = ConfigDict(extra="forbid")

    input_path: str = Field(..., description="Single nonperiodic molecular structure or a directory of such structures.")
    output_root: str = Field(..., description="Fresh empty stage directory, or fresh parent for a structure batch.")
    simple_keywords: list[str] = Field(
        ...,
        description="Complete ordered native ORCA simple-input tokens written after ! without additions.",
    )
    input_blocks: list[str] = Field(
        default_factory=list,
        description="Complete native ORCA blocks written verbatim before the xyzfile directive; omit or pass [] when unused.",
    )
    charge: int = Field(..., description="Explicit total molecular charge.")
    multiplicity: int = Field(..., ge=1, description="Explicit spin multiplicity.")

    @field_validator("simple_keywords")
    @classmethod
    def _keywords(cls, value: list[str]) -> list[str]:
        return _validate_simple_keywords(value)

    @field_validator("input_blocks")
    @classmethod
    def _blocks(cls, value: list[str]) -> list[str]:
        return _validate_native_blocks(value)


class OrcaNebTSPrepareInput(BaseModel):
    """[orca/prepare] Stage mapped ORCA NEB endpoints with an agent-authored native %neb block."""

    model_config = ConfigDict(extra="forbid")

    reactant_path: str = Field(..., description="Nonperiodic reactant structure with the intended atom ordering.")
    product_path: str = Field(..., description="Nonperiodic product structure with the same per-index elements.")
    output_root: str = Field(..., description="Fresh empty ORCA NEB stage directory.")
    simple_keywords: list[str] = Field(..., description="Complete ordered native ORCA simple-input tokens written after !.")
    neb_block: str = Field(
        ...,
        description="Complete native %neb block. Reference the staged product endpoint as product.xyz.",
    )
    input_blocks: list[str] = Field(
        default_factory=list,
        description="Other complete native ORCA blocks; do not include a second %neb block.",
    )
    charge: int = Field(..., description="Explicit total molecular charge.")
    multiplicity: int = Field(..., ge=1, description="Explicit spin multiplicity.")

    @field_validator("simple_keywords")
    @classmethod
    def _keywords(cls, value: list[str]) -> list[str]:
        return _validate_simple_keywords(value)

    @field_validator("neb_block")
    @classmethod
    def _neb(cls, value: str) -> str:
        block = str(value or "")
        matches = list(re.finditer(r"(?im)^\s*%neb\b", block))
        if len(matches) != 1:
            raise ValueError("neb_block must contain exactly one native %neb block")
        return block

    @field_validator("input_blocks")
    @classmethod
    def _blocks(cls, value: list[str]) -> list[str]:
        blocks = _validate_native_blocks(value)
        if re.search(r"(?im)^\s*%neb\b", "\n".join(blocks)):
            raise ValueError("input_blocks must not contain a second %neb block")
        return blocks


def _render_input(
    *,
    simple_keywords: list[str],
    input_blocks: list[str],
    xyz_name: str,
    charge: int,
    multiplicity: int,
) -> str:
    lines = ["! " + " ".join(simple_keywords)]
    for block in input_blocks:
        lines.append(block.rstrip())
    lines.append(f"* xyzfile {int(charge)} {int(multiplicity)} {xyz_name}")
    return "\n".join(lines) + "\n"


def _safe_stage_name(source: Path, root: Path) -> str:
    relative = source.relative_to(root).with_suffix("")
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", relative.as_posix()).strip("._") or "molecule"


def _write_stage(
    *,
    source: Path,
    atoms: Atoms,
    stage_dir: Path,
    simple_keywords: list[str],
    input_blocks: list[str],
    charge: int,
    multiplicity: int,
) -> dict[str, Any]:
    stage_dir.mkdir(parents=True, exist_ok=True)
    xyz_path = stage_dir / "input.xyz"
    input_file = stage_dir / "job.inp"
    ase_write(str(xyz_path), atoms, format="xyz")
    input_file.write_text(
        _render_input(
            simple_keywords=simple_keywords,
            input_blocks=input_blocks,
            xyz_name=xyz_path.name,
            charge=charge,
            multiplicity=multiplicity,
        ),
        encoding="utf-8",
    )
    return {
        "source_path": workspace_relpath(source),
        "stage_path": workspace_relpath(stage_dir),
        "input_path": workspace_relpath(input_file),
        "coordinate_path": workspace_relpath(xyz_path),
    }


def orca_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[orca/prepare] Prepare molecular ORCA stages from native keywords, blocks, and explicit charge/spin."""
    tool_name = "orca_prepare"
    try:
        params = OrcaPrepareInput(**payload)
        source_root = resolve_workspace_path(params.input_path, must_exist=True)
        sources = _discover_molecules(source_root)
        if not sources:
            raise ValueError("No supported nonperiodic molecular structures were found")
        output_root = resolve_workspace_path(params.output_root)
        require_fresh_stage(output_root, source_paths=sources)
        atoms_by_source = [(source, _read_molecule(source)) for source in sources]
        names = batch_names(sources, source_root, lambda rel: _safe_stage_name(source_root / rel, source_root)) if source_root.is_dir() else {}
        records: list[dict[str, Any]] = []
        for source, atoms in atoms_by_source:
            stage_dir = output_root if source_root.is_file() else output_root / names[source]
            records.append(
                _write_stage(
                    source=source,
                    atoms=atoms,
                    stage_dir=stage_dir,
                    simple_keywords=params.simple_keywords,
                    input_blocks=params.input_blocks,
                    charge=params.charge,
                    multiplicity=params.multiplicity,
                )
            )
        manifest = output_root / "orca_prepare_manifest.json"
        _write_json(manifest, {"records": records})
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            tool_name,
            f"orca_prepare failed: {exc}",
            data={"input_path": str(payload.get("input_path") or ""), "output_root": str(payload.get("output_root") or "")},
            error_code="orca_prepare_failed",
        )

    data = {
        "output_root": workspace_relpath(output_root),
        "manifest_path": workspace_relpath(manifest),
        "prepared_count": len(records),
        **compact_records_for_artifact(records, full_records_rel=workspace_relpath(manifest)),
    }
    content = (
        "orca_prepare completed.\n"
        f"prepared_count={len(records)} output_root={data['output_root']}\n"
        f"manifest_path={data['manifest_path']}"
    )
    return content, {"tool_name": tool_name, "data": data}


def orca_nebts_prepare(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[orca/prepare] Stage mapped ORCA NEB endpoints with an agent-authored native %neb block."""
    tool_name = "orca_nebts_prepare"
    try:
        params = OrcaNebTSPrepareInput(**payload)
        reactant = resolve_workspace_path(params.reactant_path, must_exist=True)
        product = resolve_workspace_path(params.product_path, must_exist=True)
        reactant_atoms = _read_molecule(reactant)
        product_atoms = _read_molecule(product)
        if len(reactant_atoms) != len(product_atoms):
            raise ValueError("NEB endpoints must have identical atom counts")
        if reactant_atoms.get_chemical_symbols() != product_atoms.get_chemical_symbols():
            raise ValueError("NEB endpoints must have identical per-index element ordering")
        output_root = resolve_workspace_path(params.output_root)
        require_fresh_stage(output_root, source_paths=(reactant, product))
        output_root.mkdir(parents=True, exist_ok=True)
        reactant_xyz = output_root / "reactant.xyz"
        product_xyz = output_root / "product.xyz"
        ase_write(str(reactant_xyz), reactant_atoms, format="xyz")
        ase_write(str(product_xyz), product_atoms, format="xyz")
        input_file = output_root / "job.inp"
        input_file.write_text(
            _render_input(
                simple_keywords=params.simple_keywords,
                input_blocks=[params.neb_block, *params.input_blocks],
                xyz_name=reactant_xyz.name,
                charge=params.charge,
                multiplicity=params.multiplicity,
            ),
            encoding="utf-8",
        )
        manifest = output_root / "orca_nebts_manifest.json"
        _write_json(
            manifest,
            {
                "input_file": "job.inp",
                "reactant_file": "reactant.xyz",
                "product_file": "product.xyz",
            },
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _tool_error(
            tool_name,
            f"orca_nebts_prepare failed: {exc}",
            data={"output_root": str(payload.get("output_root") or "")},
            error_code="orca_nebts_prepare_failed",
        )

    data = {
        "stage_path": workspace_relpath(output_root),
        "input_path": workspace_relpath(input_file),
        "reactant_path": workspace_relpath(reactant_xyz),
        "product_path": workspace_relpath(product_xyz),
        "manifest_path": workspace_relpath(manifest),
    }
    content = (
        "orca_nebts_prepare completed.\n"
        f"stage_path={data['stage_path']} input_path={data['input_path']}\n"
        f"reactant_path={data['reactant_path']} product_path={data['product_path']}"
    )
    return content, {"tool_name": tool_name, "data": data}


__all__ = ["OrcaPrepareInput", "OrcaNebTSPrepareInput", "orca_prepare", "orca_nebts_prepare"]
