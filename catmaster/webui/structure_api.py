from __future__ import annotations

from pathlib import Path, PurePosixPath
from typing import Any

from catmaster.structures.models import (
    SaveStructureRequest,
    StructureOpenRequest,
    TransformRequest,
)
from catmaster.structures.operations import transform_structure
from catmaster.structures.serialization import (
    StructureFormatLossError,
    StructureSerializationError,
    StructureVersionConflict,
    derived_summary,
    load_structure_document,
    save_structure_document,
    source_version,
    viewer_structure,
)
from catmaster.structures.trajectory import (
    trajectory_frame,
    trajectory_frame_count,
    trajectory_metadata,
)


def resolve_workspace_file(
    workspace: Path,
    relative_path: str,
    *,
    must_exist: bool,
    for_save: bool = False,
) -> tuple[Path, str]:
    raw = str(relative_path or "").replace("\\", "/").strip().lstrip("/")
    posix = PurePosixPath(raw)
    if not raw or posix.is_absolute() or ".." in posix.parts:
        raise StructureSerializationError("Choose a valid workspace-relative file path.")
    if for_save and (not posix.parts or posix.parts[0] != "files"):
        raise StructureSerializationError("Structure files must be saved inside the workspace files/ directory.")
    root = workspace.expanduser().resolve()
    candidate = (root / Path(*posix.parts)).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise StructureSerializationError("The requested file is outside this workspace.") from exc
    if must_exist and not candidate.is_file():
        raise StructureSerializationError(f"Structure file was not found: {raw}")
    return candidate, posix.as_posix()


def open_structure(workspace: Path, request: StructureOpenRequest) -> dict[str, Any]:
    path, normalized = resolve_workspace_file(workspace, request.path, must_exist=True)
    snapshot, warnings = load_structure_document(path, relative_path=normalized)
    is_vibration = path.name.upper() == "OUTCAR" or path.suffix.lower() == ".outcar"
    is_trajectory = path.name.upper() == "XDATCAR" or path.suffix.lower() in {
        ".traj",
        ".xdatcar",
    }
    if path.suffix.lower() in {".extxyz", ".xyz"}:
        is_trajectory = trajectory_frame_count(path) > 1
    return {
        "snapshot": snapshot.model_dump(mode="json"),
        "summary": derived_summary(snapshot),
        "viewer_structure": viewer_structure(snapshot),
        "warnings": warnings,
        "capabilities": {
            "editable": not is_vibration and not is_trajectory,
            "trajectory": is_trajectory,
            "vibration_fallback": is_vibration,
        },
    }


def apply_transform(request: TransformRequest) -> dict[str, Any]:
    return transform_structure(request)


def _molecule_projection_signature(projection: dict[str, Any]) -> tuple[tuple[str, ...], tuple[tuple[int, int, str], ...]]:
    atoms = tuple(
        str((site.get("species") or [{}])[0].get("element") or site.get("label") or "")
        for site in projection.get("sites") or []
    )
    bonds: list[tuple[int, int, str]] = []
    for bond in (projection.get("properties") or {}).get("bonds") or []:
        left, right = sorted(
            (int(bond.get("site_idx_1", -1)), int(bond.get("site_idx_2", -1)))
        )
        order = bond.get("order", 1)
        try:
            order_key = f"{float(order):.8g}"
        except (TypeError, ValueError):
            order_key = str(order).strip().lower()
        bonds.append((left, right, order_key))
    return atoms, tuple(sorted(bonds))


def save_structure(workspace: Path, request: SaveStructureRequest) -> dict[str, Any]:
    destination, normalized = resolve_workspace_file(
        workspace,
        request.destination_path,
        must_exist=False,
        for_save=True,
    )
    if (
        destination.is_file()
        and request.overwrite
        and request.expected_source_version.mtime_ns <= 0
    ):
        actual = source_version(destination)
        return {
            "requires_overwrite_confirmation": True,
            "path": normalized,
            "source_version": actual.model_dump(mode="json"),
        }
    snapshot = request.snapshot
    if snapshot.mode == "molecule" and request.viewer_structure:
        from catmaster.structures.molecules import snapshot_from_viewer

        expected_projection = viewer_structure(snapshot)
        if _molecule_projection_signature(expected_projection) != _molecule_projection_signature(
            request.viewer_structure
        ):
            raise StructureSerializationError(
                "The molecular 3D projection is stale relative to the MolBlock. "
                "Synchronize the 2D and 3D views before saving."
            )
        snapshot = snapshot_from_viewer(snapshot, request.viewer_structure)
    version, warnings = save_structure_document(
        snapshot,
        destination,
        overwrite=request.overwrite,
        expected_version=request.expected_source_version,
        accept_format_loss=request.accept_format_loss,
        cif_symprec=request.cif_symprec,
        cif_angle_tolerance=request.cif_angle_tolerance,
    )
    saved_snapshot, open_warnings = load_structure_document(destination, relative_path=normalized)
    return {
        "path": normalized,
        "source_version": version.model_dump(mode="json"),
        "snapshot": saved_snapshot.model_dump(mode="json"),
        "summary": derived_summary(saved_snapshot),
        "viewer_structure": viewer_structure(saved_snapshot),
        "warnings": warnings + open_warnings,
    }


def get_trajectory_meta(workspace: Path, relative_path: str) -> dict[str, Any]:
    path, normalized = resolve_workspace_file(workspace, relative_path, must_exist=True)
    payload = trajectory_metadata(path)
    payload["path"] = normalized
    return payload


def get_trajectory_frame(workspace: Path, relative_path: str, index: int) -> dict[str, Any]:
    path, normalized = resolve_workspace_file(workspace, relative_path, must_exist=True)
    payload = trajectory_frame(path, index)
    payload["path"] = normalized
    return payload


__all__ = [
    "StructureFormatLossError",
    "StructureSerializationError",
    "StructureVersionConflict",
    "apply_transform",
    "get_trajectory_frame",
    "get_trajectory_meta",
    "open_structure",
    "resolve_workspace_file",
    "save_structure",
]
