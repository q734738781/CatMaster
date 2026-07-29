from __future__ import annotations

from collections import OrderedDict
from dataclasses import dataclass
from io import StringIO
import math
from pathlib import Path
from threading import RLock
from typing import Any, Iterator

import numpy as np

from .serialization import snapshot_from_atoms, source_version, viewer_structure

MAX_PROPERTY_POINTS = 2_000
MAX_TEXT_INDEX_CACHE = 8


@dataclass(frozen=True)
class _TextFrameIndex:
    format: str
    spans: tuple[tuple[int, int], ...]
    header: bytes = b""


_TEXT_INDEX_CACHE: OrderedDict[tuple[str, int, int], _TextFrameIndex] = OrderedDict()
_TEXT_INDEX_LOCK = RLock()


def _is_extxyz(path: Path) -> bool:
    return path.suffix.lower() in {".extxyz", ".xyz"}


def _is_xdatcar(path: Path) -> bool:
    return path.name.upper() == "XDATCAR" or path.suffix.lower() == ".xdatcar"


def _scan_extxyz(path: Path) -> _TextFrameIndex:
    spans: list[tuple[int, int]] = []
    with path.open("rb") as handle:
        while True:
            start = handle.tell()
            count_line = handle.readline()
            if not count_line:
                break
            if not count_line.strip():
                continue
            try:
                atom_count = int(count_line.strip())
            except ValueError as exc:
                raise ValueError(
                    f"{path.name} is not a valid extended XYZ trajectory near byte {start}."
                ) from exc
            if atom_count < 0:
                raise ValueError(f"{path.name} contains a negative atom count.")
            if not handle.readline():
                raise ValueError(f"{path.name} ends before the frame comment line.")
            for _ in range(atom_count):
                if not handle.readline():
                    raise ValueError(f"{path.name} ends inside a frame with {atom_count} atoms.")
            spans.append((start, handle.tell() - start))
    return _TextFrameIndex(format="extxyz", spans=tuple(spans))


def _scan_xdatcar(path: Path) -> _TextFrameIndex:
    from ase.io import read as ase_read

    try:
        first = ase_read(str(path), format="vasp-xdatcar", index=0)
    except Exception as exc:
        raise ValueError(f"Could not read the first XDATCAR frame: {exc}") from exc
    atom_count = len(first)
    if atom_count <= 0:
        return _TextFrameIndex(format="vasp-xdatcar", spans=())

    spans: list[tuple[int, int]] = []
    header_end = -1
    with path.open("rb") as handle:
        while True:
            marker_start = handle.tell()
            line = handle.readline()
            if not line:
                break
            lowered = line.strip().lower()
            if b"configuration=" not in lowered:
                continue
            if not (lowered.startswith(b"direct") or lowered.startswith(b"cartesian")):
                continue
            if header_end < 0:
                header_end = marker_start
            for _ in range(atom_count):
                if not handle.readline():
                    raise ValueError(
                        f"{path.name} ends inside a configuration with {atom_count} atoms."
                    )
            spans.append((marker_start, handle.tell() - marker_start))
    if header_end < 0:
        raise ValueError(f"{path.name} contains no Direct/Cartesian configuration frames.")
    with path.open("rb") as handle:
        header = handle.read(header_end)
    return _TextFrameIndex(format="vasp-xdatcar", spans=tuple(spans), header=header)


def _text_frame_index(path: Path) -> _TextFrameIndex:
    resolved = path.expanduser().resolve()
    stat = resolved.stat()
    key = (str(resolved), int(stat.st_mtime_ns), int(stat.st_size))
    with _TEXT_INDEX_LOCK:
        cached = _TEXT_INDEX_CACHE.get(key)
        if cached is not None:
            _TEXT_INDEX_CACHE.move_to_end(key)
            return cached

    if _is_extxyz(resolved):
        index = _scan_extxyz(resolved)
    elif _is_xdatcar(resolved):
        index = _scan_xdatcar(resolved)
    else:
        raise ValueError(f"{resolved.name} does not use a CatMaster text-frame index.")

    with _TEXT_INDEX_LOCK:
        for old_key in [item for item in _TEXT_INDEX_CACHE if item[0] == str(resolved) and item != key]:
            _TEXT_INDEX_CACHE.pop(old_key, None)
        _TEXT_INDEX_CACHE[key] = index
        _TEXT_INDEX_CACHE.move_to_end(key)
        while len(_TEXT_INDEX_CACHE) > MAX_TEXT_INDEX_CACHE:
            _TEXT_INDEX_CACHE.popitem(last=False)
    return index


def _indexed_frame_at(path: Path, index: int):
    if index < 0:
        raise IndexError("Trajectory frame index cannot be negative.")
    frame_index = _text_frame_index(path)
    if index >= len(frame_index.spans):
        raise IndexError(
            f"Trajectory frame {index} is outside 0–{max(0, len(frame_index.spans) - 1)}."
        )
    offset, length = frame_index.spans[index]
    with path.open("rb") as handle:
        handle.seek(offset)
        payload = frame_index.header + handle.read(length)
    from ase.io import read as ase_read

    try:
        return ase_read(
            StringIO(payload.decode("utf-8", errors="strict")),
            format=frame_index.format,
            index=0,
        )
    except Exception as exc:
        raise ValueError(f"Could not decode trajectory frame {index}: {exc}") from exc


def _trajectory(path: Path):
    from ase.io.trajectory import Trajectory

    return Trajectory(str(path), mode="r")


def _iter_frames(path: Path) -> Iterator[Any]:
    from ase.io import iread

    yield from iread(str(path), index=":")


def _frame_count(path: Path) -> int:
    if path.suffix.lower() == ".traj":
        with _trajectory(path) as trajectory:
            return int(len(trajectory))
    if _is_extxyz(path) or _is_xdatcar(path):
        return len(_text_frame_index(path).spans)
    return sum(1 for _ in _iter_frames(path))


def _frame_at(path: Path, index: int):
    if index < 0:
        raise IndexError("Trajectory frame index cannot be negative.")
    if path.suffix.lower() == ".traj":
        with _trajectory(path) as trajectory:
            if index >= len(trajectory):
                raise IndexError(f"Trajectory frame {index} is outside 0–{max(0, len(trajectory) - 1)}.")
            return trajectory[index]
    if _is_extxyz(path) or _is_xdatcar(path):
        return _indexed_frame_at(path, index)
    from ase.io import read

    try:
        return read(str(path), index=int(index))
    except (IndexError, StopIteration) as exc:
        raise IndexError(f"Trajectory frame {index} does not exist.") from exc


def _json_scalar(value: Any) -> float | int | str | bool | None:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, (float, np.floating)):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, np.integer):
        return int(value)
    return None


def frame_properties(atoms: Any) -> dict[str, float | int | str | bool | None]:
    properties: dict[str, float | int | str | bool | None] = {}
    for key, value in dict(getattr(atoms, "info", {}) or {}).items():
        scalar = _json_scalar(value)
        if scalar is not None:
            properties[str(key)] = scalar
    calculator = getattr(atoms, "calc", None)
    results = dict(getattr(calculator, "results", {}) or {}) if calculator is not None else {}
    energy = results.get("energy", results.get("free_energy"))
    scalar_energy = _json_scalar(energy)
    if scalar_energy is not None:
        properties["energy"] = scalar_energy
    forces = results.get("forces")
    if forces is not None:
        array = np.asarray(forces, dtype=float)
        if array.shape == (len(atoms), 3) and len(array):
            norms = np.linalg.norm(array, axis=1)
            properties["max_force"] = float(norms.max())
            properties["mean_force"] = float(norms.mean())
    if atoms.has("momenta"):
        try:
            properties["temperature"] = float(atoms.get_temperature())
        except Exception:
            pass
    if any(bool(item) for item in atoms.pbc):
        properties["cell_volume"] = float(atoms.get_volume())
    return properties


def _frame_payload(atoms: Any, *, index: int) -> dict[str, Any]:
    symbols = [str(symbol) for symbol in atoms.get_chemical_symbols()]
    return {
        "index": int(index),
        "atom_count": len(atoms),
        "formula": str(atoms.get_chemical_formula() or ""),
        "symbols": symbols,
        "positions": [[float(item) for item in row] for row in atoms.get_positions()],
        "cell": [[float(item) for item in row] for row in atoms.cell.array],
        "pbc": [bool(item) for item in atoms.pbc],
        "properties": frame_properties(atoms),
    }


def trajectory_metadata(path: Path) -> dict[str, Any]:
    path = path.expanduser().resolve()
    total_frames = _frame_count(path)
    if total_frames <= 0:
        raise ValueError("Trajectory contains no frames.")
    first = _frame_at(path, 0)
    stride = max(1, math.ceil(total_frames / MAX_PROPERTY_POINTS))
    property_rows: list[dict[str, Any]] = []
    if path.suffix.lower() == ".traj" or _is_extxyz(path) or _is_xdatcar(path):
        if path.suffix.lower() == ".traj":
            with _trajectory(path) as trajectory:
                for index in range(0, total_frames, stride):
                    values = frame_properties(trajectory[index])
                    if values:
                        property_rows.append({"index": index, **values})
        else:
            for index in range(0, total_frames, stride):
                values = frame_properties(_frame_at(path, index))
                if values:
                    property_rows.append({"index": index, **values})
    else:
        for index, atoms in enumerate(_iter_frames(path)):
            if index % stride:
                continue
            values = frame_properties(atoms)
            if values:
                property_rows.append({"index": index, **values})
    if total_frames > 1 and (not property_rows or property_rows[-1].get("index") != total_frames - 1):
        last_values = frame_properties(_frame_at(path, total_frames - 1))
        if last_values:
            property_rows.append({"index": total_frames - 1, **last_values})
    version = source_version(path)
    return {
        "path": path.name,
        "source_version": version.model_dump(mode="json"),
        "total_frames": total_frames,
        "atom_count": len(first),
        "formula": str(first.get_chemical_formula() or ""),
        "pbc": [bool(item) for item in first.pbc],
        "property_stride": stride,
        "property_series": property_rows,
        "random_access": path.suffix.lower() == ".traj" or _is_extxyz(path) or _is_xdatcar(path),
    }


def trajectory_frame_count(path: Path) -> int:
    """Return the real frame count without materializing the trajectory."""
    return _frame_count(path.expanduser().resolve())


def trajectory_frame(path: Path, index: int) -> dict[str, Any]:
    path = path.expanduser().resolve()
    atoms = _frame_at(path, int(index))
    payload = _frame_payload(atoms, index=int(index))
    version = source_version(path)
    payload["source_version"] = version.model_dump(mode="json")
    snapshot = snapshot_from_atoms(
        atoms,
        fmt="trajectory-frame",
        path=f"{path.name} · frame {index}",
        version=version,
    )
    payload["snapshot"] = snapshot.model_dump(mode="json")
    payload["viewer_structure"] = viewer_structure(snapshot)
    return payload


__all__ = [
    "MAX_PROPERTY_POINTS",
    "MAX_TEXT_INDEX_CACHE",
    "frame_properties",
    "trajectory_frame_count",
    "trajectory_frame",
    "trajectory_metadata",
]
