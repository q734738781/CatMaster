from __future__ import annotations

"""Local MLFF stage validation and private run-config materialization."""

import hashlib
import json
from pathlib import Path
from typing import Any, Mapping


_STRUCTURE_SUFFIXES = {".cif", ".extxyz", ".poscar", ".traj", ".vasp", ".xyz"}
_NEB_SUFFIXES = {".cif", ".poscar", ".vasp"}
_RUN_CONFIG_SCHEMA_VERSION = 1


def _canonical_json(payload: Mapping[str, Any]) -> str:
    return json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":"))


def _sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while chunk := handle.read(1024 * 1024):
            digest.update(chunk)
    return digest.hexdigest()


def _constraint_signature(atoms: Any) -> list[dict[str, Any]]:
    import numpy as np

    out: list[dict[str, Any]] = []
    for constraint in atoms.constraints or []:
        entry: dict[str, Any] = {"type": type(constraint).__name__}
        for attr in ("index", "indices", "mask", "a"):
            if not hasattr(constraint, attr):
                continue
            value = getattr(constraint, attr)
            if isinstance(value, np.ndarray):
                value = value.tolist()
            elif isinstance(value, tuple):
                value = list(value)
            entry[attr] = value
        out.append(entry)
    return out


def _direct_structure_files(input_dir: Path) -> list[Path]:
    children = sorted(input_dir.iterdir())
    nested = [path.name for path in children if path.is_dir()]
    if nested:
        raise ValueError(
            "MLFF SP/relax/MD/VIB/TS input/ accepts structure files directly and does not recurse; "
            f"found directories: {', '.join(nested)}"
        )
    files = [
        path
        for path in children
        if path.is_file() and (path.name in {"POSCAR", "CONTCAR"} or path.suffix.lower() in _STRUCTURE_SUFFIXES)
    ]
    if not files:
        raise ValueError("MLFF stage input/ contains no ASE-readable structure files.")
    stems: dict[str, str] = {}
    for path in files:
        stem = path.stem if path.suffix else path.name
        previous = stems.setdefault(stem, path.name)
        if previous != path.name:
            raise ValueError(f"MLFF input filenames map to the same output key: {previous!r} and {path.name!r}.")
    return files


def _read_one(path: Path) -> Any:
    from ase.io import read

    atoms = read(str(path), index=-1)
    if atoms is None:
        raise ValueError(f"ASE returned no structure for {path}.")
    return atoms


def _validate_neb(stage_dir: Path) -> list[tuple[str, Any]]:
    import numpy as np

    input_dir = stage_dir / "input"
    if not input_dir.is_dir():
        raise ValueError("MLFF NEB stage requires input/path/.")
    extra = [path.name for path in input_dir.iterdir() if path.name != "path"]
    if extra:
        raise ValueError("MLFF NEB input/ must contain exactly one path/ directory.")
    path_dir = input_dir / "path"
    if not path_dir.is_dir():
        raise ValueError("MLFF NEB stage requires input/path/.")
    nested = [path.name for path in path_dir.iterdir() if path.is_dir()]
    if nested:
        raise ValueError("Nested directories are forbidden inside input/path/.")
    indexed: list[tuple[int, Path]] = []
    for path in sorted(path_dir.iterdir()):
        if not path.is_file() or path.suffix.lower() not in _NEB_SUFFIXES or not path.stem.isdigit():
            continue
        indexed.append((int(path.stem), path))
    if len(indexed) < 3:
        raise ValueError("MLFF NEB requires endpoints plus at least one locally prepared intermediate image.")
    found = [index for index, _ in indexed]
    if found != list(range(len(indexed))):
        raise ValueError(f"MLFF NEB image numbering must be contiguous from 00; found {found}.")
    width = max(2, len(str(len(indexed) - 1)))
    expected_names = [f"{index:0{width}d}" for index in range(len(indexed))]
    actual_names = [path.stem for _, path in indexed]
    if actual_names != expected_names:
        raise ValueError(f"MLFF NEB images must use consistent zero-padded names: {', '.join(expected_names)}.")

    items = [(f"path/{path.name}", _read_one(path)) for _, path in indexed]
    first = items[0][1]
    first_numbers = first.get_atomic_numbers()
    first_cell = first.cell.array
    first_pbc = first.pbc
    first_constraints = _constraint_signature(first)
    for rel, atoms in items[1:]:
        if len(atoms) != len(first):
            raise ValueError(f"NEB image {rel} has a different atom count.")
        if not np.array_equal(atoms.get_atomic_numbers(), first_numbers):
            raise ValueError(f"NEB image {rel} has a different element ordering.")
        if not np.allclose(atoms.cell.array, first_cell, atol=1e-6, rtol=0):
            raise ValueError(f"NEB image {rel} has a different cell.")
        if not np.array_equal(atoms.pbc, first_pbc):
            raise ValueError(f"NEB image {rel} has different PBC settings.")
        if _constraint_signature(atoms) != first_constraints:
            raise ValueError(f"NEB image {rel} has different constraints.")
    return items


def _validate_stage(stage_dir: Path, operation: str) -> list[tuple[str, Any]]:
    input_dir = stage_dir / "input"
    if operation == "neb":
        return _validate_neb(stage_dir)
    if not input_dir.is_dir():
        raise ValueError(f"MLFF {operation} stage requires an input/ directory.")
    files = _direct_structure_files(input_dir)
    if operation == "md" and len(files) != 1:
        raise ValueError("MLFF MD stage requires exactly one trajectory source directly under input/.")
    structures = [(path.name, _read_one(path)) for path in files]
    if operation == "ts":
        if len(structures) != 1:
            raise ValueError("MLFF TS stage requires exactly one TS-like structure directly under input/.")
    if operation in {"vib", "ts"}:
        for _rel, atoms in structures:
            _validate_vibrational_constraints(atoms, operation=operation)
    return structures


def _validate_vibrational_constraints(atoms: Any, *, operation: str) -> None:
    import numpy as np
    from ase.constraints import FixAtoms, FixCartesian, FixScaled

    ndof = 3 * len(atoms)
    rows: list[np.ndarray] = []
    unsupported: list[str] = []

    def add_row(atom_index: int, direction: Any) -> None:
        row = np.zeros(ndof, dtype=float)
        row[3 * atom_index : 3 * atom_index + 3] = np.asarray(direction, dtype=float)
        rows.append(row)

    inverse_cell = None
    for constraint in atoms.constraints or []:
        if isinstance(constraint, FixAtoms):
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis in range(3):
                    add_row(int(atom_index), np.eye(3)[axis])
        elif isinstance(constraint, FixCartesian):
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        add_row(int(atom_index), np.eye(3)[axis])
        elif isinstance(constraint, FixScaled):
            if float(atoms.cell.volume) <= 1e-12:
                raise ValueError(
                    f"FixScaled constraints in MLFF {operation.upper()} require a valid nonzero cell."
                )
            if inverse_cell is None:
                inverse_cell = np.linalg.inv(np.asarray(atoms.cell.complete(), dtype=float))
            for atom_index in np.asarray(constraint.index, dtype=int):
                for axis, fixed in enumerate(np.asarray(constraint.mask, dtype=bool)):
                    if fixed:
                        add_row(int(atom_index), inverse_cell[:, axis])
        else:
            unsupported.append(type(constraint).__name__)
    if unsupported:
        raise ValueError(
            f"MLFF {operation.upper()} supports structure-file constraints "
            "FixAtoms, FixCartesian, and FixScaled; "
            f"unsupported: {', '.join(sorted(set(unsupported)))}"
        )
    if rows and np.linalg.matrix_rank(np.vstack(rows), tol=1e-12) >= ndof:
        raise ValueError(
            f"MLFF {operation.upper()} structure has no unconstrained Cartesian degrees of freedom."
        )


def _resolve_uma_items(
    structures: list[tuple[str, Any]],
    backend_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    defaults = dict(backend_config.get("defaults") or {})
    configured_items = dict(backend_config.get("items") or {})
    unknown = sorted(set(configured_items) - {rel for rel, _ in structures})
    if unknown:
        raise ValueError(
            "fairchem_uma backend_config.items contains paths absent from input/: " + ", ".join(unknown)
        )
    resolved: dict[str, dict[str, Any]] = {}
    for rel, _atoms in structures:
        item = dict(defaults)
        item.update(dict(configured_items.get(rel) or {}))
        task = str(item["uma_task"])
        charge = int(item.get("charge", 0))
        spin = int(item.get("spin", 0))
        if task == "omol" and spin < 1:
            raise ValueError(f"UMA item {rel!r} uses omol and therefore requires multiplicity-style spin >= 1.")
        if task != "omol" and (charge != 0 or spin != 0):
            raise ValueError(f"UMA item {rel!r} resolves to {task!r} and therefore requires charge=0 and spin=0.")
        resolved[rel] = {"uma_task": task, "charge": charge, "spin": spin}
    return resolved


def _resolve_mace_items(
    structures: list[tuple[str, Any]],
    backend_config: dict[str, Any],
) -> dict[str, dict[str, Any]]:
    supports_charge_spin = bool(backend_config.get("supports_charge_spin", False))
    defaults = dict(backend_config.get("defaults") or {})
    configured_items = dict(backend_config.get("items") or {})
    unknown = sorted(set(configured_items) - {rel for rel, _ in structures})
    if unknown:
        raise ValueError("MACE backend_config.items contains paths absent from input/: " + ", ".join(unknown))
    shared = {
        key: value
        for key, value in backend_config.items()
        if key not in {"defaults", "items"}
    }
    if not supports_charge_spin:
        if configured_items or int(defaults.get("charge", 0)) != 0 or int(defaults.get("spin", 0)) != 0:
            raise ValueError(
                "The selected MACE model does not accept charge/spin metadata; use omol-0."
            )
        return {rel: dict(shared) for rel, _ in structures}

    resolved: dict[str, dict[str, Any]] = {}
    for rel, _atoms in structures:
        metadata = dict(defaults)
        metadata.update(dict(configured_items.get(rel) or {}))
        charge = int(metadata.get("charge", 0))
        spin = int(metadata.get("spin", 0))
        if spin < 1:
            raise ValueError(
                f"MACE OMOL item {rel!r} requires multiplicity-style spin >= 1."
            )
        resolved[rel] = {
            **shared,
            "charge": charge,
            "spin": spin,
        }
    return resolved


def _explicit_paths(value: Any, *, prefix: str = "") -> list[str]:
    if not isinstance(value, dict):
        return [prefix] if prefix else []
    out: list[str] = []
    for key, child in value.items():
        path = f"{prefix}.{key}" if prefix else str(key)
        if isinstance(child, dict):
            out.extend(_explicit_paths(child, prefix=path))
        else:
            out.append(path)
    return out


def materialize_mlff_run_config(
    *,
    stage_dir: Path,
    task_name: str,
    resolved: Mapping[str, Any],
    explicit_overrides: Mapping[str, Any] | None = None,
) -> Path:
    """Validate a stage and write its tool-owned immutable runner input."""

    operation = str(resolved["operation"])
    backend = str(resolved["resolved_backend"])
    normalized = dict(resolved["normalized_template_overrides"])
    backend_config = dict(normalized["backend_config"])
    task_config = dict(normalized["task_config"])
    structures = _validate_stage(stage_dir, operation)

    checkpoint = str(backend_config.get("checkpoint_artifact") or "")
    if checkpoint:
        checkpoint_path = Path(checkpoint)
        if (
            checkpoint_path.is_absolute()
            or ".." in checkpoint_path.parts
            or len(checkpoint_path.parts) < 2
            or checkpoint_path.parts[0] != "models"
        ):
            raise ValueError("checkpoint_artifact must be a safe stage-relative file under models/.")
        staged_checkpoint = stage_dir / checkpoint_path
        if staged_checkpoint.is_symlink():
            raise ValueError("checkpoint_artifact must be a regular staged file, not a symbolic link.")
        if not staged_checkpoint.is_file():
            raise FileNotFoundError(f"Staged MACE checkpoint does not exist: {checkpoint}")
        try:
            staged_checkpoint.resolve().relative_to((stage_dir / "models").resolve())
        except ValueError as exc:
            raise ValueError("checkpoint_artifact resolves outside the stage models/ directory.") from exc
        backend_config["checkpoint_sha256"] = _sha256_file(staged_checkpoint)
        backend_config["checkpoint_size_bytes"] = int(staged_checkpoint.stat().st_size)

    item_backend_config: dict[str, Any]
    if backend == "mace":
        item_backend_config = _resolve_mace_items(structures, backend_config)
    elif backend == "fairchem_uma":
        item_metadata = _resolve_uma_items(structures, backend_config)
        shared = {key: value for key, value in backend_config.items() if key not in {"defaults", "items"}}
        item_backend_config = {
            rel: {**shared, **metadata}
            for rel, metadata in item_metadata.items()
        }
    else:
        item_backend_config = {rel: dict(backend_config) for rel, _ in structures}

    digest_source = {
        "schema_version": _RUN_CONFIG_SCHEMA_VERSION,
        "task_name": task_name,
        "operation": operation,
        "backend": backend,
        "backend_config": backend_config,
        "task_config": task_config,
        "items": item_backend_config,
    }
    digest = hashlib.sha256(_canonical_json(digest_source).encode("utf-8")).hexdigest()
    payload = {
        **digest_source,
        "config_digest": digest,
        "provenance": {
            "explicit_override_paths": _explicit_paths(dict(explicit_overrides or {})),
        },
    }
    output = stage_dir / ".catmaster" / "generated" / "run_config.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(payload, ensure_ascii=False, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


__all__ = ["materialize_mlff_run_config"]
