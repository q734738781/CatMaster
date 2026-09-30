#!/usr/bin/env python3
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

try:
    from ase.db import connect
    from ase.io import write
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: ase. Install it first, e.g. `pip install ase`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Export structures from an ASE database to extxyz with MACE-compatible "
            "REF_energy / REF_forces / REF_stress labels."
        )
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        required=True,
        help="Input ASE database path.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output extxyz path.",
    )
    parser.add_argument(
        "--selection",
        type=str,
        default="",
        help="ASE DB selection string, e.g. 'source_dirname=A__S208__Na3p5__r05'.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=0,
        help="Optional maximum number of rows to export (<=0 means all).",
    )
    parser.add_argument(
        "--head",
        type=str,
        default="",
        help="Optional MACE head label to store in atoms.info['head'].",
    )
    parser.add_argument(
        "--config-type",
        type=str,
        default="dft",
        help="Value written to atoms.info['config_type'].",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output file if it already exists.",
    )
    return parser.parse_args()


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")
        path.unlink()


def _copy_reference_properties(row):
    atoms = row.toatoms()

    atoms.info["config_type"] = atoms.info.get("config_type", "dft")
    if hasattr(row, "frame_uid"):
        atoms.info["frame_uid"] = row.frame_uid
    if hasattr(row, "source_relpath"):
        atoms.info["source_relpath"] = row.source_relpath
    if hasattr(row, "source_dirname"):
        atoms.info["source_dirname"] = row.source_dirname

    try:
        atoms.info["REF_energy"] = float(atoms.get_potential_energy())
    except Exception:
        pass

    try:
        atoms.arrays["REF_forces"] = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        pass

    try:
        atoms.info["REF_stress"] = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        pass

    return atoms


def main() -> int:
    args = parse_args()

    db_path = args.db_path.resolve()
    out_path = args.out_path.resolve()
    _prepare_output_path(out_path, overwrite=args.overwrite)

    atoms_list = []
    with connect(db_path) as db:
        rows = db.select(args.selection) if args.selection else db.select()
        for index, row in enumerate(rows):
            if args.limit > 0 and index >= args.limit:
                break
            atoms = _copy_reference_properties(row)
            atoms.info["config_type"] = args.config_type
            if args.head:
                atoms.info["head"] = args.head
            atoms_list.append(atoms)

    if not atoms_list:
        print(f"No rows selected from {db_path}")
        return 1

    write(out_path, atoms_list, format="extxyz")
    print(f"db_path: {db_path}")
    print(f"out_path: {out_path}")
    print(f"frames_written: {len(atoms_list)}")
    if args.head:
        print(f"head: {args.head}")
    print("stored_keys: REF_energy, REF_forces, REF_stress, config_type")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
