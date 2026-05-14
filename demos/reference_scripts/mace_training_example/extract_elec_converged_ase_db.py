#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
import xml.etree.ElementTree as ET
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor
from dataclasses import dataclass
from itertools import repeat
from pathlib import Path
from typing import Any

try:
    from ase.db import connect
    from ase.io.vasp import read_vasp_xml
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: ase. Install it first, e.g. `pip install ase`."
    ) from exc


PARAMETER_KEYS = (
    "isif",
    "pstress",
    "ibrion",
    "nsw",
    "nelm",
    "algo",
    "lepsilon",
    "ediff",
    "ediffg",
)


@dataclass(frozen=True)
class ExtractedIonicStep:
    atoms: Any
    frame_uid: str
    source_path: str
    source_relpath: str
    source_dirname: str
    ionic_step_index: int
    ionic_step_number: int
    electronic_step_count: int | None
    nelm: int | None
    step_electronic_converged_guess: bool | None
    natoms: int
    formula: str
    energy: float | None
    free_energy: float | None
    has_constraints: bool
    constraint_types: list[str]
    selected_parameters: dict[str, Any]


@dataclass(frozen=True)
class ParseResult:
    source_path: str
    source_relpath: str
    total_ionic_steps: int
    extracted_steps: list[ExtractedIonicStep]
    warning: str = ""
    error: str = ""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Extract ionic steps from VASP vasprun.xml files into an ASE database. "
            "Structures, energies, forces, stress, constraints, and calculator "
            "metadata are read with ASE's official VASP XML reader. XML parsing is "
            "only used to read NELM and count SCF steps per ionic step."
        )
    )
    parser.add_argument(
        "--root",
        type=Path,
        default=Path("DFT_files"),
        help="Root directory searched recursively for vasprun.xml files.",
    )
    parser.add_argument(
        "--pattern",
        default="vasprun.xml",
        help="Glob pattern used under --root, e.g. vasprun.xml or vasprun.xml*.",
    )
    parser.add_argument(
        "--db-path",
        type=Path,
        default=Path("DFT_files/ionic_steps.db"),
        help="Output ASE database path.",
    )
    parser.add_argument(
        "--metadata-path",
        type=Path,
        default=None,
        help="Output metadata CSV path. Defaults to <db-path>.metadata.csv.",
    )
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers used for vasprun.xml parsing.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing DB / metadata files instead of refusing to run.",
    )
    parser.add_argument(
        "--only-electronically-converged",
        action="store_true",
        help=(
            "Keep only steps with the heuristic `electronic_step_count < NELM`. "
            "By default all ionic steps are written and the heuristic is stored as "
            "`step_electronic_converged_guess` for later filtering."
        ),
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Print one line per parsed vasprun.xml.",
    )
    parser.add_argument(
        "--alignment-check",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Validate step-by-step alignment between XML metadata and ASE frames "
            "using free energies where available. Disable with --no-alignment-check."
        ),
    )
    parser.add_argument(
        "--alignment-energy-atol",
        type=float,
        default=1e-6,
        help=(
            "Absolute tolerance in eV for XML vs ASE free-energy alignment checks."
        ),
    )
    return parser.parse_args()


def discover_vaspruns(root: Path, pattern: str) -> list[Path]:
    return sorted(path for path in root.rglob(pattern) if path.is_file())


def make_frame_uid(source_relpath: str, ionic_step_number: int) -> str:
    stem = Path(source_relpath).with_suffix("").as_posix().replace("/", "__")
    return f"{stem}__i{ionic_step_number:05d}"


def _jsonable(value: Any) -> Any:
    if hasattr(value, "tolist"):
        return value.tolist()
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, Path):
        return str(value)
    return value


def _get_parameter(parameters: dict[str, Any], key: str) -> Any:
    for candidate in (key, key.lower(), key.upper()):
        if candidate in parameters:
            return parameters[candidate]
    return None


def _extract_selected_parameters(atoms, nelm: int | None) -> dict[str, Any]:
    params = {}
    calc = getattr(atoms, "calc", None)
    calc_params = getattr(calc, "parameters", None)
    if calc_params:
        for key in PARAMETER_KEYS:
            value = _get_parameter(calc_params, key)
            if value is not None:
                params[key] = _jsonable(value)
    if nelm is not None and "nelm" not in params:
        params["nelm"] = nelm
    return params


def _parse_vasp_step_metadata(
    path: Path,
) -> tuple[int | None, list[int], list[float | None], str]:
    nelm = None
    electronic_step_counts: list[int] = []
    free_energies: list[float | None] = []
    warning = ""

    try:
        for _, elem in ET.iterparse(path, events=["end"]):
            if nelm is None and elem.tag == "i" and elem.attrib.get("name") == "NELM":
                text = (elem.text or "").strip()
                if text:
                    nelm = int(float(text))
            elif elem.tag == "calculation":
                electronic_step_counts.append(len(elem.findall("scstep")))
                e_fr_energy = None
                e_fr_node = elem.find("./energy/i[@name='e_fr_energy']")
                if e_fr_node is not None:
                    text = (e_fr_node.text or "").strip()
                    if text:
                        e_fr_energy = float(text)
                free_energies.append(e_fr_energy)
                elem.clear()
    except ET.ParseError as exc:
        warning = f"XML parse warning: {exc}"

    return nelm, electronic_step_counts, free_energies, warning


def _guess_electronic_convergence(
    electronic_step_count: int | None,
    nelm: int | None,
) -> bool | None:
    if electronic_step_count is None or nelm is None:
        return None
    return electronic_step_count < nelm


def _read_energy(atoms) -> tuple[float | None, float | None]:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None, None
    energy = calc.results.get("energy")
    free_energy = calc.results.get("free_energy")
    return (
        float(energy) if energy is not None else None,
        float(free_energy) if free_energy is not None else None,
    )


def _check_step_alignment(
    path: Path,
    images: list[Any],
    xml_free_energies: list[float | None],
    atol: float,
) -> str:
    if not xml_free_energies:
        return ""

    mismatches: list[str] = []
    n_shared = min(len(images), len(xml_free_energies))
    for index in range(n_shared):
        xml_free = xml_free_energies[index]
        if xml_free is None:
            continue
        calc = getattr(images[index], "calc", None)
        ase_free = None if calc is None else calc.results.get("free_energy")
        if ase_free is None:
            mismatches.append(
                f"step {index + 1}: XML has e_fr_energy={xml_free:.10f} but ASE frame has no free_energy"
            )
            continue
        if abs(float(ase_free) - xml_free) > atol:
            mismatches.append(
                f"step {index + 1}: XML e_fr_energy={xml_free:.10f}, "
                f"ASE free_energy={float(ase_free):.10f}"
            )
        if len(mismatches) >= 5:
            break

    if not mismatches:
        return ""
    joined = "; ".join(mismatches)
    return (
        f"Alignment check failed for {path.name}: {joined}. "
        "This suggests XML `<calculation>` ordering no longer matches ASE frames."
    )


def _build_step_payload(
    atoms,
    source_path: str,
    source_relpath: str,
    source_dirname: str,
    ionic_step_index: int,
    electronic_step_count: int | None,
    nelm: int | None,
) -> ExtractedIonicStep:
    energy, free_energy = _read_energy(atoms)
    constraints = list(getattr(atoms, "constraints", []))
    return ExtractedIonicStep(
        atoms=atoms,
        frame_uid=make_frame_uid(source_relpath, ionic_step_index + 1),
        source_path=source_path,
        source_relpath=source_relpath,
        source_dirname=source_dirname,
        ionic_step_index=ionic_step_index,
        ionic_step_number=ionic_step_index + 1,
        electronic_step_count=electronic_step_count,
        nelm=nelm,
        step_electronic_converged_guess=_guess_electronic_convergence(
            electronic_step_count, nelm
        ),
        natoms=len(atoms),
        formula=atoms.get_chemical_formula(mode="reduce"),
        energy=energy,
        free_energy=free_energy,
        has_constraints=bool(constraints),
        constraint_types=[constraint.__class__.__name__ for constraint in constraints],
        selected_parameters=_extract_selected_parameters(atoms, nelm),
    )


def _parse_one_vasprun(
    path_str: str,
    root_str: str,
    only_electronically_converged: bool,
    alignment_check: bool,
    alignment_energy_atol: float,
) -> ParseResult:
    path = Path(path_str)
    root = Path(root_str)
    relpath = path.relative_to(root).as_posix()

    try:
        nelm, electronic_step_counts, xml_free_energies, warning = _parse_vasp_step_metadata(
            path
        )
        images = list(read_vasp_xml(path, index=slice(None)))
        total_ionic_steps = len(images)

        if electronic_step_counts:
            if len(electronic_step_counts) != total_ionic_steps:
                min_len = min(len(electronic_step_counts), total_ionic_steps)
                warning = (
                    warning + "; " if warning else ""
                ) + (
                    "Ionic-step count mismatch between XML metadata and ASE reader: "
                    f"metadata={len(electronic_step_counts)}, ase={total_ionic_steps}. "
                    f"Using first {min_len} shared steps."
                )
                images = images[:min_len]
                electronic_step_counts = electronic_step_counts[:min_len]
                xml_free_energies = xml_free_energies[:min_len]
        else:
            electronic_step_counts = [None] * total_ionic_steps
            xml_free_energies = [None] * total_ionic_steps

        if alignment_check:
            alignment_error = _check_step_alignment(
                path=path,
                images=images,
                xml_free_energies=xml_free_energies,
                atol=alignment_energy_atol,
            )
            if alignment_error:
                raise ValueError(alignment_error)

        extracted_steps: list[ExtractedIonicStep] = []
        source_path = str(path.resolve())
        source_dirname = path.parent.name

        for ionic_step_index, atoms in enumerate(images):
            electronic_step_count = electronic_step_counts[ionic_step_index]
            payload = _build_step_payload(
                atoms=atoms,
                source_path=source_path,
                source_relpath=relpath,
                source_dirname=source_dirname,
                ionic_step_index=ionic_step_index,
                electronic_step_count=electronic_step_count,
                nelm=nelm,
            )
            if (
                only_electronically_converged
                and payload.step_electronic_converged_guess is not True
            ):
                continue
            extracted_steps.append(payload)

        return ParseResult(
            source_path=source_path,
            source_relpath=relpath,
            total_ionic_steps=total_ionic_steps,
            extracted_steps=extracted_steps,
            warning=warning,
            error="",
        )
    except Exception as exc:  # pragma: no cover - runtime/parsing failures
        return ParseResult(
            source_path=str(path.resolve()),
            source_relpath=relpath,
            total_ionic_steps=0,
            extracted_steps=[],
            warning="",
            error=str(exc),
        )


def _iter_parse_results(
    vaspruns: list[Path],
    root: Path,
    workers: int,
    only_electronically_converged: bool,
    alignment_check: bool,
    alignment_energy_atol: float,
):
    if workers <= 1:
        for path in vaspruns:
            yield _parse_one_vasprun(
                str(path),
                str(root),
                only_electronically_converged,
                alignment_check,
                alignment_energy_atol,
            )
        return

    def _executor_iter(executor_cls):
        with executor_cls(max_workers=max(1, workers)) as executor:
            yield from executor.map(
                _parse_one_vasprun,
                (str(path) for path in vaspruns),
                repeat(str(root)),
                repeat(only_electronically_converged),
                repeat(alignment_check),
                repeat(alignment_energy_atol),
                chunksize=1,
            )

    try:
        yield from _executor_iter(ProcessPoolExecutor)
    except (PermissionError, OSError):
        print(
            "[WARN] ProcessPool unavailable in this environment; "
            "falling back to ThreadPoolExecutor."
        )
        yield from _executor_iter(ThreadPoolExecutor)


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise SystemExit(
                f"Output already exists: {path}. Use --overwrite to replace it."
            )
        path.unlink()


def _metadata_header() -> list[str]:
    return [
        "db_id",
        "frame_uid",
        "source_path",
        "source_relpath",
        "source_dirname",
        "ionic_step_index",
        "ionic_step_number",
        "electronic_step_count",
        "nelm",
        "step_electronic_converged_guess",
        "natoms",
        "formula",
        "energy",
        "free_energy",
        "has_constraints",
        "constraint_types",
    ]


def _write_metadata_row(writer: csv.writer, db_id: int, step: ExtractedIonicStep) -> None:
    writer.writerow(
        [
            db_id,
            step.frame_uid,
            step.source_path,
            step.source_relpath,
            step.source_dirname,
            step.ionic_step_index,
            step.ionic_step_number,
            step.electronic_step_count,
            step.nelm,
            step.step_electronic_converged_guess,
            step.natoms,
            step.formula,
            step.energy,
            step.free_energy,
            step.has_constraints,
            ",".join(step.constraint_types),
        ]
    )


def _clean_key_value_pairs(raw: dict[str, Any]) -> dict[str, Any]:
    return {key: value for key, value in raw.items() if value is not None}


def main() -> int:
    args = parse_args()

    root = args.root.resolve()
    db_path = args.db_path.resolve()
    metadata_path = (
        args.metadata_path.resolve()
        if args.metadata_path is not None
        else db_path.with_suffix(db_path.suffix + ".metadata.csv")
    )

    vaspruns = discover_vaspruns(root, args.pattern)
    if not vaspruns:
        print(f"No files found: root={root} pattern={args.pattern}")
        return 1

    _prepare_output_path(db_path, overwrite=args.overwrite)
    _prepare_output_path(metadata_path, overwrite=args.overwrite)

    n_files_ok = 0
    n_files_failed = 0
    n_ionic_steps_total = 0
    n_frames_written = 0
    n_guess_converged_total = 0
    failures: list[tuple[str, str]] = []
    warnings: list[tuple[str, str]] = []

    with connect(db_path) as db, metadata_path.open(
        "w", newline="", encoding="utf-8"
    ) as meta_fp:
        metadata_writer = csv.writer(meta_fp)
        metadata_writer.writerow(_metadata_header())

        for result in _iter_parse_results(
            vaspruns=vaspruns,
            root=root,
            workers=max(1, args.workers),
            only_electronically_converged=args.only_electronically_converged,
            alignment_check=args.alignment_check,
            alignment_energy_atol=args.alignment_energy_atol,
        ):
            if result.error:
                n_files_failed += 1
                failures.append((result.source_relpath, result.error))
                if args.verbose:
                    print(f"[FAIL] {result.source_relpath}: {result.error}")
                continue

            n_files_ok += 1
            n_ionic_steps_total += result.total_ionic_steps

            if result.warning:
                warnings.append((result.source_relpath, result.warning))
                if args.verbose:
                    print(f"[WARN] {result.source_relpath}: {result.warning}")

            for step in result.extracted_steps:
                key_value_pairs = _clean_key_value_pairs(
                    {
                        "frame_uid": step.frame_uid,
                        "source_relpath": step.source_relpath,
                        "source_dirname": step.source_dirname,
                        "ionic_step": step.ionic_step_index,
                        "ionic_step_number": step.ionic_step_number,
                        "electronic_steps": step.electronic_step_count,
                        "nelm": step.nelm,
                        "step_electronic_converged_guess": (
                            int(step.step_electronic_converged_guess)
                            if step.step_electronic_converged_guess is not None
                            else None
                        ),
                        "formula_label": step.formula,
                        "has_constraints": int(step.has_constraints),
                    }
                )
                row_id = db.write(
                    step.atoms,
                    key_value_pairs=key_value_pairs,
                    data={
                        "source_path": step.source_path,
                        "source_relpath": step.source_relpath,
                        "source_dirname": step.source_dirname,
                        "ionic_step_index": step.ionic_step_index,
                        "ionic_step_number": step.ionic_step_number,
                        "electronic_step_count": step.electronic_step_count,
                        "nelm": step.nelm,
                        "step_electronic_converged_guess": step.step_electronic_converged_guess,
                        "frame_uid": step.frame_uid,
                        "has_constraints": step.has_constraints,
                        "constraint_types": step.constraint_types,
                        "selected_parameters": step.selected_parameters,
                    },
                )
                _write_metadata_row(metadata_writer, row_id, step)
                n_frames_written += 1
                if step.step_electronic_converged_guess:
                    n_guess_converged_total += 1

            if args.verbose:
                print(
                    f"[OK] {result.source_relpath}: "
                    f"{len(result.extracted_steps)}/{result.total_ionic_steps} steps written"
                )

    guess_ratio = (
        n_guess_converged_total / n_frames_written if n_frames_written else 0.0
    )
    print(f"root: {root}")
    print(f"db_path: {db_path}")
    print(f"metadata_path: {metadata_path}")
    print(f"files_found: {len(vaspruns)}")
    print(f"files_ok: {n_files_ok}")
    print(f"files_failed: {n_files_failed}")
    print(f"ionic_steps_total: {n_ionic_steps_total}")
    print(f"steps_written: {n_frames_written}")
    print(f"step_electronic_converged_guess_total: {n_guess_converged_total}")
    print(f"step_electronic_converged_guess_ratio: {guess_ratio:.6f}")
    print(f"only_electronically_converged: {args.only_electronically_converged}")

    if warnings:
        print("warnings:")
        for relpath, warning in warnings:
            print(f"- {relpath}: {warning}")

    if failures:
        print("failed_files:")
        for relpath, error in failures:
            print(f"- {relpath}: {error}")

    return 0 if n_files_failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
