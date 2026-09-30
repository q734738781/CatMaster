#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import os
from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path


@dataclass
class FileStat:
    path: str
    n_ionic_steps: int
    n_elec_converged_frames: int
    n_elec_unconverged_frames: int
    n_electronic_steps_total: int
    nelm: int | None
    converged: bool | None
    converged_electronic: bool | None
    converged_ionic: bool | None
    error: str = ""


def _parse_one_vasprun(path_str: str) -> FileStat:
    path = Path(path_str)
    try:
        from pymatgen.io.vasp.outputs import Vasprun

        vr = Vasprun(
            path,
            parse_dos=False,
            parse_eigen=False,
            parse_projected_eigen=False,
            parse_potcar_file=False,
            exception_on_bad_xml=False,
        )
        nelm_raw = vr.parameters.get("NELM", None)
        nelm = int(nelm_raw) if nelm_raw is not None else None

        n_ionic_steps = len(vr.ionic_steps)
        n_elec_converged_frames = 0
        n_elec_unconverged_frames = 0
        n_electronic_steps_total = 0

        for step in vr.ionic_steps:
            n_elec_steps = len(step.get("electronic_steps", []))
            n_electronic_steps_total += n_elec_steps
            if nelm is None:
                # Without NELM, keep conservative: do not mark as converged.
                n_elec_unconverged_frames += 1
            elif n_elec_steps < nelm:
                n_elec_converged_frames += 1
            else:
                n_elec_unconverged_frames += 1

        return FileStat(
            path=path_str,
            n_ionic_steps=n_ionic_steps,
            n_elec_converged_frames=n_elec_converged_frames,
            n_elec_unconverged_frames=n_elec_unconverged_frames,
            n_electronic_steps_total=n_electronic_steps_total,
            nelm=nelm,
            converged=bool(vr.converged),
            converged_electronic=bool(vr.converged_electronic),
            converged_ionic=bool(vr.converged_ionic),
            error="",
        )
    except Exception as exc:  # pragma: no cover - runtime/parsing failures
        return FileStat(
            path=path_str,
            n_ionic_steps=0,
            n_elec_converged_frames=0,
            n_elec_unconverged_frames=0,
            n_electronic_steps_total=0,
            nelm=None,
            converged=None,
            converged_electronic=None,
            converged_ionic=None,
            error=str(exc),
        )


def _as_int_or_none(v: str) -> int | None:
    if v == "" or v.lower() == "none":
        return None
    return int(v)


def _as_bool_or_none(v: str) -> bool | None:
    if v == "" or v.lower() == "none":
        return None
    return v.lower() == "true"


def _read_details_csv(csv_path: Path) -> list[FileStat]:
    rows: list[FileStat] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            rows.append(
                FileStat(
                    path=r["path"],
                    n_ionic_steps=int(r["n_ionic_steps"]),
                    n_elec_converged_frames=int(r["n_elec_converged_frames"]),
                    n_elec_unconverged_frames=int(r["n_elec_unconverged_frames"]),
                    n_electronic_steps_total=int(r["n_electronic_steps_total"]),
                    nelm=_as_int_or_none(r["nelm"]),
                    converged=_as_bool_or_none(r["converged"]),
                    converged_electronic=_as_bool_or_none(r["converged_electronic"]),
                    converged_ionic=_as_bool_or_none(r["converged_ionic"]),
                    error=r["error"],
                )
            )
    return rows


def _write_details_csv(csv_path: Path, stats: list[FileStat]) -> None:
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "path",
                "n_ionic_steps",
                "n_elec_converged_frames",
                "n_elec_unconverged_frames",
                "n_electronic_steps_total",
                "nelm",
                "converged",
                "converged_electronic",
                "converged_ionic",
                "error",
            ]
        )
        for s in stats:
            writer.writerow(
                [
                    s.path,
                    s.n_ionic_steps,
                    s.n_elec_converged_frames,
                    s.n_elec_unconverged_frames,
                    s.n_electronic_steps_total,
                    s.nelm,
                    s.converged,
                    s.converged_electronic,
                    s.converged_ionic,
                    s.error,
                ]
            )


def _summarize(stats: list[FileStat]) -> tuple[int, int, int, int, int]:
    n_files_ok = sum(1 for s in stats if not s.error)
    n_files_failed = len(stats) - n_files_ok
    ionic_total = sum(s.n_ionic_steps for s in stats)
    elec_conv_total = sum(s.n_elec_converged_frames for s in stats)
    elec_unconv_total = sum(s.n_elec_unconverged_frames for s in stats)
    return n_files_ok, n_files_failed, ionic_total, elec_conv_total, elec_unconv_total


def _parse_all(vaspruns: list[Path], workers: int) -> list[FileStat]:
    if workers <= 1:
        return [_parse_one_vasprun(str(p)) for p in vaspruns]

    def _collect_with_executor(executor_cls):
        out: list[FileStat] = []
        with executor_cls(max_workers=max(1, workers)) as ex:
            futures = {ex.submit(_parse_one_vasprun, str(p)): p for p in vaspruns}
            for fut in as_completed(futures):
                out.append(fut.result())
        return out

    try:
        return _collect_with_executor(ProcessPoolExecutor)
    except (PermissionError, OSError):
        print(
            "[WARN] ProcessPool unavailable in this environment; fallback to ThreadPoolExecutor."
        )
        return _collect_with_executor(ThreadPoolExecutor)


def main() -> int:
    parser = argparse.ArgumentParser(
        description=(
            "Count electronically converged frames from vasprun.xml under a root "
            "directory. A frame is considered electronically converged when "
            "len(electronic_steps) < NELM for that ionic step."
        )
    )
    parser.add_argument("--root", type=Path, default=Path("DFT_files"))
    parser.add_argument("--pattern", default="vasprun.xml")
    parser.add_argument(
        "--workers",
        type=int,
        default=max(1, (os.cpu_count() or 1) - 1),
        help="Parallel workers for parsing.",
    )
    parser.add_argument(
        "--details-csv",
        type=Path,
        default=Path("DFT_files/elec_convergence_stats.csv"),
        help="Output CSV for per-vasprun details.",
    )
    parser.add_argument(
        "--reuse-details-csv",
        action="store_true",
        help="Reuse an existing --details-csv instead of reparsing vasprun.xml files.",
    )
    args = parser.parse_args()

    if args.reuse_details_csv:
        if not args.details_csv.is_file():
            raise SystemExit(f"details csv not found: {args.details_csv}")
        stats = _read_details_csv(args.details_csv)
    else:
        vaspruns = sorted(p for p in args.root.rglob(args.pattern) if p.is_file())
        if not vaspruns:
            print(f"No files found: root={args.root} pattern={args.pattern}")
            return 1

        stats = _parse_all(vaspruns, workers=max(1, args.workers))
        stats.sort(key=lambda s: s.path)
        _write_details_csv(args.details_csv, stats)

    n_files_ok, n_files_failed, ionic_total, elec_conv_total, elec_unconv_total = _summarize(
        stats
    )
    conv_ratio = (elec_conv_total / ionic_total) if ionic_total else 0.0

    print(f"root: {args.root}")
    print(f"details_csv: {args.details_csv}")
    print(f"files_ok: {n_files_ok}")
    print(f"files_failed: {n_files_failed}")
    print(f"ionic_frames_total: {ionic_total}")
    print(f"elec_converged_frames_total: {elec_conv_total}")
    print(f"elec_unconverged_frames_total: {elec_unconv_total}")
    print(f"elec_converged_ratio: {conv_ratio:.6f}")

    if n_files_failed:
        print("failed_files:")
        for s in stats:
            if s.error:
                print(f"- {s.path}: {s.error}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
