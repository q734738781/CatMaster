from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _parse_bool(value: str) -> bool:
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def _normalize_optional_text(value: str) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "__none__", "none", "null"}:
        return ""
    return text


def _xtb_method_flags(gfn: str) -> list[str]:
    name = str(gfn or "gfn2").strip().lower()
    if name == "gfn2":
        return ["--gfn", "2"]
    if name == "gfn1":
        return ["--gfn", "1"]
    if name == "gfnff":
        return ["--gfnff"]
    raise ValueError(f"Unsupported gfn setting: {gfn}")


def _build_md_input(*, temperature: float, md_time_ps: float, timestep_fs: float, md_dump_fs: float) -> Path:
    xcontrol = Path("xtb_md.inp")
    xcontrol.write_text(
        "\n".join(
            [
                "$md",
                f"  temp={float(temperature):.6f}",
                f"  time={float(md_time_ps):.6f}",
                f"  step={float(timestep_fs):.6f}",
                f"  dump={float(md_dump_fs):.6f}",
                "$end",
                "",
            ]
        ),
        encoding="utf-8",
    )
    return xcontrol


def _collect_outputs() -> dict[str, str]:
    interesting = (
        "xtbopt.xyz",
        "xtblast.xyz",
        "xtb.trj",
        "g98.out",
        "hessian",
        "vibspectrum",
        "thermo.out",
        "charges",
        "wbo",
        "xtbtopo.mol",
        "xtbmdok",
    )
    found: dict[str, str] = {}
    for name in interesting:
        path = Path(name)
        if path.exists():
            found[name] = str(path.resolve())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="xTB boot wrapper for DPDispatcher tasks")
    parser.add_argument("--input", required=True, help="Input structure filename")
    parser.add_argument("--mode", default="opt", choices=("sp", "opt", "hess", "md"))
    parser.add_argument("--gfn", default="gfn2", choices=("gfn2", "gfn1", "gfnff"))
    parser.add_argument("--solvent-model", default="none", choices=("none", "alpb", "gbsa"))
    parser.add_argument("--solvent", default="", help="Solvent name for ALPB/GBSA")
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    parser.add_argument("--opt-level", default="normal")
    parser.add_argument("--temperature", type=float, default=298.15)
    parser.add_argument("--md-time-ps", type=float, default=5.0)
    parser.add_argument("--timestep-fs", type=float, default=1.0)
    parser.add_argument("--md-dump-fs", type=float, default=50.0)
    parser.add_argument("--xtb-bin", default="xtb", help="xTB executable name")
    parser.add_argument("--log", default="xtb_stdout.out", help="Combined stdout/stderr log")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[xtb_boot] input file missing: {input_path}\n")
        return 2

    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    base_cmd = [xtb_bin, input_path.name]
    base_cmd.extend(_xtb_method_flags(args.gfn))
    base_cmd.extend(["--chrg", str(int(args.charge)), "--uhf", str(int(args.uhf))])
    solvent = _normalize_optional_text(args.solvent)
    if args.solvent_model == "alpb" and solvent:
        base_cmd.extend(["--alpb", solvent])
    elif args.solvent_model == "gbsa" and solvent:
        base_cmd.extend(["--gbsa", solvent])

    extra_files: list[str] = []
    if args.mode == "sp":
        pass
    elif args.mode == "opt":
        base_cmd.extend(["--opt", str(args.opt_level)])
    elif args.mode == "hess":
        base_cmd.append("--hess")
    elif args.mode == "md":
        xcontrol = _build_md_input(
            temperature=args.temperature,
            md_time_ps=args.md_time_ps,
            timestep_fs=args.timestep_fs,
            md_dump_fs=args.md_dump_fs,
        )
        extra_files.append(xcontrol.name)
        base_cmd.extend(["--md", "--input", xcontrol.name])
    else:
        sys.stderr.write(f"[xtb_boot] unsupported mode: {args.mode}\n")
        return 2

    summary_path = Path("xtb_summary.json")
    started = time.time()
    completed = False
    with open(args.log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[xtb_boot] cwd={Path.cwd()}\n")
        log_handle.write(f"[xtb_boot] command={' '.join(base_cmd)}\n")
        log_handle.write(f"[xtb_boot] xtb_bin={xtb_bin}\n")
        log_handle.flush()
        proc = subprocess.run(base_cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)
        completed = proc.returncode == 0

    payload = {
        "completed": completed,
        "returncode": int(proc.returncode),
        "command": base_cmd,
        "input": input_path.name,
        "mode": args.mode,
        "gfn": args.gfn,
        "solvent_model": args.solvent_model,
        "solvent": solvent or None,
        "charge": int(args.charge),
        "uhf": int(args.uhf),
        "started_at": started,
        "finished_at": time.time(),
        "extra_files": extra_files,
        "outputs": _collect_outputs(),
        "log_file": args.log,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    try:
        os.system("ulimit -s unlimited >/dev/null 2>&1")
    except Exception:
        pass
    raise SystemExit(main())
