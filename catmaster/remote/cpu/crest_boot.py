from __future__ import annotations

import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time


def _method_flags(method: str) -> list[str]:
    name = str(method or "gfn2").strip().lower()
    if name == "gfn2":
        return ["--gfn2"]
    if name == "gfn1":
        return ["--gfn1"]
    if name == "gfnff":
        return ["--gfnff"]
    raise ValueError(f"Unsupported CREST method: {method}")


def _normalize_optional_text(value: str) -> str:
    text = str(value or "").strip()
    if text.lower() in {"", "__none__", "none", "null"}:
        return ""
    return text


def _collect_outputs() -> dict[str, str]:
    names = (
        "crest_best.xyz",
        "crest_conformers.xyz",
        "crest_rotamers.xyz",
        "crest.energies",
        "cre_members",
        "crest.ensemble.xyz",
    )
    found: dict[str, str] = {}
    for name in names:
        path = Path(name)
        if path.exists():
            found[name] = str(path.resolve())
    return found


def main() -> int:
    parser = argparse.ArgumentParser(description="CREST boot wrapper for DPDispatcher tasks")
    parser.add_argument("--input", required=True, help="Input structure filename")
    parser.add_argument("--mode", default="standard", choices=("standard", "nci", "constrained"))
    parser.add_argument("--method", default="gfn2", choices=("gfn2", "gfn1", "gfnff"))
    parser.add_argument("--ewin", type=float, default=6.0)
    parser.add_argument("--rthr", type=float, default=0.125)
    parser.add_argument("--ethr", type=float, default=0.05)
    parser.add_argument("--bthr", type=float, default=0.01)
    parser.add_argument("--charge", type=int, default=0)
    parser.add_argument("--uhf", type=int, default=0)
    parser.add_argument("--solvent", default="", help="Optional ALPB solvent")
    parser.add_argument("--constraint-file", default="", help="Constraint file in xTB syntax")
    parser.add_argument("--crest-bin", default="crest", help="CREST executable name")
    parser.add_argument("--log", default="crest_stdout.out", help="Combined stdout/stderr log")
    args = parser.parse_args()

    input_path = Path(args.input)
    if not input_path.is_file():
        sys.stderr.write(f"[crest_boot] input file missing: {input_path}\n")
        return 2

    crest_bin = shutil.which(args.crest_bin) or args.crest_bin
    cmd = [crest_bin, input_path.name]
    cmd.extend(_method_flags(args.method))
    cmd.extend(
        [
            "--ewin",
            str(float(args.ewin)),
            "--rthr",
            str(float(args.rthr)),
            "--ethr",
            str(float(args.ethr)),
            "--bthr",
            str(float(args.bthr)),
            "--chrg",
            str(int(args.charge)),
            "--uhf",
            str(int(args.uhf)),
        ]
    )
    solvent = _normalize_optional_text(args.solvent)
    if solvent:
        cmd.extend(["--alpb", solvent])
    if args.mode == "nci":
        cmd.append("--nci")
    constraint_file = _normalize_optional_text(args.constraint_file)
    if constraint_file:
        cmd.extend(["--cinp", constraint_file, "--subrmsd"])

    started = time.time()
    with open(args.log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[crest_boot] cwd={Path.cwd()}\n")
        log_handle.write(f"[crest_boot] command={' '.join(cmd)}\n")
        log_handle.write(f"[crest_boot] crest_bin={crest_bin}\n")
        log_handle.flush()
        proc = subprocess.run(cmd, stdout=log_handle, stderr=subprocess.STDOUT, check=False)

    payload = {
        "completed": proc.returncode == 0,
        "returncode": int(proc.returncode),
        "command": cmd,
        "input": input_path.name,
        "mode": args.mode,
        "method": args.method,
        "ewin": float(args.ewin),
        "rthr": float(args.rthr),
        "ethr": float(args.ethr),
        "bthr": float(args.bthr),
        "charge": int(args.charge),
        "uhf": int(args.uhf),
        "solvent": solvent or None,
        "constraint_file": constraint_file or None,
        "started_at": started,
        "finished_at": time.time(),
        "outputs": _collect_outputs(),
        "log_file": args.log,
    }
    Path("crest_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(proc.returncode)


if __name__ == "__main__":
    raise SystemExit(main())
