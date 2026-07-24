from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from typing import Any


_MODES = {"sp", "opt", "hess", "md"}
_GFN_METHODS = {"gfn2", "gfn1", "gfnff"}
_SOLVENT_MODELS = {"none", "alpb", "gbsa"}
_OPT_LEVELS = {"crude", "sloppy", "loose", "normal", "tight", "vtight", "extreme"}


def _parse_cpu_count(raw: str) -> int | None:
    text = str(raw or "").strip()
    if not text:
        return None
    head = text.split(",", 1)[0].strip()
    digits = []
    for char in head:
        if char.isdigit():
            digits.append(char)
            continue
        break
    if not digits:
        return None
    try:
        parsed = int("".join(digits))
    except Exception:
        return None
    if parsed <= 0:
        return None
    return parsed


def _resolve_threads() -> int:
    explicit = _parse_cpu_count(os.environ.get("CATMASTER_XTB_THREADS", ""))
    if explicit:
        return explicit
    for key in (
        "SLURM_CPUS_PER_TASK",
        "SLURM_NTASKS",
        "SLURM_CPUS_ON_NODE",
        "SLURM_JOB_CPUS_PER_NODE",
        "OMP_NUM_THREADS",
    ):
        parsed = _parse_cpu_count(os.environ.get(key, ""))
        if parsed:
            return parsed
    return 1


def _xtb_env(nthreads: int) -> dict[str, str]:
    env = dict(os.environ)
    threads = max(1, int(nthreads))
    env.setdefault("OMP_NUM_THREADS", f"{threads},1")
    env.setdefault("MKL_NUM_THREADS", str(threads))
    env.setdefault("OPENBLAS_NUM_THREADS", str(threads))
    env.setdefault("OMP_MAX_ACTIVE_LEVELS", "1")
    env.setdefault("OMP_STACKSIZE", "4G")
    return env


def _xtb_method_flags(gfn: str) -> list[str]:
    name = str(gfn or "gfn2").strip().lower()
    if name == "gfn2":
        return ["--gfn", "2"]
    if name == "gfn1":
        return ["--gfn", "1"]
    if name == "gfnff":
        return ["--gfnff"]
    raise ValueError(f"Unsupported gfn setting: {gfn}")


def _stage_filename(value: Any, *, field: str, required: bool) -> str:
    text = str(value or "").strip()
    if not text:
        if required:
            raise ValueError(f"manifest.{field} is required")
        return ""
    path = Path(text)
    if path.is_absolute() or path.name != text or any(part == ".." for part in path.parts):
        raise ValueError(f"manifest.{field} must be a direct stage filename: {text!r}")
    if not path.is_file():
        raise ValueError(f"manifest.{field} file is missing: {text}")
    return text


def _integer(value: Any, *, field: str, minimum: int | None = None) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"manifest.{field} must be an integer")
    parsed = int(value)
    if minimum is not None and parsed < minimum:
        raise ValueError(f"manifest.{field} must be >= {minimum}")
    return parsed


def _choice(value: Any, *, field: str, choices: set[str]) -> str:
    text = str(value or "").strip().lower()
    if text not in choices:
        raise ValueError(f"manifest.{field} must be one of {sorted(choices)}")
    return text


def _load_manifest(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"manifest file is missing: {path}")
    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        raise ValueError(f"could not parse manifest JSON: {exc}") from exc
    if not isinstance(raw, dict):
        raise ValueError("manifest must contain a JSON object")
    if raw.get("schema_version") != 1:
        raise ValueError("manifest.schema_version must be 1")
    if str(raw.get("program") or "").strip().lower() != "xtb":
        raise ValueError("manifest.program must be 'xtb'")

    config = {
        "schema_version": 1,
        "program": "xtb",
        "coordinate_file": _stage_filename(raw.get("coordinate_file"), field="coordinate_file", required=True),
        "xcontrol_file": _stage_filename(raw.get("xcontrol_file"), field="xcontrol_file", required=False),
        "mode": _choice(raw.get("mode"), field="mode", choices=_MODES),
        "gfn": _choice(raw.get("gfn"), field="gfn", choices=_GFN_METHODS),
        "solvent_model": _choice(
            raw.get("solvent_model", "none"),
            field="solvent_model",
            choices=_SOLVENT_MODELS,
        ),
        "solvent": str(raw.get("solvent") or "").strip(),
        "charge": _integer(raw.get("charge", 0), field="charge"),
        "uhf": _integer(raw.get("uhf", 0), field="uhf", minimum=0),
        "opt_level": _choice(raw.get("opt_level", "normal"), field="opt_level", choices=_OPT_LEVELS),
    }
    if config["solvent_model"] == "none" and config["solvent"]:
        raise ValueError("manifest.solvent must be empty when solvent_model=none")
    if config["solvent_model"] != "none" and not config["solvent"]:
        raise ValueError(f"manifest.solvent is required when solvent_model={config['solvent_model']}")
    return config


def _build_xtb_command(config: dict[str, Any], *, xtb_bin: str) -> list[str]:
    command = [xtb_bin, str(config["coordinate_file"])]
    command.extend(_xtb_method_flags(str(config["gfn"])))
    command.extend(["--chrg", str(int(config["charge"])), "--uhf", str(int(config["uhf"]))])
    solvent_model = str(config["solvent_model"])
    solvent = str(config["solvent"])
    if solvent_model == "alpb":
        command.extend(["--alpb", solvent])
    elif solvent_model == "gbsa":
        command.extend(["--gbsa", solvent])

    mode = str(config["mode"])
    if mode == "opt":
        command.extend(["--opt", str(config["opt_level"])])
    elif mode == "hess":
        command.append("--hess")
    elif mode == "md":
        command.append("--md")
    if config.get("xcontrol_file"):
        command.extend(["--input", str(config["xcontrol_file"])])
    return command


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
    parser = argparse.ArgumentParser(description="Manifest-driven xTB execution wrapper for DPDispatcher tasks")
    parser.add_argument("--manifest", default="manifest.json", help="Prepared xTB stage manifest")
    parser.add_argument("--xtb_bin", default="xtb", help="xTB executable name")
    parser.add_argument("--log", default="xtb_stdout.out", help="Combined stdout/stderr log")
    args = parser.parse_args()

    manifest_path = Path(args.manifest)
    try:
        config = _load_manifest(manifest_path)
    except Exception as exc:
        sys.stderr.write(f"[xtb_boot] invalid prepared stage: {exc}\n")
        return 2

    xtb_bin = shutil.which(args.xtb_bin) or args.xtb_bin
    command = _build_xtb_command(config, xtb_bin=xtb_bin)
    summary_path = Path("xtb_summary.json")
    started = time.time()
    nthreads = _resolve_threads()
    env = _xtb_env(nthreads)
    returncode = 127
    execution_error = ""
    with open(args.log, "w", encoding="utf-8") as log_handle:
        log_handle.write(f"[xtb_boot] cwd={Path.cwd()}\n")
        log_handle.write(f"[xtb_boot] manifest={manifest_path.name}\n")
        log_handle.write(f"[xtb_boot] command={' '.join(command)}\n")
        log_handle.write(f"[xtb_boot] xtb_bin={xtb_bin}\n")
        log_handle.write(f"[xtb_boot] threads={nthreads}\n")
        for key in (
            "SLURM_JOB_ID",
            "SLURM_NTASKS",
            "SLURM_CPUS_PER_TASK",
            "SLURM_CPUS_ON_NODE",
            "SLURM_JOB_CPUS_PER_NODE",
            "OMP_NUM_THREADS",
            "MKL_NUM_THREADS",
            "OPENBLAS_NUM_THREADS",
            "OMP_MAX_ACTIVE_LEVELS",
            "OMP_STACKSIZE",
        ):
            log_handle.write(f"[xtb_boot] env {key}={env.get(key, '')}\n")
        log_handle.flush()
        try:
            proc = subprocess.run(command, stdout=log_handle, stderr=subprocess.STDOUT, env=env, check=False)
            returncode = int(proc.returncode)
        except OSError as exc:
            execution_error = f"{type(exc).__name__}: {exc}"
            log_handle.write(f"[xtb_boot] execution_error={execution_error}\n")

    payload = {
        "completed": returncode == 0,
        "returncode": returncode,
        "command": command,
        "manifest": manifest_path.name,
        "coordinate_file": config["coordinate_file"],
        "xcontrol_file": config["xcontrol_file"] or None,
        "mode": config["mode"],
        "gfn": config["gfn"],
        "solvent_model": config["solvent_model"],
        "solvent": config["solvent"] or None,
        "charge": config["charge"],
        "uhf": config["uhf"],
        "started_at": started,
        "finished_at": time.time(),
        "threads": nthreads,
        "thread_env": {
            "OMP_NUM_THREADS": env.get("OMP_NUM_THREADS", ""),
            "MKL_NUM_THREADS": env.get("MKL_NUM_THREADS", ""),
            "OPENBLAS_NUM_THREADS": env.get("OPENBLAS_NUM_THREADS", ""),
            "OMP_MAX_ACTIVE_LEVELS": env.get("OMP_MAX_ACTIVE_LEVELS", ""),
            "OMP_STACKSIZE": env.get("OMP_STACKSIZE", ""),
        },
        "outputs": _collect_outputs(),
        "log_file": args.log,
        "execution_error": execution_error or None,
    }
    summary_path.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return returncode


if __name__ == "__main__":
    try:
        os.system("ulimit -s unlimited >/dev/null 2>&1")
    except Exception:
        pass
    raise SystemExit(main())
