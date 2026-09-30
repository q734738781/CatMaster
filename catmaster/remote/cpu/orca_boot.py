from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import re
import shutil
import subprocess
import sys
import time

try:
    from .node_local_scratch import ScratchStageOutError, WorkDirectory, node_local_workdir
except ImportError:  # Staged boot scripts run as task_script/orca_boot.py.
    from node_local_scratch import ScratchStageOutError, WorkDirectory, node_local_workdir

_PAL_TOKEN_RE = re.compile(
    r"^PAL([1-9][0-9]*)(?:\(([1-9][0-9]*)x([1-9][0-9]*)\))?$",
    re.IGNORECASE,
)
_NPROCS_RE = re.compile(r"\b(?:nprocs_world|nprocs)\s+([1-9][0-9]*)\b", re.IGNORECASE)
_NPROCS_GROUP_RE = re.compile(r"\bnprocs_group\s+([1-9][0-9]*)\b", re.IGNORECASE)
_SLURM_CPU_COUNT_RE = re.compile(r"^\s*([0-9]+)")


def _resolve_orca_binary(requested: str) -> str:
    if requested and requested != "auto":
        return shutil.which(requested) or requested
    for candidate in ("orca", "orca5", "orca_5_0_3", "orca-5.0.3", "orca-6.1.1"):
        resolved = shutil.which(candidate)
        if resolved:
            return resolved
    return "orca"


def _parse_cpu_count(raw: str) -> int | None:
    match = _SLURM_CPU_COUNT_RE.match(str(raw or ""))
    if not match:
        return None
    value = int(match.group(1))
    return value if value > 0 else None


def _resolve_nprocs() -> int:
    for key in ("SLURM_NTASKS", "SLURM_CPUS_ON_NODE", "SLURM_JOB_CPUS_PER_NODE", "OMP_NUM_THREADS"):
        value = _parse_cpu_count(os.environ.get(key, ""))
        if value:
            return value
    return 1


def _pal_block_spans(lines: list[str]) -> list[tuple[int, int]]:
    spans: list[tuple[int, int]] = []
    index = 0
    while index < len(lines):
        stripped = lines[index].strip()
        if not re.match(r"(?i)^%pal(?:\s|$)", stripped):
            index += 1
            continue
        if re.search(r"(?i)\bend\s*$", stripped) and stripped.lower() != "%pal":
            spans.append((index, index))
            index += 1
            continue
        stop = index + 1
        while stop < len(lines) and lines[stop].strip().lower() != "end":
            stop += 1
        if stop >= len(lines):
            raise ValueError("Unterminated %pal block")
        spans.append((index, stop))
        index = stop + 1
    return spans


def _runtime_input_text(canonical_text: str, nprocs: int) -> str:
    lines = canonical_text.splitlines()
    simple_indices: list[int] = []
    authored_counts: list[int] = []
    authored_group_sizes: list[int] = []
    for simple_index, line in enumerate(lines):
        if not line.lstrip().startswith("!"):
            continue
        simple_indices.append(simple_index)
        prefix, marker, remainder = line.partition("!")
        tokens = remainder.split()
        retained: list[str] = []
        for token in tokens:
            match = _PAL_TOKEN_RE.match(token)
            if match:
                authored_counts.append(int(match.group(1)))
                if match.group(3):
                    authored_group_sizes.append(int(match.group(3)))
            else:
                retained.append(token)
        lines[simple_index] = prefix + marker + (" " + " ".join(retained) if retained else "")

    spans = _pal_block_spans(lines)
    if len(spans) > 1:
        raise ValueError("Multiple %pal blocks are ambiguous")
    if spans:
        start, stop = spans[0]
        block_text = "\n".join(lines[start : stop + 1])
        block_counts = [int(value) for value in _NPROCS_RE.findall(block_text)]
        if len(block_counts) > 1:
            raise ValueError("Multiple nprocs entries in %pal are ambiguous")
        authored_counts.extend(block_counts)
        block_group_sizes = [int(value) for value in _NPROCS_GROUP_RE.findall(block_text)]
        if len(block_group_sizes) > 1:
            raise ValueError("Multiple nprocs_group entries in %pal are ambiguous")
        authored_group_sizes.extend(block_group_sizes)
    if len(set(authored_counts)) > 1:
        raise ValueError("Conflicting PALn and %pal nprocs values in canonical input")
    if len(set(authored_group_sizes)) > 1:
        raise ValueError("Conflicting PAL grouping and %pal nprocs_group values in canonical input")
    group_size = authored_group_sizes[0] if authored_group_sizes else None
    if group_size is not None and nprocs % group_size:
        raise ValueError(f"Runtime nprocs={nprocs} is not divisible by authored nprocs_group={group_size}")

    if not spans:
        insert_at = (simple_indices[-1] + 1) if simple_indices else 0
        block = ["%pal", f"  nprocs {int(nprocs)}"]
        if group_size is not None:
            block.append(f"  nprocs_group {group_size}")
        block.append("end")
        lines[insert_at:insert_at] = block
    else:
        start, stop = spans[0]
        if start == stop:
            line = lines[start]
            if _NPROCS_RE.search(line):
                lines[start] = _NPROCS_RE.sub(f"nprocs {int(nprocs)}", line, count=1)
            else:
                lines[start] = re.sub(r"(?i)\bend\s*$", f"nprocs {int(nprocs)} end", line, count=1)
        else:
            nprocs_lines = [index for index in range(start, stop) if _NPROCS_RE.search(lines[index])]
            if len(nprocs_lines) > 1:
                raise ValueError("Multiple nprocs entries in %pal are ambiguous")
            if nprocs_lines:
                index = nprocs_lines[0]
                lines[index] = _NPROCS_RE.sub(f"nprocs {int(nprocs)}", lines[index], count=1)
            else:
                lines.insert(stop, f"  nprocs {int(nprocs)}")
        if group_size is not None and not _NPROCS_GROUP_RE.search("\n".join(lines[start : stop + 2])):
            updated_start, updated_stop = _pal_block_spans(lines)[0]
            if updated_start == updated_stop:
                lines[updated_start] = re.sub(
                    r"(?i)\bend\s*$",
                    f"nprocs_group {group_size} end",
                    lines[updated_start],
                    count=1,
                )
            else:
                lines.insert(updated_stop, f"  nprocs_group {group_size}")
    return "\n".join(lines) + "\n"


def _collect_outputs(*, additional_paths: tuple[Path, ...] = ()) -> list[str]:
    names = {path.name for path in Path.cwd().iterdir() if path.is_file() and path.name != "job.inp"}
    names.update(path.name for path in additional_paths if path.is_file())
    return sorted(names)


def _try_orca_2json(stem: str) -> None:
    converter = shutil.which("orca_2json")
    property_text = Path(f"{stem}.property.txt")
    property_json = Path(f"{stem}.property.json")
    if not converter or not property_text.is_file() or property_json.is_file():
        return
    try:
        subprocess.run(
            [converter, stem, "-property"],
            stdout=subprocess.DEVNULL,
            stderr=subprocess.DEVNULL,
            check=False,
            shell=False,
        )
    except Exception:
        return


def _run(args: argparse.Namespace, workdir: WorkDirectory) -> int:
    canonical_input = Path(args.input)
    try:
        runtime_input = canonical_input.with_name(f"{canonical_input.stem}.runtime{canonical_input.suffix}")
        runtime_input.write_text(
            _runtime_input_text(canonical_input.read_text(encoding="utf-8", errors="replace"), _resolve_nprocs()),
            encoding="utf-8",
        )
    except Exception as exc:
        sys.stderr.write(f"[orca_boot] runtime PAL reconciliation failed: {exc}\n")
        return 2

    orca_bin = _resolve_orca_binary(args.orca_bin)
    output_path = workdir.shared_dir / "job.out" if workdir.node_local else Path("job.out")
    if workdir.node_local:
        # The stage may contain a previous job.out. Remove its scratch copy so
        # final stage-out cannot overwrite the live log opened on shared storage.
        Path("job.out").unlink(missing_ok=True)
    started = time.time()
    with Path(args.log).open("w", encoding="utf-8") as log_handle, output_path.open("w", encoding="utf-8") as output_handle:
        process = subprocess.run(
            [orca_bin, runtime_input.name],
            stdout=output_handle,
            stderr=subprocess.STDOUT,
            check=False,
        )
        log_handle.write(f"returncode={process.returncode}\n")

    if process.returncode == 0:
        _try_orca_2json(runtime_input.stem)
    output_text = output_path.read_text(encoding="utf-8", errors="replace") if output_path.is_file() else ""
    normal_termination = "ORCA TERMINATED NORMALLY" in output_text
    payload = {
        "execution_state": "completed" if process.returncode == 0 else "failed",
        "returncode": int(process.returncode),
        "normal_termination": normal_termination,
        "canonical_input": canonical_input.name,
        "runtime_input": runtime_input.name,
        "started_at": started,
        "finished_at": time.time(),
        "outputs": _collect_outputs(additional_paths=(output_path,)),
        "log_file": args.log,
    }
    Path("orca_summary.json").write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    return int(process.returncode)


def main() -> int:
    parser = argparse.ArgumentParser(description="ORCA boot wrapper for a canonical prepared input")
    parser.add_argument("--input", default="job.inp")
    parser.add_argument("--orca_bin", default="auto")
    parser.add_argument("--log", default="orca_stdout.out")
    args = parser.parse_args()

    canonical_input = Path(args.input)
    if not canonical_input.is_file():
        sys.stderr.write(f"[orca_boot] input file missing: {canonical_input}\n")
        return 2
    try:
        with node_local_workdir("orca") as workdir:
            return _run(args, workdir)
    except ScratchStageOutError as exc:
        sys.stderr.write(f"[orca_boot] {exc}\n")
        return getattr(os, "EX_IOERR", 74)


if __name__ == "__main__":
    raise SystemExit(main())
