"""Explicit analysis sources and unambiguous native-output discovery."""
from pathlib import Path
from collections.abc import Callable

from catmaster.tools.base import resolve_workspace_path


def analysis_sources(result_root: str, result_files: list[str], discover) -> tuple[Path, list[tuple[Path, Path | None]]]:
    if bool(result_root) == bool(result_files):
        raise ValueError("Provide exactly one of result_root or result_files")
    if result_files:
        paths = list(dict.fromkeys(resolve_workspace_path(path) for path in result_files))
        return paths[0], [(path.parent, path) for path in paths]
    root = resolve_workspace_path(result_root, must_exist=True)
    if root.is_file():
        return root, [(root.parent, root)]
    return root, [(directory, None) for directory in discover(root)]


def contains_signal(path: Path, signals: tuple[str, ...]) -> bool:
    with path.open(encoding="utf-8", errors="replace") as stream:
        return any(any(signal in line.upper() for signal in signals) for line in stream)


def select_output(
    directory: Path, *, names: tuple[str, ...], signals: tuple[str, ...],
    patterns: tuple[str, ...] = ("*.out", "*.log"),
) -> Path | None:
    candidates = sorted({p for pattern in patterns for p in directory.glob(pattern) if p.is_file()}
                        | {directory / name for name in names if (directory / name).is_file()})
    identified = [p for p in candidates if contains_signal(p, signals)]
    if not identified:
        identified = [p for p in candidates if p.name in names]
    if len(identified) > 1:
        raise ValueError("Multiple matching output files; pass the desired file as result_root or in result_files: "
                         + ", ".join(str(p) for p in identified))
    return identified[0] if identified else None


def discover_directories(root: Path, markers: tuple[str, ...], selector: Callable[[Path], Path | None]) -> list[Path]:
    """Include root and descendants; a root log must not shadow nested runs."""
    directories = [root, *sorted(path for path in root.rglob("*") if path.is_dir())]
    found = []
    for directory in directories:
        if any((directory / marker).is_file() for marker in markers):
            found.append(directory)
            continue
        try:
            match = selector(directory)
        except ValueError:
            # Preserve ambiguous runs for an explicit per-run parsing error.
            found.append(directory)
            continue
        if match is not None:
            found.append(directory)
    return found
