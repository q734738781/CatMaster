"""Keep batch output names distinct without changing unambiguous legacy names."""
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable


def batch_names(
    paths: Iterable[Path], root: Path, name: Callable[[Path], str],
) -> dict[Path, str]:
    """Allocate one flat name per source; manifests remain the source/name mapping.

    Reserve unique names first. Only colliding names get a numeric suffix, which
    is also checked against source names such as ``sample__2``. No file contents
    or hashes participate in identity.
    """
    sources = sorted(paths)
    proposed = {path: name(path.relative_to(root)) for path in sources}
    counts = Counter(proposed.values())
    used = {value for value, count in counts.items() if count == 1}
    allocated = {}
    for path, base in proposed.items():
        candidate = base
        if counts[base] > 1:
            index = 1
            while candidate in used or candidate in counts:
                candidate = f"{base}__{index}"
                index += 1
        used.add(candidate)
        allocated[path] = candidate
    return allocated
