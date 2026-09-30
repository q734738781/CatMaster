#!/usr/bin/env python3
# Written 2026-09-13 by Codex for CatMaster deployment.
# Install the pinned upstream runtime bundle before service startup; Python
# dependencies are owned by requirements/pc-conda.yml.
"""Install EasySlides scripts, skills, templates and references (not a PyPI package)."""

from __future__ import annotations

import argparse
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import urllib.request
import zipfile


REVISION = "e69cbdbf24479a66c8ff6a7fc1c9a33f42f563fc"
SOURCE_URL = f"https://codeload.github.com/Rimagination/easyslides/zip/{REVISION}"
DEFAULT_DEST = Path(__file__).resolve().parents[1] / "third_party" / "easyslides"
REVISION_FILE = ".catmaster-source-revision"


def install(destination: Path, *, quiet: bool = False) -> Path:
    destination = destination.resolve()
    marker = destination / REVISION_FILE
    if (
        marker.is_file()
        and marker.read_text().strip() == REVISION
        and (destination / "scripts" / "easyslides.py").is_file()
    ):
        if not quiet:
            print(f"EasySlides already installed: {destination}")
        return destination

    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(prefix=".easyslides-", dir=destination.parent) as work:
        work_dir = Path(work)
        archive = work_dir / "source.zip"
        request = urllib.request.Request(SOURCE_URL, headers={"User-Agent": "CatMaster-installer"})
        with urllib.request.urlopen(request, timeout=120) as response, archive.open("wb") as output:
            shutil.copyfileobj(response, output)
        with zipfile.ZipFile(archive) as source_zip:
            source_zip.extractall(work_dir)
        source = work_dir / f"easyslides-{REVISION}"
        bundle = work_dir / "bundle"
        # Upstream owns the complete distributable layout and relative references.
        subprocess.run(
            [sys.executable, str(source / "scripts" / "build_plugin_bundle.py"), "--out", str(bundle)],
            check=True,
            stdout=subprocess.DEVNULL if quiet else None,
        )
        (bundle / REVISION_FILE).write_text(REVISION + "\n")
        backup = work_dir / "previous"
        if destination.exists():
            destination.rename(backup)
        try:
            bundle.rename(destination)
        except OSError:
            if backup.exists():
                backup.rename(destination)
            raise
    if not quiet:
        print(f"EasySlides installed: {destination}")
    return destination


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dest", type=Path, default=DEFAULT_DEST)
    parser.add_argument("--quiet", action="store_true")
    args = parser.parse_args()
    install(args.dest, quiet=args.quiet)


if __name__ == "__main__":
    main()
