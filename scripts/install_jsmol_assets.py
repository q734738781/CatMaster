#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
import json
import shutil
import tempfile
import urllib.request
import zipfile
from datetime import datetime, timezone
from pathlib import Path

JSMOL_VERSION = "16.3.13"
JSMOL_BINARY_URL = (
    "https://downloads.sourceforge.net/project/jmol/Jmol/Version%2016.3/"
    "Jmol%2016.3.13/Jmol-16.3.13-binary.zip"
)
REQUIRED_PATHS = ["JSmol.min.js", "j2s"]


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent


def _default_target() -> Path:
    return _repo_root() / "catmaster" / "webui" / "static" / "vendor" / "jsmol"


def _static_root() -> Path:
    return _repo_root() / "catmaster" / "webui" / "static"


def install_katex_font_assets(*, quiet: bool = False) -> int:
    source_root = _repo_root() / "catmaster" / "webui" / "frontend" / "node_modules" / "katex" / "dist" / "fonts"
    if not source_root.is_dir():
        if not quiet:
            print(f"KaTeX font source not found: {source_root}")
        return 0
    target_root = _static_root()
    target_root.mkdir(parents=True, exist_ok=True)
    copied = 0
    for source in source_root.iterdir():
        if not source.is_file():
            continue
        if source.suffix.lower() not in {".woff", ".woff2", ".ttf"}:
            continue
        destination = target_root / f"asset-{source.name}"
        if not destination.exists() or source.stat().st_mtime > destination.stat().st_mtime:
            shutil.copy2(source, destination)
            copied += 1
    if copied and not quiet:
        print(f"Installed {copied} KaTeX font assets to: {target_root}")
    return copied


def _manifest_path(target: Path) -> Path:
    return target / ".install_manifest.json"


def _cache_zip_path() -> Path:
    return Path.home() / ".cache" / "catmaster" / "jsmol" / f"Jmol-{JSMOL_VERSION}-binary.zip"


def _cache_lock_path() -> Path:
    return _cache_zip_path().with_suffix(".lock")


def _is_valid_zip(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        with zipfile.ZipFile(path) as archive:
            return archive.testzip() is None
    except Exception:
        return False


def _download_cache_zip(destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(prefix="catmaster_jsmol_", suffix=".zip", dir=destination.parent, delete=False) as handle:
        temp_path = Path(handle.name)
    try:
        urllib.request.urlretrieve(JSMOL_BINARY_URL, temp_path)
        if not _is_valid_zip(temp_path):
            raise zipfile.BadZipFile(f"Downloaded archive is invalid: {temp_path}")
        os.replace(temp_path, destination)
    finally:
        if temp_path.exists():
            temp_path.unlink()


def _is_install_current(target: Path) -> bool:
    manifest_path = _manifest_path(target)
    if not manifest_path.is_file():
        return False
    try:
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    except Exception:
        return False
    if str(manifest.get("version") or "") != JSMOL_VERSION:
        return False
    return all((target / rel).exists() for rel in REQUIRED_PATHS)


def install_jsmol_assets(target: Path, *, force: bool = False, quiet: bool = False) -> Path:
    target = target.expanduser().resolve()
    if not force and _is_install_current(target):
        if not quiet:
            print(f"JSmol assets already installed: {target}")
        return target

    target.parent.mkdir(parents=True, exist_ok=True)
    cache_zip = _cache_zip_path()
    cache_zip.parent.mkdir(parents=True, exist_ok=True)
    lock_path = _cache_lock_path()
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    with lock_path.open("w", encoding="utf-8") as lock_file:
        try:
            import fcntl

            fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
        except ImportError:
            pass
        cache_valid = _is_valid_zip(cache_zip)
        if force or not cache_valid:
            if cache_zip.exists():
                cache_zip.unlink()
            if not quiet:
                print(f"Downloading JSmol {JSMOL_VERSION} from official Jmol package...")
            _download_cache_zip(cache_zip)
        elif not quiet:
            print(f"Using cached JSmol archive: {cache_zip}")

    with tempfile.TemporaryDirectory(prefix="catmaster_jsmol_") as tmp_dir:
        tmp_root = Path(tmp_dir)
        binary_zip = tmp_root / cache_zip.name
        shutil.copy2(cache_zip, binary_zip)

        extracted_root = tmp_root / "binary"
        with zipfile.ZipFile(binary_zip) as archive:
            archive.extractall(extracted_root)

        jsmol_zip = next(extracted_root.rglob("jsmol.zip"))
        jsmol_root = tmp_root / "jsmol"
        with zipfile.ZipFile(jsmol_zip) as archive:
            archive.extractall(jsmol_root)

        source_root = jsmol_root / "jsmol"
        staging = target.parent / f".{target.name}.tmp"
        if staging.exists():
            shutil.rmtree(staging)
        staging.mkdir(parents=True, exist_ok=True)

        for rel in REQUIRED_PATHS:
            source = source_root / rel
            destination = staging / rel
            if source.is_dir():
                shutil.copytree(source, destination)
            else:
                destination.parent.mkdir(parents=True, exist_ok=True)
                shutil.copy2(source, destination)

        manifest = {
            "version": JSMOL_VERSION,
            "source_url": JSMOL_BINARY_URL,
            "installed_at_utc": datetime.now(timezone.utc).isoformat(),
        }
        _manifest_path(staging).write_text(json.dumps(manifest, indent=2), encoding="utf-8")

        if target.exists():
            shutil.rmtree(target)
        staging.rename(target)

    if not quiet:
        print(f"Installed JSmol assets to: {target}")
    return target


def main() -> int:
    parser = argparse.ArgumentParser(description="Download and install local JSmol assets for the CatMaster WebUI.")
    parser.add_argument("--target", type=Path, default=_default_target(), help="Install destination. Default: catmaster/webui/static/vendor/jsmol")
    parser.add_argument("--force", action="store_true", help="Redownload and reinstall even if the expected version is already present.")
    parser.add_argument("--quiet", action="store_true", help="Suppress status output.")
    args = parser.parse_args()
    install_jsmol_assets(args.target, force=args.force, quiet=args.quiet)
    install_katex_font_assets(quiet=args.quiet)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
