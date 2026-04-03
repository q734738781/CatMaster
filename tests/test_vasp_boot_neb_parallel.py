from __future__ import annotations

import io
from pathlib import Path
from types import SimpleNamespace

from catmaster.remote.cpu import vasp_boot


def _write_incar(path: Path, text: str) -> None:
    path.write_text(text.strip() + "\n", encoding="utf-8")


def test_resolve_effective_nprocs_prefers_layout_with_ncore_gt_one(tmp_path: Path) -> None:
    incar = tmp_path / "INCAR"
    _write_incar(incar, "IMAGES = 5\nEDIFF = 1E-6")
    log = io.StringIO()

    args = SimpleNamespace(nprocs=96, incar=str(incar))

    nprocs, neb_images = vasp_boot._resolve_effective_nprocs(args, log)

    assert nprocs == 90
    assert neb_images == 5
    assert "reducing ranks from 96 to 90" in log.getvalue()
    assert "NCORE>1" in log.getvalue()


def test_resolve_effective_nprocs_keeps_best_divisible_layout_when_already_good(tmp_path: Path) -> None:
    incar = tmp_path / "INCAR"
    _write_incar(incar, "IMAGES = 6\nEDIFF = 1E-6")
    log = io.StringIO()

    args = SimpleNamespace(nprocs=96, incar=str(incar))

    nprocs, neb_images = vasp_boot._resolve_effective_nprocs(args, log)

    assert nprocs == 96
    assert neb_images == 6
    assert "ranks=96, ranks/image=16" in log.getvalue()


def test_resolve_effective_nprocs_uses_60_for_64_core_five_image_neb(tmp_path: Path) -> None:
    incar = tmp_path / "INCAR"
    _write_incar(incar, "IMAGES = 5\nEDIFF = 1E-6")
    log = io.StringIO()

    args = SimpleNamespace(nprocs=64, incar=str(incar))

    nprocs, neb_images = vasp_boot._resolve_effective_nprocs(args, log)

    assert nprocs == 60
    assert neb_images == 5
    assert "reducing ranks from 64 to 60" in log.getvalue()


def test_maybe_set_ncore_uses_ranks_per_image_for_neb(tmp_path: Path) -> None:
    incar = tmp_path / "INCAR"
    _write_incar(incar, "IMAGES = 6\nEDIFF = 1E-6")
    log = io.StringIO()

    args = SimpleNamespace(auto_ncore=True, incar=str(incar), cpu_per_node=96)

    vasp_boot._maybe_set_ncore(args, log, nprocs=96, neb_images=6)

    incar_text = incar.read_text(encoding="utf-8")
    assert "NCORE = 4" in incar_text
    assert "ranks/image=16" in log.getvalue()
    assert "ncore=4" in log.getvalue()
