from __future__ import annotations

import importlib
from pathlib import Path

from ase import Atoms
from ase.io import write
import pytest

from catmaster.tools.analysis.render_structure_views import render_structure_views
from catmaster.tools.base import ensure_project_space_layout, workspace_scope


render_module = importlib.import_module("catmaster.tools.analysis.render_structure_views")


def test_render_structure_views_generates_panel_and_tiles(tmp_path: Path) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    struct_path = files_root / "inputs" / "co.xyz"
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    atoms = Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]])
    write(struct_path, atoms)

    with workspace_scope(tmp_path):
        content, artifact = render_structure_views(
            {
                "structure_path": "inputs/co.xyz",
                "output_path": "viz/co_panel.png",
                "show_legend": True,
                "label_mode": "elements",
            }
        )

    assert "render_structure_views completed" in content
    data = artifact["data"]
    panel = files_root / data["image_path"]
    assert panel.exists()
    assert data["backend_requested"] == "ase"
    assert data["backend_used"] == "ase"
    assert set(data["legend_mapping"].keys()) == {"C", "O"}
    for rel in data["tile_paths"].values():
        assert (files_root / rel).exists()


def test_render_structure_views_auto_falls_back_to_ase(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    struct_path = files_root / "inputs" / "co.xyz"
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    write(struct_path, Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]))

    monkeypatch.setattr(render_module, "_try_render_with_ovito", lambda **_: False)

    with workspace_scope(tmp_path):
        _, artifact = render_module._render_structure_views_impl(
            {
                "structure_path": "inputs/co.xyz",
                "output_path": "viz/co_auto.png",
                "backend": "auto",
            }
        )

    data = artifact["data"]
    assert data["backend_requested"] == "auto"
    assert data["backend_used"] == "ase"


def test_render_structure_views_ovito_request_raises_when_unavailable(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    ensure_project_space_layout(tmp_path, create=True)
    files_root = tmp_path / "files"
    struct_path = files_root / "inputs" / "co.xyz"
    struct_path.parent.mkdir(parents=True, exist_ok=True)
    write(struct_path, Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]))

    monkeypatch.setattr(render_module, "_try_render_with_ovito", lambda **_: False)

    with workspace_scope(tmp_path):
        with pytest.raises(RuntimeError, match="OVITO backend requested but unavailable or render failed"):
            render_module._render_structure_views_impl(
                {
                    "structure_path": "inputs/co.xyz",
                    "output_path": "viz/co_ovito.png",
                    "backend": "ovito",
                }
            )
