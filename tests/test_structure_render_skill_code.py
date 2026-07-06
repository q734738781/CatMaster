from __future__ import annotations

import importlib.util
import json
import subprocess
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

from catmaster.tools.registry import get_tool_registry


REPO_ROOT = Path(__file__).resolve().parents[1]
SCRIPT_PATH = REPO_ROOT / "skills" / "materials_worker" / "structure-visual-inspection" / "code" / "render_structure_panel.py"


def _load_render_module():
    spec = importlib.util.spec_from_file_location("structure_visual_render_code", SCRIPT_PATH)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_render_structure_panel_code_generates_panel_and_tiles(tmp_path: Path) -> None:
    render_module = _load_render_module()
    struct_path = tmp_path / "co.xyz"
    write(struct_path, Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]))

    metadata = render_module.render_structure_panel(
        structure_path=struct_path,
        output_path=tmp_path / "co_panel.png",
        tile_size=(320, 300),
        fit_scale=1.1,
        atom_scale=0.5,
        show_cell=False,
    )

    assert Path(metadata["image_path"]).exists()
    assert set(metadata["legend_mapping"]) == {"C", "O"}
    assert [view["name"] for view in metadata["views"]] == ["front", "right", "top", "iso"]
    for rel in metadata["tile_paths"].values():
        assert Path(rel).exists()


def test_render_structure_panel_cli_accepts_custom_views_json(tmp_path: Path) -> None:
    struct_path = tmp_path / "co.xyz"
    write(struct_path, Atoms("CO", positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]]))
    views_path = tmp_path / "views.json"
    views_path.write_text(
        json.dumps(
            [
                {
                    "name": "tilted",
                    "title": "Tilted",
                    "camera_dir": [-1.0, -0.6, -0.35],
                    "camera_up": [0.0, 0.0, 1.0],
                }
            ]
        ),
        encoding="utf-8",
    )
    output_path = tmp_path / "custom.png"
    metadata_path = tmp_path / "custom.json"

    subprocess.run(
        [
            sys.executable,
            str(SCRIPT_PATH),
            str(struct_path),
            "-o",
            str(output_path),
            "--views-json",
            str(views_path),
            "--tile-size",
            "300,260",
            "--columns",
            "1",
            "--fit-scale",
            "1.2",
            "--atom-scale",
            "0.45",
            "--metadata-json",
            str(metadata_path),
        ],
        check=True,
        cwd=str(REPO_ROOT),
        capture_output=True,
        text=True,
    )

    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    assert output_path.exists()
    assert metadata["views"][0]["name"] == "tilted"
    assert metadata["fit_scale"] == 1.2
    assert metadata["atom_scale"] == 0.45


def test_render_structure_views_tool_is_not_registered() -> None:
    registry = get_tool_registry()
    assert "render_structure_views" not in registry.tools
