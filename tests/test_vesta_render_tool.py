from __future__ import annotations

import json
import sys
from pathlib import Path

from ase import Atoms
from ase.io import write

from catmaster.tools.analysis.vesta_render import _resolve_vesta_executable, render_vesta_views
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.registry import get_tool_registry


def _write_fake_vesta(path: Path) -> None:
    path.write_text(
        f"""#!{sys.executable}
import sys
from matplotlib import pyplot as plt

args = sys.argv[1:]
index = args.index('-export_img')
output = args[index + 2]
fig, axis = plt.subplots(figsize=(1, 1), dpi=96)
axis.plot([0, 1], [0, 1])
axis.set_axis_off()
fig.savefig(output, dpi=96)
plt.close(fig)
""",
        encoding="utf-8",
    )
    path.chmod(0o755)


def test_resolve_vesta_executable_prefers_explicit_config(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "custom-vesta"
    _write_fake_vesta(executable)
    monkeypatch.setenv("CATMASTER_VESTA_BIN", str(executable))
    monkeypatch.setenv("PATH", "")

    assert _resolve_vesta_executable() == executable.resolve()


def test_vesta_tool_schema_keeps_optional_arrays_non_nullable() -> None:
    schema = next(
        tool["parameters"]
        for tool in get_tool_registry().as_openai_tools(allowlist=["render_vesta_views"])
        if tool["name"] == "render_vesta_views"
    )

    assert schema["properties"]["views"]["type"] == "array"
    assert schema["properties"]["supercell"]["type"] == "array"
    assert "anyOf" not in schema["properties"]["views"]
    assert "anyOf" not in schema["properties"]["supercell"]


def test_render_vesta_views_writes_images_and_metadata(tmp_path: Path, monkeypatch) -> None:
    executable = tmp_path / "fake-vesta"
    _write_fake_vesta(executable)
    monkeypatch.setenv("CATMASTER_VESTA_BIN", str(executable))
    monkeypatch.setenv("DISPLAY", ":99")

    project_space = tmp_path / "project"
    layout = ensure_project_space_layout(project_space)
    structure_path = layout["files_root"] / "input.vasp"
    atoms = Atoms(
        "CO",
        positions=[[0.0, 0.0, 0.0], [1.15, 0.0, 0.0]],
        cell=[8.0, 8.0, 8.0],
        pbc=True,
    )
    write(structure_path, atoms, format="vasp", direct=True, vasp5=True)

    with workspace_scope(project_space):
        content, artifact = render_vesta_views(
            {
                "structure_path": "input.vasp",
                "output_dir": "structures/rendered",
                "views": ["top", "side"],
                "include_panel": False,
            }
        )

    data = artifact["data"]
    assert artifact["tool_name"] == "render_vesta_views"
    assert data["backend"] == "vesta"
    assert data["display_mode"] == "display"
    assert data["supercell"] == [1, 1, 1]
    assert data["panel_path"] == ""
    assert set(data["views"]) == {"top", "side"}
    assert "Use read_file" in content
    for relative_path in data["views"].values():
        assert (layout["files_root"] / relative_path).is_file()
    metadata = json.loads((layout["files_root"] / data["metadata_path"]).read_text(encoding="utf-8"))
    assert metadata["publication_acknowledgement_required"] is True
