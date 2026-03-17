from __future__ import annotations

import shutil
from pathlib import Path

from pymatgen.core import Structure

from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.slab_tools import (
    fix_atoms_by_height,
    fix_atoms_by_indices,
    fix_atoms_by_layers,
)


def _prepare_workspace(tmp_path: Path) -> tuple[Path, str]:
    project = tmp_path / "project_space"
    with workspace_scope(project):
        pass
    source = Path(__file__).resolve().parents[1] / "tests" / "assets" / "Fe_bcc_111__CONTCAR_h111_t0.vasp"
    structure_rel = "inputs/slab.vasp"
    target = project / "files" / structure_rel
    target.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(source, target)
    return project, structure_rel


def _load_relax_mask(output_path: Path) -> list[bool]:
    structure = Structure.from_file(output_path)
    selective = structure.site_properties["selective_dynamics"]
    return [bool(all(vector)) for vector in selective]


def test_fix_atoms_by_indices_uses_zero_based_indices(tmp_path: Path) -> None:
    project, structure_rel = _prepare_workspace(tmp_path)
    with workspace_scope(project):
        _content, artifact = fix_atoms_by_indices(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_indices.vasp",
                "indices": [0, 2, 4],
            }
        )

    data = artifact["data"]
    assert data["indices"] == [0, 2, 4]
    assert data["index_base"] == 0
    assert data["reverse"] is False
    relax_mask = _load_relax_mask(project / "files" / "outputs" / "by_indices.vasp")
    assert relax_mask[0] is False
    assert relax_mask[1] is True
    assert relax_mask[2] is False
    assert relax_mask[4] is False


def test_fix_atoms_by_indices_reverse_keeps_selected_atoms_free(tmp_path: Path) -> None:
    project, structure_rel = _prepare_workspace(tmp_path)
    with workspace_scope(project):
        _content, artifact = fix_atoms_by_indices(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_indices_reverse.vasp",
                "indices": [0, 2, 4],
                "reverse": True,
            }
        )

    data = artifact["data"]
    assert data["reverse"] is True
    relax_mask = _load_relax_mask(project / "files" / "outputs" / "by_indices_reverse.vasp")
    assert relax_mask[0] is True
    assert relax_mask[1] is False
    assert relax_mask[2] is True
    assert relax_mask[4] is True


def test_fix_atoms_by_layers_reverse_complements_selection(tmp_path: Path) -> None:
    project, structure_rel = _prepare_workspace(tmp_path)
    with workspace_scope(project):
        _content_a, artifact_a = fix_atoms_by_layers(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_layers_normal.vasp",
                "freeze_layers": 2,
                "layer_tol": 0.2,
            }
        )
        _content_b, artifact_b = fix_atoms_by_layers(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_layers_reverse.vasp",
                "freeze_layers": 2,
                "layer_tol": 0.2,
                "reverse": True,
            }
        )

    mask_normal = _load_relax_mask(project / "files" / "outputs" / "by_layers_normal.vasp")
    mask_reverse = _load_relax_mask(project / "files" / "outputs" / "by_layers_reverse.vasp")
    assert artifact_a["data"]["reverse"] is False
    assert artifact_b["data"]["reverse"] is True
    assert all(a != b for a, b in zip(mask_normal, mask_reverse))


def test_fix_atoms_by_height_reverse_complements_selection(tmp_path: Path) -> None:
    project, structure_rel = _prepare_workspace(tmp_path)
    source = Structure.from_file(project / "files" / structure_rel)
    z_values = source.cart_coords[:, 2]
    z_min = float(z_values.min())
    z_mid = z_min + 2.5

    with workspace_scope(project):
        _content_a, artifact_a = fix_atoms_by_height(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_height_normal.vasp",
                "z_ranges": [{"z_min": z_min, "z_max": z_mid}],
            }
        )
        _content_b, artifact_b = fix_atoms_by_height(
            {
                "structure_ref": structure_rel,
                "output_path": "outputs/by_height_reverse.vasp",
                "z_ranges": [{"z_min": z_min, "z_max": z_mid}],
                "reverse": True,
            }
        )

    mask_normal = _load_relax_mask(project / "files" / "outputs" / "by_height_normal.vasp")
    mask_reverse = _load_relax_mask(project / "files" / "outputs" / "by_height_reverse.vasp")
    assert artifact_a["data"]["reverse"] is False
    assert artifact_b["data"]["reverse"] is True
    assert all(a != b for a, b in zip(mask_normal, mask_reverse))
