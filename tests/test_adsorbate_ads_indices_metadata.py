from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")
from pymatgen.core import Structure

from catmaster.tools.geometry_inputs.adsorbate_tool import (
    enumerate_adsorption_sites,
    generate_batch_adsorption_structures,
    place_adsorbate,
)
from catmaster.tools.base import workspace_scope


def _copy_assets(workspace: Path) -> tuple[Path, Path]:
    root = Path(__file__).resolve().parents[1]
    slab_src = root / "tests" / "assets" / "Fe_hkl111_12A_15AVac_5ARelax.vasp"
    ads_src = root / "tests" / "assets" / "CO.xyz"
    files_root = workspace / "files"
    files_root.mkdir(parents=True, exist_ok=True)
    slab_dst = files_root / "slab.vasp"
    ads_dst = files_root / "CO.xyz"
    shutil.copy2(slab_src, slab_dst)
    shutil.copy2(ads_src, ads_dst)
    return slab_dst, ads_dst


def test_place_adsorbate_writes_ads_indices_and_meta(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        files_root = tmp_path / "files"
        base_natoms = len(Structure.from_file(str(slab_path)))

        result = place_adsorbate(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "site": "ontop_0",
                "distance": 2.0,
                "output_poscar": "out/ads_0.vasp",
            }
        )
        _, artifact = result
        data = artifact["data"]
        assert data["ads_indices_added"] == [base_natoms, base_natoms + 1]
        assert data["ads_indices"] == data["ads_indices_added"]
        assert data["ads_count_added"] == 2
        assert data["ads_count_total"] == 2

        meta_path = files_root / data["metadata_rel"]
        assert meta_path.exists()
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert meta["ads_indices"] == data["ads_indices"]
        assert meta["ads_indices_added"] == data["ads_indices_added"]

        ads_idx_json = files_root / data["ads_indices_json_rel"]
        idx_obj = json.loads(ads_idx_json.read_text(encoding="utf-8"))
        assert idx_obj["schema"] == "catmaster.ads_indices.v1"
        assert len(idx_obj["entries"]) == 1
        assert idx_obj["entries"][0]["output_poscar_rel"] == data["output_poscar_rel"]


def test_place_adsorbate_inherits_ads_indices_union(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        first = place_adsorbate(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "site": "ontop_0",
                "distance": 2.0,
                "output_poscar": "out/ads_0.vasp",
            }
        )
        _, first_artifact = first
        first_indices = first_artifact["data"]["ads_indices"]

        second = place_adsorbate(
            {
                "slab_file": first_artifact["data"]["output_poscar_rel"],
                "adsorbate_file": ads_path.name,
                "site": "bridge_0",
                "distance": 2.0,
                "output_poscar": "out/ads_1.vasp",
            }
        )
        _, second_artifact = second
        second_added = second_artifact["data"]["ads_indices_added"]
        second_total = second_artifact["data"]["ads_indices"]

        assert set(first_indices).issubset(set(second_total))
        assert set(second_added).issubset(set(second_total))
        assert len(second_total) == len(set(first_indices + second_added))


def test_generate_batch_adsorption_structures_writes_ads_indices_metadata(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        files_root = tmp_path / "files"
        result = generate_batch_adsorption_structures(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "mode": "all",
                "distance": 2.0,
                "max_structures": 3,
                "output_dir": "batch_out",
            }
        )
        _, artifact = result
        data = artifact["data"]
        assert data["metadata_files_count"] == data["generated"]
        assert "ads_indices_json_rel" in data

        batch_json = files_root / data["structures"]
        rows = json.loads(batch_json.read_text(encoding="utf-8"))
        assert rows
        for row in rows:
            assert "ads_indices_added" in row
            assert "ads_indices" in row
            assert "metadata_rel" in row
            assert (files_root / row["metadata_rel"]).exists()

        ads_indices_json = files_root / data["ads_indices_json_rel"]
        idx_obj = json.loads(ads_indices_json.read_text(encoding="utf-8"))
        assert idx_obj["schema"] == "catmaster.ads_indices.v1"
        assert len(idx_obj["entries"]) == len(rows)


def test_place_adsorbate_preserves_input_orientation_without_reorient(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, _ = _copy_assets(tmp_path)
        files_root = tmp_path / "files"
        ads_path = files_root / "H2_sideways.xyz"
        ads_path.write_text(
            "2\nH2 sideways\nH -0.5 0.0 0.0\nH 0.5 0.0 0.0\n",
            encoding="utf-8",
        )

        _, artifact = place_adsorbate(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "site": "ontop_0",
                "distance": 2.0,
                "output_poscar": "out/h2_sideways.vasp",
            }
        )
        data = artifact["data"]
        placed = Structure.from_file(str(files_root / data["output_poscar_rel"]))
        ads_indices = data["ads_indices_added"]
        coords = placed.cart_coords[ads_indices]
        delta = coords[1] - coords[0]
        assert pytest.approx(abs(float(delta[0])), rel=0.0, abs=1e-6) == 1.0
        assert pytest.approx(float(delta[1]), rel=0.0, abs=1e-6) == 0.0
        assert pytest.approx(float(delta[2]), rel=0.0, abs=1e-6) == 0.0
        assert data["geom"]["placement_reference"] == "center_of_mass_of_lowest_z_atoms"
        assert data["geom"]["reoriented"] is False


def test_enumerate_adsorption_sites_returns_cartesian_3d_coordinates(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, _ = _copy_assets(tmp_path)
        _, artifact = enumerate_adsorption_sites(
            {
                "slab_file": slab_path.name,
                "mode": "all",
                "distance": 2.0,
                "output_json": "out/sites.json",
            }
        )
        data = artifact["data"]
        rows = json.loads((tmp_path / "files" / data["sites_json_rel"]).read_text(encoding="utf-8"))
        assert rows
        first = rows[0]
        assert "cart_coords" in first
        assert len(first["cart_coords"]) == 3


def test_place_adsorbate_accepts_site_cart_coords_as_direct_coordinate(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        requested_coords = [1.234, 2.345, 8.765]

        _, artifact = place_adsorbate(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "site_cart_coords": requested_coords,
                "distance": 2.0,
                "output_poscar": "out/ads_xy.vasp",
            }
        )
        data = artifact["data"]
        assert data["site"]["label"] is None
        assert data["site"]["selection_mode"] == "cart_direct"
        assert data["site"]["requested_site_cart_coords"] == pytest.approx(requested_coords)
        assert data["site"]["cart_coords"] == pytest.approx(requested_coords)


def test_place_adsorbate_rejects_combined_site_label_and_site_cart_coords(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        with pytest.raises(Exception, match="site_label or site_cart_coords"):
            place_adsorbate(
                {
                    "slab_file": slab_path.name,
                    "adsorbate_file": ads_path.name,
                    "site_label": "ontop_0",
                    "site_cart_coords": [1.0, 2.0, 3.0],
                    "distance": 2.0,
                    "output_poscar": "out/ads_invalid.vasp",
                }
            )
