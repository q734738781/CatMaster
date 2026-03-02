from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest

pytest.importorskip("pymatgen")

from pymatgen.core import Structure

from catmaster.tools.geometry_inputs.adsorbate_tool import (
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
