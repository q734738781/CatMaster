from __future__ import annotations

import json
import shutil
from pathlib import Path

import pytest
from pymatgen.core import Molecule, Structure
from pymatgen.io.vasp import Poscar

from catmaster.tools.analysis.fragment_probe import identify_structure_fragments
from catmaster.tools.base import workspace_scope
from catmaster.tools.geometry_inputs.adsorbate_tool import ADS_META_SCHEMA, _load_inherited_ads_indices, place_adsorbate
from catmaster.tools.geometry_inputs.vasp_prepare import vasp_prepare


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


def test_identify_structure_fragments_detects_detached_co_by_ads_indices_overlap(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        slab = Structure.from_file(str(slab_path))
        co = Molecule.from_file(str(ads_path))
        z_shift = float(max(slab.cart_coords[:, 2])) + 6.0
        for specie, coords in zip(co.species, co.cart_coords):
            shifted = [float(coords[0]), float(coords[1]), float(coords[2] + z_shift)]
            slab.append(specie, shifted, coords_are_cartesian=True)
        if "selective_dynamics" in slab.site_properties:
            selective = list(slab.site_properties["selective_dynamics"])
            selective = [
                [bool(v[0]), bool(v[1]), bool(v[2])] if isinstance(v, (list, tuple)) else [True, True, True]
                for v in selective[: len(slab) - len(co)]
            ] + [[True, True, True] for _ in range(len(co))]
            slab = Structure(
                lattice=slab.lattice,
                species=list(slab.species),
                coords=slab.cart_coords,
                coords_are_cartesian=True,
                site_properties={"selective_dynamics": selective},
            )

        detached_path = tmp_path / "files" / "thermo" / "detached_co.vasp"
        detached_path.parent.mkdir(parents=True, exist_ok=True)
        Poscar(slab).write_file(str(detached_path))
        ads_indices = [len(slab) - 2, len(slab) - 1]
        meta_path = detached_path.with_suffix(".meta.json")
        meta_path.write_text(
            json.dumps(
                {
                    "schema": ADS_META_SCHEMA,
                    "output_structure_rel": "thermo/detached_co.vasp",
                    "ads_indices": ads_indices,
                    "ads_indices_added": ads_indices,
                },
                indent=2,
            ),
            encoding="utf-8",
        )

        _content, artifact = identify_structure_fragments(
            {
                "structure_path": "thermo/detached_co.vasp",
                "select_reference_overlap": True,
                "match_formula": "CO",
            }
        )

    data = artifact["data"]
    assert data["fragment_count"] == 2
    assert data["reference_source"] == "ads_indices_sidecar"
    assert data["selected_fragment_ids"] == [2]
    assert data["selected_indices"] == ads_indices
    selected_fragment = next(row for row in data["fragments"] if row["frag_id"] == 2)
    assert selected_fragment["formula"] == "CO"
    assert selected_fragment["reference_overlap_count"] == 2
    out_json = tmp_path / "files" / data["fragments_json_rel"]
    assert out_json.exists()


def test_adsorbate_indices_remain_valid_through_vasp_input_writing(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        slab_path, ads_path = _copy_assets(tmp_path)
        _, artifact = place_adsorbate(
            {
                "slab_file": slab_path.name,
                "adsorbate_file": ads_path.name,
                "site": "ontop_0",
                "distance": 2.0,
                "output_poscar": "out/ads_0.vasp",
            }
        )
        data = artifact["data"]
        placed_path = tmp_path / "files" / data["output_poscar_rel"]
        placed = Structure.from_file(str(placed_path))

        vasp_prepare(
            {
                "input_path": data["output_poscar_rel"],
                "output_root": "jobs/ads_freq",
                "preset": "freq",
                "regime": "slab",
            }
        )

    prepared = Structure.from_file(str(tmp_path / "files" / "jobs" / "ads_freq" / "POSCAR"))
    ads_indices = data["ads_indices_added"]
    assert [str(prepared[idx].specie) for idx in ads_indices] == [str(placed[idx].specie) for idx in ads_indices]
    for idx in ads_indices:
        assert prepared.cart_coords[idx] == pytest.approx(placed.cart_coords[idx], rel=0.0, abs=1e-8)
    propagated_indices, warnings = _load_inherited_ads_indices(tmp_path / "files" / "jobs" / "ads_freq" / "POSCAR")
    assert warnings == []
    assert propagated_indices == ads_indices
