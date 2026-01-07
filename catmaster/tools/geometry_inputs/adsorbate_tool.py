from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Poscar

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class EnumerateAdsorptionSitesInput(BaseModel):
    """Enumerate adsorption sites on a slab using ASF."""

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF), workspace-relative.")
    mode: str = Field("all", description="Which site families to return: all|ontop|bridge|hollow.")
    distance: float = Field(2.0, ge=0.0, description="Height above surface to sample adsorption sites (Å).")
    output_json: str = Field("adsorption/sites.json", description="Output JSON path for site list (workspace-relative).")


class PlaceAdsorbateInput(BaseModel):
    """Place an adsorbate molecule on a slab."""

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF).")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file (XYZ recommended).")
    site: str = Field("auto", description="Site label like ontop_0|bridge_1|hollow_2 or 'auto'.")
    distance: float = Field(2.0, ge=0.0, description="Height used to generate adsorption sites (Å).")
    output_poscar: str = Field("adsorption/adsorbed.vasp", description="Output POSCAR path (workspace-relative).")


class GenerateBatchAdsorptionStructuresInput(BaseModel):
    """Generate multiple adsorbed structures up to max_structures."""

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF).")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file (XYZ recommended).")
    mode: str = Field("all", description="all|ontop|bridge|hollow")
    distance: float = Field(2.0, ge=0.0, description="Height used to generate adsorption sites (Å).")
    max_structures: int = Field(12, ge=1, le=100, description="Maximum number of adsorbed structures to generate.")
    output_dir: str = Field("adsorption/batch", description="Directory to write batch POSCARs.")

_MAX_RETURN_SITES = 50


def _to_list3(x) -> List[float]:
    arr = np.array(x, dtype=float).reshape(3)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def _load_adsorbate_molecule(path: Path) -> Molecule:
    p = path
    suf = p.suffix.lower()
    if suf == ".xyz":
        return Molecule.from_file(str(p))
    if suf in {".vasp", ".poscar", ".contcar"} or p.name.upper() in {"POSCAR", "CONTCAR"}:
        s = Structure.from_file(str(p))
        return Molecule([str(sp) for sp in s.species], s.cart_coords)
    return Molecule.from_file(str(p))


def _parse_site(site: str) -> Tuple[str, int]:
    s = (site or "auto").strip().lower()
    if s in {"auto", "first"}:
        return ("auto", 0)
    parts = s.split("_")
    if len(parts) != 2:
        raise ValueError("site must be 'auto' or like 'ontop_0'")
    kind, idx_str = parts
    if kind not in {"ontop", "bridge", "hollow"}:
        raise ValueError("site kind must be ontop|bridge|hollow|auto")
    idx = int(idx_str)
    if idx < 0:
        raise ValueError("site index must be >=0")
    return kind, idx


def enumerate_adsorption_sites(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = EnumerateAdsorptionSitesInput(**payload)
        slab_path = resolve_workspace_path(params.slab_file, must_exist=True)
        out_json = resolve_workspace_path(params.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        slab = Structure.from_file(str(slab_path))
        asf = AdsorbateSiteFinder(slab)
        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]
        site_rows: List[Dict[str, Any]] = []
        for kind in kinds:
            for i, c in enumerate(ads_sites.get(kind, [])):
                site_rows.append({"label": f"{kind}_{i}", "kind": kind, "cart_coords": _to_list3(c)})

        total_found = len(site_rows)
        truncated = False
        if total_found > _MAX_RETURN_SITES:
            site_rows = site_rows[:_MAX_RETURN_SITES]
            truncated = True

        default_site = None
        for pref in ("ontop", "bridge", "hollow"):
            hit = next((s for s in site_rows if s["kind"] == pref), None)
            if hit:
                default_site = hit["label"]
                break
        default_site = default_site or (site_rows[0]["label"] if site_rows else None)

        out_json.write_text(json.dumps(site_rows, indent=2), encoding="utf-8")

        slab_cart = np.array(slab.cart_coords, dtype=float)
        z_max = float(slab_cart[:, 2].max()) if len(slab_cart) else None

        data = {
            "slab_file_rel": workspace_relpath(slab_path),
            "mode": params.mode,
            "distance": float(params.distance),
            "sites_json_rel": workspace_relpath(out_json),
            "default_site_label": default_site,
            "counts": {
                "ontop": len(ads_sites.get("ontop", [])),
                "bridge": len(ads_sites.get("bridge", [])),
                "hollow": len(ads_sites.get("hollow", [])),
                "returned": len(site_rows),
                "total_in_mode": total_found,
                "truncated": truncated,
                "max_return": _MAX_RETURN_SITES,
            },
            "sites": site_rows,
            "slab_z_max": z_max,
        }
        return create_tool_output("enumerate_adsorption_sites", success=True, data=data)
    except Exception as exc:
        return create_tool_output("enumerate_adsorption_sites", success=False, error=str(exc))


def place_adsorbate(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = PlaceAdsorbateInput(**payload)
        slab_path = resolve_workspace_path(params.slab_file, must_exist=True)
        ads_path = resolve_workspace_path(params.adsorbate_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_poscar)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        slab = Structure.from_file(str(slab_path))
        slab_sd = slab.site_properties.get("selective_dynamics") if slab.site_properties else None
        mol = _load_adsorbate_molecule(ads_path)

        asf = AdsorbateSiteFinder(slab)
        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        kind, idx = _parse_site(params.site)
        if kind == "auto":
            chosen_kind = None
            chosen_coord = None
            for pref in ("ontop", "bridge", "hollow"):
                lst = ads_sites.get(pref, [])
                if lst:
                    chosen_kind = pref
                    chosen_coord = lst[0]
                    idx = 0
                    break
            if chosen_coord is None:
                raise RuntimeError("No adsorption sites found.")
            site_label = f"{chosen_kind}_{idx}"
            site_coord = np.array(chosen_coord, dtype=float)
        else:
            lst = ads_sites.get(kind, [])
            if idx >= len(lst):
                raise ValueError(f"Requested {kind}_{idx} but only {len(lst)} {kind} sites available.")
            site_label = f"{kind}_{idx}"
            site_coord = np.array(lst[idx], dtype=float)

        ads_struct = asf.add_adsorbate(mol, site_coord, translate=True, reorient=True)

        # Selective dynamics handling: preserve slab flags and set adsorbate to [True,True,True]
        slab_sd_list: List[List[bool]]
        if slab_sd and len(slab_sd) == len(slab):
            slab_sd_list = [list(map(bool, v)) for v in slab_sd]
        else:
            slab_sd_list = [[True, True, True] for _ in range(len(slab))]
        ads_flags = [[True, True, True] for _ in range(len(mol))]
        sd_new = slab_sd_list + ads_flags
        if "selective_dynamics" in ads_struct.site_properties:
            ads_struct.remove_site_property("selective_dynamics")
        ads_struct.add_site_property("selective_dynamics", sd_new)

        Poscar(ads_struct).write_file(str(out_path))

        nat_ads = len(mol)
        ads_only_part = np.array(ads_struct.cart_coords[-nat_ads:, :], dtype=float) if nat_ads else np.zeros((0, 3))
        ads_com = ads_only_part.mean(axis=0).tolist() if len(ads_only_part) else None

        data = {
            "slab_file_rel": workspace_relpath(slab_path),
            "adsorbate_file_rel": workspace_relpath(ads_path),
            "output_poscar_rel": workspace_relpath(out_path),
            "site": {"label": site_label, "cart_coords": _to_list3(site_coord)},
            "geom": {"adsorbate_com": ads_com, "units": {"distance": "Å"}},
            "natoms": len(ads_struct),
        }
        return create_tool_output("place_adsorbate", success=True, data=data)
    except Exception as exc:
        return create_tool_output("place_adsorbate", success=False, error=str(exc))


def generate_batch_adsorption_structures(payload: Dict[str, Any]) -> Dict[str, Any]:
    try:
        params = GenerateBatchAdsorptionStructuresInput(**payload)
        slab_path = resolve_workspace_path(params.slab_file, must_exist=True)
        ads_path = resolve_workspace_path(params.adsorbate_file, must_exist=True)
        out_dir = resolve_workspace_path(params.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        slab = Structure.from_file(str(slab_path))
        slab_sd = slab.site_properties.get("selective_dynamics") if slab.site_properties else None
        mol = _load_adsorbate_molecule(ads_path)

        asf = AdsorbateSiteFinder(slab)
        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]
        results: List[Dict[str, Any]] = []
        generated = 0

        for kind in kinds:
            for idx, coord in enumerate(ads_sites.get(kind, [])):
                if generated >= params.max_structures:
                    break
                ads_struct = asf.add_adsorbate(mol, coord, translate=True, reorient=True)

                # selective dynamics propagation
                slab_sd_list = [list(map(bool, v)) for v in slab_sd] if slab_sd and len(slab_sd) == len(slab) else [
                    [True, True, True] for _ in range(len(slab))
                ]
                sd_new = slab_sd_list + [[True, True, True] for _ in range(len(mol))]
                if "selective_dynamics" in ads_struct.site_properties:
                    ads_struct.remove_site_property("selective_dynamics")
                ads_struct.add_site_property("selective_dynamics", sd_new)

                file_path = out_dir / f"{kind}_{idx}.vasp"
                Poscar(ads_struct).write_file(str(file_path))

                results.append(
                    {
                        "label": f"{kind}_{idx}",
                        "output_poscar_rel": workspace_relpath(file_path),
                    }
                )
                generated += 1
            if generated >= params.max_structures:
                break

        total_candidates = sum(len(ads_sites.get(k, [])) for k in kinds)
        if generated == 0:
            raise RuntimeError(f"No adsorption sites found for mode='{params.mode}'.")

        data = {
            "slab_file_rel": workspace_relpath(slab_path),
            "adsorbate_file_rel": workspace_relpath(ads_path),
            "mode": params.mode,
            "distance": float(params.distance),
            "max_structures": params.max_structures,
            "output_dir_rel": workspace_relpath(out_dir),
            "generated": generated,
            "total_candidates": total_candidates,
            "truncated": generated < total_candidates,
            "structures": results,
        }
        return create_tool_output("generate_batch_adsorption_structures", success=True, data=data)
    except Exception as exc:
        return create_tool_output("generate_batch_adsorption_structures", success=False, error=str(exc))


__all__ = [
    "EnumerateAdsorptionSitesInput",
    "PlaceAdsorbateInput",
    "GenerateBatchAdsorptionStructuresInput",
    "enumerate_adsorption_sites",
    "place_adsorbate",
    "generate_batch_adsorption_structures",
]
