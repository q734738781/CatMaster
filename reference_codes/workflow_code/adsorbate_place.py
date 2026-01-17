from typing import Literal, Optional
from pydantic import BaseModel, Field

class EnumerateAdsorptionSitesInput(BaseModel):
    slab_file: str = Field(..., description="Path to slab structure file (e.g., POSCAR/CONTCAR/CIF).")
    mode: Literal["all", "ontop", "bridge", "hollow"] = Field(
        "all",
        description="Which site families to return as labels. 'all' returns ontop/bridge/hollow."
    )
    distance: float = Field(
        2.0,
        ge=0.0,
        description="Distance (Å) above the surface used by ASF find_adsorption_sites(distance=...)."
    )
    output_json: str = Field(
        "adsorption/sites.json",
        description="Output JSON file path for returned site list."
    )

import json
from pathlib import Path
import numpy as np
from typing import Any, Dict, List

from pymatgen.core import Structure
from pymatgen.analysis.adsorption import AdsorbateSiteFinder


_MAX_RETURN_SITES = 1000


def _to_list3(x) -> List[float]:
    arr = np.array(x, dtype=float).reshape(3)
    return [float(arr[0]), float(arr[1]), float(arr[2])]


def enumerate_adsorption_sites(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Enumerate adsorption sites using pymatgen AdsorbateSiteFinder.

    Returns:
      - sites: list[{label, kind, cart_coords}]
      - counts by kind
      - slab basic geometry (natoms, lattice, z_max)
      - writes the sites list to output_json
    """
    try:
        params = EnumerateAdsorptionSitesInput(**payload)
        slab = Structure.from_file(params.slab_file)

        asf = AdsorbateSiteFinder(slab)
        # ASF returns a dict with keys like 'ontop', 'bridge', 'hollow', 'all' (see example usage).
        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]

        site_rows: List[Dict[str, Any]] = []
        for kind in kinds:
            coords_list = ads_sites.get(kind, [])
            for i, c in enumerate(coords_list):
                site_rows.append(
                    {"label": f"{kind}_{i}", "kind": kind, "cart_coords": _to_list3(c)}
                )

        total_found = len(site_rows)
        truncated = False
        if total_found > _MAX_RETURN_SITES:
            site_rows = site_rows[:_MAX_RETURN_SITES]
            truncated = True

        slab_cart = np.array(slab.cart_coords, dtype=float)
        z_max = float(slab_cart[:, 2].max()) if len(slab_cart) else None

        out_path = Path(params.output_json)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        out_path.write_text(json.dumps(site_rows, indent=2), encoding="utf-8")

        # 给 LLM 一个默认推荐 site（减少下一步决策难度）
        default_site = None
        if site_rows:
            # 优先 ontop_0，其次 bridge/hollow
            for pref in ("ontop", "bridge", "hollow"):
                hit = next((s for s in site_rows if s["kind"] == pref), None)
                if hit:
                    default_site = hit["label"]
                    break
            default_site = default_site or site_rows[0]["label"]

        data = {
            "slab_file": params.slab_file,
            "mode": params.mode,
            "distance": float(params.distance),
            "sites_json": str(out_path),
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
        }

        return {"tool_name": "enumerate_adsorption_sites", "success": True, "data": data}

    except Exception as exc:
        return {"tool_name": "enumerate_adsorption_sites", "success": False, "error": str(exc)}

from pydantic import BaseModel, Field

class PlaceAdsorbateInput(BaseModel):
    slab_file: str = Field(..., description="Path to slab structure file (POSCAR/CONTCAR/CIF).")
    adsorbate_file: str = Field(..., description="Path to adsorbate molecule file (XYZ recommended).")
    site: str = Field(
        "auto",
        description="Site label like 'ontop_0', 'bridge_1', 'hollow_2', or 'auto' to pick the first available site."
    )
    distance: float = Field(
        2.0,
        ge=0.0,
        description="Distance (Å) used in ASF find_adsorption_sites(distance=...) to generate the placement coordinate."
    )
    output_poscar: str = Field(
        "adsorption/adsorbed.vasp",
        description="Output POSCAR path for adsorbed slab."
    )

import numpy as np
from pathlib import Path
from typing import Any, Dict, Tuple

from pymatgen.core import Structure, Molecule
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.io.vasp.inputs import Poscar


def _load_adsorbate_molecule(path: str) -> Molecule:
    """
    Load adsorbate as pymatgen Molecule.
    - XYZ: Molecule.from_file works directly (format inferred).
    - VASP POSCAR-like: load as Structure then convert to Molecule by cart coords.
    """
    p = Path(path)
    suf = p.suffix.lower()

    if suf == ".xyz":
        return Molecule.from_file(path)

    if suf in {".vasp", ".poscar", ".contcar"} or p.name.upper() in {"POSCAR", "CONTCAR"}:
        s = Structure.from_file(path)
        return Molecule([str(sp) for sp in s.species], s.cart_coords)

    # fallback: try Molecule.from_file (may work if pymatgen supports the format)
    return Molecule.from_file(path)


def _parse_site(site: str) -> Tuple[str, int]:
    s = (site or "auto").strip().lower()
    if s in {"auto", "first"}:
        return ("auto", 0)

    parts = s.split("_")
    if len(parts) != 2:
        raise ValueError("site must be 'auto' or like 'ontop_0', 'bridge_1', 'hollow_2'")

    kind = parts[0]
    if kind not in {"ontop", "bridge", "hollow"}:
        raise ValueError("site kind must be one of: ontop, bridge, hollow (or 'auto')")

    try:
        idx = int(parts[1])
    except Exception:
        raise ValueError("site index must be an integer, e.g. 'ontop_0'")

    if idx < 0:
        raise ValueError("site index must be >= 0")

    return (kind, idx)

def set_selective_dynamics(struct: Structure, sd: List[List[bool]]):
    """Replace or add 'selective_dynamics' site property with provided flags."""
    if 'selective_dynamics' in struct.site_properties:
        struct.remove_site_property('selective_dynamics')
    struct.add_site_property('selective_dynamics', sd)


def place_adsorbate(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Place an adsorbate molecule on a slab using ASF.

    Key design points for LLM robustness:
    - site can be 'auto' (no need to enumerate first)
    - distance is the only geometry knob exposed
    - returns adsorbed POSCAR path + chosen site info + sanity metrics
    """
    try:
        params = PlaceAdsorbateInput(**payload)

        slab = Structure.from_file(params.slab_file) 
        ## TODO: Need to read selective dynamics from the structure file and treat properly after adsorbate placement.
        mol = _load_adsorbate_molecule(params.adsorbate_file)

        asf = AdsorbateSiteFinder(slab)

        kind, idx = _parse_site(params.site)

        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        if kind == "auto":
            # deterministic preference order
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
                raise RuntimeError("No adsorption sites found (ontop/bridge/hollow empty).")
            site_label = f"{chosen_kind}_{idx}"
            site_coord = np.array(chosen_coord, dtype=float)
        else:
            lst = ads_sites.get(kind, [])
            if idx >= len(lst):
                raise ValueError(f"Requested {kind}_{idx} but only {len(lst)} {kind} sites available.")
            site_label = f"{kind}_{idx}"
            site_coord = np.array(lst[idx], dtype=float)

        # Place adsorbate. translate=True and reorient=True are kept fixed to reduce LLM knobs.
        ads_struct = asf.add_adsorbate(mol, site_coord, translate=True, reorient=True)

        out_path = Path(params.output_poscar)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        Poscar(ads_struct).write_file(str(out_path))

        slab_cart = np.array(slab.cart_coords, dtype=float)
        z_max = float(slab_cart[:, 2].max()) if len(slab_cart) else None

        nat_ads = int(len(mol))

        # adsorbate is appended at the end
        ads_slab_cart = np.array(ads_struct.cart_coords, dtype=float)
        ads_only_part = ads_slab_cart[-nat_ads:, :] if nat_ads > 0 else np.zeros((0, 3))

        ads_com = ads_only_part.mean(axis=0).tolist() if len(ads_only_part) else None

        data = {
            "slab_file": params.slab_file,
            "adsorbate_file": params.adsorbate_file,
            "output_poscar": str(out_path),
            "site": {
                "label": site_label,
                "cart_coords": [float(site_coord[0]), float(site_coord[1]), float(site_coord[2])],
            },
            "geom": {
                "slab_z_max": z_max,
                "adsorbate_com": ads_com,
                "units": {"distance": "Å"},
            },
        }

        return {"tool_name": "place_adsorbate", "success": True, "data": data}

    except Exception as exc:
        return {"tool_name": "place_adsorbate", "success": False, "error": str(exc)}

class GenerateBatchAdsorptionStructuresInput(BaseModel):
    slab_file: str
    adsorbate_file: str
    mode: Literal["all", "ontop", "bridge", "hollow"] = "all"
    distance: float = 2.0
    max_structures: int = Field(12, ge=1, le=100)   # 核心：强限额
    output_dir: str = "adsorption/batch"

def generate_batch_adsorption_structures(payload: Dict[str, Any]) -> Dict[str, Any]:
    """
    Generate multiple adsorbed structures (one per adsorption site) up to max_structures with the sequence of ontop, bridge, hollow.
    """
    try:
        params = GenerateBatchAdsorptionStructuresInput(**payload)

        slab = Structure.from_file(params.slab_file)
        mol = _load_adsorbate_molecule(params.adsorbate_file)

        asf = AdsorbateSiteFinder(slab)
        ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]

        out_dir = Path(params.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        results: List[Dict[str, Any]] = []
        generated = 0

        for kind in kinds:
            coords = ads_sites.get(kind, [])
            for idx, coord in enumerate(coords):
                if generated >= params.max_structures:
                    break

                ads_struct = asf.add_adsorbate(mol, coord, translate=True, reorient=True)
                file_path = out_dir / f"{kind}_{idx}.vasp"
                Poscar(ads_struct).write_file(str(file_path))

                results.append(
                    {
                        "label": f"{kind}_{idx}",
                        "output_poscar": str(file_path),
                    }
                )
                generated += 1

            if generated >= params.max_structures:
                break

        total_candidates = sum(len(ads_sites.get(k, [])) for k in kinds)

        if generated == 0:
            raise RuntimeError(f"No adsorption sites found for mode='{params.mode}'.")

        data = {
            "slab_file": params.slab_file,
            "adsorbate_file": params.adsorbate_file,
            "mode": params.mode,
            "distance": float(params.distance),
            "max_structures": params.max_structures,
            "output_dir": str(out_dir),
            "generated": generated,
            "total_candidates": total_candidates,
            "truncated": generated < total_candidates,
            "structures": results,
        }

        return {"tool_name": "generate_batch_adsorption_structures", "success": True, "data": data}

    except Exception as exc:
        return {"tool_name": "generate_batch_adsorption_structures", "success": False, "error": str(exc)}
