from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from pydantic import BaseModel, Field
from pymatgen.analysis.adsorption import AdsorbateSiteFinder
from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Poscar

from catmaster.tools.base import create_tool_output, resolve_workspace_path, workspace_relpath


class EnumerateAdsorptionSitesInput(BaseModel):
    """
    Enumerate adsorption sites on a slab using ASF. Use it only for small scale placement.

    The tool writes a JSON list to output_json:
    [
      {"label": "ontop_0", "kind": "ontop", "cart_coords": [x, y, z]}
    ]
    The returned tool data includes the workspace-relative path in "sites" and "sites_json_rel".
    """

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF), workspace-relative.")
    mode: str = Field("all", description="Which site families to return: all|ontop|bridge|hollow.")
    distance: float = Field(2.0, ge=0.0, description="Height above surface to sample adsorption sites (Å).")
    output_json: str = Field("adsorption/sites.json", description="Output JSON path for site list (workspace-relative).")


class PlaceAdsorbateInput(BaseModel):
    """Place an adsorbate molecule on a slab. Use it only if for small scale placement.
    This tool keeps selective dynamics of the original slab structure and allows adsorbate to move freely."""

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF).")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file (XYZ).")
    site: str = Field("auto", description="Site label like ontop_0|bridge_1|hollow_2 or 'auto' (which use all[0]).")
    distance: float = Field(2.0, ge=0.0, description="Height used to generate adsorption sites (Å).")
    output_poscar: str = Field("adsorption/adsorbed.vasp", description="Output POSCAR path (workspace-relative).")


class GenerateBatchAdsorptionStructuresInput(BaseModel):
    """
    Generate multiple adsorbed structures up to max_structures. Suitable for large scale placement.
    This tool keeps selective dynamics of the original slab structure and allows adsorbate to move freely.
    Input can be a single slab file (slab_file) or a directory of slab files (slab_dir).
    When slab_dir is used, each slab gets its own subdirectory under output_dir. Max_structures applies per slab.
    For nested slab_dir layouts, slab_id encodes the relative path (without suffix) using '__'.

    The tool writes a JSON list to output_dir/batch_structures.json:
    [
      {
        "slab_file_rel": "slabs/fe111.vasp",
        "slab_id": "fe111",
        "label": "ontop_0",
        "output_poscar_rel": "adsorption/batch/fe111/ontop_0.vasp"
      }
    ]
    The returned tool data replaces "structures" with the workspace-relative path to this JSON file.
    """

    slab_file: Optional[str] = Field(None, description="Slab structure file (POSCAR/CONTCAR/CIF).")
    slab_dir: Optional[str] = Field(None, description="Directory containing slab files for high-throughput placement.")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file (XYZ).")
    mode: str = Field("all", description="all|ontop|bridge|hollow")
    distance: float = Field(2.0, ge=0.0, description="Height used to generate adsorption sites (Å).")
    max_structures: int = Field(1000, ge=1, description="Maximum number of adsorbed structures to generate.")
    output_dir: str = Field("adsorption/batch", description="Directory to write batch POSCARs.")

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


def _collect_slab_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for path in root.rglob("*"):
        if not path.is_file():
            continue
        name = path.name
        if name in {"POSCAR", "CONTCAR"}:
            files.append(path)
            continue
        if path.suffix.lower() in {".vasp", ".poscar", ".cif"}:
            files.append(path)
    return sorted(files, key=lambda p: str(p))


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
                "returned": total_found,
                "total_in_mode": total_found,
                "truncated": False,
            },
            "sites": workspace_relpath(out_json),
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
        ads_path = resolve_workspace_path(params.adsorbate_file, must_exist=True)
        out_dir = resolve_workspace_path(params.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mol = _load_adsorbate_molecule(ads_path)

        slab_paths: List[Path] = []
        slab_root: Optional[Path] = None
        if (params.slab_file is None) == (params.slab_dir is None):
            return create_tool_output(
                "generate_batch_adsorption_structures",
                success=False,
                error="Provide exactly one of slab_file or slab_dir.",
            )
        if params.slab_file is not None:
            slab_paths = [resolve_workspace_path(params.slab_file, must_exist=True)]
        else:
            slab_root = resolve_workspace_path(params.slab_dir, must_exist=True)
            if not slab_root.is_dir():
                return create_tool_output(
                    "generate_batch_adsorption_structures",
                    success=False,
                    error=f"slab_dir is not a directory: {slab_root}",
                )
            slab_paths = _collect_slab_files(slab_root)
            if not slab_paths:
                return create_tool_output(
                    "generate_batch_adsorption_structures",
                    success=False,
                    error="No slab files found in slab_dir.",
                )

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]
        results: List[Dict[str, Any]] = []
        generated_total = 0
        total_candidates = 0
        slabs_processed = 0
        slabs_failed = 0
        errors: List[Dict[str, str]] = []
        truncated_any = False

        for slab_path in slab_paths:
            try:
                slab = Structure.from_file(str(slab_path))
                slab_sd = slab.site_properties.get("selective_dynamics") if slab.site_properties else None

                asf = AdsorbateSiteFinder(slab)
                ads_sites = asf.find_adsorption_sites(distance=float(params.distance))

                if slab_root is None:
                    slab_id = slab_path.stem
                    slab_out_dir = out_dir
                    slab_rel = workspace_relpath(slab_path)
                else:
                    rel_path = slab_path.relative_to(slab_root).with_suffix("")
                    slab_id = "__".join(rel_path.parts)
                    slab_out_dir = out_dir / slab_id
                    slab_rel = workspace_relpath(slab_path)

                slab_out_dir.mkdir(parents=True, exist_ok=True)

                generated = 0
                for kind in kinds:
                    for idx, coord in enumerate(ads_sites.get(kind, [])):
                        if generated >= params.max_structures:
                            break
                        ads_struct = asf.add_adsorbate(mol, coord, translate=True, reorient=True)

                        slab_sd_list = [list(map(bool, v)) for v in slab_sd] if slab_sd and len(slab_sd) == len(slab) else [
                            [True, True, True] for _ in range(len(slab))
                        ]
                        sd_new = slab_sd_list + [[True, True, True] for _ in range(len(mol))]
                        if "selective_dynamics" in ads_struct.site_properties:
                            ads_struct.remove_site_property("selective_dynamics")
                        ads_struct.add_site_property("selective_dynamics", sd_new)

                        file_path = slab_out_dir / f"{kind}_{idx}.vasp"
                        Poscar(ads_struct).write_file(str(file_path))

                        results.append(
                            {
                                "slab_file_rel": slab_rel,
                                "slab_id": slab_id,
                                "label": f"{kind}_{idx}",
                                "output_poscar_rel": workspace_relpath(file_path),
                            }
                        )
                        generated += 1
                    if generated >= params.max_structures:
                        break

                total_candidates_slab = sum(len(ads_sites.get(k, [])) for k in kinds)
                total_candidates += total_candidates_slab
                generated_total += generated
                truncated_any = truncated_any or (generated < total_candidates_slab)
                slabs_processed += 1
            except Exception as exc:
                slabs_failed += 1
                errors.append({"slab_file_rel": workspace_relpath(slab_path), "error": str(exc)})

        if slabs_processed == 0:
            return create_tool_output(
                "generate_batch_adsorption_structures",
                success=False,
                error="No slabs processed successfully.",
            )

        data = {
            "adsorbate_file_rel": workspace_relpath(ads_path),
            "mode": params.mode,
            "distance": float(params.distance),
            "max_structures": params.max_structures,
            "output_dir_rel": workspace_relpath(out_dir),
            "generated": generated_total,
            "total_candidates": total_candidates,
            "truncated": truncated_any,
            "slabs_processed": slabs_processed,
            "slabs_failed": slabs_failed,
        }
        if slab_root is None:
            data["slab_file_rel"] = workspace_relpath(slab_paths[0])
        else:
            data["slab_dir_rel"] = workspace_relpath(slab_root)

        structures_path = out_dir / "batch_structures.json"
        structures_path.write_text(json.dumps(results, indent=2), encoding="utf-8")
        data["structures"] = workspace_relpath(structures_path)
        if errors:
            data["errors"] = errors

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
