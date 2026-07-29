from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np
from pydantic import BaseModel, Field, model_validator
from pymatgen.core import Structure, Molecule
from pymatgen.io.vasp.inputs import Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.structures.adsorption import (
    enumerate_adsorption_sites as enumerate_adsorption_sites_domain,
    place_adsorbate_at_site,
)
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath

ADS_META_SCHEMA = "catmaster.adsorbate_meta.v1"
ADS_INDICES_SCHEMA = "catmaster.ads_indices.v1"


def _success(
    tool_name: str,
    *,
    content: str,
    data: dict[str, Any],
    warnings: list[str] | None = None,
) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    return content, artifact


def _fail(
    tool_name: str,
    *,
    message: str,
    data: dict[str, Any] | None = None,
    error_code: str = "",
) -> None:
    details: list[str] = [str(message).strip()]
    if isinstance(data, dict):
        for key in (
            "slab_file_rel",
            "slab_dir_rel",
            "output_poscar_rel",
            "output_dir_rel",
            "sites_json_rel",
            "structures",
            "ads_indices_json_rel",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            details.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(details),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


class EnumerateAdsorptionSitesInput(BaseModel):
    """
    [adsorption/modeling] Enumerate adsorption sites on a slab using ASF. Use it only for small-scale placement.
    The returned site list is the deduplicated representative-site result from ASF, not a symmetry-expanded list of every equivalent site.

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
    """[adsorption/modeling] Place an adsorbate molecule on a slab. Use it only for small-scale placement.
    This tool keeps selective dynamics of the original slab structure and allows adsorbate to move freely.
    It writes metadata sidecar <output>.meta.json and an ads_indices index JSON."""

    slab_file: str = Field(..., description="Slab structure file (POSCAR/CONTCAR/CIF).")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file (XYZ). Internal relative geometry is preserved; no automatic reorientation is applied.")
    site_label: Optional[str] = Field(
        None,
        description=(
            "Optional ASF site label like ontop_0|bridge_1|hollow_2 or 'auto'. "
            "These labels are the ones returned by enumerate_adsorption_sites."
        ),
    )
    site: Optional[str] = Field(
        None,
        description="Deprecated alias for site_label. Prefer site_label in new calls.",
    )
    site_cart_coords: Optional[List[float]] = Field(
        None,
        description=(
            "Optional Cartesian [x, y, z] in Angstrom. If provided, the tool places the adsorbate directly at "
            "that Cartesian site coordinate. "
            "This is mutually exclusive with site_label."
        ),
    )
    distance: float = Field(2.0, ge=0.0, description="Height above the surface used when enumerating adsorption site coordinates (Å). The molecule is translated so its placement reference point lands on that site coordinate.")
    output_poscar: str = Field("adsorption/adsorbed.vasp", description="Output POSCAR path (workspace-relative).")

    @model_validator(mode="after")
    def _validate_site_input(self) -> "PlaceAdsorbateInput":
        if self.site is not None and self.site_label is not None:
            raise ValueError("Provide only one of site_label or legacy site.")
        if self.site_label is None and self.site is not None:
            self.site_label = self.site
        if self.site_cart_coords is not None and self.site_label is not None:
            raise ValueError("Provide exactly one of site_label or site_cart_coords.")
        if self.site_cart_coords is None and self.site_label is None:
            self.site_label = "auto"
        if self.site_cart_coords is not None:
            if len(self.site_cart_coords) != 3:
                raise ValueError("site_cart_coords must be a 3-element Cartesian [x, y, z] list in Angstrom.")
            self.site_cart_coords = [float(v) for v in self.site_cart_coords]
        return self


class GenerateBatchAdsorptionStructuresInput(BaseModel):
    """[adsorption/modeling] Generate a capped batch of adsorbed structures from one slab file or a slab directory."""

    slab_file: Optional[str] = Field(None, description="Slab structure file (POSCAR/CONTCAR/CIF).")
    slab_dir: Optional[str] = Field(None, description="Directory containing slab files for high-throughput placement.")
    adsorbate_file: str = Field(..., description="Adsorbate molecule file, usually XYZ; geometry is translated without automatic reorientation.")
    mode: str = Field("all", description="all|ontop|bridge|hollow")
    distance: float = Field(2.0, ge=0.0, description="Height above the surface used for site coordinates (Å).")
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


def _choose_site(
    *,
    ads_sites: Dict[str, Any],
    site_label: str,
    site_cart_coords: Optional[List[float]],
) -> tuple[str, np.ndarray, str, Optional[List[float]]]:
    if site_cart_coords is not None:
        coord = np.array(site_cart_coords, dtype=float).reshape(3)
        return "", coord, "cart_direct", [float(v) for v in coord.tolist()]

    kind, idx = _parse_site(site_label)
    if kind == "auto":
        for pref in ("ontop", "bridge", "hollow"):
            lst = ads_sites.get(pref, [])
            if lst:
                return f"{pref}_0", np.array(lst[0], dtype=float), "auto", None
        raise RuntimeError("No adsorption sites found.")

    lst = ads_sites.get(kind, [])
    if idx >= len(lst):
        raise ValueError(f"Requested {kind}_{idx} but only {len(lst)} {kind} sites available.")
    return f"{kind}_{idx}", np.array(lst[idx], dtype=float), "label", None


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


def _meta_sidecar_path(structure_path: Path) -> Path:
    try:
        return structure_path.with_suffix(".meta.json")
    except ValueError:
        return Path(f"{structure_path}.meta.json")


def _meta_path_candidates(structure_path: Path) -> List[Path]:
    primary = _meta_sidecar_path(structure_path)
    alt = Path(f"{structure_path}.meta.json")
    if alt == primary:
        return [primary]
    return [primary, alt]


def _normalize_indices(raw: Any) -> List[int]:
    if not isinstance(raw, list):
        return []
    values: List[int] = []
    for item in raw:
        try:
            idx = int(item)
        except Exception:
            continue
        if idx < 0:
            continue
        values.append(idx)
    return sorted(set(values))


def _merge_indices(existing: List[int], added: List[int]) -> List[int]:
    return sorted(set(_normalize_indices(existing) + _normalize_indices(added)))


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_indices_from_meta(slab_path: Path, warnings: List[str]) -> List[int]:
    for meta_path in _meta_path_candidates(slab_path):
        if not meta_path.exists():
            continue
        try:
            data = _load_json(meta_path)
        except Exception as exc:
            warnings.append(f"Failed to parse metadata {workspace_relpath(meta_path)}: {exc}")
            continue
        if not isinstance(data, dict):
            warnings.append(f"Ignoring non-object metadata file: {workspace_relpath(meta_path)}")
            continue
        indices = _normalize_indices(data.get("ads_indices"))
        if "ads_indices" in data and not indices:
            warnings.append(f"ads_indices in {workspace_relpath(meta_path)} is empty/invalid; ignored.")
        if indices:
            return indices
    return []


def _entry_matches_structure(entry: Dict[str, Any], slab_path: Path, slab_rel: str) -> bool:
    candidates = []
    for key in ("output_poscar_rel", "output_structure_rel", "structure_rel", "path_rel"):
        val = entry.get(key)
        if isinstance(val, str):
            candidates.append(val)
    if any(val == slab_rel for val in candidates):
        return True
    return any(Path(val).name == slab_path.name for val in candidates)


def _load_indices_from_ads_json(slab_path: Path, warnings: List[str]) -> List[int]:
    ads_json = slab_path.parent / "ads_indices.json"
    if not ads_json.exists():
        return []
    try:
        data = _load_json(ads_json)
    except Exception as exc:
        warnings.append(f"Failed to parse ads_indices index {workspace_relpath(ads_json)}: {exc}")
        return []

    slab_rel = workspace_relpath(slab_path)
    entries: List[Dict[str, Any]] = []
    if isinstance(data, dict) and isinstance(data.get("entries"), list):
        entries = [e for e in data["entries"] if isinstance(e, dict)]
    elif isinstance(data, dict) and ("output_poscar_rel" in data or "ads_indices" in data):
        entries = [data]
    elif isinstance(data, list):
        entries = [e for e in data if isinstance(e, dict)]
    else:
        warnings.append(f"Ignoring unsupported ads_indices index schema: {workspace_relpath(ads_json)}")
        return []

    merged: List[int] = []
    for ent in entries:
        if _entry_matches_structure(ent, slab_path, slab_rel):
            merged.extend(_normalize_indices(ent.get("ads_indices")))
    return sorted(set(merged))


def _load_inherited_ads_indices(slab_path: Path) -> Tuple[List[int], List[str]]:
    warnings: List[str] = []
    from_meta = _load_indices_from_meta(slab_path, warnings)
    from_index = _load_indices_from_ads_json(slab_path, warnings)
    return _merge_indices(from_meta, from_index), warnings


def _load_existing_ads_meta(structure_path: Path, warnings: List[str]) -> Dict[str, Any]:
    for meta_path in _meta_path_candidates(structure_path):
        if not meta_path.exists():
            continue
        try:
            raw = _load_json(meta_path)
        except Exception as exc:
            warnings.append(f"Failed to parse metadata {workspace_relpath(meta_path)}: {exc}")
            continue
        if isinstance(raw, dict):
            return raw
        warnings.append(f"Ignoring non-object metadata file: {workspace_relpath(meta_path)}")
    return {}


def propagate_adsorbate_metadata(
    *,
    input_structure_path: Path,
    output_structure_path: Path,
    tool_name: str,
) -> tuple[Dict[str, Any] | None, List[str]]:
    warnings: List[str] = []
    ads_indices, inherited_warnings = _load_inherited_ads_indices(input_structure_path)
    warnings.extend(inherited_warnings)
    if not ads_indices:
        return None, warnings

    existing_meta = _load_existing_ads_meta(input_structure_path, warnings)
    ads_indices_added = _normalize_indices(existing_meta.get("ads_indices_added"))
    meta_payload: Dict[str, Any] = {
        "schema": ADS_META_SCHEMA,
        "source_tool": tool_name,
        "parent_structure_rel": workspace_relpath(input_structure_path),
        "output_structure_rel": workspace_relpath(output_structure_path),
        "ads_indices": ads_indices,
        "ads_count_total": len(ads_indices),
        "propagated_from_rel": workspace_relpath(input_structure_path),
    }
    if ads_indices_added:
        meta_payload["ads_indices_added"] = ads_indices_added
        meta_payload["ads_count_added"] = len(ads_indices_added)
    for key in ("adsorbate_file_rel", "site_label", "site_cart_coords", "site_input_mode"):
        value = existing_meta.get(key)
        if value not in (None, "", [], {}):
            meta_payload[key] = value

    meta_path = _meta_sidecar_path(output_structure_path)
    meta_path.write_text(json.dumps(meta_payload, indent=2, ensure_ascii=False), encoding="utf-8")
    entry: Dict[str, Any] = {
        "output_poscar_rel": workspace_relpath(output_structure_path),
        "metadata_rel": workspace_relpath(meta_path),
        "ads_indices": ads_indices,
        "ads_count_total": len(ads_indices),
    }
    if ads_indices_added:
        entry["ads_indices_added"] = ads_indices_added
        entry["ads_count_added"] = len(ads_indices_added)
    ads_indices_json = _write_ads_indices_index(output_structure_path.parent / "ads_indices.json", [entry])
    return {
        "ads_indices": ads_indices,
        "metadata_rel": workspace_relpath(meta_path),
        "ads_indices_json_rel": workspace_relpath(ads_indices_json),
    }, warnings


def _write_adsorbate_meta(
    *,
    output_structure_path: Path,
    parent_structure_path: Path,
    adsorbate_path: Path,
    tool_name: str,
    site_label: str | None,
    site_input_mode: str,
    site_cart_coords_input: List[float] | None,
    site_cart_coords: List[float],
    distance: float,
    ads_indices_added: List[int],
    ads_indices: List[int],
) -> Path:
    meta_path = _meta_sidecar_path(output_structure_path)
    existing: Dict[str, Any] = {}
    if meta_path.exists():
        try:
            raw = _load_json(meta_path)
            if isinstance(raw, dict):
                existing = raw
        except Exception:
            existing = {}

    existing.update(
        {
            "schema": ADS_META_SCHEMA,
            "source_tool": tool_name,
            "parent_structure_rel": workspace_relpath(parent_structure_path),
            "output_structure_rel": workspace_relpath(output_structure_path),
            "adsorbate_file_rel": workspace_relpath(adsorbate_path),
            "site_label": site_label,
            "site_input_mode": site_input_mode,
            "site_cart_coords_input": [float(v) for v in site_cart_coords_input] if site_cart_coords_input is not None else None,
            "site_cart_coords": [float(v) for v in site_cart_coords],
            "distance": float(distance),
            "ads_indices_added": _normalize_indices(ads_indices_added),
            "ads_indices": _normalize_indices(ads_indices),
            "ads_count_added": len(_normalize_indices(ads_indices_added)),
            "ads_count_total": len(_normalize_indices(ads_indices)),
        }
    )
    meta_path.write_text(json.dumps(existing, indent=2, ensure_ascii=False), encoding="utf-8")
    return meta_path


def _write_ads_indices_index(index_path: Path, entries: List[Dict[str, Any]]) -> Path:
    current_entries: List[Dict[str, Any]] = []
    if index_path.exists():
        try:
            raw = _load_json(index_path)
            if isinstance(raw, dict) and isinstance(raw.get("entries"), list):
                current_entries = [e for e in raw["entries"] if isinstance(e, dict)]
            elif isinstance(raw, list):
                current_entries = [e for e in raw if isinstance(e, dict)]
            elif isinstance(raw, dict) and "output_poscar_rel" in raw:
                current_entries = [raw]
        except Exception:
            current_entries = []

    merged: Dict[str, Dict[str, Any]] = {}
    for ent in current_entries + entries:
        key = str(ent.get("output_poscar_rel") or ent.get("output_structure_rel") or "")
        if not key:
            continue
        merged[key] = ent

    payload = {
        "schema": ADS_INDICES_SCHEMA,
        "entries": [merged[key] for key in sorted(merged.keys())],
    }
    index_path.write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return index_path


def enumerate_adsorption_sites(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[adsorption/modeling] Enumerate deduplicated representative ASF adsorption sites on a slab and persist them as JSON."""
    try:
        params = EnumerateAdsorptionSitesInput(**payload)
        slab_path = resolve_workspace_path(params.slab_file, must_exist=True)
        out_json = resolve_workspace_path(params.output_json)
        out_json.parent.mkdir(parents=True, exist_ok=True)

        slab = Structure.from_file(str(slab_path))
        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]
        site_rows = enumerate_adsorption_sites_domain(
            slab,
            distance=float(params.distance),
            site_kinds=kinds,
        )

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
                "ontop": sum(row["kind"] == "ontop" for row in site_rows),
                "bridge": sum(row["kind"] == "bridge" for row in site_rows),
                "hollow": sum(row["kind"] == "hollow" for row in site_rows),
                "returned": total_found,
                "total_in_mode": total_found,
                "truncated": False,
            },
            "sites": workspace_relpath(out_json),
            "slab_z_max": z_max,
        }
        lines = [
            "enumerate_adsorption_sites completed.",
            f"returned={total_found} mode={params.mode}",
            f"sites_json_rel={data['sites_json_rel']}",
            f"slab_file_rel={data['slab_file_rel']}",
        ]
        if default_site:
            lines[1] = f"{lines[1]} default_site_label={default_site}"
        content = "\n".join(lines)
        return _success("enumerate_adsorption_sites", content=content, data=data)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail(
            "enumerate_adsorption_sites",
            message=f"enumerate_adsorption_sites failed: {exc}",
            error_code="enumerate_failed",
        )


def place_adsorbate(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[adsorption/modeling] Place one adsorbate on one slab and preserve inherited selective dynamics."""
    try:
        params = PlaceAdsorbateInput(**payload)
        slab_path = resolve_workspace_path(params.slab_file, must_exist=True)
        ads_path = resolve_workspace_path(params.adsorbate_file, must_exist=True)
        out_path = resolve_workspace_path(params.output_poscar)
        out_path.parent.mkdir(parents=True, exist_ok=True)

        slab = Structure.from_file(str(slab_path))
        mol = _load_adsorbate_molecule(ads_path)
        site_rows = enumerate_adsorption_sites_domain(
            slab,
            distance=float(params.distance),
            site_kinds=["ontop", "bridge", "hollow"],
        )
        ads_sites = {
            kind: [
                np.asarray(row["cart_coords"], dtype=float)
                for row in site_rows
                if row["kind"] == kind
            ]
            for kind in ("ontop", "bridge", "hollow")
        }

        site_label, site_coord, selection_mode, requested_site_cart_coords = _choose_site(
            ads_sites=ads_sites,
            site_label=str(params.site_label),
            site_cart_coords=params.site_cart_coords,
        )

        ads_struct = place_adsorbate_at_site(
            slab,
            mol,
            site_coord,
            reorient=False,
        )

        Poscar(ads_struct).write_file(str(out_path))

        nat_ads = len(mol)
        ads_indices_added = list(range(len(slab), len(slab) + nat_ads))
        inherited_indices, inherit_warnings = _load_inherited_ads_indices(slab_path)
        ads_indices = _merge_indices(inherited_indices, ads_indices_added)
        meta_path = _write_adsorbate_meta(
            output_structure_path=out_path,
            parent_structure_path=slab_path,
            adsorbate_path=ads_path,
            tool_name="place_adsorbate",
            site_label=site_label,
            site_input_mode=selection_mode,
            site_cart_coords_input=requested_site_cart_coords,
            site_cart_coords=_to_list3(site_coord),
            distance=float(params.distance),
            ads_indices_added=ads_indices_added,
            ads_indices=ads_indices,
        )
        index_entry = {
            "slab_file_rel": workspace_relpath(slab_path),
            "label": site_label,
            "output_poscar_rel": workspace_relpath(out_path),
            "metadata_rel": workspace_relpath(meta_path),
            "ads_indices_added": ads_indices_added,
            "ads_indices": ads_indices,
            "ads_count_added": len(ads_indices_added),
            "ads_count_total": len(ads_indices),
        }
        ads_indices_json = _write_ads_indices_index(out_path.parent / "ads_indices.json", [index_entry])

        ads_only_part = np.array(ads_struct.cart_coords[-nat_ads:, :], dtype=float) if nat_ads else np.zeros((0, 3))
        ads_com = ads_only_part.mean(axis=0).tolist() if len(ads_only_part) else None

        data = {
            "slab_file_rel": workspace_relpath(slab_path),
            "adsorbate_file_rel": workspace_relpath(ads_path),
            "output_poscar_rel": workspace_relpath(out_path),
            "site": {
                "label": site_label or None,
                "cart_coords": _to_list3(site_coord),
                "selection_mode": selection_mode,
                "requested_site_cart_coords": requested_site_cart_coords,
                "label_source": "enumerate_adsorption_sites" if site_label else None,
            },
            "geom": {
                "adsorbate_com": ads_com,
                "placement_reference": "center_of_mass_of_lowest_z_atoms",
                "site_height_distance": float(params.distance),
                "reoriented": False,
                "units": {"distance": "Å"},
            },
            "natoms": len(ads_struct),
            "ads_indices_added": ads_indices_added,
            "ads_indices": ads_indices,
            "ads_count_added": len(ads_indices_added),
            "ads_count_total": len(ads_indices),
            "metadata_rel": workspace_relpath(meta_path),
            "ads_indices_json_rel": workspace_relpath(ads_indices_json),
        }
        content = (
            "place_adsorbate completed.\n"
            f"site_label={site_label or 'none'} selection_mode={selection_mode} output_poscar_rel={data['output_poscar_rel']}\n"
            "placement_reference=center_of_mass_of_lowest_z_atoms reoriented=False\n"
            f"ads_count_added={data['ads_count_added']} ads_count_total={data['ads_count_total']}\n"
            f"metadata_rel={data['metadata_rel']} ads_indices_json_rel={data['ads_indices_json_rel']}"
        )
        return _success("place_adsorbate", content=content, data=data, warnings=inherit_warnings)
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail(
            "place_adsorbate",
            message=f"place_adsorbate failed: {exc}",
            error_code="place_adsorbate_failed",
        )


def generate_batch_adsorption_structures(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[adsorption/modeling] Convenience wrapper to generate many adsorbed structures from ASF site candidates."""
    try:
        params = GenerateBatchAdsorptionStructuresInput(**payload)
        ads_path = resolve_workspace_path(params.adsorbate_file, must_exist=True)
        out_dir = resolve_workspace_path(params.output_dir)
        out_dir.mkdir(parents=True, exist_ok=True)

        mol = _load_adsorbate_molecule(ads_path)

        slab_paths: List[Path] = []
        slab_root: Optional[Path] = None
        if (params.slab_file is None) == (params.slab_dir is None):
            _fail(
                "generate_batch_adsorption_structures",
                message="Provide exactly one of slab_file or slab_dir.",
                error_code="invalid_input_mode",
            )
        if params.slab_file is not None:
            slab_paths = [resolve_workspace_path(params.slab_file, must_exist=True)]
        else:
            slab_root = resolve_workspace_path(params.slab_dir, must_exist=True)
            if not slab_root.is_dir():
                _fail(
                    "generate_batch_adsorption_structures",
                    message=f"slab_dir is not a directory: {slab_root}",
                    data={"slab_dir_rel": workspace_relpath(slab_root)},
                    error_code="invalid_slab_dir",
                )
            slab_paths = _collect_slab_files(slab_root)
            if not slab_paths:
                _fail(
                    "generate_batch_adsorption_structures",
                    message="No slab files found in slab_dir.",
                    data={"slab_dir_rel": workspace_relpath(slab_root)},
                    error_code="no_slab_files",
                )

        kinds = ["ontop", "bridge", "hollow"] if params.mode == "all" else [params.mode]
        results: List[Dict[str, Any]] = []
        generated_total = 0
        total_candidates = 0
        slabs_processed = 0
        slabs_failed = 0
        errors: List[Dict[str, str]] = []
        truncated_any = False
        warnings: List[str] = []
        ads_index_entries: List[Dict[str, Any]] = []

        for slab_path in slab_paths:
            try:
                slab = Structure.from_file(str(slab_path))
                inherited_indices, inherit_warnings = _load_inherited_ads_indices(slab_path)
                warnings.extend([f"{workspace_relpath(slab_path)}: {msg}" for msg in inherit_warnings])

                site_rows = enumerate_adsorption_sites_domain(
                    slab,
                    distance=float(params.distance),
                    site_kinds=kinds,
                )

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
                for site_row in site_rows[: params.max_structures]:
                    site_label = str(site_row["label"])
                    coord = site_row["cart_coords"]
                    ads_struct = place_adsorbate_at_site(
                        slab,
                        mol,
                        coord,
                        reorient=False,
                    )

                    file_path = slab_out_dir / f"{site_label}.vasp"
                    Poscar(ads_struct).write_file(str(file_path))
                    ads_indices_added = list(range(len(slab), len(slab) + len(mol)))
                    ads_indices = _merge_indices(inherited_indices, ads_indices_added)
                    meta_path = _write_adsorbate_meta(
                        output_structure_path=file_path,
                        parent_structure_path=slab_path,
                        adsorbate_path=ads_path,
                        tool_name="generate_batch_adsorption_structures",
                        site_label=site_label,
                        site_input_mode="label",
                        site_cart_coords_input=None,
                        site_cart_coords=_to_list3(coord),
                        distance=float(params.distance),
                        ads_indices_added=ads_indices_added,
                        ads_indices=ads_indices,
                    )

                    result_row = {
                        "slab_file_rel": slab_rel,
                        "slab_id": slab_id,
                        "label": site_label,
                        "output_poscar_rel": workspace_relpath(file_path),
                        "ads_indices_added": ads_indices_added,
                        "ads_indices": ads_indices,
                        "ads_count_added": len(ads_indices_added),
                        "ads_count_total": len(ads_indices),
                        "metadata_rel": workspace_relpath(meta_path),
                    }
                    results.append(result_row)
                    ads_index_entries.append(dict(result_row))
                    generated += 1

                total_candidates_slab = len(site_rows)
                total_candidates += total_candidates_slab
                generated_total += generated
                truncated_any = truncated_any or (generated < total_candidates_slab)
                slabs_processed += 1
            except Exception as exc:
                slabs_failed += 1
                errors.append({"slab_file_rel": workspace_relpath(slab_path), "error": str(exc)})

        errors_path: Path | None = None
        if errors:
            errors_path = out_dir / "batch_errors.json"
            try:
                errors_path.write_text(json.dumps(errors, indent=2, ensure_ascii=False), encoding="utf-8")
            except Exception:
                errors_path = None

        if slabs_processed == 0:
            _fail(
                "generate_batch_adsorption_structures",
                message="No slabs processed successfully.",
                data={
                    "output_dir_rel": workspace_relpath(out_dir),
                    **compact_list_for_artifact(
                        errors,
                        count_key="errors_count",
                        inline_key="errors",
                        preview_key="errors_preview",
                        truncated_key="errors_truncated",
                        full_rel_key="errors_full_rel",
                        full_rel=workspace_relpath(errors_path) if errors_path else None,
                    ),
                },
                error_code="all_slabs_failed",
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
        ads_indices_json = _write_ads_indices_index(out_dir / "ads_indices.json", ads_index_entries)
        data["structures"] = workspace_relpath(structures_path)
        data["ads_indices_json_rel"] = workspace_relpath(ads_indices_json)
        data["metadata_files_count"] = len(ads_index_entries)
        data["ads_indices_summary"] = {
            "structures": len(ads_index_entries),
            "ads_atoms_added_total": int(sum(len(ent.get("ads_indices_added", [])) for ent in ads_index_entries)),
            "ads_atoms_total_sum": int(sum(len(ent.get("ads_indices", [])) for ent in ads_index_entries)),
        }
        if errors:
            data.update(
                compact_list_for_artifact(
                    errors,
                    count_key="errors_count",
                    inline_key="errors",
                    preview_key="errors_preview",
                    truncated_key="errors_truncated",
                    full_rel_key="errors_full_rel",
                    full_rel=workspace_relpath(errors_path) if errors_path else None,
                )
            )

        content = (
            "generate_batch_adsorption_structures completed.\n"
            f"generated={generated_total} slabs_processed={slabs_processed} slabs_failed={slabs_failed}\n"
            f"output_dir_rel={data['output_dir_rel']} structures={data['structures']}\n"
            f"ads_indices_json_rel={data['ads_indices_json_rel']} truncated={data['truncated']}"
        )
        return _success(
            "generate_batch_adsorption_structures",
            content=content,
            data=data,
            warnings=warnings,
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail(
            "generate_batch_adsorption_structures",
            message=f"generate_batch_adsorption_structures failed: {exc}",
            error_code="batch_adsorption_failed",
        )


__all__ = [
    "EnumerateAdsorptionSitesInput",
    "PlaceAdsorbateInput",
    "GenerateBatchAdsorptionStructuresInput",
    "enumerate_adsorption_sites",
    "place_adsorbate",
    "generate_batch_adsorption_structures",
]
