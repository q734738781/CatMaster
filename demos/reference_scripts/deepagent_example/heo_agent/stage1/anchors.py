from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Any

from ..core.config import DEFAULT_STAGE1_ANCHOR_ROOT, DEFAULT_STAGE1_ANCHOR_TABLE


def get_s208_anchor_cache_path(anchor_root: str | Path = DEFAULT_STAGE1_ANCHOR_ROOT) -> Path:
    return Path(anchor_root).expanduser().resolve() / "s208_single_dopant_anchor_cache.json"


def get_s208_anchor_table_path(anchor_source: str | Path = DEFAULT_STAGE1_ANCHOR_TABLE) -> Path:
    source = Path(anchor_source).expanduser().resolve()
    if source.is_dir():
        replacement_json = source / "s208_single_dopant_mace_substitution_table.json"
        if replacement_json.exists():
            return replacement_json
        cache_json = source / "s208_single_dopant_anchor_cache.json"
        if cache_json.exists():
            return cache_json
    return source


def _normalize_anchor_payload(payload: dict[str, Any], source_path: Path) -> dict[str, Any]:
    rows = payload.get("rows")
    if not isinstance(rows, list) or not rows:
        raise ValueError(f"Anchor table at {source_path} does not contain any rows.")

    if "constants" in payload:
        constants = {str(key): float(value) for key, value in payload["constants"].items()}
        normalized_rows = []
        for row in rows:
            normalized_rows.append(
                {
                    "element": str(row["element"]),
                    "reference_dir": str(row.get("reference_dir", "")),
                    "reference_family": str(row.get("reference_family", "B1__S208__Na4__dop_*__base")),
                    "substitution_energy_eV_per_dopant": float(row["substitution_energy_eV_per_dopant"]),
                    "source": str(row.get("energy_source", "dft_cache")),
                    "electronic_converged": row.get("electronic_converged"),
                    "ionic_converged": row.get("ionic_converged"),
                }
            )
        return {
            "source_path": str(source_path),
            "source_kind": "legacy_cache",
            "rows": normalized_rows,
            "constants": constants,
        }

    constants = {}
    normalized_rows = []
    for row in rows:
        element = str(row["element"])
        energy = float(row["substitution_energy_eV_per_dopant"])
        constants[element] = energy
        normalized_rows.append(
            {
                "element": element,
                "reference_dir": str(row.get("reference_dir", "")),
                "reference_family": str(row.get("reference_family", "B1__S208__Na4__dop_*__base")),
                "substitution_energy_eV_per_dopant": energy,
                "source": "nfpp_mace_raw",
                "electronic_converged": None,
                "ionic_converged": None,
                "fmax": row.get("fmax"),
                "max_steps": row.get("max_steps"),
                "gpu_id": row.get("gpu_id"),
                "mace_structure_path": row.get("mace_structure_path"),
            }
        )
    return {
        "source_path": str(source_path),
        "source_kind": "mace_replacement_table",
        "rows": normalized_rows,
        "constants": constants,
        "note": payload.get("note"),
        "model_path": payload.get("model_path"),
    }


@lru_cache(maxsize=4)
def load_s208_single_dopant_anchor_table(anchor_source: str | Path = DEFAULT_STAGE1_ANCHOR_TABLE) -> dict[str, Any]:
    table_path = get_s208_anchor_table_path(anchor_source)
    if not table_path.exists():
        raise FileNotFoundError(
            f"Missing S208 anchor JSON: {table_path}. "
            "Point stage1.anchor_table_path to a valid NFPP anchor table or rebuild the legacy cache if needed."
        )
    payload = json.loads(table_path.read_text(encoding="utf-8"))
    return _normalize_anchor_payload(payload, table_path)
