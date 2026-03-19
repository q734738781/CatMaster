from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any, Literal

import networkx as nx
from pydantic import BaseModel, Field, field_validator, model_validator
from pymatgen.analysis.graphs import StructureGraph
from pymatgen.analysis.local_env import CrystalNN, JmolNN, MinimumDistanceNN
from pymatgen.core import Composition, Structure

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.geometry_inputs.adsorbate_tool import _load_inherited_ads_indices


class IdentifyStructureFragmentsInput(BaseModel):
    """[structure/analysis] Probe connected fragments in a periodic structure and return fragment-index summaries."""

    structure_path: str = Field(..., description="Workspace-relative structure path to analyze.")
    strategy: Literal["jmolnn", "crystalnn", "mindistance"] = Field(
        "jmolnn",
        description=(
            "Connectivity heuristic. jmolnn is the default probe mode and is often useful for weakly bound "
            "molecular fragments on slabs; crystalnn and mindistance are available for comparison."
        ),
    )
    output_json: str | None = Field(
        None,
        description="Optional workspace-relative JSON output path. Defaults to analysis/<stem>_fragments.json.",
    )
    reference_indices: list[int] = Field(
        default_factory=list,
        description=(
            "Optional zero-based reference atom indices, typically an existing ads_indices list. "
            "The tool annotates fragment overlap against these indices."
        ),
    )
    load_ads_indices: bool = Field(
        True,
        description=(
            "If true, also load inherited ads_indices metadata/sidecar information next to structure_path and "
            "merge it with any explicit reference_indices."
        ),
    )
    select_reference_overlap: bool = Field(
        False,
        description="If true, mark fragments with nonzero overlap against the resolved reference_indices as selected.",
    )
    select_ids: list[int] | None = Field(
        None,
        description="Optional fragment IDs to select after probing.",
    )
    match_natoms: int | None = Field(
        None,
        ge=1,
        description="Optional exact atom-count selector for fragment selection.",
    )
    match_formula: str | None = Field(
        None,
        description=(
            "Optional exact full-composition selector, e.g. CO or C2H4O. "
            "This is an exact formula match, not a reduced-formula match."
        ),
    )

    @field_validator("structure_path")
    @classmethod
    def _validate_structure_path(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("structure_path must not be empty")
        return text

    @field_validator("reference_indices")
    @classmethod
    def _validate_reference_indices(cls, value: list[int]) -> list[int]:
        return _normalize_indices(value)

    @field_validator("select_ids")
    @classmethod
    def _validate_select_ids(cls, value: list[int] | None) -> list[int] | None:
        if value is None:
            return None
        cleaned = sorted({int(v) for v in value if int(v) >= 1})
        return cleaned or None

    @model_validator(mode="after")
    def _validate_selection(self) -> "IdentifyStructureFragmentsInput":
        if self.select_reference_overlap and not self.load_ads_indices and not self.reference_indices:
            raise ValueError(
                "select_reference_overlap=True requires explicit reference_indices or load_ads_indices=True"
            )
        return self


def _normalize_indices(raw: Any) -> list[int]:
    if not isinstance(raw, list):
        return []
    out: list[int] = []
    for item in raw:
        try:
            idx = int(item)
        except Exception:
            continue
        if idx < 0:
            continue
        out.append(idx)
    return sorted(set(out))


def _default_output_json(structure_path: Path) -> Path:
    return resolve_workspace_path(f"analysis/{structure_path.stem}_fragments.json")


def _build_structure_graph(structure: Structure, strategy: str) -> StructureGraph:
    if strategy == "jmolnn":
        nn_strategy = JmolNN()
    elif strategy == "crystalnn":
        nn_strategy = CrystalNN()
    elif strategy == "mindistance":
        nn_strategy = MinimumDistanceNN()
    else:  # pragma: no cover - guarded by pydantic
        raise ValueError(f"Unsupported strategy: {strategy}")
    return StructureGraph.from_local_env_strategy(structure, nn_strategy)


def _hill_formula(comp: Composition) -> str:
    return comp.hill_formula.replace(" ", "")


def _normalize_formula(formula: str) -> tuple[tuple[str, int], ...]:
    parsed = Composition(formula)
    amounts: list[tuple[str, int]] = []
    for el, amt in parsed.get_el_amt_dict().items():
        rounded = round(amt)
        if abs(float(amt) - float(rounded)) > 1e-8:
            raise ValueError(f"Formula contains a non-integer amount for {el}: {amt}")
        amounts.append((el, int(rounded)))
    return tuple(sorted(amounts))


def _extract_fragments(
    structure: Structure,
    *,
    strategy: str,
    reference_indices: list[int],
) -> list[dict[str, Any]]:
    sg = _build_structure_graph(structure, strategy)
    graph = sg.graph.to_undirected()
    reference_set = set(reference_indices)
    rows: list[dict[str, Any]] = []

    for frag_id, component in enumerate(
        sorted(nx.connected_components(graph), key=len, reverse=True),
        start=1,
    ):
        indices = tuple(sorted(int(idx) for idx in component))
        counts = Counter(str(structure[idx].specie) for idx in indices)
        comp = Composition(counts)
        overlap = sorted(reference_set.intersection(indices))
        rows.append(
            {
                "frag_id": frag_id,
                "indices": list(indices),
                "indices_1based": [idx + 1 for idx in indices],
                "natoms": len(indices),
                "formula": _hill_formula(comp),
                "reduced_formula": comp.reduced_formula.replace(" ", ""),
                "reference_overlap_indices": overlap,
                "reference_overlap_indices_1based": [idx + 1 for idx in overlap],
                "reference_overlap_count": len(overlap),
                "contains_reference_indices": bool(overlap),
            }
        )
    return rows


def _select_fragments(
    fragments: list[dict[str, Any]],
    *,
    select_ids: list[int] | None,
    match_natoms: int | None,
    match_formula: str | None,
    select_reference_overlap: bool,
) -> list[dict[str, Any]]:
    formula_key = _normalize_formula(match_formula) if match_formula else None
    selected: list[dict[str, Any]] = []
    requested = bool(select_ids or match_natoms is not None or formula_key is not None or select_reference_overlap)

    for frag in fragments:
        if select_ids and int(frag["frag_id"]) not in select_ids:
            continue
        if match_natoms is not None and int(frag["natoms"]) != int(match_natoms):
            continue
        if formula_key is not None and _normalize_formula(str(frag["formula"])) != formula_key:
            continue
        if select_reference_overlap and int(frag["reference_overlap_count"]) <= 0:
            continue
        if requested:
            selected.append(frag)
    return selected


def identify_structure_fragments(payload: dict[str, object]) -> tuple[str, dict[str, Any]]:
    """[structure/analysis] Probe connected fragments in a structure and return fragment/index summaries."""
    tool_name = "identify_structure_fragments"
    try:
        params = IdentifyStructureFragmentsInput(**payload)
        structure_path = resolve_workspace_path(params.structure_path, must_exist=True)
        output_json = (
            resolve_workspace_path(params.output_json)
            if params.output_json
            else _default_output_json(structure_path)
        )
        output_json.parent.mkdir(parents=True, exist_ok=True)

        structure = Structure.from_file(str(structure_path))
        sidecar_indices: list[int] = []
        warnings: list[str] = []
        if params.load_ads_indices:
            sidecar_indices, warnings = _load_inherited_ads_indices(structure_path)
        reference_indices = sorted(set(params.reference_indices + sidecar_indices))
        if params.reference_indices and sidecar_indices:
            reference_source = "explicit+ads_indices_sidecar"
        elif params.reference_indices:
            reference_source = "explicit"
        elif sidecar_indices:
            reference_source = "ads_indices_sidecar"
        else:
            reference_source = None

        fragments = _extract_fragments(
            structure,
            strategy=params.strategy,
            reference_indices=reference_indices,
        )
        selected = _select_fragments(
            fragments,
            select_ids=params.select_ids,
            match_natoms=params.match_natoms,
            match_formula=params.match_formula,
            select_reference_overlap=params.select_reference_overlap,
        )
        selected_indices = sorted(
            {
                int(idx)
                for frag in selected
                for idx in frag.get("indices", [])
            }
        )

        data = {
            "structure_path_rel": workspace_relpath(structure_path),
            "strategy": params.strategy,
            "fragment_count": len(fragments),
            "fragments_json_rel": workspace_relpath(output_json),
            "reference_indices": reference_indices,
            "reference_indices_1based": [idx + 1 for idx in reference_indices],
            "reference_source": reference_source,
            "selected_fragment_ids": [int(frag["frag_id"]) for frag in selected],
            "selected_indices": selected_indices,
            "selected_indices_1based": [idx + 1 for idx in selected_indices],
            "fragments": fragments,
        }
        output_json.write_text(json.dumps(data, indent=2, ensure_ascii=False), encoding="utf-8")

        content_lines = [
            "identify_structure_fragments completed.",
            f"fragment_count={data['fragment_count']} strategy={params.strategy}",
            f"fragments_json_rel={data['fragments_json_rel']}",
        ]
        if reference_indices:
            content_lines.append(
                f"reference_source={reference_source} selected_fragment_ids={data['selected_fragment_ids'] or []}"
            )
        elif data["selected_fragment_ids"]:
            content_lines.append(f"selected_fragment_ids={data['selected_fragment_ids']}")

        artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
        if warnings:
            artifact["warnings"] = warnings
        return "\n".join(content_lines), artifact
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"structure_path": payload.get("structure_path")}},
            error_code="identify_structure_fragments_failed",
        ) from exc


__all__ = ["IdentifyStructureFragmentsInput", "identify_structure_fragments"]
