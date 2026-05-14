from __future__ import annotations

import itertools
import random
from collections import defaultdict
from math import log, sqrt
from statistics import mean, pstdev
import json
from typing import Any

from ..core.config import (
    DEFAULT_ELEMENT_POOL,
    DEFAULT_GPU_IDS,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_HEAD,
    DEFAULT_MACE_MODEL,
    DEFAULT_STAGE1_ACTIVE_POOL_LIMIT,
    DEFAULT_STAGE1_ANCHOR_TABLE,
    DEFAULT_STAGE1_BASE_STRUCTURE,
    DEFAULT_STAGE1_ROUND_LIMIT,
    CampaignPaths,
)
from ..core.schemas import ElementPosterior
from ..core.storage import (
    append_jsonl,
    read_csv_records,
    read_yaml,
    write_dataframe_bundle,
    write_json,
    write_markdown,
    write_rows_csv,
    write_yaml,
)
from .anchors import load_s208_single_dopant_anchor_table
from .mace_eval import (
    get_fe_site_indices,
    load_base_structure,
    screen_candidates_in_parallel,
    substitution_count_from_x_total,
)


ELEMENT_DATA: dict[str, dict[str, float | int | bool]] = {
    "Ti": {"radius": 0.605, "chi": 1.54, "valence": 4, "redox": True},
    "V": {"radius": 0.64, "chi": 1.63, "valence": 3, "redox": True},
    "Cr": {"radius": 0.615, "chi": 1.66, "valence": 3, "redox": True},
    "Mn": {"radius": 0.83, "chi": 1.55, "valence": 2, "redox": True},
    "Co": {"radius": 0.745, "chi": 1.88, "valence": 2, "redox": True},
    "Ni": {"radius": 0.69, "chi": 1.91, "valence": 2, "redox": True},
    "Mo": {"radius": 0.65, "chi": 2.16, "valence": 4, "redox": True},
    "Ru": {"radius": 0.62, "chi": 2.20, "valence": 4, "redox": True},
    "Pd": {"radius": 0.86, "chi": 2.20, "valence": 2, "redox": True},
    "Mg": {"radius": 0.72, "chi": 1.31, "valence": 2, "redox": False},
    "Al": {"radius": 0.535, "chi": 1.61, "valence": 3, "redox": False},
    "Sc": {"radius": 0.745, "chi": 1.36, "valence": 3, "redox": False},
    "Cu": {"radius": 0.73, "chi": 1.90, "valence": 2, "redox": False},
    "Zn": {"radius": 0.74, "chi": 1.65, "valence": 2, "redox": False},
    "Ga": {"radius": 0.62, "chi": 1.81, "valence": 3, "redox": False},
    "Sr": {"radius": 1.18, "chi": 0.95, "valence": 2, "redox": False},
    "Y": {"radius": 0.90, "chi": 1.22, "valence": 3, "redox": False},
    "Zr": {"radius": 0.72, "chi": 1.33, "valence": 4, "redox": False},
    "Nb": {"radius": 0.64, "chi": 1.60, "valence": 5, "redox": False},
    "Rh": {"radius": 0.665, "chi": 2.28, "valence": 3, "redox": False},
    "In": {"radius": 0.80, "chi": 1.78, "valence": 3, "redox": False},
    "Sn": {"radius": 0.69, "chi": 1.96, "valence": 4, "redox": False},
    "Ca": {"radius": 1.00, "chi": 1.00, "valence": 2, "redox": False},
}

FE_RADIUS = 0.645
FE_CHI = 1.83
FE_VALENCE = 3.0
KB_EV_PER_K = 8.617333262145e-5
DEFAULT_STAGE1_TEMPERATURE_K = 298.0


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(", ".join(str(path) for path in paths))


def _load_manifest(paths: CampaignPaths) -> dict[str, Any]:
    manifest_path = paths.root / "manifest.yaml"
    return read_yaml(manifest_path) if manifest_path.exists() else {}


def _read_jsonl_records(path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows = []
    for line in path.read_text(encoding="utf-8").splitlines():
        line = line.strip()
        if line:
            rows.append(json.loads(line))
    return rows


def _normalize_dopants(raw_dopants: Any) -> list[str]:
    if isinstance(raw_dopants, list):
        return [str(item) for item in raw_dopants]
    if isinstance(raw_dopants, str):
        return [item for item in raw_dopants.strip("[]").replace("'", "").replace(" ", "").split(",") if item]
    return []


def _normalize_counts(raw_counts: Any) -> dict[str, int]:
    if isinstance(raw_counts, dict):
        return {str(key): int(value) for key, value in raw_counts.items()}
    if isinstance(raw_counts, str) and raw_counts.strip():
        tokens = [token for token in raw_counts.strip("{}").split(",") if ":" in token]
        counts = {}
        for token in tokens:
            key, value = token.split(":", 1)
            counts[key.strip().strip("'\"")] = int(value.strip())
        return counts
    return {}


def _distribute_counts(total_substitutions: int, dopants: list[str], seed: int) -> dict[str, int]:
    shuffled = list(dopants)
    rng = random.Random(seed)
    rng.shuffle(shuffled)
    base = total_substitutions // len(shuffled)
    remainder = total_substitutions % len(shuffled)
    counts = {element: base for element in shuffled}
    for index in range(remainder):
        counts[shuffled[index]] += 1
    return counts


def _compute_l0_metrics(combo: tuple[str, ...], x_total: float, counts: dict[str, int]) -> dict[str, float]:
    total_dopant_fraction = x_total / 3.0
    weights = [1.0 - total_dopant_fraction] + [
        total_dopant_fraction * counts[element] / sum(counts.values()) for element in combo
    ]
    radii = [FE_RADIUS] + [float(ELEMENT_DATA[element]["radius"]) for element in combo]
    chis = [FE_CHI] + [float(ELEMENT_DATA[element]["chi"]) for element in combo]
    valences = [FE_VALENCE] + [float(ELEMENT_DATA[element]["valence"]) for element in combo]

    mean_radius = sum(weight * radius for weight, radius in zip(weights, radii, strict=False))
    radius_mismatch = 100.0 * sqrt(
        sum(weight * (1.0 - radius / mean_radius) ** 2 for weight, radius in zip(weights, radii, strict=False))
    )
    mean_chi = sum(weight * chi for weight, chi in zip(weights, chis, strict=False))
    chi_spread = sqrt(sum(weight * (chi - mean_chi) ** 2 for weight, chi in zip(weights, chis, strict=False)))
    mean_valence = sum(weight * valence for weight, valence in zip(weights, valences, strict=False))
    valence_pressure = abs(mean_valence - FE_VALENCE)
    return {
        "radius_mismatch_pct": round(radius_mismatch, 6),
        "chi_spread": round(chi_spread, 6),
        "valence_pressure": round(valence_pressure, 6),
        "redox_active_count": sum(1 for element in combo if ELEMENT_DATA[element]["redox"]),
    }


def _ideal_config_entropy_term_per_dopant_eV(counts: dict[str, int], temperature_k: float = DEFAULT_STAGE1_TEMPERATURE_K) -> float:
    total = sum(int(value) for value in counts.values())
    if total <= 0:
        return 0.0
    fractions = [int(value) / total for value in counts.values() if int(value) > 0]
    if len(fractions) <= 1:
        return 0.0
    entropy_per_dopant = -KB_EV_PER_K * sum(fraction * log(fraction) for fraction in fractions)
    return temperature_k * entropy_per_dopant


def _render_chemistry_rationale(posterior_rows: list[dict[str, Any]], max_elements: int = 6) -> str:
    selected = posterior_rows[:max_elements]
    lines = [
        "# Stage 1 Chemistry Rationale",
        "",
        "The current rationale is based on L0 descriptors plus unified MACE screening on MC-sampled S416 configurations.",
        "",
    ]
    for row in selected:
        element = row["element"]
        props = ELEMENT_DATA[element]
        lines.append(
            f"- {element}: p_keep={float(row['p_keep']):.2f}, score_mean={float(row['score_mean']):.3f}, "
            f"uncertainty={float(row['uncertainty']):.3f}; radius={props['radius']}, "
            f"chi={props['chi']}, valence={props['valence']}, redox_active={bool(props['redox'])}."
        )
    return "\n".join(lines)


def _screening_row_summary(row: dict[str, Any]) -> dict[str, Any]:
    return {
        "dopants": _normalize_dopants(row.get("dopants", [])),
        "round_score": round(float(row["score"]), 6),
        "delta_g_mix_298k_proxy_eV_per_dopant": round(float(row["delta_g_mix_298k_proxy_eV_per_dopant"]), 6),
        "mixing_enthalpy_proxy_eV_per_dopant": round(float(row["mixing_enthalpy_proxy_eV_per_dopant"]), 6),
        "ideal_entropy_stabilization_298k_eV_per_dopant": round(
            float(row.get("ideal_entropy_stabilization_298k_eV_per_dopant", 0.0)),
            6,
        ),
        "substitution_energy_eV_per_dopant": round(float(row["raw_substitution_energy_eV_per_dopant"]), 6),
        "single_dopant_anchor_baseline_eV_per_dopant": round(float(row["single_dopant_baseline_eV_per_dopant"]), 6),
        "configuration_uncertainty_eV_per_dopant": round(float(row.get("mace_energy_std_eV", 0.0)), 6),
        "radius_mismatch_pct": round(float(row.get("radius_mismatch_pct", 0.0)), 6),
        "chi_spread": round(float(row.get("chi_spread", 0.0)), 6),
        "valence_pressure": round(float(row.get("valence_pressure", 0.0)), 6),
        "redox_active_count": int(row.get("redox_active_count", 0)),
        "mace_structure_path": row.get("mace_structure_path"),
    }


def build_stage1_tools(paths: CampaignPaths):
    def sample_low_doping_compositions(
        candidate_elements: list[str] | None = None,
        sample_size: int | None = None,
        x_total: float | None = None,
        num_dopants: int | None = None,
        random_seed: int = 13,
        round_index: int | None = None,
    ) -> list[dict[str, Any]]:
        """Sample low-doping five-element combinations for Na4Fe3-xMx(PO4)2(P2O7), where x_total is the total x."""

        manifest = _load_manifest(paths)
        stage1_cfg = manifest.get("stage1", {})
        base_atoms = load_base_structure(stage1_cfg.get("base_structure_path", str(DEFAULT_STAGE1_BASE_STRUCTURE)))
        n_fe = len(get_fe_site_indices(base_atoms))
        x_total = float(x_total if x_total is not None else stage1_cfg.get("x_total", 0.5))
        num_dopants = int(num_dopants if num_dopants is not None else stage1_cfg.get("num_dopants", 5))
        sample_size = int(sample_size if sample_size is not None else stage1_cfg.get("sample_size", 96))
        active_pool_limit = int(stage1_cfg.get("active_pool_limit", DEFAULT_STAGE1_ACTIVE_POOL_LIMIT))
        pool = list(candidate_elements) if candidate_elements else list(DEFAULT_ELEMENT_POOL[:active_pool_limit])
        if len(pool) > active_pool_limit:
            raise ValueError(
                f"Active element pool size {len(pool)} exceeds the hard stage1 limit of {active_pool_limit}. "
                "Shrink the candidate pool before sampling."
            )
        if len(pool) < num_dopants:
            raise ValueError(
                f"Active element pool size {len(pool)} is smaller than num_dopants={num_dopants}; "
                "at least five elements are required to form a five-dopant sample."
            )
        total_substitutions = substitution_count_from_x_total(n_fe, x_total)
        combos = list(itertools.combinations(pool, num_dopants))
        rng = random.Random(random_seed)
        sampled = rng.sample(combos, min(sample_size, len(combos)))

        rows = []
        for index, combo in enumerate(sampled):
            counts = _distribute_counts(total_substitutions, list(combo), seed=random_seed + index)
            rows.append(
                {
                    "formula": f"Na4Fe{3 - x_total:.4g}M{x_total:.4g}(PO4)2(P2O7)",
                    "dopants": list(combo),
                    "dopant_counts": counts,
                    "round_index": int(round_index or 0),
                    "x_total": x_total,
                    "num_dopants": num_dopants,
                    "total_substitutions": total_substitutions,
                }
            )

        bundle = write_dataframe_bundle(paths.stage1 / "sampled_candidates", rows)
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "sample_low_doping_compositions",
                "sample_size": len(rows),
                "x_total": x_total,
                "num_dopants": num_dopants,
                "total_substitutions": total_substitutions,
                "active_pool_limit": active_pool_limit,
                "candidate_elements": pool,
                "artifacts": {name: str(path) for name, path in bundle.items()},
            },
        )
        return rows

    def load_stage1_context() -> dict[str, Any]:
        """Load concise stage1 context from saved artifacts so the agent can resume planning without relying on raw chat history."""

        manifest = _load_manifest(paths)
        stage1_cfg = manifest.get("stage1", {})
        round_history = _read_jsonl_records(paths.stage1 / "round_history.jsonl")
        latest_summary_path = paths.stage1 / "latest_round_summary.md"
        context = {
            "candidate_universe": list(DEFAULT_ELEMENT_POOL),
            "active_pool_limit": int(stage1_cfg.get("active_pool_limit", DEFAULT_STAGE1_ACTIVE_POOL_LIMIT)),
            "target_rounds": int(stage1_cfg.get("round_limit", DEFAULT_STAGE1_ROUND_LIMIT)),
            "completed_rounds": len(round_history),
            "next_round_index": len(round_history) + 1,
            "round_history": round_history[-5:],
            "round_summaries": [f"[Round {row['round_index']:03d}] {row.get('summary_line', '')}".strip() for row in round_history[-5:]],
            "all_round_summaries": [
                f"[Round {row['round_index']:03d}] {row.get('summary_line', '')}".strip() for row in round_history
            ],
            "latest_round_summary": latest_summary_path.read_text(encoding="utf-8") if latest_summary_path.exists() else "",
            "screening_history_csv": str(paths.stage1 / "stage1_mace_screening.csv"),
        }
        top10_path = paths.shared / "top10_pool.yaml"
        shadow_path = paths.shared / "shadow_pool.yaml"
        if top10_path.exists():
            context["latest_top10_pool"] = read_yaml(top10_path)["top10_pool"]
        if shadow_path.exists():
            context["latest_shadow_pool"] = read_yaml(shadow_path)["shadow_pool"]
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "load_stage1_context",
                "completed_rounds": context["completed_rounds"],
                "next_round_index": context["next_round_index"],
            },
        )
        return context

    def screen_multidopant_configurations_with_mace(
        sampled_rows: list[dict[str, Any]] | None = None,
        mc_samples: int | None = None,
        mc_steps: int | None = None,
        mc_temperature_ev: float | None = None,
        random_seed: int = 13,
        relax_mode: str | None = None,
        model_path: str | None = None,
        head: str | None = None,
        default_dtype: str | None = None,
        light_relax_fmax: float | None = None,
        light_relax_steps: int | None = None,
    ) -> list[dict[str, Any]]:
        """Compute explicit composition descriptors, MC-sample S416 Fe-site configurations, and evaluate them with unified MACE."""

        manifest = _load_manifest(paths)
        stage1_cfg = manifest.get("stage1", {})
        mace_cfg = manifest.get("mace", {})
        if sampled_rows is None:
            sampled_rows = read_csv_records(paths.stage1 / "sampled_candidates.csv")
        anchor_payload = load_s208_single_dopant_anchor_table(
            stage1_cfg.get("anchor_table_path") or stage1_cfg.get("anchor_root", str(DEFAULT_STAGE1_ANCHOR_TABLE))
        )
        anchor_constants: dict[str, float] = anchor_payload["constants"]
        anchor_bundle = write_dataframe_bundle(paths.stage1 / "s208_single_dopant_anchors", anchor_payload["rows"])

        base_atoms = load_base_structure(stage1_cfg.get("base_structure_path", str(DEFAULT_STAGE1_BASE_STRUCTURE)))
        mc_samples = int(mc_samples if mc_samples is not None else stage1_cfg.get("mc_samples", 2))
        mc_steps = int(mc_steps if mc_steps is not None else stage1_cfg.get("mc_steps", 25))
        mc_temperature_ev = float(
            mc_temperature_ev if mc_temperature_ev is not None else stage1_cfg.get("mc_temperature_ev", 0.03)
        )
        relax_mode = relax_mode or stage1_cfg.get("relax_mode", "light_relax")
        if stage1_cfg.get("gpu_ids"):
            gpu_pool = [str(value) for value in stage1_cfg.get("gpu_ids")]
        elif stage1_cfg.get("gpu_id") is not None:
            gpu_pool = [str(stage1_cfg.get("gpu_id"))]
        else:
            gpu_pool = list(DEFAULT_GPU_IDS)
        model_path = model_path or mace_cfg.get("model_path") or str(DEFAULT_MACE_MODEL)
        head = head if head is not None else mace_cfg.get("head", DEFAULT_MACE_HEAD)
        default_dtype = default_dtype or mace_cfg.get("default_dtype", DEFAULT_MACE_DTYPE)
        light_relax_fmax = float(light_relax_fmax if light_relax_fmax is not None else stage1_cfg.get("light_relax_fmax", 0.05))
        light_relax_steps = int(light_relax_steps if light_relax_steps is not None else stage1_cfg.get("light_relax_steps", 20))
        n_fe_sites = len(get_fe_site_indices(base_atoms))

        rows = []
        worker_rows = []
        for index, row in enumerate(sampled_rows):
            dopants = _normalize_dopants(row["dopants"])
            counts = _normalize_counts(row["dopant_counts"])
            x_total = float(row["x_total"])
            missing = [element for element in dopants if element not in anchor_constants]
            if missing:
                raise ValueError(f"Missing S208 single-dopant anchors for elements: {missing}")
            metrics = _compute_l0_metrics(tuple(dopants), x_total=x_total, counts=counts)
            rows.append(
                {
                    **row,
                    **metrics,
                }
            )
            worker_rows.append(
                {
                    "index": index,
                    "round_index": int(row.get("round_index", 0)),
                    "dopants": dopants,
                    "x_total": x_total,
                    "total_substitutions": int(row["total_substitutions"]),
                    "random_seed": random_seed + index * 17,
                    "relax_mode": relax_mode,
                    "mc_samples": mc_samples,
                    "mc_steps": mc_steps,
                    "mc_temperature_ev": mc_temperature_ev,
                    "fmax": light_relax_fmax,
                    "max_steps": light_relax_steps,
                    "output_dir": str(
                        paths.stage1 / "mace_screening" / f"round_{int(row.get('round_index', 0)):03d}" / f"candidate_{index:03d}"
                    ),
                }
            )

        worker_results = screen_candidates_in_parallel(
            base_structure_path=str(stage1_cfg.get("base_structure_path", str(DEFAULT_STAGE1_BASE_STRUCTURE))),
            rows=worker_rows,
            model_path=str(model_path),
            head=head,
            default_dtype=default_dtype,
            gpu_ids=gpu_pool,
        )
        base_energy = float(worker_results[0]["base_energy_eV"]) if worker_results else math.nan
        for result in worker_results:
            index = int(result["index"])
            row = rows[index]
            candidate = result["candidate"]
            total_substitutions = int(worker_rows[index]["total_substitutions"])
            raw_substitution_energy = (float(candidate["energy_eV"]) - float(result["base_energy_eV"])) / max(total_substitutions, 1)
            counts = _normalize_counts(row["dopant_counts"])
            anchor_baseline = sum(
                (int(counts[element]) / max(total_substitutions, 1)) * anchor_constants[element] for element in counts
            )
            mixing_enthalpy_proxy = raw_substitution_energy - anchor_baseline
            ideal_entropy_stabilization = _ideal_config_entropy_term_per_dopant_eV(counts)
            delta_g_mix_298k_proxy = mixing_enthalpy_proxy - ideal_entropy_stabilization
            energy_std = float(candidate.get("replica_energy_std_eV", 0.0)) / max(total_substitutions, 1)
            score = -delta_g_mix_298k_proxy - 0.5 * energy_std
            row.update(
                {
                    "gpu_id": result["gpu_id"],
                    "n_fe_sites": n_fe_sites,
                    "anchor_reference_family": "B1__S208__Na4__dop_*__base",
                    "mace_structure_path": candidate["structure_path"],
                    "mace_energy_eV": round(float(candidate["energy_eV"]), 6),
                    "mace_energy_per_atom_eV": round(float(candidate["energy_per_atom_eV"]), 8),
                    "raw_substitution_energy_eV_per_dopant": round(raw_substitution_energy, 6),
                    "single_dopant_baseline_eV_per_dopant": round(anchor_baseline, 6),
                    "mixing_enthalpy_proxy_eV_per_dopant": round(mixing_enthalpy_proxy, 6),
                    "ideal_entropy_stabilization_298k_eV_per_dopant": round(ideal_entropy_stabilization, 6),
                    "delta_g_mix_298k_proxy_eV_per_dopant": round(delta_g_mix_298k_proxy, 6),
                    "mace_energy_std_eV": round(energy_std, 6),
                    "score": round(score, 6),
                }
            )

        screening_bundle = write_dataframe_bundle(paths.stage1 / "stage1_mace_screening", rows)
        explored_bundle = write_dataframe_bundle(paths.stage1 / "explored_candidates", rows)
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "screen_multidopant_configurations_with_mace",
                "evaluated_count": len(rows),
                "base_energy_eV": base_energy,
                "gpu_ids": gpu_pool,
                "mc_samples": mc_samples,
                "mc_steps": mc_steps,
                "mc_temperature_ev": mc_temperature_ev,
                "artifacts": {
                    **{f"anchors_{name}": str(path) for name, path in anchor_bundle.items()},
                    **{f"screening_{name}": str(path) for name, path in screening_bundle.items()},
                    **{f"explored_{name}": str(path) for name, path in explored_bundle.items()},
                },
            },
        )
        return rows

    def analyze_stage1_results(
        screened_rows: list[dict[str, Any]] | None = None,
        keep_fraction: float = 0.45,
        shadow_size: int = 3,
    ) -> dict[str, Any]:
        """Analyze screened rows, write stage1 statistics, and update the element posterior and candidate pools."""

        if screened_rows is None:
            screened_rows = read_csv_records(paths.stage1 / "stage1_mace_screening.csv")

        rows = []
        for row in screened_rows:
            rows.append(
                {
                    **row,
                    "score": float(row["score"]),
                    "delta_g_mix_298k_proxy_eV_per_dopant": float(
                        row.get("delta_g_mix_298k_proxy_eV_per_dopant", row.get("excess_mixing_energy_proxy_eV_per_dopant"))
                    ),
                    "mixing_enthalpy_proxy_eV_per_dopant": float(
                        row.get("mixing_enthalpy_proxy_eV_per_dopant", row.get("excess_mixing_energy_proxy_eV_per_dopant"))
                    ),
                    "mace_energy_std_eV": float(row.get("mace_energy_std_eV", 0.0)),
                }
            )
        ranked_rows = sorted(rows, key=lambda row: row["score"], reverse=True)

        statistics = {
            "candidate_count": len(ranked_rows),
            "score_mean": round(mean(row["score"] for row in ranked_rows), 6),
            "score_max": round(ranked_rows[0]["score"], 6),
            "delta_g_mix_298k_proxy_mean": round(
                mean(row["delta_g_mix_298k_proxy_eV_per_dopant"] for row in ranked_rows), 6
            ),
            "mixing_enthalpy_proxy_mean": round(
                mean(row["mixing_enthalpy_proxy_eV_per_dopant"] for row in ranked_rows), 6
            ),
            "top_formulas": [",".join(_normalize_dopants(row["dopants"])) for row in ranked_rows[:5]],
        }
        stats_text = (
            "# Stage 1 Statistics\n\n"
            f"Candidates evaluated: {statistics['candidate_count']}\n\n"
            f"Mean score: {statistics['score_mean']}\n\n"
            f"Best score: {statistics['score_max']}\n\n"
            f"Mean ideal-configurational Delta G_mix(298 K) proxy: {statistics['delta_g_mix_298k_proxy_mean']} eV per substituted Fe site\n\n"
            f"Mean anchor-referenced mixing enthalpy proxy: {statistics['mixing_enthalpy_proxy_mean']} eV per substituted Fe site\n\n"
            "Top combinations:\n"
            + "\n".join(f"- {formula}" for formula in statistics["top_formulas"])
        )
        write_markdown(paths.stage1 / "stage1_statistics.md", stats_text)

        cutoff = max(1, int(len(ranked_rows) * keep_fraction))
        support: dict[str, list[float]] = defaultdict(list)
        top_hits: dict[str, int] = defaultdict(int)
        kept_hits: dict[str, int] = defaultdict(int)
        for index, row in enumerate(ranked_rows):
            dopants = _normalize_dopants(row["dopants"])
            for element in dopants:
                support[element].append(float(row["score"]))
                if index < cutoff:
                    kept_hits[element] += 1
                if index < 10:
                    top_hits[element] += 1

        posterior_rows = []
        for element, scores in support.items():
            support_count = len(scores)
            spread = pstdev(scores) if support_count > 1 else 0.0
            posterior_rows.append(
                ElementPosterior(
                    element=element,
                    p_keep=round(min(0.99, kept_hits[element] / support_count), 4),
                    p_top10=round(min(0.99, top_hits[element] / support_count), 4),
                    score_mean=round(mean(scores), 4),
                    score_std=round(spread, 4),
                    uncertainty=round(max(0.01, 1 / sqrt(support_count + 1) + spread * 0.3), 4),
                    support_count=support_count,
                ).model_dump()
            )
        posterior_rows.sort(key=lambda row: (row["p_keep"], row["score_mean"]), reverse=True)
        top10_pool = [row["element"] for row in posterior_rows[:10]]
        experimental_pool5 = [row["element"] for row in posterior_rows[:5]]
        shadow_pool = [row["element"] for row in posterior_rows[10 : 10 + shadow_size]]

        posterior_bundle = write_dataframe_bundle(paths.shared / "element_posterior", posterior_rows)
        write_yaml(paths.shared / "top10_pool.yaml", {"top10_pool": top10_pool})
        write_yaml(paths.shared / "experimental_pool5.yaml", {"experimental_pool5": experimental_pool5})
        write_yaml(paths.shared / "shadow_pool.yaml", {"shadow_pool": shadow_pool})
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "analyze_stage1_results",
                "top10_pool": top10_pool,
                "experimental_pool5": experimental_pool5,
                "shadow_pool": shadow_pool,
                "statistics": statistics,
                "artifacts": {name: str(path) for name, path in posterior_bundle.items()},
            },
        )
        return {
            "posterior_rows": posterior_rows,
            "top10_pool": top10_pool,
            "experimental_pool5": experimental_pool5,
            "shadow_pool": shadow_pool,
            "statistics": statistics,
            "top_candidates": [_screening_row_summary(row) for row in ranked_rows[:12]],
            "bottom_candidates": [_screening_row_summary(row) for row in ranked_rows[-5:]],
            "screening_csv_path": str(paths.stage1 / "stage1_mace_screening.csv"),
        }

    def run_stage1_round(
        candidate_elements: list[str],
        round_index: int | None = None,
        sample_size: int | None = None,
        x_total: float | None = None,
        num_dopants: int | None = None,
        random_seed: int = 13,
        keep_fraction: float = 0.45,
        shadow_size: int = 3,
        mc_samples: int | None = None,
        mc_steps: int | None = None,
        mc_temperature_ev: float | None = None,
        relax_mode: str | None = None,
        light_relax_fmax: float | None = None,
        light_relax_steps: int | None = None,
        literature_notes: str | None = None,
        round_objective: str | None = None,
    ) -> dict[str, Any]:
        """Run one controlled stage1 round: sample -> screen -> analyze, then persist concise round history artifacts."""

        history_rows = _read_jsonl_records(paths.stage1 / "round_history.jsonl")
        round_index = int(round_index or (len(history_rows) + 1))
        round_dir = paths.stage1 / "rounds" / f"round_{round_index:03d}"

        previous_rows = read_csv_records(paths.stage1 / "stage1_mace_screening.csv") if (paths.stage1 / "stage1_mace_screening.csv").exists() else []
        sampled_rows = sample_low_doping_compositions(
            candidate_elements=candidate_elements,
            sample_size=sample_size,
            x_total=x_total,
            num_dopants=num_dopants,
            random_seed=random_seed,
            round_index=round_index,
        )
        screened_current = screen_multidopant_configurations_with_mace(
            sampled_rows=sampled_rows,
            mc_samples=mc_samples,
            mc_steps=mc_steps,
            mc_temperature_ev=mc_temperature_ev,
            random_seed=random_seed,
            relax_mode=relax_mode,
            light_relax_fmax=light_relax_fmax,
            light_relax_steps=light_relax_steps,
        )
        cumulative_rows = list(previous_rows) + list(screened_current)
        write_dataframe_bundle(paths.stage1 / "stage1_mace_screening", cumulative_rows)
        write_dataframe_bundle(paths.stage1 / "explored_candidates", cumulative_rows)

        analysis_payload = analyze_stage1_results(
            screened_rows=cumulative_rows,
            keep_fraction=keep_fraction,
            shadow_size=shadow_size,
        )

        write_rows_csv(round_dir / "sampled_candidates.csv", sampled_rows)
        write_rows_csv(round_dir / "screened_candidates.csv", screened_current)
        write_rows_csv(round_dir / "cumulative_screening.csv", cumulative_rows)
        write_rows_csv(round_dir / "element_posterior.csv", analysis_payload["posterior_rows"])

        round_summary = (
            f"[Round {round_index:03d}]\n"
            f"Objective: {(round_objective or 'shrink the active pool with literature-informed stage1 screening').strip()}\n"
            f"Active pool: {', '.join(candidate_elements)}\n"
            f"Current-round sample count: {len(sampled_rows)}\n"
            f"Cumulative candidate count: {analysis_payload['statistics']['candidate_count']}\n"
            f"Top10 pool: {', '.join(analysis_payload['top10_pool'])}\n"
            f"Shadow pool: {', '.join(analysis_payload['shadow_pool'])}\n"
            f"Mean score: {analysis_payload['statistics']['score_mean']}\n"
            f"Mean ideal-configurational Delta G_mix(298 K) proxy: {analysis_payload['statistics']['delta_g_mix_298k_proxy_mean']} eV/site\n"
            f"Top candidate snippets: {json.dumps(analysis_payload['top_candidates'][:3], ensure_ascii=True)}\n"
        )
        if literature_notes and literature_notes.strip():
            round_summary += f"Literature notes: {literature_notes.strip()}\n"

        round_record = {
            "round_index": round_index,
            "candidate_elements": list(candidate_elements),
            "sample_size": len(sampled_rows),
            "cumulative_candidate_count": analysis_payload["statistics"]["candidate_count"],
            "top10_pool": analysis_payload["top10_pool"],
            "experimental_pool5": analysis_payload["experimental_pool5"],
            "shadow_pool": analysis_payload["shadow_pool"],
            "score_mean": analysis_payload["statistics"]["score_mean"],
            "delta_g_mix_298k_proxy_mean": analysis_payload["statistics"]["delta_g_mix_298k_proxy_mean"],
            "round_summary_path": str(round_dir / "round_summary.md"),
            "summary_line": (
                f"pool={','.join(candidate_elements)} | top10={','.join(analysis_payload['top10_pool'])} | "
                f"mean_score={analysis_payload['statistics']['score_mean']} | "
                f"mean_deltaG298={analysis_payload['statistics']['delta_g_mix_298k_proxy_mean']} eV/site"
            ),
            "literature_notes": literature_notes or "",
        }
        write_markdown(round_dir / "round_summary.md", round_summary)
        write_json(round_dir / "round_summary.json", round_record)
        write_markdown(paths.stage1 / "latest_round_summary.md", round_summary)
        append_jsonl(paths.stage1 / "round_history.jsonl", round_record)
        recent_round_history = (_read_jsonl_records(paths.stage1 / "round_history.jsonl"))[-5:]
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "run_stage1_round",
                "round_index": round_index,
                "candidate_elements": list(candidate_elements),
                "sample_size": len(sampled_rows),
                "top10_pool": analysis_payload["top10_pool"],
                "experimental_pool5": analysis_payload["experimental_pool5"],
                "shadow_pool": analysis_payload["shadow_pool"],
            },
        )
        return {
            **analysis_payload,
            "round_index": round_index,
            "round_summary": round_summary,
            "round_summary_path": str(round_dir / "round_summary.md"),
            "round_history_jsonl": str(paths.stage1 / "round_history.jsonl"),
            "recent_round_history": recent_round_history,
            "recent_round_summaries": [
                f"[Round {row['round_index']:03d}] {row.get('summary_line', '')}".strip() for row in recent_round_history
            ],
            "all_round_summaries": [
                f"[Round {row['round_index']:03d}] {row.get('summary_line', '')}".strip()
                for row in _read_jsonl_records(paths.stage1 / "round_history.jsonl")
            ],
            "planning_context": {
                "completed_rounds": len(recent_round_history),
                "remaining_rounds": max(0, int(_load_manifest(paths).get("stage1", {}).get("round_limit", DEFAULT_STAGE1_ROUND_LIMIT)) - round_index),
                "latest_top10_pool": analysis_payload["top10_pool"],
                "latest_shadow_pool": analysis_payload["shadow_pool"],
                "screening_csv_path": str(paths.stage1 / "stage1_mace_screening.csv"),
            },
        }

    def explain_element_chemistry(
        analysis_payload: dict[str, Any] | None = None,
        max_elements: int = 6,
        report_markdown: str | None = None,
    ) -> str:
        """Write a stage1 chemistry analysis report. Prefer passing agent-authored Markdown that synthesizes literature and screening evidence."""

        posterior_rows = analysis_payload["posterior_rows"] if analysis_payload else None
        if posterior_rows is None:
            csv_rows = read_csv_records(paths.shared / "element_posterior.csv")
            posterior_rows = [
                {
                    **row,
                    "p_keep": float(row["p_keep"]),
                    "score_mean": float(row["score_mean"]),
                    "uncertainty": float(row["uncertainty"]),
                }
                for row in csv_rows
            ]

        report = report_markdown.strip() if report_markdown and report_markdown.strip() else _render_chemistry_rationale(
            posterior_rows,
            max_elements=max_elements,
        )
        write_markdown(paths.stage1 / "chemistry_rationale.md", report)
        write_markdown(paths.shared / "chemistry_rationale.md", report)
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {
                "action": "explain_element_chemistry",
                "elements": [row["element"] for row in posterior_rows[:max_elements]],
                "used_agent_authored_report": bool(report_markdown and report_markdown.strip()),
            },
        )
        return report

    def export_stage1_report(
        analysis_payload: dict[str, Any] | None = None,
        chemistry_rationale: str | None = None,
    ) -> dict[str, str]:
        """Materialize the stage1 markdown summary consumed by stage2."""

        manifest = _load_manifest(paths)
        stage1_cfg = manifest.get("stage1", {})
        if analysis_payload is None:
            top10_pool = read_yaml(paths.shared / "top10_pool.yaml")["top10_pool"]
            experimental_pool5 = read_yaml(paths.shared / "experimental_pool5.yaml")["experimental_pool5"]
            shadow_pool = read_yaml(paths.shared / "shadow_pool.yaml")["shadow_pool"]
            statistics = None
        else:
            top10_pool = analysis_payload["top10_pool"]
            experimental_pool5 = analysis_payload["experimental_pool5"]
            shadow_pool = analysis_payload["shadow_pool"]
            statistics = analysis_payload.get("statistics")
        if chemistry_rationale is None:
            try:
                chemistry_rationale = _first_existing(
                    paths.shared / "chemistry_rationale.md",
                    paths.stage1 / "chemistry_rationale.md",
                ).read_text(encoding="utf-8")
            except FileNotFoundError:
                csv_rows = read_csv_records(paths.shared / "element_posterior.csv")
                chemistry_rationale = _render_chemistry_rationale(csv_rows)

        stats_line = ""
        if statistics:
            stats_line = (
                f"Candidates evaluated: {statistics['candidate_count']}; "
                f"mean score: {statistics['score_mean']}; "
                f"mean ideal-configurational Delta G_mix(298 K) proxy: {statistics['delta_g_mix_298k_proxy_mean']} eV/site.\n\n"
            )
        summary = (
            "# Summary For Stage2\n\n"
            f"Base structure: {stage1_cfg.get('base_structure_path', str(DEFAULT_STAGE1_BASE_STRUCTURE))}\n\n"
            f"Fixed supercell: {stage1_cfg.get('supercell', 'S416')}\n\n"
            f"Stage1 total doping x_total: {stage1_cfg.get('x_total', 0.5)} in Na4Fe3-xMx(PO4)2(P2O7)\n\n"
            f"{stats_line}"
            f"Experimental direct-start pool (5 elements): {', '.join(experimental_pool5)}\n\n"
            f"Top10 pool: {', '.join(top10_pool)}\n\n"
            f"Shadow pool: {', '.join(shadow_pool)}\n\n"
            "Stage1 followed a sample -> evaluate -> analyze loop with L0 descriptors and unified MACE screening.\n"
            "Energy ranking used an ideal-configurational Delta G_mix(298 K) proxy built from an anchor-referenced mixing enthalpy proxy plus an ideal entropy approximation.\n"
        )
        full_report = (
            "# Stage 1 Full Report\n\n"
            "This report captures stage1 low-cost screening on the fixed S416 NFPP base structure.\n\n"
            f"## Experimental direct-start pool (5 elements)\n{', '.join(experimental_pool5)}\n\n"
            f"## Main pool\n{', '.join(top10_pool)}\n\n"
            f"## Shadow pool\n{', '.join(shadow_pool)}\n\n"
            "## Written analysis\n\n"
            f"{chemistry_rationale}\n"
        )
        write_markdown(paths.stage1 / "chemistry_rationale.md", chemistry_rationale)
        write_markdown(paths.shared / "chemistry_rationale.md", chemistry_rationale)
        summary_path = write_markdown(paths.stage1 / "summary_for_stage2.md", summary)
        write_markdown(paths.shared / "summary_for_stage2.md", summary)
        report_path = write_markdown(paths.stage1 / "stage1_full_report.md", full_report)
        append_jsonl(
            paths.stage1 / "decision_log.jsonl",
            {"action": "export_stage1_report", "summary_path": str(summary_path), "report_path": str(report_path)},
        )
        return {
            "summary_for_stage2": str(paths.shared / "summary_for_stage2.md"),
            "stage1_full_report": str(report_path),
            "chemistry_rationale": str(paths.shared / "chemistry_rationale.md"),
            "element_posterior_csv": str(paths.shared / "element_posterior.csv"),
            "experimental_pool5_yaml": str(paths.shared / "experimental_pool5.yaml"),
            "stage1_mace_screening_csv": str(paths.stage1 / "stage1_mace_screening.csv"),
            "stage1_statistics_md": str(paths.stage1 / "stage1_statistics.md"),
        }

    return [load_stage1_context, run_stage1_round, explain_element_chemistry, export_stage1_report]
