from __future__ import annotations

import itertools
import math
from pathlib import Path
from typing import Any

from ..core.config import (
    CampaignPaths,
    DEFAULT_GPU_IDS,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_HEAD,
    DEFAULT_MACE_MODEL,
    DEFAULT_MD_TEMPERATURES_K,
)
from ..core.storage import append_jsonl, read_csv_records, read_yaml, write_dataframe_bundle, write_markdown, write_yaml
from ..stage1.tools import ELEMENT_DATA
from .ase_md import evaluate_volume_change, run_multitemperature_md


def _first_existing(*paths):
    for path in paths:
        if path.exists():
            return path
    raise FileNotFoundError(", ".join(str(path) for path in paths))


def _load_manifest(paths: CampaignPaths) -> dict[str, Any]:
    manifest_path = paths.root / "manifest.yaml"
    return read_yaml(manifest_path) if manifest_path.exists() else {}


def _load_candidate_library(paths: CampaignPaths) -> list[dict[str, Any]]:
    for candidate_path in (
        paths.shared / "candidate_library.yaml",
        paths.shared / "candidate_library.yml",
        paths.root / "candidate_library.yaml",
        paths.root / "candidate_library.yml",
    ):
        if candidate_path.exists():
            payload = read_yaml(candidate_path)
            if isinstance(payload, dict):
                return payload.get("candidates", [])
            return payload
    return []


def _normalize_dopants(raw_dopants: Any) -> list[str]:
    if isinstance(raw_dopants, list):
        return [str(item) for item in raw_dopants]
    if isinstance(raw_dopants, str):
        return [item for item in raw_dopants.strip("[]").replace("'", "").replace(" ", "").split(",") if item]
    return []


def _candidate_metrics(combo: tuple[str, ...], mode: str) -> dict[str, float]:
    size_penalty = sum(abs(float(ELEMENT_DATA[element]["radius"]) - 0.645) for element in combo) / len(combo)
    redox_density = sum(1 for element in combo if ELEMENT_DATA[element]["redox"]) / len(combo)
    chi_mean = sum(float(ELEMENT_DATA[element]["chi"]) for element in combo) / len(combo)

    stability = round(1.25 - 0.75 * size_penalty + 0.12 * redox_density, 4)
    diffusion = round(0.42 + 0.35 * redox_density + 0.08 * (2.0 - abs(1.75 - chi_mean)), 4)
    barrier = round(max(0.12, 0.72 - diffusion * 0.55), 4)
    deformation = round(max(0.01, 0.035 + size_penalty * 0.08), 4)
    if mode == "explore":
        diffusion = round(diffusion * 0.97, 4)
        barrier = round(barrier * 1.03, 4)
    if mode == "surprise":
        stability = round(stability * 0.94, 4)
        diffusion = round(diffusion * 1.04, 4)
    return {
        "stability_score": stability,
        "diffusion_score": diffusion,
        "activation_barrier_ev": barrier,
        "volume_deformation": deformation,
    }


def _dominates(left: dict[str, Any], right: dict[str, Any]) -> bool:
    no_worse = (
        float(left["stability_score"]) >= float(right["stability_score"])
        and float(left["diffusion_score"]) >= float(right["diffusion_score"])
        and float(left["activation_barrier_ev"]) <= float(right["activation_barrier_ev"])
        and float(left["volume_deformation"]) <= float(right["volume_deformation"])
    )
    strictly_better = (
        float(left["stability_score"]) > float(right["stability_score"])
        or float(left["diffusion_score"]) > float(right["diffusion_score"])
        or float(left["activation_barrier_ev"]) < float(right["activation_barrier_ev"])
        or float(left["volume_deformation"]) < float(right["volume_deformation"])
    )
    return no_worse and strictly_better


def _score_from_real_metrics(volume_deformation: float, diffusion_m2_s: float, activation_barrier_ev: float) -> float:
    if not math.isfinite(volume_deformation):
        volume_deformation = 0.2
    if not math.isfinite(diffusion_m2_s) or diffusion_m2_s <= 0:
        diffusion_term = -20.0
    else:
        diffusion_term = math.log10(diffusion_m2_s)
    return round(2.0 - abs(volume_deformation) * 12.0 - activation_barrier_ev * 1.5 + (diffusion_term + 12.0) * 0.15, 6)


def build_stage2_tools(paths: CampaignPaths):
    def load_stage1_prior() -> dict[str, Any]:
        """Load stage1 posterior, pools, manifest settings, and optional candidate library."""

        prior_rows = read_csv_records(paths.shared / "element_posterior.csv")
        summary_path = _first_existing(paths.shared / "summary_for_stage2.md", paths.stage1 / "summary_for_stage2.md")
        payload = {
            "posterior": prior_rows,
            "top10_pool": read_yaml(paths.shared / "top10_pool.yaml")["top10_pool"],
            "shadow_pool": read_yaml(paths.shared / "shadow_pool.yaml")["shadow_pool"],
            "summary_for_stage2": summary_path.read_text(encoding="utf-8"),
            "manifest": _load_manifest(paths),
            "candidate_library": _load_candidate_library(paths),
        }
        append_jsonl(paths.stage2 / "decision_log.jsonl", {"action": "load_stage1_prior"})
        return payload

    def propose_stage2_candidates(
        top_k: int = 8,
        explore_k: int = 3,
        surprise_k: int = 2,
        prior_payload: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Propose exploit and exploration candidates, optionally enriching with structure paths."""

        if prior_payload is None:
            prior_payload = load_stage1_prior()
        top_pool = prior_payload["top10_pool"]
        shadow_pool = prior_payload["shadow_pool"]
        candidate_library = prior_payload.get("candidate_library", [])

        if candidate_library:
            scored = []
            for candidate in candidate_library:
                dopants = _normalize_dopants(candidate.get("dopants"))
                overlap_top = len(set(dopants) & set(top_pool))
                overlap_shadow = len(set(dopants) & set(shadow_pool))
                exploration_mode = "exploit"
                if overlap_shadow:
                    exploration_mode = "explore"
                if overlap_top <= 2:
                    exploration_mode = "surprise"
                scored.append(
                    {
                        **candidate,
                        "dopants": dopants,
                        "exploration_mode": exploration_mode,
                        "_priority": overlap_top * 10 - overlap_shadow,
                    }
                )
            scored.sort(key=lambda row: row["_priority"], reverse=True)
            proposed = []
            buckets = {
                "exploit": [row for row in scored if row["exploration_mode"] == "exploit"][:top_k],
                "explore": [row for row in scored if row["exploration_mode"] == "explore"][:explore_k],
                "surprise": [row for row in scored if row["exploration_mode"] == "surprise"][:surprise_k],
            }
            for mode in ("exploit", "explore", "surprise"):
                proposed.extend([{key: value for key, value in row.items() if key != "_priority"} for row in buckets[mode]])
        else:
            proposed = []
            for combo in itertools.islice(itertools.combinations(top_pool[:8], 5), top_k):
                proposed.append(
                    {
                        "formula": f"Na4Fe3-0.5({'-'.join(combo)})0.5(PO4)2(P2O7)",
                        "dopants": list(combo),
                        "exploration_mode": "exploit",
                    }
                )
            for combo in itertools.islice(itertools.combinations(top_pool[:5] + shadow_pool, 5), explore_k):
                proposed.append(
                    {
                        "formula": f"Na4Fe3-0.5({'-'.join(combo)})0.5(PO4)2(P2O7)",
                        "dopants": list(combo),
                        "exploration_mode": "explore",
                    }
                )
            surprise_pool = shadow_pool + top_pool[-3:]
            for combo in itertools.islice(itertools.combinations(surprise_pool, 5), surprise_k):
                proposed.append(
                    {
                        "formula": f"Na4Fe3-0.5({'-'.join(combo)})0.5(PO4)2(P2O7)",
                        "dopants": list(combo),
                        "exploration_mode": "surprise",
                    }
                )

        bundle = write_dataframe_bundle(paths.stage2 / "proposed_candidates", proposed)
        append_jsonl(
            paths.stage2 / "decision_log.jsonl",
            {
                "action": "propose_stage2_candidates",
                "candidate_count": len(proposed),
                "artifacts": {name: str(path) for name, path in bundle.items()},
            },
        )
        return proposed

    def evaluate_volume_deformation_with_mlff(
        sodiated_structure_path: str,
        desodiated_structure_path: str,
        model_path: str | None = None,
        head: str | None = None,
        default_dtype: str | None = None,
        gpu_id: str | None = None,
        fmax: float = 0.05,
        max_steps: int = 200,
    ) -> dict[str, Any]:
        """Relax the sodiated/desodiated pair and compute ΔV/V."""

        manifest = _load_manifest(paths)
        mace_cfg = manifest.get("mace", {})
        md_cfg = manifest.get("md", {})
        result = evaluate_volume_change(
            sodiated_structure_path=sodiated_structure_path,
            desodiated_structure_path=desodiated_structure_path,
            output_dir=str(paths.stage2 / "volume_change" / Path(sodiated_structure_path).stem),
            model_path=model_path or mace_cfg.get("model_path") or str(DEFAULT_MACE_MODEL),
            head=head if head is not None else mace_cfg.get("head", DEFAULT_MACE_HEAD),
            default_dtype=default_dtype or mace_cfg.get("default_dtype", DEFAULT_MACE_DTYPE),
            device=f"cuda:{gpu_id or (md_cfg.get('gpu_ids') or DEFAULT_GPU_IDS)[0]}",
            fmax=fmax,
            max_steps=max_steps,
        )
        append_jsonl(
            paths.stage2 / "decision_log.jsonl",
            {
                "action": "evaluate_volume_deformation_with_mlff",
                "sodiated_structure_path": sodiated_structure_path,
                "desodiated_structure_path": desodiated_structure_path,
                "volume_deformation": result["volume_deformation"],
            },
        )
        return result

    def run_mlff_md_batch(
        candidates: list[dict[str, Any]] | None = None,
        reference_temperature_K: float = 800.0,
        steps: int | None = None,
        timestep_fs: float | None = None,
        sample_interval: int | None = None,
        temperatures_K: list[float] | None = None,
        gpu_ids: list[str] | None = None,
        friction: float | None = None,
    ) -> list[dict[str, Any]]:
        """Run 4-temperature ASE MD when structure paths are available, otherwise fallback to heuristics."""

        manifest = _load_manifest(paths)
        mace_cfg = manifest.get("mace", {})
        md_cfg = manifest.get("md", {})
        if candidates is None:
            candidates = read_csv_records(paths.stage2 / "proposed_candidates.csv")

        rows = []
        for index, candidate in enumerate(candidates):
            dopants = _normalize_dopants(candidate.get("dopants"))
            exploration_mode = candidate.get("exploration_mode", "exploit")
            structure_path = candidate.get("structure_path")
            desodiated_path = candidate.get("desodiated_structure_path")

            if structure_path and Path(structure_path).expanduser().exists():
                run_root = paths.stage2 / "md_runs" / (candidate.get("formula") or Path(structure_path).stem).replace("/", "_")
                diffusion = run_multitemperature_md(
                    structure_path=structure_path,
                    output_dir=str(run_root / "multitemp_md"),
                    model_path=mace_cfg.get("model_path") or str(DEFAULT_MACE_MODEL),
                    head=mace_cfg.get("head", DEFAULT_MACE_HEAD),
                    default_dtype=mace_cfg.get("default_dtype", DEFAULT_MACE_DTYPE),
                    temperatures_K=temperatures_K or md_cfg.get("temperatures_K") or DEFAULT_MD_TEMPERATURES_K,
                    gpu_ids=gpu_ids or md_cfg.get("gpu_ids") or DEFAULT_GPU_IDS,
                    timestep_fs=timestep_fs or md_cfg.get("timestep_fs", 1.0),
                    steps=int(steps or md_cfg.get("steps", 4000)),
                    sample_interval=int(sample_interval or md_cfg.get("sample_interval", 20)),
                    friction=float(friction or md_cfg.get("friction", 0.02)),
                    reference_temperature_K=reference_temperature_K,
                )
                if desodiated_path and Path(desodiated_path).expanduser().exists():
                    volume = evaluate_volume_change(
                        sodiated_structure_path=structure_path,
                        desodiated_structure_path=desodiated_path,
                        output_dir=str(run_root / "volume_change"),
                        model_path=mace_cfg.get("model_path") or str(DEFAULT_MACE_MODEL),
                        head=mace_cfg.get("head", DEFAULT_MACE_HEAD),
                        default_dtype=mace_cfg.get("default_dtype", DEFAULT_MACE_DTYPE),
                        device=f"cuda:{(gpu_ids or md_cfg.get('gpu_ids') or DEFAULT_GPU_IDS)[0]}",
                    )
                    volume_deformation = float(volume["volume_deformation"])
                    volume_summary_path = str(run_root / "volume_change" / "volume_change_summary.json")
                else:
                    volume_deformation = float(_candidate_metrics(tuple(dopants), exploration_mode)["volume_deformation"])
                    volume_summary_path = ""

                diffusion_score = float(diffusion["diffusion_at_reference_m2_s"])
                activation_barrier_ev = float(diffusion["activation_barrier_ev"])
                stability_score = _score_from_real_metrics(volume_deformation, diffusion_score, activation_barrier_ev)
                rows.append(
                    {
                        **candidate,
                        "dopants": dopants,
                        "exploration_mode": exploration_mode,
                        "stability_score": stability_score,
                        "diffusion_score": diffusion_score,
                        "activation_barrier_ev": activation_barrier_ev,
                        "volume_deformation": volume_deformation,
                        "reference_temperature_K": reference_temperature_K,
                        "md_summary_path": str(run_root / "multitemp_md" / "arrhenius_summary.json"),
                        "volume_summary_path": volume_summary_path,
                    }
                )
            else:
                metrics = _candidate_metrics(tuple(dopants), exploration_mode)
                rows.append(
                    {
                        **candidate,
                        "dopants": dopants,
                        **metrics,
                        "reference_temperature_K": reference_temperature_K,
                        "md_summary_path": "",
                        "volume_summary_path": "",
                    }
                )

        bundle = write_dataframe_bundle(paths.stage2 / "md_screening", rows)
        append_jsonl(
            paths.stage2 / "decision_log.jsonl",
            {
                "action": "run_mlff_md_batch",
                "evaluated_count": len(rows),
                "real_md_count": sum(1 for row in rows if row.get("md_summary_path")),
                "artifacts": {name: str(path) for name, path in bundle.items()},
            },
        )
        return rows

    def update_pareto_archive(screened_rows: list[dict[str, Any]] | None = None) -> dict[str, Any]:
        """Compute a Pareto archive and queue recommendations."""

        if screened_rows is None:
            screened_rows = read_csv_records(paths.stage2 / "md_screening.csv")

        normalized_rows = []
        for row in screened_rows:
            row = dict(row)
            row["dopants"] = _normalize_dopants(row.get("dopants"))
            row["stability_score"] = float(row["stability_score"])
            row["diffusion_score"] = float(row["diffusion_score"])
            row["activation_barrier_ev"] = float(row["activation_barrier_ev"])
            row["volume_deformation"] = float(row["volume_deformation"])
            normalized_rows.append(row)

        archive = []
        for row in normalized_rows:
            if any(_dominates(other, row) for other in normalized_rows if other is not row):
                continue
            archive.append(row)

        archive.sort(
            key=lambda row: (
                row["activation_barrier_ev"],
                -row["diffusion_score"],
                -row["stability_score"],
                row["volume_deformation"],
            )
        )
        archive_bundle = write_dataframe_bundle(paths.stage2 / "pareto_archive", archive)
        dft_queue = [row["formula"] for row in archive[: min(5, len(archive))]]
        experiment_queue = [row["formula"] for row in archive[: min(3, len(archive))]]
        write_yaml(paths.stage2 / "recommended_dft_queue.yaml", {"dft_queue": dft_queue})
        write_yaml(paths.stage2 / "recommended_experiment_queue.yaml", {"experiment_queue": experiment_queue})
        append_jsonl(
            paths.stage2 / "decision_log.jsonl",
            {
                "action": "update_pareto_archive",
                "pareto_count": len(archive),
                "dft_queue": dft_queue,
                "experiment_queue": experiment_queue,
                "artifacts": {name: str(path) for name, path in archive_bundle.items()},
            },
        )
        return {
            "archive": archive,
            "dft_queue": dft_queue,
            "experiment_queue": experiment_queue,
        }

    def export_stage2_report(
        archive_payload: dict[str, Any] | None = None,
        prior_payload: dict[str, Any] | None = None,
    ) -> dict[str, str]:
        """Write the final stage2 report and recommendation queues."""

        if prior_payload is None:
            prior_payload = load_stage1_prior()
        if archive_payload is None:
            archive_payload = update_pareto_archive()

        report_lines = [
            "# Stage 2 Full Report",
            "",
            "## Prior summary",
            "",
            prior_payload["summary_for_stage2"].strip(),
            "",
            "## Pareto archive",
            "",
        ]
        for row in archive_payload["archive"]:
            report_lines.append(
                f"- {row['formula']}: stability={float(row['stability_score']):.3f}, "
                f"D@{int(float(row.get('reference_temperature_K', 800.0)))}K={float(row['diffusion_score']):.3e} m^2/s, "
                f"Ea={float(row['activation_barrier_ev']):.3f} eV, ΔV/V={float(row['volume_deformation']):.3f}, "
                f"mode={row['exploration_mode']}."
            )
            if row.get("md_summary_path"):
                report_lines.append(f"  MD summary: {row['md_summary_path']}")
            if row.get("volume_summary_path"):
                report_lines.append(f"  Volume summary: {row['volume_summary_path']}")
        report_lines.extend(
            [
                "",
                "## Recommended DFT queue",
                "",
                ", ".join(archive_payload["dft_queue"]) or "None",
                "",
                "## Recommended experiment queue",
                "",
                ", ".join(archive_payload["experiment_queue"]) or "None",
            ]
        )
        report_path = write_markdown(paths.stage2 / "stage2_full_report.md", "\n".join(report_lines))
        append_jsonl(
            paths.stage2 / "decision_log.jsonl",
            {"action": "export_stage2_report", "report_path": str(report_path)},
        )
        return {
            "pareto_archive_csv": str(paths.stage2 / "pareto_archive.csv"),
            "stage2_full_report": str(report_path),
            "recommended_dft_queue": str(paths.stage2 / "recommended_dft_queue.yaml"),
            "recommended_experiment_queue": str(paths.stage2 / "recommended_experiment_queue.yaml"),
        }

    return [
        load_stage1_prior,
        propose_stage2_candidates,
        evaluate_volume_deformation_with_mlff,
        run_mlff_md_batch,
        update_pareto_archive,
        export_stage2_report,
    ]
