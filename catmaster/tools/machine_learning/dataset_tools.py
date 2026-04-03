from __future__ import annotations

import json
import xml.etree.ElementTree as ET
from collections import OrderedDict
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Optional

import numpy as np
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io.vasp import read_vasp_xml
from ase.io import write as ase_write
from pydantic import BaseModel, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath

def _error(message: str, *, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    lines = [str(message).strip()]
    if data:
        for key in ("result_root_rel", "output_dir_rel", "summary_json_rel"):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name="build_dataset_from_runs",
        public_message="\n".join(lines),
        artifact={"tool_name": "build_dataset_from_runs", "data": data or {}},
        error_code=error_code,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _discover_vaspruns(root: Path) -> list[Path]:
    if root.is_file() and root.name == "vasprun.xml":
        return [root]
    files = sorted(path for path in root.rglob("vasprun.xml") if path.is_file())
    return files


def _local_tag(tag: str) -> str:
    return tag.rsplit("}", 1)[-1]


def _parse_vasp_step_metadata(path: Path) -> tuple[int | None, list[int | None], list[float | None], str]:
    nelm = None
    electronic_step_counts: list[int | None] = []
    free_energies: list[float | None] = []
    warning = ""
    try:
        for _, elem in ET.iterparse(path, events=["end"]):
            tag = _local_tag(elem.tag)
            if nelm is None and tag == "i" and elem.attrib.get("name") == "NELM":
                text = (elem.text or "").strip()
                if text:
                    nelm = int(float(text))
            elif tag == "calculation":
                electronic_step_counts.append(sum(1 for child in elem if _local_tag(child.tag) == "scstep"))
                e_fr_energy = None
                for energy_node in elem:
                    if _local_tag(energy_node.tag) != "energy":
                        continue
                    for child in energy_node:
                        if _local_tag(child.tag) == "i" and child.attrib.get("name") == "e_fr_energy":
                            text = (child.text or "").strip()
                            if text:
                                try:
                                    e_fr_energy = float(text)
                                except Exception:
                                    e_fr_energy = None
                            break
                free_energies.append(e_fr_energy)
                elem.clear()
    except ET.ParseError as exc:
        warning = f"xml_parse_warning: {exc}"
    return nelm, electronic_step_counts, free_energies, warning


def _read_energy(atoms: Atoms) -> tuple[float | None, float | None]:
    calc = getattr(atoms, "calc", None)
    if calc is None:
        return None, None
    energy = calc.results.get("energy")
    free_energy = calc.results.get("free_energy")
    return (
        float(energy) if energy is not None else None,
        float(free_energy) if free_energy is not None else None,
    )


def _check_step_alignment(
    path: Path,
    images: list[Atoms],
    xml_free_energies: list[float | None],
    atol: float,
) -> str:
    if not xml_free_energies:
        return ""

    mismatches: list[str] = []
    n_shared = min(len(images), len(xml_free_energies))
    for index in range(n_shared):
        xml_free = xml_free_energies[index]
        if xml_free is None:
            continue
        _, ase_free = _read_energy(images[index])
        if ase_free is None:
            mismatches.append(
                f"step {index + 1}: XML has e_fr_energy={xml_free:.10f} but ASE frame has no free_energy"
            )
            continue
        if abs(float(ase_free) - xml_free) > atol:
            mismatches.append(
                f"step {index + 1}: XML e_fr_energy={xml_free:.10f}, "
                f"ASE free_energy={float(ase_free):.10f}"
            )
        if len(mismatches) >= 5:
            break

    if not mismatches:
        return ""
    joined = "; ".join(mismatches)
    return (
        f"Alignment check failed for {path.name}: {joined}. "
        "This suggests XML <calculation> ordering no longer matches ASE frames."
    )


def _guess_electronic_convergence(electronic_step_count: int | None, nelm: int | None) -> bool | None:
    if electronic_step_count is None or nelm is None:
        return None
    return electronic_step_count < nelm


def _attach_labels(
    *,
    atoms: Atoms,
    source_run_rel: str,
    frame_index: int,
    config_type: str,
    head_label: str | None,
    electronic_step_count: int | None,
    nelm: int | None,
    step_electronic_converged_guess: bool | None,
):
    atoms = atoms.copy()
    kwargs: dict[str, Any] = {}
    energy = None
    forces_array = None
    stress_voigt = None
    try:
        energy = float(atoms.get_potential_energy())
    except Exception:
        energy = None
    try:
        forces_array = np.asarray(atoms.get_forces(), dtype=float)
    except Exception:
        forces_array = None
    try:
        stress_voigt = np.asarray(atoms.get_stress(voigt=True), dtype=float)
    except Exception:
        stress_voigt = None
    if energy is not None:
        kwargs["energy"] = energy
        atoms.info["REF_energy"] = energy
    if forces_array is not None:
        kwargs["forces"] = forces_array
        atoms.arrays["REF_forces"] = forces_array
    if stress_voigt is not None:
        kwargs["stress"] = stress_voigt
        atoms.info["REF_stress"] = stress_voigt.tolist()
    if kwargs:
        atoms.calc = SinglePointCalculator(atoms, **kwargs)
    atoms.info["source_run_rel"] = source_run_rel
    atoms.info["frame_index"] = int(frame_index)
    atoms.info["ionic_step_index"] = int(frame_index)
    atoms.info["ionic_step_number"] = int(frame_index) + 1
    atoms.info["config_type"] = str(config_type)
    if electronic_step_count is not None:
        atoms.info["electronic_step_count"] = int(electronic_step_count)
    if nelm is not None:
        atoms.info["nelm"] = int(nelm)
    if step_electronic_converged_guess is not None:
        atoms.info["step_electronic_converged_guess"] = bool(step_electronic_converged_guess)
    if head_label:
        atoms.info["head"] = str(head_label)
    return atoms


class BuildDatasetFromRunsInput(BaseModel):
    """[ml/prepare] Build an extxyz training dataset from one VASP result directory or a batch of VASP result directories using ASE's vasprun.xml parser."""

    result_root: str = Field(..., description="Single VASP result directory, a batch root, or a vasprun.xml file.")
    output_dir: str = Field(..., description="Output dataset directory.")
    frame_mode: Literal["final", "all_ionic_steps"] = Field(
        "all_ionic_steps",
        description="Use only final relaxed frames or all ionic steps from each vasprun.xml.",
    )
    require_converged: bool = Field(
        False,
        description="If true, keep only frames whose heuristic step_electronic_converged_guess evaluates to True from vasprun.xml metadata.",
    )
    alignment_check: bool = Field(
        True,
        description="Validate XML step ordering against ASE free energies and skip malformed vasprun.xml files when alignment fails.",
    )
    alignment_energy_atol: float = Field(
        1e-6,
        gt=0.0,
        description="Absolute tolerance in eV for XML vs ASE free-energy alignment checks.",
    )
    split_unit: Literal["source_run", "frame"] = Field(
        "source_run",
        description="Split by source VASP run to avoid trajectory leakage, or by individual frame for deliberate frame-level splitting.",
    )
    config_type: str = Field("dft", description="Value written to atoms.info['config_type'] for exported frames.")
    head_label: Optional[str] = Field(
        None,
        description="Optional value written to atoms.info['head'], matching the validated reference export script convention.",
    )
    train_fraction: float = Field(0.8, ge=0.0, le=1.0, description="Fraction of frames assigned to train.extxyz.")
    valid_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of frames assigned to valid.extxyz.")
    test_fraction: float = Field(0.1, ge=0.0, le=1.0, description="Fraction of frames assigned to test.extxyz.")
    shuffle: bool = Field(True, description="Shuffle split units before splitting.")
    seed: int = Field(7, description="Random seed used when shuffle=true.")

    @model_validator(mode="after")
    def _validate_split(self) -> "BuildDatasetFromRunsInput":
        total = float(self.train_fraction + self.valid_fraction + self.test_fraction)
        if abs(total - 1.0) > 1e-8:
            raise ValueError("train_fraction + valid_fraction + test_fraction must sum to 1.0")
        return self


def _split_items(items: list[Any], *, params: BuildDatasetFromRunsInput) -> tuple[list[Any], list[Any], list[Any]]:
    if params.shuffle:
        rng = np.random.default_rng(params.seed)
        order = np.arange(len(items))
        rng.shuffle(order)
        items = [items[idx] for idx in order.tolist()]
    n = len(items)
    n_train = int(round(n * params.train_fraction))
    n_valid = int(round(n * params.valid_fraction))
    if n_train + n_valid > n:
        n_valid = max(0, n - n_train)
    n_test = n - n_train - n_valid
    train = items[:n_train]
    valid = items[n_train : n_train + n_valid]
    test = items[n_train + n_valid : n_train + n_valid + n_test]
    return train, valid, test


def _split_grouped_items(
    groups: list[tuple[str, list[Any]]], *, params: BuildDatasetFromRunsInput
) -> tuple[list[Any], list[Any], list[Any]]:
    group_items: list[tuple[str, list[Any]]] = list(groups)
    if params.shuffle:
        rng = np.random.default_rng(params.seed)
        order = np.arange(len(group_items))
        rng.shuffle(order)
        group_items = [group_items[idx] for idx in order.tolist()]
    train_groups, valid_groups, test_groups = _split_items(group_items, params=params)

    def _flatten(items: list[tuple[str, list[Any]]]) -> list[Any]:
        frames: list[Any] = []
        for _, members in items:
            frames.extend(members)
        return frames

    return _flatten(train_groups), _flatten(valid_groups), _flatten(test_groups)


def _write_extxyz(path: Path, frames: Iterable[Any]) -> None:
    frames = list(frames)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not frames:
        path.write_text("", encoding="utf-8")
        return
    ase_write(str(path), frames, format="extxyz")


def build_dataset_from_runs(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ml/prepare] Build an extxyz dataset with train/valid/test splits from VASP run outputs."""
    try:
        params = BuildDatasetFromRunsInput(**payload)
        result_root = resolve_workspace_path(params.result_root, must_exist=True)
        output_dir = resolve_workspace_path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        vaspruns = _discover_vaspruns(result_root)
        if not vaspruns:
            _error(
                "No vasprun.xml files found under result_root.",
                data={"result_root_rel": workspace_relpath(result_root)},
                error_code="no_vaspruns",
            )

        frame_records: list[dict[str, Any]] = []
        skipped: list[dict[str, Any]] = []
        metadata_warnings: list[dict[str, Any]] = []
        filtered_frames = 0
        for vasprun_path in vaspruns:
            try:
                ionic_frames = list(read_vasp_xml(str(vasprun_path), index=slice(None)))
            except Exception as exc:
                skipped.append({"vasprun_rel": workspace_relpath(vasprun_path), "reason": f"parse_failed: {exc}"})
                continue
            if not ionic_frames:
                skipped.append({"vasprun_rel": workspace_relpath(vasprun_path), "reason": "no_ionic_steps"})
                continue
            nelm, electronic_step_counts, xml_free_energies, metadata_warning = _parse_vasp_step_metadata(vasprun_path)
            if metadata_warning:
                metadata_warnings.append({"vasprun_rel": workspace_relpath(vasprun_path), "warning": metadata_warning})
                if params.alignment_check:
                    skipped.append({"vasprun_rel": workspace_relpath(vasprun_path), "reason": metadata_warning})
                    continue
            if electronic_step_counts:
                if len(electronic_step_counts) != len(ionic_frames):
                    skipped.append(
                        {
                            "vasprun_rel": workspace_relpath(vasprun_path),
                            "reason": (
                                "ionic_step_count_mismatch: "
                                f"metadata={len(electronic_step_counts)} ase={len(ionic_frames)}"
                            ),
                        }
                    )
                    continue
            else:
                electronic_step_counts = [None] * len(ionic_frames)
                xml_free_energies = [None] * len(ionic_frames)
            if params.alignment_check:
                alignment_error = _check_step_alignment(
                    vasprun_path,
                    ionic_frames,
                    xml_free_energies,
                    params.alignment_energy_atol,
                )
                if alignment_error:
                    skipped.append({"vasprun_rel": workspace_relpath(vasprun_path), "reason": alignment_error})
                    continue
            indices = [len(ionic_frames) - 1] if params.frame_mode == "final" else list(range(len(ionic_frames)))
            source_run_rel = workspace_relpath(vasprun_path.parent)
            for idx in indices:
                electronic_step_count = electronic_step_counts[idx] if idx < len(electronic_step_counts) else None
                converged_guess = _guess_electronic_convergence(electronic_step_count, nelm)
                if params.require_converged and converged_guess is not True:
                    filtered_frames += 1
                    continue
                frame_records.append(
                    {
                        "source_run_rel": source_run_rel,
                        "atoms": _attach_labels(
                            atoms=ionic_frames[idx],
                            source_run_rel=source_run_rel,
                            frame_index=idx,
                            config_type=params.config_type,
                            head_label=params.head_label,
                            electronic_step_count=electronic_step_count,
                            nelm=nelm,
                            step_electronic_converged_guess=converged_guess,
                        ),
                    }
                )

        if not frame_records:
            _error(
                "No frames were extracted from the supplied VASP results.",
                data={"result_root_rel": workspace_relpath(result_root)},
                error_code="no_frames",
            )

        dataset_path = output_dir / "dataset.extxyz"
        train_path = output_dir / "train.extxyz"
        valid_path = output_dir / "valid.extxyz"
        test_path = output_dir / "test.extxyz"
        frames = [record["atoms"] for record in frame_records]
        if params.split_unit == "source_run":
            grouped: "OrderedDict[str, list[Atoms]]" = OrderedDict()
            for record in frame_records:
                grouped.setdefault(str(record["source_run_rel"]), []).append(record["atoms"])
            train_frames, valid_frames, test_frames = _split_grouped_items(list(grouped.items()), params=params)
        else:
            train_frames, valid_frames, test_frames = _split_items(frames, params=params)

        _write_extxyz(dataset_path, frames)
        _write_extxyz(train_path, train_frames)
        _write_extxyz(valid_path, valid_frames)
        _write_extxyz(test_path, test_frames)

        summary_path = output_dir / "dataset_summary.json"
        summary = {
            "result_root_rel": workspace_relpath(result_root),
            "output_dir_rel": workspace_relpath(output_dir),
            "frame_mode": params.frame_mode,
            "require_converged": params.require_converged,
            "alignment_check": params.alignment_check,
            "alignment_energy_atol": params.alignment_energy_atol,
            "split_unit": params.split_unit,
            "vaspruns_found": len(vaspruns),
            "frames_written": len(frames),
            "train_frames": len(train_frames),
            "valid_frames": len(valid_frames),
            "test_frames": len(test_frames),
            "dataset_rel": workspace_relpath(dataset_path),
            "train_rel": workspace_relpath(train_path),
            "valid_rel": workspace_relpath(valid_path),
            "test_rel": workspace_relpath(test_path),
            "parser": "ase.io.vasp.read_vasp_xml",
            "convergence_filter_applied": bool(params.require_converged),
            "filtered_frames": filtered_frames,
            "config_type": params.config_type,
            "head_label": params.head_label,
            "skipped": skipped,
            "metadata_warnings": metadata_warnings,
        }
        _write_json(summary_path, summary)
        data = {
            "result_root_rel": workspace_relpath(result_root),
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(summary_path),
            "frames_written": len(frames),
            "dataset_rel": workspace_relpath(dataset_path),
            "train_rel": workspace_relpath(train_path),
            "valid_rel": workspace_relpath(valid_path),
            "test_rel": workspace_relpath(test_path),
            "skipped_runs": len(skipped),
        }
        content = (
            "build_dataset_from_runs completed.\n"
            f"result_root_rel={data['result_root_rel']} frames_written={data['frames_written']} "
            f"summary_json_rel={data['summary_json_rel']}"
        )
        return content, {"tool_name": "build_dataset_from_runs", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _error(f"build_dataset_from_runs failed: {exc}", error_code="build_dataset_from_runs_failed")


__all__ = ["BuildDatasetFromRunsInput", "build_dataset_from_runs"]
