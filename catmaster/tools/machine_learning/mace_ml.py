from __future__ import annotations

import json
import shlex
import shutil
from pathlib import Path
from typing import Any, Dict, Literal, Optional

import numpy as np
from ase import Atoms
from ase.io import iread as ase_iread
from ase.io import read as ase_read
from ase.io import write as ase_write
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import compact_list_for_artifact, resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    BatchDispatchRequest,
    TaskSpec,
    dispatch_submission,
    make_work_base,
    remote_context_from_exception,
    remote_context_from_result,
)
from catmaster.tools.execution.mace_dispatch import (
    _collect_mace_outputs,
    _resolve_machine_for_resources,
    _resolve_mace_head,
    _stage_mace_model,
    _write_batch_state,
)
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskRegistry


def _success(tool_name: str, *, content: str, data: dict[str, Any], warnings: list[str] | None = None, execution_time: float | None = None):
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    if execution_time is not None:
        artifact["execution_time"] = execution_time
    return content, artifact


def _fail(tool_name: str, *, message: str, data: dict[str, Any] | None = None, warnings: list[str] | None = None, error_code: str = "") -> None:
    lines = [str(message).strip()]
    if data:
        for key in (
            "dataset_dir_rel",
            "output_root_rel",
            "output_dir_rel",
            "batch_state_rel",
            "summary_json_rel",
            "csv_rel",
            "work_base",
        ):
            value = data.get(key)
            if value in (None, "", [], {}):
                continue
            lines.append(f"{key}={value}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(lines),
        artifact={"tool_name": tool_name, "data": data or {}, "warnings": warnings or []},
        error_code=error_code,
    )


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    import csv

    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return
    fieldnames: list[str] = []
    seen: set[str] = set()
    for row in rows:
        for key in row.keys():
            if key in seen:
                continue
            seen.add(key)
            fieldnames.append(key)
    with path.open("w", encoding="utf-8", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)


def _is_structure_file(path: Path) -> bool:
    if not path.is_file():
        return False
    if path.name in {"POSCAR", "CONTCAR"}:
        return True
    return path.suffix.lower() in {".vasp", ".poscar", ".cif", ".xyz", ".extxyz"}


def _collect_structure_files(root: Path) -> list[Path]:
    return sorted(path for path in root.rglob("*") if _is_structure_file(path))


def _read_dataset_frames(path: Path) -> list[Atoms]:
    if path.is_dir():
        candidates = [path / "dataset.extxyz", path / "train.extxyz", path / "valid.extxyz", path / "test.extxyz"]
        dataset_file = next((candidate for candidate in candidates if candidate.exists()), None)
        if dataset_file is None:
            raise FileNotFoundError(f"No extxyz dataset file found under {path}")
        path = dataset_file
    return list(ase_read(str(path), index=":"))


def _collect_dataset_atomic_numbers(dataset_dir: Path, split_files: list[str | None]) -> list[int]:
    atomic_numbers: set[int] = set()
    for filename in split_files:
        text = str(filename or "").strip()
        if not text:
            continue
        dataset_path = dataset_dir / text
        if not dataset_path.exists():
            continue
        try:
            for atoms in ase_iread(str(dataset_path), index=":"):
                atomic_numbers.update(int(number) for number in atoms.numbers)
        except Exception:
            continue
    return sorted(atomic_numbers)


def _feature_vector(atoms: Atoms, *, species_order: list[str], bins: int = 80, r_max: float = 6.0) -> np.ndarray:
    symbols = atoms.get_chemical_symbols()
    counts = np.array([symbols.count(symbol) for symbol in species_order], dtype=float)
    if counts.sum() > 0:
        counts /= counts.sum()
    cell = atoms.cell
    volume_per_atom = float(abs(cell.volume) / max(len(atoms), 1)) if cell is not None else 0.0
    distances = atoms.get_all_distances(mic=True)
    tri = distances[np.triu_indices(len(atoms), k=1)]
    tri = tri[(tri > 1e-8) & (tri <= r_max)]
    if tri.size:
        hist, _ = np.histogram(tri, bins=bins, range=(0.0, r_max), density=True)
    else:
        hist = np.zeros(bins, dtype=float)
    return np.concatenate([counts, np.array([volume_per_atom], dtype=float), hist.astype(float)])


def _committee_energies(frames: list[Atoms], model_paths: list[str], *, head: Optional[str], device: str) -> np.ndarray:
    if len(model_paths) < 2:
        return np.zeros(len(frames), dtype=float)
    try:
        from mace.calculators import MACECalculator, mace_mp
    except Exception as exc:
        raise RuntimeError(f"MACE calculators are unavailable locally: {exc}") from exc

    calculators = []
    for raw_model in model_paths:
        model_text = str(raw_model).strip()
        try:
            local_path = resolve_workspace_path(model_text, must_exist=True)
        except Exception:
            local_path = None
        if local_path is not None:
            try:
                calculators.append(MACECalculator(model_paths=[str(local_path)], device=device))
                continue
            except Exception:
                calculators.append(MACECalculator(model_paths=str(local_path), device=device))
                continue
        kwargs = {"model": model_text, "device": device}
        if head:
            kwargs["head"] = head
        calculators.append(mace_mp(**kwargs))

    energies = np.zeros((len(calculators), len(frames)), dtype=float)
    for calc_idx, calc in enumerate(calculators):
        for frame_idx, atoms in enumerate(frames):
            sample = atoms.copy()
            sample.calc = calc
            energies[calc_idx, frame_idx] = float(sample.get_potential_energy()) / max(len(sample), 1)
    return np.std(energies, axis=0)


def _greedy_select(features: np.ndarray, *, selection_size: int, disagreement: np.ndarray | None = None) -> list[dict[str, Any]]:
    n = len(features)
    if n == 0:
        return []
    distances = np.linalg.norm(features[:, None, :] - features[None, :, :], axis=2)
    disagreement = np.asarray(disagreement if disagreement is not None else np.zeros(n), dtype=float)
    if np.ptp(disagreement) > 0:
        disagreement_norm = (disagreement - disagreement.min()) / np.ptp(disagreement)
    else:
        disagreement_norm = np.zeros(n, dtype=float)

    selected: list[int] = []
    ranking: list[dict[str, Any]] = []
    remaining = set(range(n))
    mean_feature = np.mean(features, axis=0)
    first = int(np.argmax(np.linalg.norm(features - mean_feature[None, :], axis=1) + disagreement_norm))
    while remaining and len(selected) < min(selection_size, n):
        if not selected:
            current = first
            min_distance = float(np.max(distances[first]))
        else:
            candidates = sorted(remaining)
            best_score = None
            current = candidates[0]
            min_distance = 0.0
            for idx in candidates:
                diversity = float(np.min(distances[idx, selected]))
                score = diversity + disagreement_norm[idx]
                if best_score is None or score > best_score:
                    best_score = score
                    current = idx
                    min_distance = diversity
        remaining.remove(current)
        selected.append(current)
        ranking.append(
            {
                "selection_rank": len(selected),
                "candidate_index": current,
                "diversity_score": float(min_distance),
                "disagreement_score": float(disagreement[current]) if disagreement is not None else 0.0,
                "combined_score": float(min_distance + disagreement_norm[current]),
            }
        )
    return ranking


class MaceTrainInput(BaseModel):
    """[ml/execute] Launch a remote MACE training or fine-tuning job using the validated reference-script style MACE CLI contract."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str = Field(..., description="Dataset directory containing train.extxyz and optional valid/test splits.")
    output_root: str = Field(..., description="Output directory where trained model artifacts and logs will be collected.")
    train_file: str = Field("train.extxyz", description="Training dataset filename relative to dataset_dir.")
    valid_file: Optional[str] = Field("valid.extxyz", description="Validation dataset filename relative to dataset_dir.")
    test_file: Optional[str] = Field("test.extxyz", description="Optional test dataset filename relative to dataset_dir.")
    name: str = Field("mace_finetune", description="Run name passed unchanged to the MACE --name flag.")
    foundation_model: Optional[str] = Field(
        "mh-1",
        description=(
            "Pretrained model name or workspace-local model path used as the MACE foundation model. "
            "Defaults to the validated repo baseline `mh-1`."
        ),
    )
    foundation_head: Optional[str] = Field(
        "omat_pbe",
        description="Foundation-model head for multi-head models. Use empty string to disable.",
    )
    E0s: str = Field(
        "estimated",
        description="Value passed unchanged to the MACE --E0s flag: 'estimated' or a workspace-local E0 JSON file.",
    )
    multiheads_finetuning: bool = Field(
        True,
        description="Whether to use the multihead replay-style finetuning path validated in the reference scripts.",
    )
    pt_train_file: Optional[str] = Field(
        "omat",
        description="Replay dataset shortcut or path passed to --pt_train_file for replay finetuning.",
    )
    num_samples_pt: int = Field(50000, ge=0, description="Replay sample count passed to --num_samples_pt.")
    filter_type_pt: str = Field("combinations", description="Replay filtering mode passed to --filter_type_pt.")
    subselect_pt: str = Field("fps", description="Replay subselect mode passed to --subselect_pt.")
    weight_pt_head: float = Field(1.0, ge=0.0, description="Replay-head weight passed to --weight_pt_head.")
    atomic_numbers: list[int] = Field(
        default_factory=list,
        description="Explicit atomic numbers list passed to --atomic_numbers for replay finetuning.",
    )
    compute_stress: bool = Field(True, description="Whether to train with stress labels, passed to --compute_stress.")
    energy_weight: float = Field(1.0, ge=0.0, description="Energy loss weight.")
    forces_weight: float = Field(10.0, ge=0.0, description="Forces loss weight.")
    stress_weight: float = Field(1.0, ge=0.0, description="Stress loss weight.")
    max_num_epochs: int = Field(25, ge=1, description="Maximum training epochs passed to the MACE training CLI.")
    batch_size: int = Field(4, ge=1, description="Batch size passed to the MACE training CLI.")
    lr: float = Field(1e-4, gt=0.0, description="Learning rate passed unchanged to the MACE --lr flag.")
    default_dtype: Literal["float32", "float64"] = Field("float32", description="Torch dtype used for training.")
    device: str = Field("cuda", description="Training device passed to the MACE CLI, typically cuda.")
    seed: int = Field(42, description="Random seed passed to the MACE CLI.")
    restart_latest: bool = Field(True, description="Whether to pass --restart_latest for resumable training runs.")
    weight_decay: float | None = Field(None, ge=0.0, description="Optional official MACE CLI optimizer weight decay.")
    scheduler: str | None = Field(None, description="Optional official MACE CLI scheduler name.")
    patience: int | None = Field(None, ge=0, description="Optional official MACE CLI early-stopping patience.")
    eval_interval: int | None = Field(None, ge=1, description="Optional official MACE CLI eval interval.")
    valid_batch_size: int | None = Field(None, ge=1, description="Optional official MACE CLI validation batch size.")
    save_all_checkpoints: bool | None = Field(None, description="Optional official MACE CLI save_all_checkpoints flag.")
    keep_checkpoints: bool | None = Field(None, description="Optional official MACE CLI keep_checkpoints flag.")
    cli_args: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Additional official MACE CLI flags keyed by the exact long-option name without leading '--'. "
            "Use underscores exactly as the CLI defines them; aliases and hyphen-to-underscore rewriting are not applied. "
            "Workspace-local file/dir values are automatically staged and rewritten for the remote job."
        ),
    )
    check_interval: int = Field(30, description="Polling interval in seconds while waiting for DPDispatcher.")


class MaceEvaluateInput(BaseModel):
    """[ml/analysis] Run remote MACE evaluation on an extxyz dataset and collect error metrics and per-configuration outputs."""

    model_config = ConfigDict(extra="forbid")

    dataset_dir: str = Field(..., description="Dataset directory containing the extxyz file to evaluate.")
    output_root: str = Field(..., description="Output directory where evaluation metrics and plots will be collected.")
    model: str = Field(..., description="Pretrained model name or workspace-local model path to evaluate.")
    dataset_file: str = Field("test.extxyz", description="Dataset filename relative to dataset_dir.")
    head: Optional[str] = Field("omat_pbe", description="Model head for multi-head models. Use empty string to disable.")
    default_dtype: Literal["float32", "float64"] = Field("float32", description="Torch dtype used by the evaluator.")
    device: str = Field("cuda", description="Device used by the evaluator, e.g. cuda or cpu.")
    check_interval: int = Field(30, description="Polling interval in seconds while waiting for DPDispatcher.")


class CalculateALCandidatesInput(BaseModel):
    """[ml/analysis] Rank candidate structures for active learning using diversity and optional MACE committee disagreement."""

    structure_dir: Optional[str] = Field(
        None,
        description="Directory containing candidate structures. Provide either structure_dir or dataset_path.",
    )
    dataset_path: Optional[str] = Field(
        None,
        description="extxyz dataset file or dataset directory. Provide either structure_dir or dataset_path.",
    )
    output_dir: str = Field(..., description="Directory to write candidate ranking artifacts.")
    selection_size: int = Field(20, ge=1, description="Number of candidates to mark as selected.")
    model_paths: list[str] = Field(
        default_factory=list,
        description="Optional committee of model names or workspace-local model paths used for disagreement scoring.",
    )
    head: Optional[str] = Field("omat_pbe", description="Model head used for pretrained committee models.")
    device: str = Field("cpu", description="Device used for local committee scoring, e.g. cpu or cuda.")

    @model_validator(mode="after")
    def _validate_source(self) -> "CalculateALCandidatesInput":
        if bool(self.structure_dir) == bool(self.dataset_path):
            raise ValueError("Provide exactly one of structure_dir or dataset_path.")
        return self


def _stage_remote_script(script_name: str, stage_root: Path) -> Path:
    script_src = Path(__file__).resolve().parents[2] / "remote" / "gpu" / script_name
    if not script_src.is_file():
        raise FileNotFoundError(f"Missing ML remote script: {script_src}")
    script_dst = stage_root / "task_script" / script_name
    script_dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, script_dst)
    return script_dst


def _stage_optional_input_file(raw_value: str | None, *, stage_root: Path, subdir: str) -> str | None:
    text = str(raw_value or "").strip()
    if not text:
        return None
    try:
        src = resolve_workspace_path(text, must_exist=True)
    except Exception:
        return text
    if not src.is_file():
        raise ValueError(f"Expected a file path for {text!r}, got {src}")
    dest_root = stage_root / "assets" / subdir
    dest_root.mkdir(parents=True, exist_ok=True)
    dest = dest_root / src.name
    shutil.copy2(src, dest)
    return dest.relative_to(stage_root).as_posix()


def _stage_optional_input_value(raw_value: Any, *, stage_root: Path, subdir: str) -> Any:
    if isinstance(raw_value, dict):
        return {
            str(key): _stage_optional_input_value(value, stage_root=stage_root, subdir=subdir)
            for key, value in raw_value.items()
        }
    if isinstance(raw_value, tuple):
        return [_stage_optional_input_value(value, stage_root=stage_root, subdir=subdir) for value in raw_value]
    if isinstance(raw_value, list):
        return [_stage_optional_input_value(value, stage_root=stage_root, subdir=subdir) for value in raw_value]
    if raw_value is None or isinstance(raw_value, (bool, int, float)):
        return raw_value

    text = str(raw_value).strip()
    if not text:
        return raw_value
    try:
        src = resolve_workspace_path(text, must_exist=True)
    except Exception:
        return raw_value

    rel_src = Path(workspace_relpath(src))
    dest_root = stage_root / "assets" / subdir
    dest = dest_root / rel_src
    dest.parent.mkdir(parents=True, exist_ok=True)
    if src.is_file():
        shutil.copy2(src, dest)
    elif src.is_dir():
        if dest.exists():
            shutil.rmtree(dest)
        shutil.copytree(src, dest)
    else:
        raise ValueError(f"Unsupported staged input path type: {src}")
    return dest.relative_to(stage_root).as_posix()


def mace_train(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ml/execute] Submit a MACE training or fine-tuning job through DPDispatcher."""
    params = MaceTrainInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("mace_train_dir")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("mace_train_dir missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    dataset_dir = resolve_workspace_path(params.dataset_dir, must_exist=True)
    if not dataset_dir.is_dir():
        _fail("mace_train", message=f"dataset_dir is not a directory: {dataset_dir}", error_code="invalid_dataset_dir")
    output_root = resolve_workspace_path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    work_base = make_work_base("mace_train")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_dataset = stage_root / "dataset"
    stage_output = stage_root / "output"
    stage_output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_dir, stage_dataset)
    _stage_remote_script("mace_train.py", stage_root)

    foundation_head = _resolve_mace_head(params.foundation_head)
    foundation_model = None
    model_spec = None
    if params.foundation_model:
        model_spec = _stage_mace_model(params.foundation_model, stage_root=stage_root)
        foundation_model = model_spec.command_arg
    e0s_arg = _stage_optional_input_value(params.E0s, stage_root=stage_root, subdir="e0s")
    pt_train_file_arg = _stage_optional_input_value(params.pt_train_file, stage_root=stage_root, subdir="replay")
    cli_args = _stage_optional_input_value(params.cli_args, stage_root=stage_root, subdir="cli_args")
    atomic_numbers = list(params.atomic_numbers or [])

    for filename in [params.train_file, params.valid_file, params.test_file]:
        if filename and not (stage_dataset / filename).exists():
            if filename == params.valid_file and params.valid_file == "valid.extxyz":
                continue
            if filename == params.test_file and params.test_file == "test.extxyz":
                continue
            _fail(
                "mace_train",
                message=f"Dataset split file missing under dataset_dir: {filename}",
                data={"dataset_dir_rel": workspace_relpath(dataset_dir)},
                error_code="missing_dataset_split",
            )

    if (
        params.multiheads_finetuning
        and str(params.filter_type_pt or "").strip().lower() not in {"", "none"}
        and not atomic_numbers
    ):
        atomic_numbers = _collect_dataset_atomic_numbers(
            stage_dataset,
            [params.train_file, params.valid_file, params.test_file],
        )
        if not atomic_numbers:
            _fail(
                "mace_train",
                message=(
                    "Replay finetuning with filter_type_pt requires atomic_numbers, "
                    "and none could be inferred from the dataset extxyz splits."
                ),
                data={"dataset_dir_rel": workspace_relpath(dataset_dir)},
                error_code="missing_atomic_numbers",
            )

    params_dir = stage_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    params_path = params_dir / "train_params.json"
    _write_json(
        params_path,
        {
            "name": params.name,
            "train_file": params.train_file,
            "valid_file": params.valid_file if (stage_dataset / str(params.valid_file or "")).exists() else None,
            "test_file": params.test_file if (stage_dataset / str(params.test_file or "")).exists() else None,
            "foundation_model": foundation_model,
            "foundation_head": foundation_head,
            "E0s": e0s_arg,
            "multiheads_finetuning": params.multiheads_finetuning,
            "pt_train_file": pt_train_file_arg,
            "num_samples_pt": params.num_samples_pt,
            "filter_type_pt": params.filter_type_pt,
            "subselect_pt": params.subselect_pt,
            "weight_pt_head": params.weight_pt_head,
            "atomic_numbers": atomic_numbers,
            "compute_stress": params.compute_stress,
            "energy_weight": params.energy_weight,
            "forces_weight": params.forces_weight,
            "stress_weight": params.stress_weight,
            "max_num_epochs": params.max_num_epochs,
            "batch_size": params.batch_size,
            "lr": params.lr,
            "weight_decay": params.weight_decay,
            "scheduler": params.scheduler,
            "patience": params.patience,
            "eval_interval": params.eval_interval,
            "valid_batch_size": params.valid_batch_size,
            "save_all_checkpoints": params.save_all_checkpoints,
            "keep_checkpoints": params.keep_checkpoints,
            "default_dtype": params.default_dtype,
            "device": params.device,
            "seed": params.seed,
            "restart_latest": params.restart_latest,
            "cli_args": cli_args,
        },
    )
    ctx = {
        "dataset_root": "dataset",
        "output_root": "output",
        "params": "params/train_params.json",
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    if model_spec and model_spec.asset_dir_rel and model_spec.asset_dir_rel not in rendered["forward_files"]:
        rendered["forward_files"].append(model_spec.asset_dir_rel)
    if (
        (isinstance(e0s_arg, str) and e0s_arg.startswith("assets/"))
        or (isinstance(pt_train_file_arg, str) and pt_train_file_arg.startswith("assets/"))
        or ("assets/" in json.dumps(cli_args, ensure_ascii=False))
    ) and "assets" not in rendered["forward_files"]:
        rendered["forward_files"].append("assets")
    task = TaskSpec(
        command=rendered["command"],
        task_work_path=".",
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
    )
    batch_req = BatchDispatchRequest(
        machine=machine,
        resources=resources_key,
        work_base=work_base,
        local_root=str(output_root),
        tasks=[task],
        clean_remote=False,
        check_interval=params.check_interval,
        tool_name="mace_train",
    )
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": 1})
    dispatch_error = None
    result = None
    collect_info: dict[str, Any] = {}
    collect_warnings: list[str] = []
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": 1})
        result = dispatch_submission(batch_req)
    except Exception as exc:
        dispatch_error = exc
    finally:
        collect_info, collect_warnings = _collect_mace_outputs(stage_root, stage_output, output_root)
        _write_batch_state(
            output_root,
            work_base=work_base,
            state="collected_partial" if dispatch_error else "collected_complete",
            details={"tasks": 1, "errors": [str(dispatch_error)] if dispatch_error else []},
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")
    if dispatch_error is not None:
        _fail(
            "mace_train",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            data={
                "dataset_dir_rel": workspace_relpath(dataset_dir),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                "work_base": work_base,
                **remote_context_from_exception(dispatch_error),
                **collect_info,
            },
            warnings=collect_warnings,
            error_code="dispatch_failed",
        )
    states = result.task_states if result else []
    data = {
        "dataset_dir_rel": workspace_relpath(dataset_dir),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "work_base": work_base,
        "name": params.name,
        "foundation_model": params.foundation_model,
        "foundation_head": foundation_head,
        "E0s": params.E0s,
        "multiheads_finetuning": params.multiheads_finetuning,
        "atomic_numbers": atomic_numbers,
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            max_inline=20,
        ),
        **remote_context_from_result(result),
        **collect_info,
    }
    content = (
        "mace_train completed.\n"
        f"dataset_dir_rel={data['dataset_dir_rel']} output_root_rel={data['output_root_rel']} "
        f"batch_state_rel={data['batch_state_rel']}"
    )
    return _success("mace_train", content=content, data=data, warnings=collect_warnings, execution_time=result.duration_s if result else None)


def mace_evaluate(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ml/analysis] Submit a remote MACE evaluation job for an extxyz dataset through DPDispatcher."""
    params = MaceEvaluateInput(**payload)
    reg = TaskRegistry()
    cfg = reg.get("mace_eval_dir")
    resources_key = cfg.resources
    if not resources_key:
        raise KeyError("mace_eval_dir missing resources in task config")
    machine = _resolve_machine_for_resources(resources_key)
    dataset_dir = resolve_workspace_path(params.dataset_dir, must_exist=True)
    if not dataset_dir.is_dir():
        _fail("mace_evaluate", message=f"dataset_dir is not a directory: {dataset_dir}", error_code="invalid_dataset_dir")
    if not (dataset_dir / params.dataset_file).exists():
        _fail(
            "mace_evaluate",
            message=f"dataset_file does not exist under dataset_dir: {params.dataset_file}",
            data={"dataset_dir_rel": workspace_relpath(dataset_dir)},
            error_code="missing_dataset_file",
        )
    output_root = resolve_workspace_path(params.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    work_base = make_work_base("mace_eval")
    stage_root = output_root / work_base
    if stage_root.exists():
        shutil.rmtree(stage_root)
    stage_dataset = stage_root / "dataset"
    stage_output = stage_root / "output"
    stage_output.mkdir(parents=True, exist_ok=True)
    shutil.copytree(dataset_dir, stage_dataset)
    _stage_remote_script("mace_evaluate.py", stage_root)

    model_spec = _stage_mace_model(params.model, stage_root=stage_root)
    params_dir = stage_root / "params"
    params_dir.mkdir(parents=True, exist_ok=True)
    _write_json(
        params_dir / "eval_params.json",
        {
            "dataset_file": params.dataset_file,
            "model": model_spec.command_arg,
            "head": _resolve_mace_head(params.head),
            "default_dtype": params.default_dtype,
            "device": params.device,
        },
    )
    ctx = {
        "dataset_root": "dataset",
        "output_root": "output",
        "params": "params/eval_params.json",
    }
    rendered = render_task_fields(cfg, ctx, stage_root)
    if model_spec.asset_dir_rel and model_spec.asset_dir_rel not in rendered["forward_files"]:
        rendered["forward_files"].append(model_spec.asset_dir_rel)
    task = TaskSpec(
        command=rendered["command"],
        task_work_path=".",
        forward_files=rendered["forward_files"],
        backward_files=rendered["backward_files"],
    )
    batch_req = BatchDispatchRequest(
        machine=machine,
        resources=resources_key,
        work_base=work_base,
        local_root=str(output_root),
        tasks=[task],
        clean_remote=False,
        check_interval=params.check_interval,
        tool_name="mace_evaluate",
    )
    state_path = _write_batch_state(output_root, work_base=work_base, state="submitted", details={"tasks": 1})
    dispatch_error = None
    result = None
    collect_info: dict[str, Any] = {}
    collect_warnings: list[str] = []
    try:
        _write_batch_state(output_root, work_base=work_base, state="running", details={"tasks": 1})
        result = dispatch_submission(batch_req)
    except Exception as exc:
        dispatch_error = exc
    finally:
        collect_info, collect_warnings = _collect_mace_outputs(stage_root, stage_output, output_root)
        _write_batch_state(
            output_root,
            work_base=work_base,
            state="collected_partial" if dispatch_error else "collected_complete",
            details={"tasks": 1, "errors": [str(dispatch_error)] if dispatch_error else []},
        )
        if stage_root.exists():
            try:
                shutil.rmtree(stage_root)
            except Exception as exc:
                collect_warnings.append(f"staging cleanup failed: {type(exc).__name__}: {exc}")
    if dispatch_error is not None:
        _fail(
            "mace_evaluate",
            message=f"DPDispatcher submission failed: {dispatch_error}",
            data={
                "dataset_dir_rel": workspace_relpath(dataset_dir),
                "output_root_rel": workspace_relpath(output_root),
                "batch_state_rel": workspace_relpath(state_path),
                "work_base": work_base,
                **remote_context_from_exception(dispatch_error),
                **collect_info,
            },
            warnings=collect_warnings,
            error_code="dispatch_failed",
        )
    states = result.task_states if result else []
    data = {
        "dataset_dir_rel": workspace_relpath(dataset_dir),
        "output_root_rel": workspace_relpath(output_root),
        "batch_state_rel": workspace_relpath(state_path),
        "work_base": work_base,
        "model": params.model,
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        **compact_list_for_artifact(
            states,
            count_key="task_states_count",
            inline_key="task_states",
            preview_key="task_states_preview",
            truncated_key="task_states_truncated",
            max_inline=20,
        ),
        **remote_context_from_result(result),
        **collect_info,
    }
    content = (
        "mace_evaluate completed.\n"
        f"dataset_dir_rel={data['dataset_dir_rel']} output_root_rel={data['output_root_rel']} "
        f"batch_state_rel={data['batch_state_rel']}"
    )
    return _success("mace_evaluate", content=content, data=data, warnings=collect_warnings, execution_time=result.duration_s if result else None)


def calculate_al_candidates(payload: Dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[ml/analysis] Rank candidate structures for active learning using diversity and optional committee disagreement."""
    try:
        params = CalculateALCandidatesInput(**payload)
        output_dir = resolve_workspace_path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)

        frames: list[Atoms] = []
        source_records: list[dict[str, Any]] = []
        if params.structure_dir:
            structure_dir = resolve_workspace_path(params.structure_dir, must_exist=True)
            structure_files = _collect_structure_files(structure_dir)
            if not structure_files:
                _fail(
                    "calculate_al_candidates",
                    message="No candidate structure files found under structure_dir.",
                    data={"output_dir_rel": workspace_relpath(output_dir)},
                    error_code="no_structures",
                )
            for path in structure_files:
                atoms = ase_read(str(path))
                frames.append(atoms)
                source_records.append({"source_rel": workspace_relpath(path)})
        else:
            dataset_path = resolve_workspace_path(params.dataset_path, must_exist=True)
            dataset_frames = _read_dataset_frames(dataset_path)
            if not dataset_frames:
                _fail(
                    "calculate_al_candidates",
                    message="No frames found in dataset_path.",
                    data={"output_dir_rel": workspace_relpath(output_dir)},
                    error_code="no_dataset_frames",
                )
            frames.extend(dataset_frames)
            for idx in range(len(dataset_frames)):
                source_records.append({"source_rel": workspace_relpath(dataset_path), "frame_index": idx})

        species_order = sorted({symbol for atoms in frames for symbol in atoms.get_chemical_symbols()})
        features = np.vstack([_feature_vector(atoms, species_order=species_order) for atoms in frames])
        disagreement = None
        if len(params.model_paths) >= 2:
            disagreement = _committee_energies(frames, params.model_paths, head=_resolve_mace_head(params.head), device=params.device)

        ranking = _greedy_select(features, selection_size=params.selection_size, disagreement=disagreement)
        rows: list[dict[str, Any]] = []
        selected_indices = [entry["candidate_index"] for entry in ranking]
        selected_frames = [frames[idx] for idx in selected_indices]
        for entry in ranking:
            idx = int(entry["candidate_index"])
            row = {
                **source_records[idx],
                **entry,
                "natoms": len(frames[idx]),
                "formula": frames[idx].get_chemical_formula(),
            }
            rows.append(row)

        json_path = output_dir / "al_candidates.json"
        csv_path = output_dir / "al_candidates.csv"
        selected_dataset = output_dir / "selected.extxyz"
        _write_json(
            json_path,
            {
                "selection_size": params.selection_size,
                "committee_size": len(params.model_paths),
                "selected": rows,
            },
        )
        _write_csv(csv_path, rows)
        if selected_frames:
            ase_write(str(selected_dataset), selected_frames, format="extxyz")
        if params.structure_dir:
            selected_root = output_dir / "selected_structures"
            selected_root.mkdir(parents=True, exist_ok=True)
            structure_dir = resolve_workspace_path(params.structure_dir, must_exist=True)
            structure_files = _collect_structure_files(structure_dir)
            for rank, idx in enumerate(selected_indices, start=1):
                src = structure_files[idx]
                dst = selected_root / f"{rank:03d}_{src.name}"
                shutil.copy2(src, dst)
        data = {
            "output_dir_rel": workspace_relpath(output_dir),
            "summary_json_rel": workspace_relpath(json_path),
            "csv_rel": workspace_relpath(csv_path),
            "selected_count": len(rows),
            "used_committee": len(params.model_paths) >= 2,
        }
        content = (
            "calculate_al_candidates completed.\n"
            f"output_dir_rel={data['output_dir_rel']} selected_count={data['selected_count']} "
            f"summary_json_rel={data['summary_json_rel']}"
        )
        return content, {"tool_name": "calculate_al_candidates", "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail("calculate_al_candidates", message=f"calculate_al_candidates failed: {exc}", error_code="calculate_al_candidates_failed")


__all__ = [
    "MaceTrainInput",
    "MaceEvaluateInput",
    "CalculateALCandidatesInput",
    "mace_train",
    "mace_evaluate",
    "calculate_al_candidates",
]
