from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Any


def _read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _cli_runner() -> list[str]:
    if shutil.which("mace_run_train"):
        return ["mace_run_train"]
    return [sys.executable, "-m", "mace.cli.run_train"]


def _append_flag(cmd: list[str], key: str, value: Any) -> None:
    if value in (None, "", [], {}):
        return
    flag = "--" + str(key).replace("_", "-")
    if isinstance(value, (list, tuple)):
        cmd.extend([flag, json.dumps(list(value))])
        return
    if isinstance(value, bool):
        cmd.extend([flag, "true" if value else "false"])
        return
    cmd.extend([flag, str(value)])


def _resolve_stage_path(stage_root: Path, raw: str | None, *, must_exist: bool = False) -> str | None:
    text = str(raw or "").strip()
    if not text:
        return None
    path = Path(text)
    if not path.is_absolute():
        path = stage_root / path
    resolved = path.resolve()
    if must_exist and not resolved.exists():
        raise FileNotFoundError(f"Missing staged path: {resolved}")
    if resolved.exists():
        return str(resolved)
    return None


def _collect_model_artifacts(root: Path) -> list[str]:
    out: list[str] = []
    for path in sorted(root.rglob("*")):
        if not path.is_file():
            continue
        if path.suffix.lower() in {".model", ".pt", ".pth", ".ckpt"}:
            out.append(str(path.relative_to(root)))
    return out


def run_training(dataset_root: Path, output_root: Path, params_path: Path) -> dict[str, Any]:
    stage_root = params_path.resolve().parents[1]
    if not dataset_root.is_absolute():
        dataset_root = (stage_root / dataset_root).resolve()
    else:
        dataset_root = dataset_root.resolve()
    if not output_root.is_absolute():
        output_root = (stage_root / output_root).resolve()
    else:
        output_root = output_root.resolve()
    if not params_path.is_absolute():
        params_path = (stage_root / params_path).resolve()
    else:
        params_path = params_path.resolve()
    params = _read_json(params_path)
    output_root.mkdir(parents=True, exist_ok=True)
    checkpoints_dir = output_root / "checkpoints"
    checkpoints_dir.mkdir(parents=True, exist_ok=True)

    train_file = dataset_root / str(params["train_file"])
    valid_value = params.get("valid_file")
    test_value = params.get("test_file")
    valid_file = dataset_root / str(valid_value) if valid_value else None
    test_file = dataset_root / str(test_value) if test_value else None

    cmd = _cli_runner()
    _append_flag(cmd, "name", params.get("model_name", "mace_finetune"))
    _append_flag(cmd, "train_file", train_file)
    _append_flag(cmd, "valid_file", valid_file if valid_file and valid_file.exists() else None)
    _append_flag(cmd, "test_file", test_file if test_file and test_file.exists() else None)
    _append_flag(cmd, "foundation_model", _resolve_stage_path(stage_root, params.get("foundation_model")))
    _append_flag(cmd, "foundation_head", params.get("foundation_head"))
    _append_flag(cmd, "E0s", _resolve_stage_path(stage_root, params.get("e0s")) or params.get("e0s"))
    _append_flag(cmd, "multiheads_finetuning", params.get("multiheads_finetuning"))
    _append_flag(cmd, "pt_train_file", _resolve_stage_path(stage_root, params.get("pt_train_file")) or params.get("pt_train_file"))
    _append_flag(cmd, "num_samples_pt", params.get("num_samples_pt"))
    _append_flag(cmd, "filter_type_pt", params.get("filter_type_pt"))
    _append_flag(cmd, "subselect_pt", params.get("subselect_pt"))
    _append_flag(cmd, "weight_pt", params.get("weight_pt"))
    _append_flag(cmd, "atomic_numbers", params.get("atomic_numbers"))
    _append_flag(cmd, "compute_stress", params.get("compute_stress"))
    _append_flag(cmd, "energy_weight", params.get("energy_weight"))
    _append_flag(cmd, "forces_weight", params.get("forces_weight"))
    _append_flag(cmd, "stress_weight", params.get("stress_weight"))
    _append_flag(cmd, "max_num_epochs", params.get("max_num_epochs"))
    _append_flag(cmd, "batch_size", params.get("batch_size"))
    _append_flag(cmd, "lr", params.get("learning_rate"))
    _append_flag(cmd, "default_dtype", params.get("default_dtype"))
    _append_flag(cmd, "device", params.get("device"))
    _append_flag(cmd, "seed", params.get("seed"))
    _append_flag(cmd, "checkpoints_dir", checkpoints_dir)
    _append_flag(cmd, "energy_key", "REF_energy")
    _append_flag(cmd, "forces_key", "REF_forces")
    _append_flag(cmd, "stress_key", "REF_stress")
    if params.get("restart_latest"):
        cmd.append("--restart_latest")

    for key, value in (params.get("extra_cli_args") or {}).items():
        _append_flag(cmd, str(key), value)

    completed = subprocess.run(cmd, check=True, cwd=str(output_root))
    model_artifacts = _collect_model_artifacts(output_root)
    summary = {
        "command": cmd,
        "returncode": int(completed.returncode),
        "train_file": str(train_file.relative_to(dataset_root)),
        "valid_file": str(valid_file.relative_to(dataset_root)) if valid_file and valid_file.exists() else None,
        "test_file": str(test_file.relative_to(dataset_root)) if test_file and test_file.exists() else None,
        "checkpoints_dir": str(checkpoints_dir.relative_to(output_root)),
        "model_artifacts": model_artifacts,
    }
    _write_json(output_root / "batch_summary.json", summary)
    return summary


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run MACE training/fine-tuning in a staged DPDispatcher task.")
    parser.add_argument("--dataset_root", required=True, help="Staged dataset directory")
    parser.add_argument("--output_root", required=True, help="Output directory for model artifacts")
    parser.add_argument("--params", required=True, help="Training params JSON path")
    args = parser.parse_args()

    summary = run_training(Path(args.dataset_root), Path(args.output_root), Path(args.params))
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    _cli()
