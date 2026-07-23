from __future__ import annotations

import json
import os
import re
from dataclasses import dataclass
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any, Dict, Iterable, List

VALID_UMA_TASKS = {"omat", "omol", "oc20", "oc22", "oc25", "odac", "omc"}
_NON_OMOL_TASKS = VALID_UMA_TASKS - {"omol"}
_STRUCTURE_SUFFIXES = {".xyz", ".extxyz", ".vasp", ".poscar", ".cif"}
_SKIP_PREFIXES = (
    "mace_batch_",
    "mace_sp_batch_",
    "mace_md_batch_",
    "uma_batch_",
    "uma_sp_batch_",
    "uma_relax_batch_",
    "vasp_batch_",
)
_INTERNAL_DIRS = {"metadata", ".catmaster", "__pycache__"}


@dataclass(frozen=True)
class UmaItemConfig:
    uma_task: str
    charge: int
    spin: int


def parse_bool(value: str | bool) -> bool:
    if isinstance(value, bool):
        return value
    text = str(value).strip().lower()
    if text in {"1", "true", "t", "yes", "y", "on"}:
        return True
    if text in {"0", "false", "f", "no", "n", "off"}:
        return False
    raise ValueError(f"Invalid boolean value: {value!r}")


def normalize_uma_task(value: Any) -> str:
    task = str(value or "omat").strip().lower() or "omat"
    if task not in VALID_UMA_TASKS:
        raise ValueError(
            "uma_task must be one of: " + ", ".join(sorted(VALID_UMA_TASKS))
        )
    return task


def resolve_device(preference: str) -> str:
    import torch

    device = str(preference or "auto").strip().lower() or "auto"
    if device == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if device.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError(
            "CUDA device was requested but torch.cuda.is_available() is false; "
            "check the remote driver/CUDA/PyTorch UMA environment."
        )
    return device


def collect_structure_files(root: Path) -> List[Path]:
    files: List[Path] = []
    for dirpath, dirnames, filenames in os.walk(root):
        path = Path(dirpath)
        if any(part.startswith(_SKIP_PREFIXES) for part in path.parts):
            dirnames[:] = []
            continue
        if any(part in _INTERNAL_DIRS for part in path.parts):
            dirnames[:] = []
            continue
        if "summary.json" in filenames:
            dirnames[:] = []
            continue
        dirnames[:] = [
            d for d in dirnames
            if d not in _INTERNAL_DIRS and not d.startswith(_SKIP_PREFIXES)
        ]
        for fname in filenames:
            p = path / fname
            if fname in {"POSCAR", "CONTCAR"}:
                files.append(p)
                continue
            if p.suffix.lower() in _STRUCTURE_SUFFIXES:
                files.append(p)
    return sorted(files, key=lambda p: str(p))


def is_within(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
        return True
    except ValueError:
        return False


def has_periodic_cell(atoms: Any) -> bool:
    try:
        volume = float(getattr(atoms.cell, "volume", 0.0) or 0.0)
    except Exception:
        volume = 0.0
    try:
        pbc_any = bool(any(bool(x) for x in atoms.pbc))
    except Exception:
        pbc_any = bool(getattr(atoms, "pbc", False))
    return pbc_any and volume > 1e-6


def fairchem_version() -> str:
    for name in ("fairchem-core", "fairchem"):
        try:
            return importlib_metadata.version(name)
        except Exception:
            continue
    return "unknown"


def load_metadata(metadata_path: str | None) -> dict[str, Any]:
    raw = str(metadata_path or "").strip()
    if not raw or raw == "__none__":
        return {}
    path = Path(raw)
    if not path.is_file():
        raise FileNotFoundError(f"UMA metadata file does not exist: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict):
        raise ValueError("UMA metadata must be a JSON object.")
    unknown = sorted(str(key) for key in data if key not in {"defaults", "items"})
    if unknown:
        raise ValueError(f"Unknown UMA metadata key(s): {', '.join(unknown)}")
    defaults = data.get("defaults")
    if defaults is not None:
        if not isinstance(defaults, dict):
            raise ValueError("UMA metadata.defaults must be an object.")
        _validate_item_config_keys(defaults, path="UMA metadata.defaults")
    items = data.get("items")
    if items is not None:
        if not isinstance(items, dict):
            raise ValueError("UMA metadata.items must be an object keyed by staged structure path.")
        for item_name, item in items.items():
            if not isinstance(item, dict):
                raise ValueError(f"UMA metadata.items[{item_name!r}] must be an object.")
            _validate_item_config_keys(item, path=f"UMA metadata.items[{item_name!r}]")
    return data


def _validate_item_config_keys(data: dict[str, Any], *, path: str) -> None:
    unknown = sorted(str(key) for key in data if key not in {"uma_task", "charge", "spin"})
    if unknown:
        raise ValueError(f"Unknown {path} key(s): {', '.join(unknown)}")


def _coerce_item_config(data: dict[str, Any], *, fallback: UmaItemConfig) -> UmaItemConfig:
    task = data.get("uma_task", fallback.uma_task)
    charge = data.get("charge", fallback.charge)
    spin = data.get("spin", fallback.spin)
    return UmaItemConfig(
        uma_task=normalize_uma_task(task),
        charge=int(charge),
        spin=int(spin),
    )


def resolve_item_config(
    *,
    structure_path: Path,
    input_root: Path,
    metadata: dict[str, Any],
    default_task: str,
    default_charge: int,
    default_spin: int,
) -> UmaItemConfig:
    cfg = UmaItemConfig(
        uma_task=normalize_uma_task(default_task),
        charge=int(default_charge),
        spin=int(default_spin),
    )
    defaults = metadata.get("defaults")
    if isinstance(defaults, dict):
        cfg = _coerce_item_config(defaults, fallback=cfg)

    items = metadata.get("items")
    if isinstance(items, dict):
        rel = structure_path.relative_to(input_root).as_posix()
        candidates = (rel, structure_path.name, Path(rel).with_suffix("").as_posix())
        for key in candidates:
            item = items.get(key)
            if isinstance(item, dict):
                cfg = _coerce_item_config(item, fallback=cfg)
                break
    return cfg


def apply_charge_spin(atoms: Any, cfg: UmaItemConfig) -> None:
    if cfg.uma_task == "omol" and cfg.spin < 1:
        raise ValueError("UMA task 'omol' requires multiplicity-style spin >= 1.")
    if cfg.uma_task in _NON_OMOL_TASKS and (cfg.charge != 0 or cfg.spin != 0):
        raise ValueError(
            f"UMA task {cfg.uma_task!r} expects charge=0 and spin=0 in CatMaster; "
            "use omol for charged or spin-specific molecular predictions."
        )
    atoms.info.update({"charge": int(cfg.charge), "spin": int(cfg.spin)})


def output_structure_path(output_dir: Path, atoms: Any, *, stem: str) -> Path:
    if has_periodic_cell(atoms):
        return output_dir / f"{stem}.vasp"
    return output_dir / f"{stem}.xyz"


def write_structure(path: Path, atoms: Any) -> None:
    from ase.io import write

    fmt = "vasp" if path.suffix.lower() == ".vasp" else "xyz"
    write(str(path), atoms, format=fmt)


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _redact_secrets(text: str) -> str:
    out = str(text)
    for key in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN", "HUGGINGFACE_HUB_TOKEN"):
        value = os.environ.get(key)
        if value:
            out = out.replace(value, "<redacted>")
    out = re.sub(r"hf_[A-Za-z0-9_=-]{12,}", "hf_<redacted>", out)
    return out


def _model_load_hint(exc: BaseException) -> str:
    text = f"{type(exc).__name__}: {exc}"
    lowered = text.lower()
    if "no module named 'fairchem'" in lowered or 'no module named "fairchem"' in lowered:
        return (
            "FairChem is not importable in the remote job. Check uma_gpu.source_list, "
            "the catmaster-uma conda environment, and whether the source script is visible "
            "from the scheduler execution host."
        )
    if "gatedrepoerror" in lowered or "access to model facebook/uma is restricted" in lowered or "401 client error" in lowered:
        return (
            "The UMA checkpoint is gated on Hugging Face. Configure a remote HF_TOKEN or "
            "$HOME/.config/huggingface/token for an account with access to facebook/UMA, "
            "then prewarm HF_HUB_CACHE before enabling offline mode."
        )
    if "offline" in lowered or "cannot find" in lowered and "snapshot" in lowered:
        return (
            "The model was not found in the local Hugging Face cache. Disable HF_HUB_OFFLINE "
            "until the checkpoint is downloaded, or prewarm HF_HUB_CACHE from a network-enabled node."
        )
    return (
        "Check the remote UMA environment, Hugging Face model access, HF_TOKEN, network access, "
        "and HF_HOME/HF_HUB_CACHE."
    )


class UmaCalculatorFactory:
    def __init__(self, *, model: str, device: str) -> None:
        self.model = str(model or "uma-s-1p2").strip() or "uma-s-1p2"
        self.device = resolve_device(device)
        self._predictor: Any = None
        self._calculators: dict[str, Any] = {}

    def _get_predictor(self) -> Any:
        if self._predictor is not None:
            return self._predictor
        try:
            from fairchem.core import pretrained_mlip

            self._predictor = pretrained_mlip.get_predict_unit(self.model, device=self.device)
        except Exception as exc:
            hint = _model_load_hint(exc)
            raise RuntimeError(
                f"Failed to load FairChem UMA model. {hint} "
                f"Original error: {type(exc).__name__}: {_redact_secrets(str(exc))}"
            ) from exc
        return self._predictor

    def get(self, task_name: str) -> Any:
        task = normalize_uma_task(task_name)
        if task == "auto":
            raise ValueError("Calculator task_name cannot be auto after structure inspection.")
        if task not in self._calculators:
            from fairchem.core import FAIRChemCalculator

            self._calculators[task] = FAIRChemCalculator(
                self._get_predictor(),
                task_name=task,
            )
        return self._calculators[task]


def max_force_eva(values: Any) -> float:
    import numpy as np

    arr = np.asarray(values, dtype=float)
    if arr.size == 0:
        return 0.0
    if arr.ndim != 2 or arr.shape[1] != 3:
        raise ValueError("Force arrays must have shape (n_atoms, 3).")
    return float(np.max(np.linalg.norm(arr, axis=1)))


def forces_payload(forces: Any) -> list[list[float]]:
    import numpy as np

    return np.asarray(forces, dtype=float).tolist()


def stress_payload(atoms: Any) -> list[float] | None:
    import numpy as np

    try:
        stress = atoms.get_stress()
    except Exception:
        return None
    try:
        return np.asarray(stress, dtype=float).reshape(-1).tolist()
    except Exception:
        return None


def validate_batch_paths(input_path: str, output_root: str | None) -> tuple[Path, Path]:
    input_root = Path(input_path)
    if not input_root.is_dir():
        raise ValueError("input_path must be a directory for UMA batch execution.")
    if not output_root:
        raise ValueError("output_root is required for UMA batch execution.")
    output_root_path = Path(output_root)
    if is_within(output_root_path.resolve(), input_root.resolve()):
        raise ValueError("output_root must not be inside input_path.")
    output_root_path.mkdir(parents=True, exist_ok=True)
    return input_root, output_root_path


def summarize_batch(
    *,
    input_root: Path,
    output_root: Path,
    model: str,
    default_task: str,
    device: str,
    metadata_path: str,
    results: Iterable[dict[str, Any]],
    errors: Iterable[dict[str, Any]],
    mode: str,
) -> dict[str, Any]:
    payload = {
        "mode": mode,
        "input_root": str(input_root),
        "output_root": str(output_root),
        "model": model,
        "default_uma_task": default_task,
        "device": device,
        "metadata_path": metadata_path,
        "fairchem_version": fairchem_version(),
        "results": list(results),
        "errors": list(errors),
    }
    write_json(output_root / "batch_summary.json", payload)
    return payload
