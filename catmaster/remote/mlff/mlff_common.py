# Code writing date: 2026-07-17
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: normalize provider outputs without changing the
# provider-specific calculator contract or ASE optimizer semantics.
# Purpose: shared managed MLFF SP/relax adapters and result normalization.
from __future__ import annotations

"""Provider adapters and shared SP/relax execution for managed MLFF tasks."""

import argparse
import json
from importlib import metadata as importlib_metadata
from pathlib import Path
from typing import Any

import numpy as np


def _package_version(*names: str) -> str:
    for name in names:
        try:
            return importlib_metadata.version(name)
        except Exception:
            continue
    return "unknown"


def _resolve_device(preference: str) -> str:
    import torch

    requested = str(preference or "auto").strip().lower() or "auto"
    if requested == "auto":
        return "cuda" if torch.cuda.is_available() else "cpu"
    if requested.startswith("cuda") and not torch.cuda.is_available():
        raise RuntimeError("CUDA was requested but torch.cuda.is_available() is false.")
    return requested


def _config_key(config: dict[str, Any]) -> str:
    return json.dumps(config, sort_keys=True, separators=(",", ":"))


class MaceAdapter:
    def __init__(self) -> None:
        self._calculators: dict[str, Any] = {}

    def calculator_for(self, atoms: Any, config: dict[str, Any]) -> Any:
        supports_charge_spin = bool(config.get("supports_charge_spin", False))
        if supports_charge_spin:
            charge = int(config.get("charge", 0))
            spin = int(config.get("spin", 0))
            if spin < 1:
                raise ValueError("MACE OMOL requires multiplicity-style spin >= 1.")
            atoms.info.update({"charge": charge, "spin": spin})
        calculator_config = {
            key: value
            for key, value in config.items()
            if key not in {"charge", "spin"}
        }
        key = _config_key(calculator_config)
        if key in self._calculators:
            return self._calculators[key]
        from mace.calculators import MACECalculator, mace_mp, mace_omol

        device = _resolve_device(str(config.get("device") or "auto"))
        kwargs: dict[str, Any] = {
            "device": device,
            "default_dtype": str(config.get("default_dtype") or "float64"),
        }
        loader = str(config.get("loader") or "mace_mp")
        head = str(config.get("head") or "").strip()
        if head and loader != "mace_omol":
            kwargs["head"] = head
        checkpoint = str(config.get("checkpoint_artifact") or "").strip()
        if loader == "checkpoint" or checkpoint:
            if config.get("dispersion"):
                raise ValueError("MACE dispersion is not supported with checkpoint_artifact.")
            kwargs["model_paths"] = checkpoint
            calc = MACECalculator(**kwargs)
        elif loader == "mace_mp":
            kwargs.update(
                {
                    "model": str(config.get("provider_model") or config["model"]),
                    "dispersion": bool(config.get("dispersion", False)),
                }
            )
            if config.get("enable_cueq"):
                if not device.startswith("cuda"):
                    raise ValueError("MACE enable_cueq requires CUDA.")
                kwargs["enable_cueq"] = True
            compile_mode = str(config.get("compile_mode") or "")
            if compile_mode:
                kwargs["compile_mode"] = compile_mode
            calc = mace_mp(**kwargs)
        elif loader == "mace_omol":
            if config.get("dispersion"):
                raise ValueError("MACE omol-0 does not support the mace_mp dispersion wrapper.")
            if config.get("enable_cueq"):
                if not device.startswith("cuda"):
                    raise ValueError("MACE enable_cueq requires CUDA.")
                kwargs["enable_cueq"] = True
            compile_mode = str(config.get("compile_mode") or "")
            if compile_mode:
                kwargs["compile_mode"] = compile_mode
            calc = mace_omol(
                model=str(config.get("provider_model") or "extra_large"),
                **kwargs,
            )
        else:
            raise ValueError(f"Unsupported registered MACE loader: {loader!r}.")
        self._calculators[key] = calc
        return calc

    def provider_metadata(self, atoms: Any, config: dict[str, Any], calculator: Any) -> dict[str, Any]:
        metadata = {
            "loader": str(config.get("loader") or "mace_mp"),
            "provider_model": str(config.get("provider_model") or config.get("model") or ""),
            "head": str(config.get("head") or ""),
            "dispersion": bool(config.get("dispersion", False)),
            "default_dtype": str(config.get("default_dtype") or "float64"),
            "enable_cueq": bool(config.get("enable_cueq", False)),
            "compile_mode": str(config.get("compile_mode") or ""),
            "checkpoint_artifact": str(config.get("checkpoint_artifact") or ""),
            "checkpoint_sha256": str(config.get("checkpoint_sha256") or ""),
            "checkpoint_size_bytes": int(config.get("checkpoint_size_bytes") or 0),
        }
        if config.get("supports_charge_spin"):
            metadata.update(
                {
                    "charge": int(config.get("charge", 0)),
                    "spin": int(config.get("spin", 0)),
                }
            )
        return metadata

    @property
    def provider_version(self) -> str:
        return _package_version("mace-torch")


class FairChemUmaAdapter:
    def __init__(self) -> None:
        self._predictors: dict[tuple[str, str, str], Any] = {}
        self._calculators: dict[tuple[str, str, str, str], Any] = {}

    def calculator_for(self, atoms: Any, config: dict[str, Any]) -> Any:
        from fairchem.core import FAIRChemCalculator, pretrained_mlip

        model = str(config["provider_model"])
        device = _resolve_device(str(config.get("device") or "auto"))
        task = str(config["uma_task"])
        inference_settings = str(config.get("inference_settings") or "default")
        atoms.info.update({"charge": int(config.get("charge", 0)), "spin": int(config.get("spin", 0))})
        predictor_key = (model, device, inference_settings)
        if predictor_key not in self._predictors:
            self._predictors[predictor_key] = pretrained_mlip.get_predict_unit(
                model,
                inference_settings=inference_settings,
                device=device,
            )
        key = (model, device, task, inference_settings)
        if key not in self._calculators:
            self._calculators[key] = FAIRChemCalculator(self._predictors[predictor_key], task_name=task)
        return self._calculators[key]

    def provider_metadata(self, atoms: Any, config: dict[str, Any], calculator: Any) -> dict[str, Any]:
        return {
            "provider_model": str(config["provider_model"]),
            "uma_task": str(config["uma_task"]),
            "charge": int(config.get("charge", 0)),
            "spin": int(config.get("spin", 0)),
            "inference_settings": str(config.get("inference_settings") or "default"),
        }

    @property
    def provider_version(self) -> str:
        return _package_version("fairchem-core", "fairchem")


class MatterSimAdapter:
    def __init__(self) -> None:
        self._calculators: dict[str, Any] = {}

    def calculator_for(self, atoms: Any, config: dict[str, Any]) -> Any:
        key = _config_key(config)
        if key in self._calculators:
            return self._calculators[key]
        try:
            from mattersim.forcefield import MatterSimCalculator
        except ImportError:
            from mattersim.forcefield.potential import MatterSimCalculator

        model = str(config["provider_model"])
        kwargs: dict[str, Any] = {
            "device": _resolve_device(str(config.get("device") or "auto")),
            "dtype": str(config.get("dtype") or "float32"),
            "compute_stress": bool(config.get("compute_stress", True)),
            "direct_graph": bool(config.get("direct_graph", False)),
            "compile": bool(config.get("compile", False)),
        }
        kwargs["load_path"] = model
        calc = MatterSimCalculator(**kwargs)
        self._calculators[key] = calc
        return calc

    def provider_metadata(self, atoms: Any, config: dict[str, Any], calculator: Any) -> dict[str, Any]:
        return {
            "checkpoint_identity": str(config["provider_model"]),
            "dtype": str(config.get("dtype") or "float32"),
            "compute_stress": bool(config.get("compute_stress", True)),
            "direct_graph": bool(config.get("direct_graph", False)),
            "compile": bool(config.get("compile", False)),
        }

    @property
    def provider_version(self) -> str:
        return _package_version("mattersim")


class OrbV3Adapter:
    def __init__(self) -> None:
        self._calculators: dict[str, Any] = {}

    def calculator_for(self, atoms: Any, config: dict[str, Any]) -> Any:
        key = _config_key(config)
        if key in self._calculators:
            return self._calculators[key]
        from orb_models.forcefield import pretrained
        from orb_models.forcefield.inference.calculator import ORBCalculator

        model = str(config["provider_model"])
        loader_name = model.replace("-", "_")
        loader = getattr(pretrained, loader_name, None)
        if loader is None:
            raise ValueError(f"ORB pretrained loader is unavailable for configured official model {model!r}.")
        device = _resolve_device(str(config.get("device") or "auto"))
        compile_mode = str(config.get("compile_mode") or "auto")
        compile_value = {"auto": None, "on": True, "off": False}[compile_mode]
        orbff, atoms_adapter = loader(
            device=device,
            precision=str(config["precision"]),
            compile=compile_value,
        )
        half_supercell_mode = str(config.get("half_supercell") or "auto")
        half_supercell = {"auto": None, "on": True, "off": False}[half_supercell_mode]
        calc = ORBCalculator(
            orbff,
            atoms_adapter=atoms_adapter,
            device=device,
            edge_method=str(config.get("edge_method") or "knn_alchemi"),
            half_supercell=half_supercell,
        )
        self._calculators[key] = calc
        return calc

    def provider_metadata(self, atoms: Any, config: dict[str, Any], calculator: Any) -> dict[str, Any]:
        results = getattr(calculator, "results", {}) or {}
        confidence = results.get("confidence")
        metadata: dict[str, Any] = {
            "provider_model": str(config["provider_model"]),
            "precision": str(config["precision"]),
            "compile_mode": str(config.get("compile_mode") or "auto"),
            "edge_method": str(config.get("edge_method") or "knn_alchemi"),
            "half_supercell": str(config.get("half_supercell") or "auto"),
        }
        if confidence is not None:
            array = np.asarray(confidence)
            metadata["confidence_shape"] = list(array.shape)
            if array.ndim >= 2 and array.shape[-1]:
                metadata["confidence_peak_bin_per_atom"] = np.argmax(array, axis=-1).tolist()
        return metadata

    @property
    def provider_version(self) -> str:
        return _package_version("orb-models")


_ADAPTER_TYPES = {
    "mace": MaceAdapter,
    "fairchem_uma": FairChemUmaAdapter,
    "mattersim": MatterSimAdapter,
    "orb_v3": OrbV3Adapter,
}


def _load_run_config(path: str) -> dict[str, Any]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, dict) or int(payload.get("schema_version", 0)) != 1:
        raise ValueError("Unsupported or malformed MLFF run_config.json.")
    required = {"config_digest", "operation", "backend", "backend_config", "task_config", "items"}
    missing = sorted(required - set(payload))
    if missing:
        raise ValueError("MLFF run config is missing: " + ", ".join(missing))
    return payload


def _stress(atoms: Any) -> list[float] | None:
    try:
        return np.asarray(atoms.get_stress(), dtype=float).reshape(-1).tolist()
    except Exception:
        return None


def _extxyz_move_mask(atoms: Any) -> np.ndarray:
    """Return ASE's Cartesian move mask and reject constraints extxyz cannot encode."""

    from ase.constraints import FixAtoms, FixCartesian

    move_mask = np.ones((len(atoms), 3), dtype=bool)
    unsupported: list[str] = []
    for constraint in atoms.constraints or []:
        if isinstance(constraint, FixAtoms):
            move_mask[np.asarray(constraint.index, dtype=int)] = False
        elif isinstance(constraint, FixCartesian):
            move_mask[np.asarray(constraint.index, dtype=int)] &= ~np.asarray(
                constraint.mask,
                dtype=bool,
            )
        else:
            unsupported.append(type(constraint).__name__)
    if unsupported:
        raise ValueError(
            "extxyz output can preserve only ASE FixAtoms and FixCartesian constraints; "
            f"unsupported: {', '.join(sorted(set(unsupported)))}"
        )
    return move_mask


def _write_extxyz_preserving_constraints(path: Path, atoms: Any) -> None:
    from ase.io import read, write

    expected_move_mask = _extxyz_move_mask(atoms)
    write(str(path), atoms, format="extxyz")
    restored = read(str(path), index=-1)
    restored_move_mask = _extxyz_move_mask(restored)
    if not np.array_equal(restored_move_mask, expected_move_mask):
        raise RuntimeError(
            "extxyz constraint round-trip changed the ASE FixAtoms/FixCartesian move mask"
        )


def _output_structure(
    output_dir: Path,
    atoms: Any,
    *,
    stem: str,
    source_name: str = "",
) -> str:
    from ase.io import write

    if Path(str(source_name or "")).suffix.lower() == ".extxyz":
        name = f"{stem}.extxyz"
        _write_extxyz_preserving_constraints(output_dir / name, atoms)
        return name

    periodic = bool(any(bool(value) for value in atoms.pbc)) and float(atoms.cell.volume) > 1e-6
    name = f"{stem}.vasp" if periodic else f"{stem}.xyz"
    write(str(output_dir / name), atoms, format="vasp" if periodic else "xyz")
    return name


def _base_summary(
    *,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
    atoms: Any,
    calculator: Any,
) -> dict[str, Any]:
    forces = np.asarray(atoms.get_forces(), dtype=float)
    force_norms = np.linalg.norm(forces, axis=1) if forces.size else np.asarray([], dtype=float)
    backend = str(config["backend"])
    return {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": str(config["operation"]),
        "backend": backend,
        "model": str(item_config.get("model") or item_config.get("checkpoint_artifact") or ""),
        "provider_version": adapter.provider_version,
        "device": str(item_config.get("device") or "auto"),
        "max_force_eVA": float(np.max(force_norms)) if force_norms.size else 0.0,
        "stress_voigt_eVA3": _stress(atoms),
        "warnings": [],
        "provider_metadata": adapter.provider_metadata(atoms, item_config, calculator),
    }


def _run_sp(
    *,
    atoms: Any,
    output_dir: Path,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
    source_name: str = "",
) -> dict[str, Any]:
    calculator = adapter.calculator_for(atoms, item_config)
    atoms.calc = calculator
    energy = float(atoms.get_potential_energy())
    summary = _base_summary(
        config=config,
        item_config=item_config,
        adapter=adapter,
        atoms=atoms,
        calculator=calculator,
    )
    summary.update(
        {
            "energy_eV": energy,
            "output_structure": _output_structure(
                output_dir,
                atoms,
                stem="sp",
                source_name=source_name,
            ),
        }
    )
    return summary


def _run_relax(
    *,
    atoms: Any,
    output_dir: Path,
    config: dict[str, Any],
    item_config: dict[str, Any],
    adapter: Any,
    source_name: str = "",
) -> dict[str, Any]:
    from ase.filters import FrechetCellFilter
    from ase.io.trajectory import Trajectory
    from ase.optimize import BFGS, FIRE, LBFGS

    task = dict(config["task_config"])
    calculator = adapter.calculator_for(atoms, item_config)
    atoms.calc = calculator
    relax_cell = bool(task["relax_cell"])
    if relax_cell and not (bool(all(bool(value) for value in atoms.pbc)) and float(atoms.cell.volume) > 1e-6):
        raise ValueError("relax_cell=true requires a fully periodic structure with a nonzero cell.")
    target = FrechetCellFilter(atoms) if relax_cell else atoms
    optimizer_type = {"FIRE": FIRE, "BFGS": BFGS, "LBFGS": LBFGS}[str(task["optimizer"])]
    trajectory = Trajectory(str(output_dir / "opt.traj"), "w", atoms)
    optimizer = optimizer_type(target, logfile=str(output_dir / "opt.log"))
    optimizer.attach(trajectory.write)
    try:
        optimizer.run(fmax=float(task["fmax"]), steps=int(task["steps"]))
    finally:
        trajectory.close()
    converged = bool(optimizer.converged())
    final_energy = float(atoms.get_potential_energy())
    summary = _base_summary(
        config=config,
        item_config=item_config,
        adapter=adapter,
        atoms=atoms,
        calculator=calculator,
    )
    summary.update(
        {
            "final_energy_eV": final_energy,
            "converged": converged,
            "nsteps": int(optimizer.nsteps),
            "output_structure": _output_structure(
                output_dir,
                atoms,
                stem="opt",
                source_name=source_name,
            ),
        }
    )
    return summary


def run(operation: str, run_config: str) -> dict[str, Any]:
    from ase.io import read

    config = _load_run_config(run_config)
    if config["operation"] != operation:
        raise ValueError(f"Runner operation {operation!r} does not match run config {config['operation']!r}.")
    adapter = _ADAPTER_TYPES[str(config["backend"])]()
    input_root = Path("input")
    output_root = Path("output")
    output_root.mkdir(parents=True, exist_ok=True)
    overwrite = bool(dict(config.get("task_config") or {}).get("output", {}).get("overwrite", False))
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    for rel, item_config in sorted(dict(config["items"]).items()):
        source = input_root / rel
        output_dir = output_root / Path(rel).with_suffix("")
        if output_dir.exists() and any(output_dir.iterdir()) and not overwrite:
            errors.append({"input_rel": rel, "error": f"Output already exists: {output_dir}"})
            continue
        output_dir.mkdir(parents=True, exist_ok=True)
        try:
            atoms = read(str(source), index=-1)
            if operation == "sp":
                summary = _run_sp(
                    atoms=atoms,
                    output_dir=output_dir,
                    config=config,
                    item_config=dict(item_config),
                    adapter=adapter,
                    source_name=rel,
                )
            elif operation == "relax":
                summary = _run_relax(
                    atoms=atoms,
                    output_dir=output_dir,
                    config=config,
                    item_config=dict(item_config),
                    adapter=adapter,
                    source_name=rel,
                )
            else:
                raise ValueError(f"Unsupported common MLFF operation: {operation}")
            (output_dir / "summary.json").write_text(
                json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
            )
            results.append({"input_rel": rel, "output_rel": output_dir.relative_to(output_root).as_posix(), "summary": summary})
        except Exception as exc:
            errors.append({"input_rel": rel, "error": f"{type(exc).__name__}: {exc}"})
    batch = {
        "schema_version": 1,
        "config_digest": config["config_digest"],
        "operation": operation,
        "backend": config["backend"],
        "results": results,
        "errors": errors,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n", encoding="utf-8"
    )
    if errors:
        raise RuntimeError(f"{len(errors)} of {len(results) + len(errors)} MLFF inputs failed; see output/batch_summary.json")
    return batch


def cli(operation: str) -> None:
    parser = argparse.ArgumentParser(description=f"Run managed MLFF {operation}.")
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(operation, args.run_config), ensure_ascii=False, indent=2))


__all__ = ["cli", "run"]
