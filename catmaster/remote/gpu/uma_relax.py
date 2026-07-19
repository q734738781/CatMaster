from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ase.io import read
from ase.io.trajectory import Trajectory

from uma_common import (
    UmaCalculatorFactory,
    apply_charge_spin,
    auto_uma_task,
    collect_structure_files,
    fairchem_version,
    max_force_eva,
    forces_payload,
    has_periodic_cell,
    load_metadata,
    normalize_uma_task,
    output_structure_path,
    parse_bool,
    resolve_item_config,
    summarize_batch,
    validate_batch_paths,
    write_json,
    write_structure,
)


def _optimizer_class(name: str) -> Any:
    from ase.optimize import BFGS, FIRE, LBFGS

    normalized = str(name or "FIRE").strip().upper()
    if normalized == "FIRE":
        return FIRE
    if normalized == "BFGS":
        return BFGS
    if normalized == "LBFGS":
        return LBFGS
    raise ValueError("optimizer must be one of FIRE, BFGS, or LBFGS.")


def _cell_filter(atoms: Any) -> Any:
    try:
        from ase.filters import FrechetCellFilter

        return FrechetCellFilter(atoms)
    except Exception:
        from ase.filters import UnitCellFilter

        return UnitCellFilter(atoms)


def _run_relax(
    *,
    structure_path: Path,
    input_root: Path,
    output_root: Path,
    calc_factory: UmaCalculatorFactory,
    model: str,
    metadata: dict[str, Any],
    default_task: str,
    default_charge: int,
    default_spin: int,
    fmax: float,
    steps: int,
    optimizer: str,
    relax_cell: bool,
) -> dict[str, Any]:
    atoms = read(str(structure_path))
    cfg = resolve_item_config(
        structure_path=structure_path,
        input_root=input_root,
        metadata=metadata,
        default_task=default_task,
        default_charge=default_charge,
        default_spin=default_spin,
    )
    task_name = auto_uma_task(atoms) if cfg.uma_task == "auto" else cfg.uma_task
    cfg = cfg.__class__(uma_task=task_name, charge=cfg.charge, spin=cfg.spin)
    apply_charge_spin(atoms, cfg)

    if relax_cell:
        if task_name != "omat":
            raise ValueError("relax_cell=True is only enabled for UMA task omat in CatMaster.")
        if not has_periodic_cell(atoms):
            raise ValueError("relax_cell=True requires a periodic structure with a valid cell.")

    atoms.calc = calc_factory.get(task_name)

    rel_path = structure_path.relative_to(input_root)
    out_dir = output_root / rel_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    traj_path = out_dir / "opt.traj"
    log_path = out_dir / "opt.log"

    target = _cell_filter(atoms) if relax_cell else atoms
    traj = Trajectory(str(traj_path), "w", atoms)
    opt_cls = _optimizer_class(optimizer)
    opt = opt_cls(target, logfile=str(log_path))
    opt.attach(traj)
    try:
        opt.run(fmax=float(fmax), steps=int(steps))
    finally:
        traj.close()

    final_energy = float(atoms.get_potential_energy())
    final_forces = atoms.get_forces()
    max_force = max_force_eva(final_forces)
    structure_out = output_structure_path(out_dir, atoms, stem="opt")
    write_structure(structure_out, atoms)
    write_json(out_dir / "forces.json", {"forces_eVA": forces_payload(final_forces)})

    summary = {
        "mode": "relax",
        "input_rel": rel_path.as_posix(),
        "output_structure": structure_out.name,
        "model": model,
        "uma_task": task_name,
        "device": calc_factory.device,
        "charge": cfg.charge,
        "spin": cfg.spin,
        "fairchem_version": fairchem_version(),
        "relax_cell": relax_cell,
        "optimizer": str(optimizer),
        "final_energy_eV": final_energy,
        "fmax": float(fmax),
        "max_force_eVA": max_force,
        "steps": int(steps),
        "nsteps": int(getattr(opt, "nsteps", 0)),
        "converged": bool(max_force < float(fmax)),
    }
    write_json(out_dir / "summary.json", summary)
    return {
        "input_rel": rel_path.as_posix(),
        "output_rel": out_dir.relative_to(output_root).as_posix(),
        "summary": summary,
    }


def run_uma_relax_batch(
    input_path: str,
    *,
    output_root: str,
    model: str = "uma-s-1p2",
    uma_task: str = "auto",
    charge: int = 0,
    spin: int = 0,
    metadata_path: str = "__none__",
    device: str = "auto",
    fmax: float = 0.02,
    steps: int = 500,
    optimizer: str = "FIRE",
    relax_cell: bool = False,
) -> Dict[str, object]:
    input_root, output_root_path = validate_batch_paths(input_path, output_root)
    default_task = normalize_uma_task(uma_task)
    metadata = load_metadata(metadata_path)
    structures = collect_structure_files(input_root)
    if not structures:
        raise ValueError(f"No structure files found in directory: {input_root}")

    calc_factory = UmaCalculatorFactory(model=model, device=device)
    results: list[dict[str, Any]] = []
    errors: list[dict[str, Any]] = []
    for structure in structures:
        try:
            results.append(
                _run_relax(
                    structure_path=structure,
                    input_root=input_root,
                    output_root=output_root_path,
                    calc_factory=calc_factory,
                    model=model,
                    metadata=metadata,
                    default_task=default_task,
                    default_charge=int(charge),
                    default_spin=int(spin),
                    fmax=float(fmax),
                    steps=int(steps),
                    optimizer=optimizer,
                    relax_cell=bool(relax_cell),
                )
            )
        except Exception as exc:
            errors.append(
                {
                    "input_rel": structure.relative_to(input_root).as_posix(),
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )

    return summarize_batch(
        input_root=input_root,
        output_root=output_root_path,
        model=model,
        default_task=default_task,
        device=calc_factory.device,
        metadata_path=metadata_path,
        results=results,
        errors=errors,
        mode="relax",
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run FairChem UMA relaxation batch for staged structures.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    parser.add_argument("--model", default="uma-s-1p2", help="FairChem UMA model name")
    parser.add_argument("--uma_task", default="auto", help="auto|omat|omol|oc20|oc22|oc25|odac|omc")
    parser.add_argument("--charge", type=int, default=0, help="Charge for omol task inputs")
    parser.add_argument("--spin", type=int, default=0, help="Spin value for omol task inputs")
    parser.add_argument("--metadata", default="__none__", help="Optional params/uma_metadata.json path")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    parser.add_argument("--fmax", type=float, default=0.02, help="Force convergence threshold")
    parser.add_argument("--steps", type=int, default=500, help="Maximum optimizer steps")
    parser.add_argument("--optimizer", default="FIRE", help="FIRE|BFGS|LBFGS")
    parser.add_argument("--relax_cell", default="false", help="Whether to relax periodic cell, true|false")
    args = parser.parse_args()

    result = run_uma_relax_batch(
        input_path=args.input,
        output_root=args.output_root,
        model=args.model,
        uma_task=args.uma_task,
        charge=args.charge,
        spin=args.spin,
        metadata_path=args.metadata,
        device=args.device,
        fmax=args.fmax,
        steps=args.steps,
        optimizer=args.optimizer,
        relax_cell=parse_bool(args.relax_cell),
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
