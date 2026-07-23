from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Dict

from ase.io import read

from uma_common import (
    UmaCalculatorFactory,
    apply_charge_spin,
    collect_structure_files,
    fairchem_version,
    max_force_eva,
    forces_payload,
    load_metadata,
    normalize_uma_task,
    output_structure_path,
    resolve_item_config,
    stress_payload,
    summarize_batch,
    validate_batch_paths,
    write_json,
    write_structure,
)


def _run_single_point(
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
    task_name = cfg.uma_task
    apply_charge_spin(atoms, cfg)

    atoms.calc = calc_factory.get(task_name)
    energy = float(atoms.get_potential_energy())
    forces = atoms.get_forces()
    stress = stress_payload(atoms)

    rel_path = structure_path.relative_to(input_root)
    out_dir = output_root / rel_path.with_suffix("")
    out_dir.mkdir(parents=True, exist_ok=True)
    structure_out = output_structure_path(out_dir, atoms, stem="sp")
    write_structure(structure_out, atoms)
    write_json(out_dir / "forces.json", {"forces_eVA": forces_payload(forces)})
    if stress is not None:
        write_json(out_dir / "stress.json", {"stress_voigt_eVA3": stress})

    summary = {
        "mode": "sp",
        "input_rel": rel_path.as_posix(),
        "output_structure": structure_out.name,
        "model": model,
        "uma_task": task_name,
        "device": calc_factory.device,
        "charge": cfg.charge,
        "spin": cfg.spin,
        "fairchem_version": fairchem_version(),
        "energy_eV": energy,
        "max_force_eVA": max_force_eva(forces),
    }
    if stress is not None:
        summary["stress_voigt_eVA3"] = stress
    write_json(out_dir / "summary.json", summary)
    return {
        "input_rel": rel_path.as_posix(),
        "output_rel": out_dir.relative_to(output_root).as_posix(),
        "summary": summary,
    }


def run_uma_sp_batch(
    input_path: str,
    *,
    output_root: str,
    model: str = "uma-s-1p2",
    uma_task: str = "omat",
    charge: int = 0,
    spin: int = 0,
    metadata_path: str = "__none__",
    device: str = "auto",
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
                _run_single_point(
                    structure_path=structure,
                    input_root=input_root,
                    output_root=output_root_path,
                    calc_factory=calc_factory,
                    model=model,
                    metadata=metadata,
                    default_task=default_task,
                    default_charge=int(charge),
                    default_spin=int(spin),
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
        mode="sp",
    )


def _cli() -> None:
    parser = argparse.ArgumentParser(description="Run FairChem UMA single-point batch for staged structures.")
    parser.add_argument("--input", required=True, help="Input root directory")
    parser.add_argument("--output_root", required=True, help="Output root directory")
    parser.add_argument("--model", default="uma-s-1p2", help="FairChem UMA model name")
    parser.add_argument("--uma_task", default="omat", help="omat|omol|oc20|oc22|oc25|odac|omc")
    parser.add_argument("--charge", type=int, default=0, help="Charge for omol task inputs")
    parser.add_argument("--spin", type=int, default=0, help="Spin value for omol task inputs")
    parser.add_argument("--metadata", default="__none__", help="Optional params/uma_metadata.json path")
    parser.add_argument("--device", default="auto", help="auto|cpu|cuda|cuda:0")
    args = parser.parse_args()

    result = run_uma_sp_batch(
        input_path=args.input,
        output_root=args.output_root,
        model=args.model,
        uma_task=args.uma_task,
        charge=args.charge,
        spin=args.spin,
        metadata_path=args.metadata,
        device=args.device,
    )
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    _cli()
