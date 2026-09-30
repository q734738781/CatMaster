#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

try:
    import torch
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: torch. Install it first, e.g. `pip install torch`."
    ) from exc

try:
    from mace import data as mace_data
    from mace.tools import AtomicNumberTable
    from mace.tools.scripts_utils import remove_pt_head
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: mace-torch. Install it first, e.g. `pip install mace-torch`."
    ) from exc


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Estimate MACE E0s from a foundation model on a training extxyz file "
            "using MACE's official least-squares residual correction."
        )
    )
    parser.add_argument(
        "--train-file",
        type=Path,
        required=True,
        help="Training extxyz file containing REF_energy / REF_forces / REF_stress.",
    )
    parser.add_argument(
        "--foundation-model",
        type=Path,
        required=True,
        help="Foundation model checkpoint path, e.g. models/mace-mh-1.model.",
    )
    parser.add_argument(
        "--out-path",
        type=Path,
        required=True,
        help="Output JSON path for estimated E0s.",
    )
    parser.add_argument(
        "--foundation-head",
        type=str,
        default=None,
        help="Head to keep when the foundation model is multi-head, e.g. omat_pbe.",
    )
    parser.add_argument(
        "--head-name",
        type=str,
        default="Default",
        help="Head label assigned when reading the training extxyz file.",
    )
    parser.add_argument(
        "--energy-key",
        type=str,
        default="REF_energy",
        help="Energy key in extxyz info.",
    )
    parser.add_argument(
        "--forces-key",
        type=str,
        default="REF_forces",
        help="Forces key in extxyz arrays.",
    )
    parser.add_argument(
        "--stress-key",
        type=str,
        default="REF_stress",
        help="Stress key in extxyz info.",
    )
    parser.add_argument(
        "--head-key",
        type=str,
        default="head",
        help="Head key in extxyz info.",
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cpu",
        help="Torch device used for foundation-model inference during E0 estimation.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite output JSON if it already exists.",
    )
    return parser.parse_args()


def _prepare_output_path(path: Path, overwrite: bool) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if not overwrite:
            raise SystemExit(f"Output already exists: {path}. Use --overwrite to replace it.")
        path.unlink()


def _load_foundation_model(model_path: Path, device: str, foundation_head: str | None):
    model = torch.load(model_path, map_location=device)
    if hasattr(model, "heads") and len(model.heads) > 1:
        logging.info("Foundation model heads: %s", ", ".join(model.heads))
        model = remove_pt_head(model, foundation_head)
        kept_head = model.heads[0] if hasattr(model, "heads") and model.heads else "unknown"
        logging.info("Keeping foundation head: %s", kept_head)
    return model


def _extract_foundation_e0s(model) -> dict[int, float]:
    z_table_foundation = AtomicNumberTable([int(z) for z in model.atomic_numbers])
    foundation_atomic_energies = model.atomic_energies_fn.atomic_energies
    if foundation_atomic_energies.ndim > 1:
        foundation_atomic_energies = foundation_atomic_energies.squeeze()
        if foundation_atomic_energies.ndim == 2:
            foundation_atomic_energies = foundation_atomic_energies[0]
            logging.info(
                "Foundation model atomic energies still have multiple heads; using the first one."
            )
    return {
        z: foundation_atomic_energies[z_table_foundation.z_to_index(z)].item()
        for z in z_table_foundation.zs
    }


def _load_train_configs(args: argparse.Namespace):
    keyspec = mace_data.KeySpecification.from_defaults()
    keyspec.update(
        info_keys={
            "energy": args.energy_key,
            "stress": args.stress_key,
            "head": args.head_key,
        },
        arrays_keys={"forces": args.forces_key},
    )
    _, train_configs = mace_data.load_from_xyz(
        file_path=str(args.train_file),
        key_specification=keyspec,
        head_name=args.head_name,
        extract_atomic_energies=False,
    )
    if not train_configs:
        raise SystemExit(f"No training configurations loaded from {args.train_file}")
    zs = sorted({int(z) for config in train_configs for z in config.atomic_numbers})
    z_table = AtomicNumberTable(zs)
    return train_configs, z_table


def main() -> int:
    args = parse_args()

    logging.basicConfig(level=logging.INFO, format="[%(levelname)s] %(message)s")

    train_file = args.train_file.resolve()
    foundation_model = args.foundation_model.resolve()
    out_path = args.out_path.resolve()
    _prepare_output_path(out_path, overwrite=args.overwrite)

    model = _load_foundation_model(
        foundation_model,
        device=args.device,
        foundation_head=args.foundation_head,
    )
    foundation_e0s = _extract_foundation_e0s(model)
    train_configs, z_table = _load_train_configs(args)
    estimated_e0s = mace_data.estimate_e0s_from_foundation(
        foundation_model=model,
        foundation_e0s=foundation_e0s,
        collections_train=train_configs,
        z_table=z_table,
        device=args.device,
    )

    with out_path.open("w", encoding="utf-8") as f:
        json.dump({str(key): value for key, value in estimated_e0s.items()}, f, indent=2)
        f.write("\n")

    print(f"train_file: {train_file}")
    print(f"foundation_model: {foundation_model}")
    print(f"out_path: {out_path}")
    print(f"elements: {z_table.zs}")
    if args.foundation_head:
        print(f"foundation_head: {args.foundation_head}")
    print("estimated_e0s:")
    for z in z_table.zs:
        base = foundation_e0s.get(z)
        est = estimated_e0s[z]
        if base is None:
            print(f"- Z={z}: estimated={est:.10f} eV")
        else:
            print(
                f"- Z={z}: foundation={base:.10f} eV, "
                f"estimated={est:.10f} eV, delta={est - base:.10f} eV"
            )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
