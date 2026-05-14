from __future__ import annotations

import json
import math
import multiprocessing as mp
import os
import random
import time
import traceback
from pathlib import Path
from queue import Empty
from typing import Any

from ase.io import read, write
from ase.optimize import FIRE

from ..core.config import DEFAULT_MACE_DTYPE, DEFAULT_MACE_HEAD, DEFAULT_MACE_MODEL, DEFAULT_STAGE1_BASE_STRUCTURE


def build_mace_calculator(model_path: str, head: str | None, default_dtype: str, device: str):
    from mace.calculators import mace_mp

    kwargs: dict[str, Any] = {
        "model": model_path,
        "device": device,
        "default_dtype": default_dtype,
    }
    if head:
        kwargs["head"] = head
    return mace_mp(**kwargs)


def load_base_structure(base_structure_path: str | Path = DEFAULT_STAGE1_BASE_STRUCTURE):
    return read(Path(base_structure_path).expanduser().resolve())


def get_fe_site_indices(atoms) -> list[int]:
    return [index for index, symbol in enumerate(atoms.get_chemical_symbols()) if symbol == "Fe"]


def substitution_count_from_x_total(n_fe_sites: int, x_total: float) -> int:
    count = int(round(n_fe_sites * x_total / 3.0))
    if count <= 0:
        raise ValueError(f"x_total={x_total} produces zero substitutions for {n_fe_sites} Fe sites.")
    return count


def distribute_dopant_counts(total_substitutions: int, dopants: list[str], rng: random.Random) -> dict[str, int]:
    if len(dopants) == 0:
        raise ValueError("dopants must not be empty.")
    shuffled = list(dopants)
    rng.shuffle(shuffled)
    base = total_substitutions // len(shuffled)
    remainder = total_substitutions % len(shuffled)
    counts = {element: base for element in shuffled}
    for index in range(remainder):
        counts[shuffled[index]] += 1
    return counts


def decorate_fe_sublattice(base_atoms, fe_indices: list[int], dopants: list[str], x_total: float, rng: random.Random):
    atoms = base_atoms.copy()
    counts = distribute_dopant_counts(substitution_count_from_x_total(len(fe_indices), x_total), dopants, rng)
    labels = ["Fe"] * len(fe_indices)
    doped_slots = list(range(len(fe_indices)))
    rng.shuffle(doped_slots)
    cursor = 0
    for element, count in counts.items():
        for _ in range(count):
            labels[doped_slots[cursor]] = element
            cursor += 1
    symbols = atoms.get_chemical_symbols()
    for local_index, atom_index in enumerate(fe_indices):
        symbols[atom_index] = labels[local_index]
    atoms.set_chemical_symbols(symbols)
    return atoms, counts, labels


def labels_to_atoms(base_atoms, fe_indices: list[int], labels: list[str]):
    atoms = base_atoms.copy()
    symbols = atoms.get_chemical_symbols()
    for local_index, atom_index in enumerate(fe_indices):
        symbols[atom_index] = labels[local_index]
    atoms.set_chemical_symbols(symbols)
    return atoms


def evaluate_atoms(atoms, calc, output_dir: str | Path, relax_mode: str, fmax: float, max_steps: int) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    atoms = atoms.copy()
    atoms.calc = calc

    if relax_mode == "light_relax":
        optimizer = FIRE(
            atoms,
            logfile=str(output_dir / "light_relax.log"),
            trajectory=str(output_dir / "light_relax.traj"),
        )
        optimizer.run(fmax=fmax, steps=max_steps)

    energy_eV = float(atoms.get_potential_energy())
    payload = {
        "relax_mode": relax_mode,
        "energy_eV": energy_eV,
        "energy_per_atom_eV": energy_eV / len(atoms),
        "n_atoms": len(atoms),
        "formula": atoms.get_chemical_formula(),
        "volume_A3": float(atoms.get_volume()),
    }
    write(output_dir / "final.vasp", atoms, format="vasp")
    (output_dir / "energy_summary.json").write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return payload


def monte_carlo_screen(
    *,
    base_atoms,
    fe_indices: list[int],
    dopants: list[str],
    x_total: float,
    calc,
    output_dir: str | Path,
    mc_samples: int,
    mc_steps: int,
    mc_temperature_ev: float,
    random_seed: int,
    relax_mode: str,
    fmax: float,
    max_steps: int,
) -> dict[str, Any]:
    output_dir = Path(output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    best_payload: dict[str, Any] | None = None
    replica_energies: list[float] = []

    for replica in range(mc_samples):
        rng = random.Random(random_seed + replica)
        atoms, counts, labels = decorate_fe_sublattice(base_atoms, fe_indices, dopants, x_total, rng)
        current = evaluate_atoms(
            atoms,
            calc=calc,
            output_dir=output_dir / f"replica_{replica:02d}" / "initial",
            relax_mode="singlepoint",
            fmax=fmax,
            max_steps=max_steps,
        )
        current_energy = float(current["energy_eV"])
        best_labels = list(labels)
        best_energy = current_energy

        for step in range(mc_steps):
            proposal = list(labels)
            swap_i, swap_j = rng.sample(range(len(proposal)), 2)
            if proposal[swap_i] == proposal[swap_j]:
                continue
            proposal[swap_i], proposal[swap_j] = proposal[swap_j], proposal[swap_i]
            proposal_atoms = labels_to_atoms(base_atoms, fe_indices, proposal)
            trial = evaluate_atoms(
                proposal_atoms,
                calc=calc,
                output_dir=output_dir / f"replica_{replica:02d}" / f"mc_{step:03d}",
                relax_mode="singlepoint",
                fmax=fmax,
                max_steps=max_steps,
            )
            trial_energy = float(trial["energy_eV"])
            accept = trial_energy <= current_energy
            if not accept and mc_temperature_ev > 0:
                accept = rng.random() < math.exp(-(trial_energy - current_energy) / mc_temperature_ev)
            if accept:
                labels = proposal
                current_energy = trial_energy
            if trial_energy < best_energy:
                best_energy = trial_energy
                best_labels = list(proposal)

        best_atoms = labels_to_atoms(base_atoms, fe_indices, best_labels)
        final = evaluate_atoms(
            best_atoms,
            calc=calc,
            output_dir=output_dir / f"replica_{replica:02d}" / "final",
            relax_mode=relax_mode,
            fmax=fmax,
            max_steps=max_steps,
        )
        final.update(
            {
                "dopants": dopants,
                "dopant_counts": counts,
                "mc_steps": mc_steps,
                "mc_temperature_ev": mc_temperature_ev,
                "structure_path": str(output_dir / f"replica_{replica:02d}" / "final" / "final.vasp"),
            }
        )
        replica_energies.append(float(final["energy_eV"]))
        if best_payload is None or float(final["energy_eV"]) < float(best_payload["energy_eV"]):
            best_payload = final

    assert best_payload is not None
    best_payload["replica_energy_mean_eV"] = sum(replica_energies) / len(replica_energies)
    if len(replica_energies) > 1:
        mean_energy = best_payload["replica_energy_mean_eV"]
        best_payload["replica_energy_std_eV"] = (
            sum((energy - mean_energy) ** 2 for energy in replica_energies) / len(replica_energies)
        ) ** 0.5
    else:
        best_payload["replica_energy_std_eV"] = 0.0
    return best_payload


def _screen_worker(
    gpu_id: str,
    base_structure_path: str,
    rows_chunk: list[dict[str, Any]],
    model_path: str,
    head: str | None,
    default_dtype: str,
    queue: mp.Queue,
) -> None:
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    os.environ.setdefault("MKL_NUM_THREADS", "1")

    try:
        base_atoms = load_base_structure(base_structure_path)
        fe_indices = get_fe_site_indices(base_atoms)
        calc = build_mace_calculator(
            model_path=model_path,
            head=head,
            default_dtype=default_dtype,
            device="cuda",
        )
        base_eval = evaluate_atoms(
            base_atoms,
            calc=calc,
            output_dir=Path(rows_chunk[0]["output_root"]).expanduser().resolve() / f"base_reference_gpu{gpu_id}",
            relax_mode=str(rows_chunk[0]["relax_mode"]),
            fmax=float(rows_chunk[0]["fmax"]),
            max_steps=int(rows_chunk[0]["max_steps"]),
        )
        base_energy = float(base_eval["energy_eV"])
        results = []
        for row in rows_chunk:
            candidate = monte_carlo_screen(
                base_atoms=base_atoms,
                fe_indices=fe_indices,
                dopants=list(row["dopants"]),
                x_total=float(row["x_total"]),
                calc=calc,
                output_dir=Path(str(row["output_dir"])).expanduser().resolve(),
                mc_samples=int(row["mc_samples"]),
                mc_steps=int(row["mc_steps"]),
                mc_temperature_ev=float(row["mc_temperature_ev"]),
                random_seed=int(row["random_seed"]),
                relax_mode=str(row["relax_mode"]),
                fmax=float(row["fmax"]),
                max_steps=int(row["max_steps"]),
            )
            results.append(
                {
                    "index": int(row["index"]),
                    "gpu_id": str(gpu_id),
                    "base_energy_eV": base_energy,
                    "candidate": candidate,
                }
            )
        queue.put({"ok": True, "gpu_id": str(gpu_id), "results": results})
    except Exception as exc:
        queue.put(
            {
                "ok": False,
                "gpu_id": str(gpu_id),
                "error": repr(exc),
                "traceback": traceback.format_exc(),
            }
        )


def screen_candidates_in_parallel(
    *,
    base_structure_path: str,
    rows: list[dict[str, Any]],
    model_path: str,
    head: str | None,
    default_dtype: str,
    gpu_ids: list[str],
) -> list[dict[str, Any]]:
    if not rows:
        return []
    gpu_pool = [str(gpu_id) for gpu_id in gpu_ids if str(gpu_id).strip()]
    if not gpu_pool:
        raise ValueError("At least one GPU id is required for stage1 screening.")
    if len(gpu_pool) == 1 or len(rows) == 1:
        queue_rows = [{"output_root": str(Path(rows[0]["output_dir"]).expanduser().resolve().parent), **row} for row in rows]
        queue: mp.Queue = mp.get_context("spawn").Queue()
        _screen_worker(
            gpu_id=gpu_pool[0],
            base_structure_path=base_structure_path,
            rows_chunk=queue_rows,
            model_path=model_path,
            head=head,
            default_dtype=default_dtype,
            queue=queue,
        )
        payload = queue.get(timeout=5.0)
        if not payload.get("ok"):
            raise RuntimeError(f"Stage1 GPU worker failed: {payload['error']}")
        return payload["results"]

    chunks: list[list[dict[str, Any]]] = [[] for _ in gpu_pool]
    output_root = str(Path(rows[0]["output_dir"]).expanduser().resolve().parent)
    for index, row in enumerate(rows):
        chunks[index % len(gpu_pool)].append({"output_root": output_root, **row})

    ctx = mp.get_context("spawn")
    queue: mp.Queue = ctx.Queue()
    procs: list[mp.Process] = []
    active_gpu_ids: list[str] = []
    for gpu_id, chunk in zip(gpu_pool, chunks, strict=False):
        if not chunk:
            continue
        proc = ctx.Process(
            target=_screen_worker,
            args=(gpu_id, base_structure_path, chunk, model_path, head, default_dtype, queue),
            name=f"stage1_gpu{gpu_id}",
        )
        proc.start()
        procs.append(proc)
        active_gpu_ids.append(gpu_id)

    for proc in procs:
        proc.join()

    results = []
    deadline = time.time() + 10.0
    while len(results) < len(active_gpu_ids) and time.time() < deadline:
        try:
            results.append(queue.get(timeout=0.5))
        except Empty:
            if all(not proc.is_alive() for proc in procs):
                break

    if len(results) != len(active_gpu_ids):
        exit_codes = [proc.exitcode for proc in procs]
        raise RuntimeError(
            f"Missing stage1 GPU worker results. Expected {len(active_gpu_ids)}, got {len(results)}. Exit codes: {exit_codes}"
        )
    errors = [result for result in results if not result.get("ok")]
    if errors:
        raise RuntimeError(f"Stage1 parallel screening failed: {errors[0]['error']}")

    merged = []
    for result in results:
        merged.extend(result["results"])
    merged.sort(key=lambda item: int(item["index"]))
    return merged
