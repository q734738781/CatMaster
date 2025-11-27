"""
Structured wrapper for inserting H2 gas-phase molecules into slab vacuum.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Mapping, Tuple

import numpy as np
from ase import Atoms
from ase.io import read, write

K_B_SI = 1.380649e-23
ANG3_TO_M3 = 1.0e-30


def random_unit_vector(rng: np.random.Generator) -> np.ndarray:
    u = rng.uniform(-1.0, 1.0)
    phi = rng.uniform(0.0, 2.0 * np.pi)
    s = math.sqrt(max(0.0, 1.0 - u * u))
    return np.array([s * math.cos(phi), s * math.sin(phi), u])


def largest_periodic_gap(zfrac: np.ndarray) -> Tuple[float, float]:
    zs = np.sort(np.mod(zfrac, 1.0))
    diffs = np.diff(zs, append=zs[0] + 1.0)
    idx = int(np.argmax(diffs))
    return float(diffs[idx]), float(zs[idx])


def place_h2_gas(
    slab: Atoms,
    n_h2: int,
    gap_start_frac: float,
    gap_len_frac: float,
    buffer_A: float,
    bond_A: float,
    rng: np.random.Generator,
    max_trials_per_mol: int = 5000,
    min_dist_A: float = 1.6,
) -> Tuple[Atoms, int]:
    atoms = slab.copy()
    cell = atoms.get_cell()
    c_len = np.linalg.norm(cell[2])
    if c_len <= 1e-8:
        raise RuntimeError("c axis length too small for vacuum placement.")
    extra = 0.5 * bond_A / c_len
    usable_len_frac = max(0.0, gap_len_frac - 2.0 * (buffer_A / c_len + extra))
    if usable_len_frac <= 0.0:
        return atoms, 0

    def frac_to_cart(frac):
        return frac[0] * cell[0] + frac[1] * cell[1] + frac[2] * cell[2]

    placed = 0
    for _ in range(n_h2):
        success = False
        for _ in range(max_trials_per_mol):
            xf = rng.uniform(0.0, 1.0)
            yf = rng.uniform(0.0, 1.0)
            zf_local = rng.uniform(0.0, usable_len_frac)
            zf = (gap_start_frac + buffer_A / c_len + extra + zf_local) % 1.0

            center = frac_to_cart(np.array([xf, yf, zf]))
            disp = random_unit_vector(rng)
            disp = disp / np.linalg.norm(disp) * (0.5 * bond_A)
            h1 = center - disp
            h2 = center + disp

            trial = atoms.copy()
            trial += Atoms("H2", positions=[h1, h2], cell=trial.cell, pbc=True)
            i1 = len(trial) - 2
            i2 = len(trial) - 1

            ok = True
            for j in range(len(trial) - 2):
                if trial.get_distance(i1, j, mic=True) < min_dist_A:
                    ok = False
                    break
                if trial.get_distance(i2, j, mic=True) < min_dist_A:
                    ok = False
                    break
            if ok and abs(trial.get_distance(i1, i2, mic=True) - bond_A) > 0.1:
                ok = False
            if ok:
                atoms = trial
                placed += 1
                success = True
                break
        if not success:
            break
    return atoms, placed


def choose_replication_for_targetN(
    base_expected_N: float,
    targetN: float,
    max_rep: int,
    prefer_square: bool = True,
) -> Tuple[int, int]:
    if targetN is None or targetN <= 0 or base_expected_N <= 0:
        return 1, 1
    best = None
    best_score = 1e99
    for mx in range(1, max_rep + 1):
        for my in range(1, max_rep + 1):
            N = base_expected_N * mx * my
            if N + 1e-12 < targetN:
                continue
            ratio = max(mx, my) / min(mx, my)
            score = (N - targetN) + (0.1 * abs(ratio - 1.0) if prefer_square else 0.0)
            if score < best_score:
                best_score = score
                best = (mx, my)
    return best if best else (1, 1)


def integerize(expected_N: float, mode: str, rng: np.random.Generator, min_molecules: int = 0) -> int:
    if mode == "round":
        n = int(round(expected_N))
    elif mode == "ceil":
        n = int(math.ceil(expected_N))
    elif mode == "floor":
        n = int(math.floor(expected_N))
    elif mode == "poisson":
        n = rng.poisson(lam=max(0.0, expected_N))
    else:
        raise ValueError("Unknown rounding mode.")
    return max(n, int(min_molecules))


def fill_h2(config: Mapping[str, object]) -> Mapping[str, object]:
    rng = np.random.default_rng(int(config.get("seed", 42)))
    input_path = Path(config["input"])
    output_path = Path(config.get("output") or input_path.with_suffix(".with_h2.vasp"))
    temperature = float(config.get("temperature", 673.0))
    pressure_mpa = float(config.get("pressure_mpa", 1.0))
    buffer_A = float(config.get("buffer_A", 2.0))
    bond_A = float(config.get("bond_A", 0.74))
    replicate = config.get("replicate")
    targetN = config.get("targetN")
    max_rep = int(config.get("max_rep", 16))
    rounding = config.get("rounding", "round")
    min_molecules = int(config.get("min_molecules", 0))
    summary_json = Path(config["summary_json"]) if config.get("summary_json") else None

    atoms = read(input_path)
    atoms.wrap(eps=1e-9)

    cell = atoms.get_cell()
    area_AB = float(np.linalg.norm(np.cross(cell[0], cell[1])))
    c_len = float(np.linalg.norm(cell[2]))
    zfrac = atoms.get_scaled_positions()[:, 2]
    gap_frac, gap_start = largest_periodic_gap(zfrac)
    vacuum_thick_A = float(gap_frac * c_len)
    usable_len_A = max(0.0, vacuum_thick_A - 2.0 * (buffer_A + 0.5 * bond_A))
    v_gas_A3_base = area_AB * usable_len_A
    v_gas_m3_base = v_gas_A3_base * ANG3_TO_M3

    pressure_pa = pressure_mpa * 1.0e6
    expected_N_base = pressure_pa * v_gas_m3_base / (K_B_SI * temperature)

    if replicate:
        mx, my = int(replicate[0]), int(replicate[1])
    else:
        mx, my = choose_replication_for_targetN(expected_N_base, targetN, max_rep, prefer_square=True)

    if mx != 1 or my != 1:
        atoms = atoms.repeat((mx, my, 1))
        cell = atoms.get_cell()
        area_AB = float(np.linalg.norm(np.cross(cell[0], cell[1])))
        c_len = float(np.linalg.norm(cell[2]))
        zfrac = atoms.get_scaled_positions()[:, 2]
        gap_frac, gap_start = largest_periodic_gap(zfrac)
        vacuum_thick_A = float(gap_frac * c_len)
        usable_len_A = max(0.0, vacuum_thick_A - 2.0 * (buffer_A + 0.5 * bond_A))

    v_gas_A3 = area_AB * usable_len_A
    v_gas_m3 = v_gas_A3 * ANG3_TO_M3
    expected_N = pressure_pa * v_gas_m3 / (K_B_SI * temperature)

    initial_H = sum(1 for s in atoms.get_chemical_symbols() if s == "H")
    n_h2 = integerize(expected_N, mode=rounding, rng=rng, min_molecules=min_molecules)
    atoms_with_gas, placed = place_h2_gas(
        slab=atoms,
        n_h2=n_h2,
        gap_start_frac=gap_start,
        gap_len_frac=gap_frac,
        buffer_A=buffer_A,
        bond_A=bond_A,
        rng=rng,
    )
    final_H = sum(1 for s in atoms_with_gas.get_chemical_symbols() if s == "H")
    placed_H_atoms = max(0, final_H - initial_H)
    placed_molecules = placed_H_atoms // 2

    output_path.parent.mkdir(parents=True, exist_ok=True)
    write(output_path, atoms_with_gas, format="vasp")

    summary = {
        "input": input_path.as_posix(),
        "output": output_path.as_posix(),
        "temperature_K": temperature,
        "pressure_MPa": pressure_mpa,
        "buffer_A": buffer_A,
        "bond_A": bond_A,
        "replicate_mx": mx,
        "replicate_my": my,
        "rounding_mode": rounding,
        "min_molecules": min_molecules,
        "area_AB_A2": area_AB,
        "c_len_A": c_len,
        "vacuum_thick_A_total": vacuum_thick_A,
        "usable_gas_len_A": usable_len_A,
        "gas_volume_A3": v_gas_A3,
        "gas_volume_m3": v_gas_m3,
        "expected_molecules": expected_N,
        "requested_molecules": n_h2,
        "placed_molecules": placed_molecules,
        "placed_H_atoms": placed_H_atoms,
    }

    if summary_json:
        summary_json.parent.mkdir(parents=True, exist_ok=True)
        summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    return {
        "output": summary["output"],
        "placed_molecules": summary["placed_molecules"],
        "placed_H_atoms": summary["placed_H_atoms"],
        "expected_molecules": summary["expected_molecules"],
        "gas_volume_A3": summary["gas_volume_A3"],
        "summary": summary,
    }


