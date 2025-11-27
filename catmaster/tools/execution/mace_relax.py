#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Fast prescreen relaxation utilities (MACE).
"""
from __future__ import annotations

import sys
import logging
from typing import List
import torch
import pandas as pd
from tqdm import tqdm
from ase.io import read as ase_read, write as ase_write
from ase import Atoms
from ase.optimize import FIRE, LBFGS

from .gpu_common import apply_all_constraints, make_mace_calc


def read_structure_poscar(path: str) -> Atoms:
    """
    Read POSCAR/CONTCAR using ASE only.
    ASE will handle Selective dynamics as implemented in its VASP reader.
    """
    return ase_read(path)


def mace_relax(
    structure_files: List[str],
    device: str = 'auto',
    use_d3: bool = False,
    fmax: float = 0.08,
    maxsteps: int = 200,
    optimizer: str = 'FIRE',
    model: str = "medium-mpa-0",
    mode: str = "molecular",
    constraint_entities: str = "auto",
    bond_rt_scale: float = 0.10,
    hooke_k: float = 15.0,
    amp_mode: str = 'auto',
    amp_dtype: str = 'fp16',
    tf32: bool = False,
    ase_log: str = 'stdout',
) -> pd.DataFrame:
    """
    Fast relaxation for all structures, already attaching Hookean constraints when requested.
    Shows tqdm progress and can print ASE optimizer logs to stdout.
    """
    if device == 'auto':
        is_cuda = bool(torch and torch.cuda.is_available())
        device = 'cuda' if is_cuda else 'cpu'
    logging.info(
        f"[prescreen] device={device}, D3={use_d3}, model={model}, amp={amp_mode}/{amp_dtype}, tf32={tf32}"
    )

    calc = make_mace_calc(model=model, use_d3=use_d3, device=device, amp_mode=amp_mode, amp_dtype=amp_dtype, tf32=tf32)

    logfile = None
    if ase_log == 'stdout':
        logfile = sys.stdout
    elif ase_log == 'file':
        logfile = 'prescreen_ase.log'  # per-process single log

    rows = []
    pbar = tqdm(structure_files, desc="Prescreen relax", unit="structure")
    for p in pbar:
        at = read_structure_poscar(p)
        # Constraints from metadata & mode
        apply_all_constraints(
            at,
            seed_poscar=p,
            mode=mode,
            constraint_entities=constraint_entities,
            bond_rt_scale=bond_rt_scale,
            hooke_k=hooke_k,
        )

        at.calc = calc
        opt = FIRE(at, logfile=logfile) if optimizer.upper() == 'FIRE' else LBFGS(at, logfile=logfile)
        try:
            opt.run(fmax=fmax, steps=maxsteps)
            conv = True
        except Exception:
            conv = False
        E = float(at.get_potential_energy())
        # robust relaxed filenames, independent of original suffix
        from pathlib import Path as _Path
        _base = _Path(p)
        p_rel = (_base.with_suffix("").as_posix() + "_relaxed.traj")
        ase_write(p_rel, at)  # constraints usually persisted in .traj
        # Also, write a vasp file if need for downstream use
        vasp_p_rel = (_base.with_suffix("").as_posix() + "_relaxed.vasp")
        ase_write(vasp_p_rel, at, format='vasp')
        rows.append(dict(structure_file=p, relaxed=p_rel, energy=E, converged=conv))
        pbar.set_postfix_str(f"E={E:.3f} eV, conv={conv}")

    df = pd.DataFrame(rows).sort_values("energy").reset_index(drop=True)
    logging.info(f"[prescreen] finished: {len(df)} structures")
    return df


