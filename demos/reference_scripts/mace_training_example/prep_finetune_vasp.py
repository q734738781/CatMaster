#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import gc
import hashlib
import json
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path

import numpy as np
from ase.data import atomic_numbers
from ase.io import write
from ase.io.trajectory import Trajectory

try:
    from dscribe.descriptors import SOAP
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: dscribe. Please install it before running prep_finetune.py."
    ) from exc

try:
    from sklearn.cluster import MiniBatchKMeans
    from sklearn.decomposition import PCA
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - runtime dependency check
    raise SystemExit(
        "Missing dependency: scikit-learn. Please install it before running prep_finetune.py."
    ) from exc


@dataclass(frozen=True)
class TrajInfo:
    traj_id: int
    path: Path
    relpath: Path
    traj_code: str
    system_code: str
    strain_code: str
    group_code: str
    n_frames: int


@dataclass(frozen=True)
class CandidateRef:
    candidate_id: int
    traj_id: int
    frame_index: int


_WORKER_SOAP = None
_WORKER_TRAJ_PATHS: list[str] = []
_WORKER_TRAJ_CACHE: dict[int, Trajectory] = {}


def _sanitize_token(token: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.+\-]+", "-", token.strip())


def _build_traj_code(relpath: Path) -> str:
    parts = list(relpath.parts)
    if parts:
        parts[-1] = Path(parts[-1]).stem
    return "__".join(_sanitize_token(p) for p in parts)


def _infer_system_and_strain(relpath: Path) -> tuple[str, str, str]:
    system_code = ""
    strain_code = ""
    for part in relpath.parts:
        if part.startswith("scale_"):
            strain_code = part
        elif "__" in part and not part.endswith(".traj"):
            system_code = part
    if not system_code:
        system_code = relpath.parent.name
    if not strain_code:
        strain_code = "unknown_strain"
    return system_code, strain_code, f"{system_code}__{strain_code}"


def discover_trajectories(
    root: Path, pattern: str, exclude_regex: str | None
) -> list[TrajInfo]:
    regex = re.compile(exclude_regex) if exclude_regex else None
    traj_paths = sorted(p for p in root.rglob(pattern) if p.is_file())
    infos: list[TrajInfo] = []
    for path in traj_paths:
        rel = path.relative_to(root)
        if regex and regex.search(rel.as_posix()):
            continue
        try:
            traj = Trajectory(str(path))
            n_frames = len(traj)
        except Exception as exc:
            print(f"[WARN] Skip unreadable trajectory: {rel} ({exc})")
            continue
        if n_frames <= 0:
            continue
        traj_code = _build_traj_code(rel)
        system_code, strain_code, group_code = _infer_system_and_strain(rel)
        infos.append(
            TrajInfo(
                traj_id=len(infos),
                path=path,
                relpath=rel,
                traj_code=traj_code,
                system_code=system_code,
                strain_code=strain_code,
                group_code=group_code,
                n_frames=n_frames,
            )
        )
    return infos


def stratified_temporal_sample_indices(
    n_frames: int,
    burn_frames: int,
    stride: int,
    window_frames: int,
    rng: np.random.Generator,
) -> list[int]:
    if n_frames <= burn_frames:
        return []
    stride = max(1, stride)
    window_frames = max(1, window_frames)
    sampled: list[int] = []
    for ws in range(burn_frames, n_frames, window_frames):
        we = min(ws + window_frames, n_frames)
        if ws >= we:
            continue
        offset = int(rng.integers(0, stride))
        first = ws + offset
        if first >= we:
            first = we - 1
        sampled.extend(range(first, we, stride))
    return sampled


def _frame_id(traj_code: str, frame_index: int) -> str:
    return f"{traj_code}__f{frame_index:06d}"


def _safe_output_name(frame_id: str, max_len: int = 180) -> str:
    safe = _sanitize_token(frame_id)
    if len(safe) <= max_len:
        return safe
    digest = hashlib.sha1(safe.encode("utf-8")).hexdigest()[:12]
    return f"{safe[:max_len - 14]}__{digest}"


def _collect_species(traj_infos: list[TrajInfo]) -> list[str]:
    species = set()
    for info in traj_infos:
        traj = Trajectory(str(info.path))
        symbols = traj[0].get_chemical_symbols()
        species.update(symbols)
    return sorted(species, key=lambda s: atomic_numbers.get(s, 999))


def _build_soap(
    species: list[str],
    soap_rcut: float,
    soap_nmax: int,
    soap_lmax: int,
    soap_sigma: float,
) -> SOAP:
    return SOAP(
        r_cut=soap_rcut,
        n_max=soap_nmax,
        l_max=soap_lmax,
        species=species,
        periodic=True,
        sparse=False,
        sigma=soap_sigma,
        compression={"mode": "mu1nu1"},
        average="off",
    )


def _get_soap_dim(soap: SOAP, atoms) -> int:
    if hasattr(soap, "get_number_of_features"):
        return int(soap.get_number_of_features())
    centers = np.arange(len(atoms), dtype=int)
    mat = np.asarray(soap.create(atoms, centers=centers))
    if mat.ndim == 1:
        return int(mat.shape[0])
    return int(mat.shape[1])


def _descriptor_from_atomwise_soap(mat: np.ndarray) -> np.ndarray:
    if mat.ndim == 1:
        desc = mat
    elif mat.ndim == 2:
        desc = np.mean(mat, axis=0)
    else:
        raise RuntimeError(
            f"Unexpected SOAP output shape for single frame: {mat.shape}. "
            "Expected 1D or 2D SOAP output."
        )
    return desc.astype(np.float32, copy=False)


def _init_soap_worker(
    traj_paths: list[str],
    species: list[str],
    soap_rcut: float,
    soap_nmax: int,
    soap_lmax: int,
    soap_sigma: float,
) -> None:
    global _WORKER_SOAP, _WORKER_TRAJ_PATHS, _WORKER_TRAJ_CACHE
    _WORKER_TRAJ_PATHS = traj_paths
    _WORKER_TRAJ_CACHE = {}
    _WORKER_SOAP = _build_soap(
        species=species,
        soap_rcut=soap_rcut,
        soap_nmax=soap_nmax,
        soap_lmax=soap_lmax,
        soap_sigma=soap_sigma,
    )


def _process_soap_chunk(
    chunk: list[tuple[int, int, int]],
) -> tuple[np.ndarray, np.ndarray, float, float, float, int]:
    candidate_ids = []
    descriptors = []
    t_prepare = 0.0
    t_soap = 0.0
    t_reduce = 0.0

    for candidate_id, traj_id, frame_index in chunk:
        t0 = time.perf_counter()
        traj = _WORKER_TRAJ_CACHE.get(traj_id)
        if traj is None:
            traj = Trajectory(_WORKER_TRAJ_PATHS[traj_id])
            _WORKER_TRAJ_CACHE[traj_id] = traj
        atoms = traj[frame_index]
        centers = np.arange(len(atoms), dtype=int)
        t_prepare += time.perf_counter() - t0

        t1 = time.perf_counter()
        atomwise = np.asarray(_WORKER_SOAP.create(atoms, centers=centers))
        t_soap += time.perf_counter() - t1
        if atomwise.ndim != 2:
            raise RuntimeError(
                f"Unexpected SOAP output shape for single frame: {atomwise.shape}. "
                "Expected (n_centers, n_features)."
            )
        if atomwise.shape[0] != len(atoms):
            raise RuntimeError(
                f"Unexpected center count: got {atomwise.shape[0]}, expected {len(atoms)}."
            )

        t2 = time.perf_counter()
        desc = _descriptor_from_atomwise_soap(atomwise)
        t_reduce += time.perf_counter() - t2

        candidate_ids.append(candidate_id)
        descriptors.append(desc)

    ids = np.asarray(candidate_ids, dtype=np.int64)
    desc_arr = np.asarray(descriptors, dtype=np.float32)
    return ids, desc_arr, t_prepare, t_soap, t_reduce, len(candidate_ids)


def _compute_descriptors_serial_for_refs(
    traj_infos: list[TrajInfo],
    refs: list[CandidateRef],
    soap: SOAP,
) -> np.ndarray:
    if not refs:
        return np.empty((0, 0), dtype=np.float32)

    traj_cache: dict[int, Trajectory] = {}
    out = []
    for ref in refs:
        traj = traj_cache.get(ref.traj_id)
        if traj is None:
            traj = Trajectory(str(traj_infos[ref.traj_id].path))
            traj_cache[ref.traj_id] = traj
        atoms = traj[ref.frame_index]
        centers = np.arange(len(atoms), dtype=int)
        atomwise = np.asarray(soap.create(atoms, centers=centers))
        if atomwise.ndim != 2:
            raise RuntimeError(
                f"Unexpected SOAP output shape for single frame: {atomwise.shape}. "
                "Expected (n_centers, n_features)."
            )
        if atomwise.shape[0] != len(atoms):
            raise RuntimeError(
                f"Unexpected center count: got {atomwise.shape[0]}, expected {len(atoms)}."
            )
        out.append(_descriptor_from_atomwise_soap(atomwise))
    return np.asarray(out, dtype=np.float32)


def _compute_descriptors_parallel_for_refs(
    traj_infos: list[TrajInfo],
    refs: list[CandidateRef],
    soap_workers: int,
    soap_chunk_size: int,
    soap_species: list[str],
    soap_rcut: float,
    soap_nmax: int,
    soap_lmax: int,
    soap_sigma: float,
) -> np.ndarray:
    if not refs:
        return np.empty((0, 0), dtype=np.float32)
    if soap_workers <= 1:
        raise ValueError("soap_workers must be >1 for parallel descriptor verification.")

    first_ref = refs[0]
    first_atoms = Trajectory(str(traj_infos[first_ref.traj_id].path))[first_ref.frame_index]
    feat_dim = _get_soap_dim(
        _build_soap(
            species=soap_species,
            soap_rcut=soap_rcut,
            soap_nmax=soap_nmax,
            soap_lmax=soap_lmax,
            soap_sigma=soap_sigma,
        ),
        first_atoms,
    )
    out = np.empty((len(refs), feat_dim), dtype=np.float32)

    work_items = [(i, ref.traj_id, ref.frame_index) for i, ref in enumerate(refs)]
    chunk_size = max(1, int(soap_chunk_size))
    chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]
    traj_paths = [str(info.path) for info in traj_infos]

    with ProcessPoolExecutor(
        max_workers=soap_workers,
        initializer=_init_soap_worker,
        initargs=(
            traj_paths,
            soap_species,
            soap_rcut,
            soap_nmax,
            soap_lmax,
            soap_sigma,
        ),
    ) as ex:
        futures = [ex.submit(_process_soap_chunk, chunk) for chunk in chunks]
        for fut in as_completed(futures):
            ids, desc_arr, _, _, _, _ = fut.result()
            out[ids, :] = desc_arr

    return out


def _verify_parallel_consistency(
    traj_infos: list[TrajInfo],
    candidates: list[CandidateRef],
    soap: SOAP,
    soap_workers: int,
    soap_chunk_size: int,
    soap_species: list[str],
    soap_rcut: float,
    soap_nmax: int,
    soap_lmax: int,
    soap_sigma: float,
    verify_samples: int,
    verify_atol: float,
    verify_rtol: float,
    rng: np.random.Generator,
) -> None:
    if soap_workers <= 1:
        print("[Verify] Skipped: --soap-workers <= 1, no parallel path to compare.")
        return
    if not candidates:
        print("[Verify] Skipped: no candidates.")
        return

    n = min(max(1, int(verify_samples)), len(candidates))
    sample_idx = rng.choice(len(candidates), size=n, replace=False)
    sample_idx.sort()
    sample_refs = [candidates[int(i)] for i in sample_idx]

    t0 = time.perf_counter()
    serial_desc = _compute_descriptors_serial_for_refs(
        traj_infos=traj_infos,
        refs=sample_refs,
        soap=soap,
    )
    t_serial = time.perf_counter() - t0

    t1 = time.perf_counter()
    parallel_desc = _compute_descriptors_parallel_for_refs(
        traj_infos=traj_infos,
        refs=sample_refs,
        soap_workers=soap_workers,
        soap_chunk_size=soap_chunk_size,
        soap_species=soap_species,
        soap_rcut=soap_rcut,
        soap_nmax=soap_nmax,
        soap_lmax=soap_lmax,
        soap_sigma=soap_sigma,
    )
    t_parallel = time.perf_counter() - t1

    abs_diff = np.abs(serial_desc - parallel_desc)
    max_abs = float(abs_diff.max()) if abs_diff.size else 0.0
    mean_abs = float(abs_diff.mean()) if abs_diff.size else 0.0
    ok = np.allclose(
        serial_desc,
        parallel_desc,
        atol=float(verify_atol),
        rtol=float(verify_rtol),
    )

    serial_fps = len(sample_refs) / max(t_serial, 1e-12)
    parallel_fps = len(sample_refs) / max(t_parallel, 1e-12)
    speedup = parallel_fps / max(serial_fps, 1e-12)
    print(
        f"[Verify] samples={len(sample_refs)} | "
        f"serial={t_serial:.3f}s ({serial_fps:.2f} frame/s) | "
        f"parallel={t_parallel:.3f}s ({parallel_fps:.2f} frame/s) | "
        f"speedup={speedup:.2f}x"
    )
    print(
        f"[Verify] allclose={ok} | max_abs_diff={max_abs:.3e} | mean_abs_diff={mean_abs:.3e} | "
        f"atol={verify_atol:.1e} rtol={verify_rtol:.1e}"
    )
    if not ok:
        raise RuntimeError(
            "Parallel descriptor verification failed: serial and parallel results differ."
        )


def _compute_soap_features(
    out_dir: Path,
    traj_infos: list[TrajInfo],
    candidates: list[CandidateRef],
    soap: SOAP,
    soap_workers: int,
    soap_chunk_size: int,
    soap_species: list[str],
    soap_rcut: float,
    soap_nmax: int,
    soap_lmax: int,
    soap_sigma: float,
    log_every: int,
) -> tuple[np.memmap, int]:
    if not candidates:
        raise ValueError("No candidates for SOAP feature generation.")

    first_ref = candidates[0]
    first_info = traj_infos[first_ref.traj_id]
    first_atoms = Trajectory(str(first_info.path))[first_ref.frame_index]
    feat_dim = _get_soap_dim(soap, first_atoms)

    feat_path = out_dir / "soap_features.float32.dat"
    feats = np.memmap(
        feat_path,
        mode="w+",
        dtype=np.float32,
        shape=(len(candidates), feat_dim),
    )

    traj_ids = np.array([c.traj_id for c in candidates], dtype=np.int64)
    frame_indices = np.array([c.frame_index for c in candidates], dtype=np.int64)
    order = np.lexsort((frame_indices, traj_ids))

    total = len(candidates)
    done = 0
    t_start = time.perf_counter()
    t_last = t_start
    done_last = 0
    t_prepare_sum = 0.0
    t_soap_sum = 0.0
    t_reduce_sum = 0.0
    t_prepare_last = 0.0
    t_soap_last = 0.0
    t_reduce_last = 0.0
    next_log = log_every if log_every > 0 else total + 1

    if soap_workers <= 1:
        traj_cache: dict[int, Trajectory] = {}
        for candidate_idx in order:
            t_prepare_0 = time.perf_counter()
            ref = candidates[int(candidate_idx)]
            info = traj_infos[ref.traj_id]
            traj = traj_cache.get(ref.traj_id)
            if traj is None:
                traj = Trajectory(str(info.path))
                traj_cache[ref.traj_id] = traj
            atoms = traj[ref.frame_index]
            centers = np.arange(len(atoms), dtype=int)
            t_prepare_sum += time.perf_counter() - t_prepare_0

            t_soap_0 = time.perf_counter()
            atomwise = np.asarray(soap.create(atoms, centers=centers))
            t_soap_sum += time.perf_counter() - t_soap_0
            if atomwise.ndim != 2:
                raise RuntimeError(
                    f"Unexpected SOAP output shape for single frame: {atomwise.shape}. "
                    "Expected (n_centers, n_features)."
                )
            if atomwise.shape[0] != len(atoms):
                raise RuntimeError(
                    f"Unexpected center count: got {atomwise.shape[0]}, expected {len(atoms)}."
                )

            t_reduce_0 = time.perf_counter()
            feats[ref.candidate_id, :] = _descriptor_from_atomwise_soap(atomwise)
            t_reduce_sum += time.perf_counter() - t_reduce_0
            done += 1

            while done >= next_log:
                t_now = time.perf_counter()
                elapsed = max(t_now - t_start, 1e-12)
                interval = max(t_now - t_last, 1e-12)
                fps_avg = done / elapsed
                fps_recent = (done - done_last) / interval

                prof_total = max(t_prepare_sum + t_soap_sum + t_reduce_sum, 1e-12)
                prep_pct = 100.0 * t_prepare_sum / prof_total
                soap_pct = 100.0 * t_soap_sum / prof_total
                reduce_pct = 100.0 * t_reduce_sum / prof_total

                prep_recent = t_prepare_sum - t_prepare_last
                soap_recent = t_soap_sum - t_soap_last
                reduce_recent = t_reduce_sum - t_reduce_last
                prof_recent = max(prep_recent + soap_recent + reduce_recent, 1e-12)
                prep_recent_pct = 100.0 * prep_recent / prof_recent
                soap_recent_pct = 100.0 * soap_recent / prof_recent
                reduce_recent_pct = 100.0 * reduce_recent / prof_recent

                print(
                    f"[Step2] SOAP progress: {done}/{total} | "
                    f"avg {fps_avg:.2f} frame/s | recent {fps_recent:.2f} frame/s"
                )
                print(
                    f"[Step2] Time split total: prep={t_prepare_sum:.2f}s ({prep_pct:.1f}%), "
                    f"soap={t_soap_sum:.2f}s ({soap_pct:.1f}%), "
                    f"frame-avg={t_reduce_sum:.2f}s ({reduce_pct:.1f}%)"
                )
                print(
                    f"[Step2] Time split recent: prep={prep_recent_pct:.1f}%, "
                    f"soap={soap_recent_pct:.1f}%, frame-avg={reduce_recent_pct:.1f}%"
                )
                print(
                    f"[Step2] Time/frame(ms): prep={1000.0 * t_prepare_sum / done:.3f}, "
                    f"soap={1000.0 * t_soap_sum / done:.3f}, "
                    f"frame-avg={1000.0 * t_reduce_sum / done:.3f}"
                )
                t_last = t_now
                done_last = done
                t_prepare_last = t_prepare_sum
                t_soap_last = t_soap_sum
                t_reduce_last = t_reduce_sum
                next_log += log_every
    else:
        traj_paths = [str(info.path) for info in traj_infos]
        ordered_refs = [candidates[int(i)] for i in order]
        work_items = [
            (ref.candidate_id, ref.traj_id, ref.frame_index) for ref in ordered_refs
        ]
        chunk_size = max(1, int(soap_chunk_size))
        chunks = [work_items[i : i + chunk_size] for i in range(0, len(work_items), chunk_size)]

        with ProcessPoolExecutor(
            max_workers=soap_workers,
            initializer=_init_soap_worker,
            initargs=(
                traj_paths,
                soap_species,
                soap_rcut,
                soap_nmax,
                soap_lmax,
                soap_sigma,
            ),
        ) as ex:
            futures = [ex.submit(_process_soap_chunk, chunk) for chunk in chunks]
            for fut in as_completed(futures):
                ids, desc_arr, t_p, t_s, t_r, n_done = fut.result()
                feats[ids, :] = desc_arr
                done += n_done
                t_prepare_sum += t_p
                t_soap_sum += t_s
                t_reduce_sum += t_r

                while done >= next_log:
                    t_now = time.perf_counter()
                    elapsed = max(t_now - t_start, 1e-12)
                    interval = max(t_now - t_last, 1e-12)
                    fps_avg = done / elapsed
                    fps_recent = (done - done_last) / interval

                    prof_total = max(t_prepare_sum + t_soap_sum + t_reduce_sum, 1e-12)
                    prep_pct = 100.0 * t_prepare_sum / prof_total
                    soap_pct = 100.0 * t_soap_sum / prof_total
                    reduce_pct = 100.0 * t_reduce_sum / prof_total

                    prep_recent = t_prepare_sum - t_prepare_last
                    soap_recent = t_soap_sum - t_soap_last
                    reduce_recent = t_reduce_sum - t_reduce_last
                    prof_recent = max(prep_recent + soap_recent + reduce_recent, 1e-12)
                    prep_recent_pct = 100.0 * prep_recent / prof_recent
                    soap_recent_pct = 100.0 * soap_recent / prof_recent
                    reduce_recent_pct = 100.0 * reduce_recent / prof_recent

                    print(
                        f"[Step2] SOAP progress: {done}/{total} | "
                        f"avg {fps_avg:.2f} frame/s | recent {fps_recent:.2f} frame/s"
                    )
                    print(
                        f"[Step2] Time split total: prep={t_prepare_sum:.2f}s ({prep_pct:.1f}%), "
                        f"soap={t_soap_sum:.2f}s ({soap_pct:.1f}%), "
                        f"frame-avg={t_reduce_sum:.2f}s ({reduce_pct:.1f}%)"
                    )
                    print(
                        f"[Step2] Time split recent: prep={prep_recent_pct:.1f}%, "
                        f"soap={soap_recent_pct:.1f}%, frame-avg={reduce_recent_pct:.1f}%"
                    )
                    print(
                        f"[Step2] Time/frame(ms): prep={1000.0 * t_prepare_sum / done:.3f}, "
                        f"soap={1000.0 * t_soap_sum / done:.3f}, "
                        f"frame-avg={1000.0 * t_reduce_sum / done:.3f}"
                    )
                    t_last = t_now
                    done_last = done
                    t_prepare_last = t_prepare_sum
                    t_soap_last = t_soap_sum
                    t_reduce_last = t_reduce_sum
                    next_log += log_every

    if done > 0:
        elapsed_total = max(time.perf_counter() - t_start, 1e-12)
        print(f"[Step2] SOAP done: {done}/{total} | avg {done / elapsed_total:.2f} frame/s")
        prof_total = t_prepare_sum + t_soap_sum + t_reduce_sum
        misc_time = max(0.0, elapsed_total - prof_total)
        denom = max(prof_total + misc_time, 1e-12)
        stages = {
            "prep": t_prepare_sum,
            "soap": t_soap_sum,
            "frame-avg": t_reduce_sum,
            "misc": misc_time,
        }
        bottleneck = max(stages, key=stages.get)
        print(
            f"[Step2] Final timing: prep={t_prepare_sum:.3f}s, soap={t_soap_sum:.3f}s, "
            f"frame-avg={t_reduce_sum:.3f}s, misc={misc_time:.3f}s"
        )
        print(
            f"[Step2] Final share: prep={100.0 * t_prepare_sum / denom:.1f}%, "
            f"soap={100.0 * t_soap_sum / denom:.1f}%, "
            f"frame-avg={100.0 * t_reduce_sum / denom:.1f}%, "
            f"misc={100.0 * misc_time / denom:.1f}% | bottleneck={bottleneck}"
        )
        print(
            f"[Step2] Final time/frame(ms): prep={1000.0 * t_prepare_sum / done:.3f}, "
            f"soap={1000.0 * t_soap_sum / done:.3f}, "
            f"frame-avg={1000.0 * t_reduce_sum / done:.3f}"
        )

    feats.flush()
    return feats, feat_dim


def _fit_and_transform_pca(
    out_dir: Path,
    feats: np.memmap,
    pca_dim: int,
    pca_fit_max: int,
    batch_size: int,
    seed: int,
) -> tuple[np.memmap, PCA, np.ndarray]:
    n_samples, feat_dim = feats.shape
    if n_samples < 2:
        raise ValueError("Need at least 2 samples for PCA.")

    scaler = StandardScaler(with_mean=True, with_std=True)
    scaler.fit(feats)

    rng = np.random.default_rng(seed)
    if pca_fit_max > 0 and n_samples > pca_fit_max:
        fit_idx = rng.choice(n_samples, size=pca_fit_max, replace=False)
        fit_idx.sort()
    else:
        fit_idx = np.arange(n_samples, dtype=np.int64)

    n_comp = min(int(pca_dim), int(feat_dim), int(fit_idx.size))
    if n_comp < 1:
        raise ValueError("Invalid PCA dimension after clipping.")

    x_fit = np.asarray(feats[fit_idx], dtype=np.float32)
    x_fit = (x_fit - scaler.mean_) / scaler.scale_
    pca = PCA(n_components=n_comp, svd_solver="randomized", random_state=seed)
    pca.fit(x_fit)

    z_path = out_dir / "pca_features.float32.dat"
    z = np.memmap(z_path, mode="w+", dtype=np.float32, shape=(n_samples, n_comp))
    for start in range(0, n_samples, batch_size):
        end = min(start + batch_size, n_samples)
        x_batch = np.asarray(feats[start:end], dtype=np.float32)
        x_batch = (x_batch - scaler.mean_) / scaler.scale_
        z[start:end] = pca.transform(x_batch).astype(np.float32, copy=False)
    z.flush()
    return z, pca, fit_idx


def _pick_cluster_representatives(
    z: np.memmap,
    k_dft: int,
    kmeans_batch: int,
    kmeans_max_iter: int,
    seed: int,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    n_samples = z.shape[0]
    k = min(int(k_dft), int(n_samples))
    if k < 1:
        raise ValueError("k_dft must be >= 1.")

    kmeans = MiniBatchKMeans(
        n_clusters=k,
        batch_size=kmeans_batch,
        max_iter=kmeans_max_iter,
        n_init=10,
        random_state=seed,
    )
    kmeans.fit(z)

    best_dist2 = np.full(k, np.inf, dtype=np.float64)
    best_idx = np.full(k, -1, dtype=np.int64)

    for start in range(0, n_samples, kmeans_batch):
        end = min(start + kmeans_batch, n_samples)
        batch = np.asarray(z[start:end], dtype=np.float32)
        labels = kmeans.predict(batch)
        centers = kmeans.cluster_centers_[labels]
        dist2 = np.einsum("ij,ij->i", batch - centers, batch - centers)
        for i, cid in enumerate(labels):
            d = float(dist2[i])
            if d < best_dist2[cid]:
                best_dist2[cid] = d
                best_idx[cid] = start + i

    valid_clusters = np.where(best_idx >= 0)[0]
    return valid_clusters, best_idx, best_dist2


def _write_candidate_pool_csv(
    out_csv: Path,
    traj_infos: list[TrajInfo],
    candidates: list[CandidateRef],
    timestep_fs: float,
) -> None:
    out_csv.parent.mkdir(parents=True, exist_ok=True)
    with out_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "candidate_id",
                "frame_id",
                "traj_id",
                "traj_relpath",
                "system_code",
                "strain_code",
                "group_code",
                "frame_index",
                "time_ps",
            ]
        )
        for ref in candidates:
            info = traj_infos[ref.traj_id]
            writer.writerow(
                [
                    ref.candidate_id,
                    _frame_id(info.traj_code, ref.frame_index),
                    ref.traj_id,
                    info.relpath.as_posix(),
                    info.system_code,
                    info.strain_code,
                    info.group_code,
                    ref.frame_index,
                    f"{ref.frame_index * timestep_fs / 1000.0:.6f}",
                ]
            )


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Prepare representative fine-tuning frames from MD trajectories."
    )
    parser.add_argument(
        "--traj-root",
        type=Path,
        default=Path("MD_Files"),
        help="Root directory for recursive *.traj search.",
    )
    parser.add_argument(
        "--traj-pattern",
        type=str,
        default="*.traj",
        help="Glob pattern used under --traj-root.",
    )
    parser.add_argument(
        "--exclude-regex",
        type=str,
        default=None,
        help="Optional regex on relative traj path to exclude files.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("DFT_files/finetune_prep"),
        help="Output directory for pool metadata and selected structures.",
    )
    parser.add_argument("--seed", type=int, default=42, help="RNG seed.")

    parser.add_argument("--timestep-fs", type=float, default=1.0, help="MD timestep (fs).")
    parser.add_argument(
        "--burn-ps", type=float, default=2.0, help="Drop initial burn-in window (ps)."
    )
    parser.add_argument(
        "--stride",
        type=int,
        default=15,
        help="Base stride in Step1 coarse temporal sampling.",
    )
    parser.add_argument(
        "--window-ps",
        type=float,
        default=1.0,
        help="Temporal stratification window length (ps).",
    )
    parser.add_argument(
        "--max-candidates",
        type=int,
        default=0,
        help="Global candidate cap after Step1 (<=0 means no cap).",
    )

    parser.add_argument("--soap-rcut", type=float, default=6.0, help="SOAP rcut / r_cut.")
    parser.add_argument("--soap-nmax", type=int, default=8, help="SOAP nmax / n_max.")
    parser.add_argument("--soap-lmax", type=int, default=6, help="SOAP lmax / l_max.")
    parser.add_argument("--soap-sigma", type=float, default=0.5, help="SOAP sigma.")
    parser.add_argument(
        "--soap-workers",
        type=int,
        default=1,
        help="Number of outer-process workers for SOAP/descriptor calculation.",
    )
    parser.add_argument(
        "--soap-chunk-size",
        type=int,
        default=32,
        help="Number of frames handled per worker task in outer parallel SOAP mode.",
    )

    parser.add_argument("--pca-dim", type=int, default=256, help="PCA output dimension.")
    parser.add_argument(
        "--pca-fit-max",
        type=int,
        default=0,
        help="Max samples used to fit randomized PCA (0 means fit on all candidates).",
    )
    parser.add_argument(
        "--transform-batch-size",
        type=int,
        default=4096,
        help="Batch size for PCA transform over all candidates.",
    )

    parser.add_argument(
        "--k-dft",
        type=int,
        default=4000,
        help="Target selected_for_DFT size (K clusters).",
    )
    parser.add_argument(
        "--kmeans-batch-size",
        type=int,
        default=4096,
        help="MiniBatchKMeans batch size.",
    )
    parser.add_argument(
        "--kmeans-max-iter",
        type=int,
        default=200,
        help="MiniBatchKMeans max_iter.",
    )
    parser.add_argument(
        "--log-every",
        type=int,
        default=2000,
        help="Progress log interval for Step2 SOAP.",
    )
    parser.add_argument(
        "--verify-parallel",
        action="store_true",
        help="Verify serial vs parallel descriptor consistency on a random candidate subset.",
    )
    parser.add_argument(
        "--verify-samples",
        type=int,
        default=128,
        help="Number of sampled candidates for --verify-parallel.",
    )
    parser.add_argument(
        "--verify-atol",
        type=float,
        default=1e-6,
        help="Absolute tolerance for --verify-parallel allclose check.",
    )
    parser.add_argument(
        "--verify-rtol",
        type=float,
        default=1e-5,
        help="Relative tolerance for --verify-parallel allclose check.",
    )
    parser.add_argument(
        "--keep-memmap",
        action="store_true",
        help="Keep intermediate SOAP/PCA memmap files (default: clean them after selection).",
    )

    args = parser.parse_args()

    if not args.traj_root.exists():
        raise FileNotFoundError(f"--traj-root not found: {args.traj_root}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(args.seed)
    burn_frames = int(round(args.burn_ps * 1000.0 / args.timestep_fs))
    window_frames = max(1, int(round(args.window_ps * 1000.0 / args.timestep_fs)))

    print("[Init] Discovering trajectories...")
    traj_infos = discover_trajectories(args.traj_root, args.traj_pattern, args.exclude_regex)
    if not traj_infos:
        raise RuntimeError("No valid trajectories found.")
    print(f"[Init] Trajectories found: {len(traj_infos)}")

    print("[Step1] Temporal stratified coarse sampling...")
    candidates: list[CandidateRef] = []
    for info in traj_infos:
        sampled_idx = stratified_temporal_sample_indices(
            n_frames=info.n_frames,
            burn_frames=burn_frames,
            stride=args.stride,
            window_frames=window_frames,
            rng=rng,
        )
        for idx in sampled_idx:
            candidates.append(
                CandidateRef(
                    candidate_id=len(candidates),
                    traj_id=info.traj_id,
                    frame_index=int(idx),
                )
            )
    if not candidates:
        raise RuntimeError("Step1 produced 0 candidates; check burn/stride/window settings.")
    print(f"[Step1] Candidates before cap: {len(candidates)}")

    if args.max_candidates > 0 and len(candidates) > args.max_candidates:
        keep = rng.choice(len(candidates), size=args.max_candidates, replace=False)
        keep.sort()
        candidates = [
            CandidateRef(
                candidate_id=new_id,
                traj_id=candidates[int(old_id)].traj_id,
                frame_index=candidates[int(old_id)].frame_index,
            )
            for new_id, old_id in enumerate(keep)
        ]
        print(f"[Step1] Candidates capped to: {len(candidates)}")

    candidate_csv = args.out_dir / "candidate_pool.csv"
    _write_candidate_pool_csv(candidate_csv, traj_infos, candidates, args.timestep_fs)
    print(f"[Step1] candidate_pool saved: {candidate_csv}")

    print("[Step2] Building SOAP descriptor...")
    species = _collect_species(traj_infos)
    print(f"[Step2] Species used in SOAP: {species}")
    mode = "outer-process parallel" if args.soap_workers > 1 else "single-frame serial"
    print(
        f"[Step2] SOAP mode: {mode} | workers={args.soap_workers} | "
        f"chunk_size={args.soap_chunk_size}"
    )
    soap = _build_soap(
        species=species,
        soap_rcut=args.soap_rcut,
        soap_nmax=args.soap_nmax,
        soap_lmax=args.soap_lmax,
        soap_sigma=args.soap_sigma,
    )
    if args.verify_parallel:
        _verify_parallel_consistency(
            traj_infos=traj_infos,
            candidates=candidates,
            soap=soap,
            soap_workers=args.soap_workers,
            soap_chunk_size=args.soap_chunk_size,
            soap_species=species,
            soap_rcut=args.soap_rcut,
            soap_nmax=args.soap_nmax,
            soap_lmax=args.soap_lmax,
            soap_sigma=args.soap_sigma,
            verify_samples=args.verify_samples,
            verify_atol=args.verify_atol,
            verify_rtol=args.verify_rtol,
            rng=rng,
        )
    soap_memmap_path = args.out_dir / "soap_features.float32.dat"
    pca_memmap_path = args.out_dir / "pca_features.float32.dat"

    feats, feat_dim = _compute_soap_features(
        out_dir=args.out_dir,
        traj_infos=traj_infos,
        candidates=candidates,
        soap=soap,
        soap_workers=args.soap_workers,
        soap_chunk_size=args.soap_chunk_size,
        soap_species=species,
        soap_rcut=args.soap_rcut,
        soap_nmax=args.soap_nmax,
        soap_lmax=args.soap_lmax,
        soap_sigma=args.soap_sigma,
        log_every=args.log_every,
    )
    print(f"[Step2] SOAP feature matrix shape: ({len(candidates)}, {feat_dim})")

    print("[Step3] Standardize + PCA (randomized)...")
    z, pca, fit_idx = _fit_and_transform_pca(
        out_dir=args.out_dir,
        feats=feats,
        pca_dim=args.pca_dim,
        pca_fit_max=args.pca_fit_max,
        batch_size=args.transform_batch_size,
        seed=args.seed,
    )
    print(f"[Step3] PCA feature matrix shape: {z.shape}")
    pca_dim_out = int(z.shape[1])

    print("[Step4] MiniBatchKMeans representative selection...")
    valid_clusters, rep_idx, rep_dist2 = _pick_cluster_representatives(
        z=z,
        k_dft=args.k_dft,
        kmeans_batch=args.kmeans_batch_size,
        kmeans_max_iter=args.kmeans_max_iter,
        seed=args.seed,
    )

    selected_csv = args.out_dir / "selected_for_DFT.csv"
    structures_dir = args.out_dir / "selected_structures_vasp"
    structures_dir.mkdir(parents=True, exist_ok=True)
    with selected_csv.open("w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(
            [
                "selected_rank",
                "cluster_id",
                "dist2_to_centroid",
                "candidate_id",
                "frame_id",
                "traj_id",
                "traj_relpath",
                "system_code",
                "strain_code",
                "group_code",
                "frame_index",
                "time_ps",
                "vasp_file",
            ]
        )
        rank = 0
        for cid in valid_clusters:
            idx = int(rep_idx[cid])
            if idx < 0:
                continue
            rank += 1
            ref = candidates[idx]
            info = traj_infos[ref.traj_id]
            frame_id = _frame_id(info.traj_code, ref.frame_index)
            vasp_name = f"{rank:05d}__{_safe_output_name(frame_id)}.vasp"
            vasp_path = structures_dir / vasp_name
            traj = Trajectory(str(info.path))
            atoms = traj[ref.frame_index]
            write(vasp_path, atoms, format="vasp", vasp5=True, direct=True, sort=True)

            writer.writerow(
                [
                    rank,
                    int(cid),
                    f"{rep_dist2[cid]:.8e}",
                    ref.candidate_id,
                    frame_id,
                    ref.traj_id,
                    info.relpath.as_posix(),
                    info.system_code,
                    info.strain_code,
                    info.group_code,
                    ref.frame_index,
                    f"{ref.frame_index * args.timestep_fs / 1000.0:.6f}",
                    vasp_path.relative_to(args.out_dir).as_posix(),
                ]
            )

    removed_memmap_files: list[str] = []
    if args.keep_memmap:
        print("[Done] Keep intermediate memmap files (--keep-memmap enabled).")
    else:
        print("[Done] Cleaning intermediate memmap files...")
        del feats
        del z
        gc.collect()
        for mm_path in (soap_memmap_path, pca_memmap_path):
            if not mm_path.exists():
                continue
            try:
                mm_path.unlink()
                removed_memmap_files.append(mm_path.as_posix())
                print(f"[Done] Removed memmap: {mm_path}")
            except OSError as exc:
                print(f"[WARN] Failed to remove memmap: {mm_path} ({exc})")

    params_for_json = {
        k: (v.as_posix() if isinstance(v, Path) else v) for k, v in vars(args).items()
    }

    summary = {
        "traj_root": args.traj_root.as_posix(),
        "traj_pattern": args.traj_pattern,
        "n_trajectories": len(traj_infos),
        "n_candidates": len(candidates),
        "n_selected": int(len(valid_clusters)),
        "soap_species": species,
        "soap_dim": int(feat_dim),
        "pca_dim": pca_dim_out,
        "pca_fit_samples": int(len(fit_idx)),
        "paths": {
            "candidate_pool_csv": candidate_csv.as_posix(),
            "selected_for_DFT_csv": selected_csv.as_posix(),
            "selected_structures_dir": structures_dir.as_posix(),
            "soap_features_memmap": soap_memmap_path.as_posix(),
            "pca_features_memmap": pca_memmap_path.as_posix(),
        },
        "memmap_cleanup": {
            "enabled": not args.keep_memmap,
            "removed_files": removed_memmap_files,
        },
        "params": params_for_json,
        "explained_variance_ratio_sum": float(np.sum(pca.explained_variance_ratio_)),
        "explained_variance_ratio": pca.explained_variance_ratio_.tolist(),
    }

    summary_json = args.out_dir / "prep_summary.json"
    summary_json.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print("[Done] selected_for_DFT completed.")
    print(f"[Done] candidate pool: {candidate_csv}")
    print(f"[Done] selected csv: {selected_csv}")
    print(f"[Done] selected structures: {structures_dir}")
    print(f"[Done] summary: {summary_json}")
    return 0


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted.", file=sys.stderr)
        raise SystemExit(130)
