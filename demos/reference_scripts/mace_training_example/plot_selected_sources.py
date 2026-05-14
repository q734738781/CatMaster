#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

try:
    import matplotlib.pyplot as plt
except Exception:  # pragma: no cover - runtime dependency check
    plt = None


def _shorten(text: str, max_len: int = 64) -> str:
    if len(text) <= max_len:
        return text
    return text[: max_len - 3] + "..."


def _read_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open(newline="") as f:
        reader = csv.DictReader(f)
        if reader.fieldnames is None:
            raise ValueError(f"CSV has no header: {csv_path}")
        rows = list(reader)
    if not rows:
        raise ValueError(f"CSV has no data rows: {csv_path}")
    return rows


def _require_columns(rows: list[dict[str, str]], required: list[str]) -> None:
    columns = set(rows[0].keys())
    missing = [c for c in required if c not in columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def _plot_top_counter(
    ax,
    counter: Counter[str],
    title: str,
    top_n: int,
    color: str,
) -> None:
    total = sum(counter.values())
    items = counter.most_common(max(1, int(top_n)))
    labels = [_shorten(k) for k, _ in items]
    counts = [v for _, v in items]

    y = np.arange(len(items))
    ax.barh(y, counts, color=color, alpha=0.9)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=8)
    ax.invert_yaxis()
    ax.set_xlabel("Count")
    covered = 100.0 * (sum(counts) / max(total, 1))
    ax.set_title(f"{title} (Top {len(items)}, cover {covered:.1f}%)", fontsize=10)


def _write_counter_csv(path: Path, name: str, counter: Counter[str]) -> None:
    total = sum(counter.values())
    with path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([name, "count", "fraction"])
        for key, count in counter.most_common():
            frac = count / max(total, 1)
            w.writerow([key, count, f"{frac:.6f}"])


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Analyze source composition of selected_for_DFT.csv and generate plots."
    )
    parser.add_argument(
        "--csv",
        type=Path,
        default=Path("MD_Files/v0/MD_sample/selected_for_DFT.csv"),
        help="Input selected_for_DFT.csv path.",
    )
    parser.add_argument(
        "--out-dir",
        type=Path,
        default=Path("MD_Files/v0/MD_sample/source_analysis"),
        help="Output directory for plots and summary files.",
    )
    parser.add_argument(
        "--top-n",
        type=int,
        default=20,
        help="Top-N categories to show in source bar plots.",
    )
    args = parser.parse_args()

    if plt is None:
        raise SystemExit(
            "Missing dependency: matplotlib. Please install it before running "
            "plot_selected_sources.py."
        )

    if not args.csv.exists():
        raise FileNotFoundError(f"Input CSV not found: {args.csv}")
    args.out_dir.mkdir(parents=True, exist_ok=True)

    rows = _read_rows(args.csv)
    _require_columns(
        rows,
        required=[
            "traj_relpath",
            "system_code",
            "strain_code",
            "group_code",
            "time_ps",
        ],
    )

    system_counter = Counter(r["system_code"] for r in rows)
    strain_counter = Counter(r["strain_code"] for r in rows)
    group_counter = Counter(r["group_code"] for r in rows)
    traj_counter = Counter(r["traj_relpath"] for r in rows)

    times = []
    time_by_strain: dict[str, list[float]] = defaultdict(list)
    for r in rows:
        try:
            t = float(r["time_ps"])
        except (TypeError, ValueError):
            continue
        times.append(t)
        time_by_strain[r["strain_code"]].append(t)

    # Figure 1: source composition
    fig, axes = plt.subplots(2, 2, figsize=(16, 12), constrained_layout=True)
    _plot_top_counter(axes[0, 0], system_counter, "By system_code", args.top_n, "#1f77b4")
    _plot_top_counter(axes[0, 1], strain_counter, "By strain_code", args.top_n, "#2ca02c")
    _plot_top_counter(axes[1, 0], group_counter, "By group_code", args.top_n, "#ff7f0e")
    _plot_top_counter(axes[1, 1], traj_counter, "By traj_relpath", args.top_n, "#d62728")
    fig.suptitle("selected_for_DFT source composition", fontsize=14)
    fig1_path = args.out_dir / "source_topn.png"
    fig.savefig(fig1_path, dpi=180)
    plt.close(fig)

    # Figure 2: time distribution (overall + by top strains)
    fig2, axes2 = plt.subplots(1, 2, figsize=(16, 6), constrained_layout=True)
    if times:
        axes2[0].hist(times, bins=40, color="#4c78a8", edgecolor="white")
        axes2[0].set_xlabel("time_ps")
        axes2[0].set_ylabel("Count")
        axes2[0].set_title("Overall time_ps distribution")
    else:
        axes2[0].text(0.5, 0.5, "No valid time_ps", ha="center", va="center")
        axes2[0].set_axis_off()

    top_strains = [k for k, _ in strain_counter.most_common(8)]
    box_data = [time_by_strain[s] for s in top_strains if time_by_strain[s]]
    box_labels = [_shorten(s, 42) for s in top_strains if time_by_strain[s]]
    if box_data:
        axes2[1].boxplot(box_data, labels=box_labels, showfliers=False)
        axes2[1].set_ylabel("time_ps")
        axes2[1].set_xlabel("strain_code (top 8 by count)")
        axes2[1].set_title("time_ps by top strain_code")
        axes2[1].tick_params(axis="x", labelrotation=35)
    else:
        axes2[1].text(0.5, 0.5, "No valid strain/time data", ha="center", va="center")
        axes2[1].set_axis_off()
    fig2_path = args.out_dir / "time_distribution.png"
    fig2.savefig(fig2_path, dpi=180)
    plt.close(fig2)

    # Detailed tables
    _write_counter_csv(args.out_dir / "count_by_system_code.csv", "system_code", system_counter)
    _write_counter_csv(args.out_dir / "count_by_strain_code.csv", "strain_code", strain_counter)
    _write_counter_csv(args.out_dir / "count_by_group_code.csv", "group_code", group_counter)
    _write_counter_csv(args.out_dir / "count_by_traj_relpath.csv", "traj_relpath", traj_counter)

    summary = {
        "input_csv": args.csv.as_posix(),
        "n_selected": len(rows),
        "n_unique_system_code": len(system_counter),
        "n_unique_strain_code": len(strain_counter),
        "n_unique_group_code": len(group_counter),
        "n_unique_traj_relpath": len(traj_counter),
        "top_system_code": system_counter.most_common(10),
        "top_strain_code": strain_counter.most_common(10),
        "top_group_code": group_counter.most_common(10),
        "top_traj_relpath": traj_counter.most_common(10),
        "time_ps_min": float(np.min(times)) if times else None,
        "time_ps_median": float(np.median(times)) if times else None,
        "time_ps_max": float(np.max(times)) if times else None,
        "plots": {
            "source_topn": fig1_path.as_posix(),
            "time_distribution": fig2_path.as_posix(),
        },
    }
    summary_path = args.out_dir / "source_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    print(f"[Done] Input: {args.csv}")
    print(f"[Done] n_selected: {len(rows)}")
    print(f"[Done] Output dir: {args.out_dir}")
    print(f"[Done] Plot: {fig1_path}")
    print(f"[Done] Plot: {fig2_path}")
    print(f"[Done] Summary: {summary_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
