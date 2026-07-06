#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
from matplotlib import patches
from matplotlib import pyplot as plt
from matplotlib.gridspec import GridSpec


@dataclass(frozen=True)
class ViewSpec:
    name: str
    title: str
    camera_dir: tuple[float, float, float]
    camera_up: tuple[float, float, float]


DEFAULT_VIEWS = [
    ViewSpec("front", "Front", (-1.0, 0.0, 0.0), (0.0, 0.0, 1.0)),
    ViewSpec("right", "Right", (0.0, -1.0, 0.0), (0.0, 0.0, 1.0)),
    ViewSpec("top", "Top", (0.0, 0.0, -1.0), (0.0, 1.0, 0.0)),
    ViewSpec("iso", "Iso", (-1.0, -1.0, -1.0), (0.0, 0.0, 1.0)),
]


def _as_vector(value: Any, *, name: str) -> np.ndarray:
    arr = np.array(value, dtype=float)
    if arr.shape != (3,):
        raise ValueError(f"{name} must contain exactly 3 numbers")
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        raise ValueError(f"{name} must not be a zero vector")
    return arr / norm


def _view_basis(view: ViewSpec) -> np.ndarray:
    depth = -_as_vector(view.camera_dir, name=f"{view.name}.camera_dir")
    up = _as_vector(view.camera_up, name=f"{view.name}.camera_up")
    screen_x = np.cross(up, depth)
    if np.linalg.norm(screen_x) <= 1e-10:
        raise ValueError(f"{view.name}: camera_up is parallel to camera_dir")
    screen_x /= np.linalg.norm(screen_x)
    screen_y = np.cross(depth, screen_x)
    screen_y /= np.linalg.norm(screen_y)
    return np.vstack([screen_x, screen_y, depth])


def _load_views(path: Path | None) -> list[ViewSpec]:
    if path is None:
        return DEFAULT_VIEWS
    raw = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(raw, list) or not raw:
        raise ValueError("views JSON must be a non-empty list")
    views: list[ViewSpec] = []
    for idx, item in enumerate(raw, start=1):
        if not isinstance(item, dict):
            raise ValueError(f"view #{idx} must be an object")
        name = str(item.get("name") or f"view_{idx}")
        title = str(item.get("title") or name)
        camera_dir = tuple(float(v) for v in item["camera_dir"])
        camera_up = tuple(float(v) for v in item.get("camera_up", [0.0, 0.0, 1.0]))
        if len(camera_dir) != 3 or len(camera_up) != 3:
            raise ValueError(f"{name}: camera_dir and camera_up must each have length 3")
        views.append(ViewSpec(name=name, title=title, camera_dir=camera_dir, camera_up=camera_up))
    return views


def _load_atoms(structure_path: Path, supercell: tuple[int, int, int]) -> Atoms:
    atoms = ase_read(str(structure_path))
    if isinstance(atoms, list):
        if not atoms:
            raise ValueError(f"No structures found in {structure_path}")
        atoms = atoms[0]
    if not isinstance(atoms, Atoms):
        raise ValueError(f"Unsupported structure object from {structure_path}")
    if supercell != (1, 1, 1):
        atoms = atoms.repeat(supercell)
    return atoms


def _symbol_color(symbol: str) -> str:
    try:
        atomic_number = chemical_symbols.index(symbol)
    except ValueError:
        atomic_number = 0
    if atomic_number <= 0 or atomic_number >= len(jmol_colors):
        rgb = (0.55, 0.55, 0.55)
    else:
        rgb = tuple(float(v) for v in jmol_colors[atomic_number])
    return "#{:02x}{:02x}{:02x}".format(
        int(max(0.0, min(1.0, rgb[0])) * 255),
        int(max(0.0, min(1.0, rgb[1])) * 255),
        int(max(0.0, min(1.0, rgb[2])) * 255),
    )


def _legend_mapping(symbols: list[str]) -> dict[str, str]:
    ordered: list[str] = []
    seen: set[str] = set()
    for symbol in symbols:
        if symbol not in seen:
            ordered.append(symbol)
            seen.add(symbol)
    return {symbol: _symbol_color(symbol) for symbol in ordered}


def _cell_segments(cell: np.ndarray) -> np.ndarray:
    a, b, c = cell
    corners = np.array(
        [
            [0.0, 0.0, 0.0],
            a,
            b,
            c,
            a + b,
            a + c,
            b + c,
            a + b + c,
        ],
        dtype=float,
    )
    edge_ids = [
        (0, 1),
        (0, 2),
        (0, 3),
        (1, 4),
        (1, 5),
        (2, 4),
        (2, 6),
        (3, 5),
        (3, 6),
        (4, 7),
        (5, 7),
        (6, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edge_ids], dtype=float)


def _project(points: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = points @ basis.T
    return projected[:, :2], projected[:, 2]


def _render_view(
    *,
    atoms: Atoms,
    view: ViewSpec,
    tile_path: Path,
    tile_size: tuple[int, int],
    fit_scale: float,
    padding: float,
    atom_scale: float,
    bond_width: float,
    background: str,
    show_cell: bool,
    title: bool,
    legend_mapping: dict[str, str],
) -> None:
    basis = _view_basis(view)
    positions = atoms.get_positions()
    center = positions.mean(axis=0) if len(positions) else np.zeros(3)
    centered = positions - center
    projected, depth = _project(centered, basis)
    symbols = atoms.get_chemical_symbols()
    radii = np.array(
        [
            max(0.20, float(covalent_radii[chemical_symbols.index(symbol)]) * atom_scale)
            if symbol in chemical_symbols
            else 0.7 * atom_scale
            for symbol in symbols
        ],
        dtype=float,
    )

    fig = plt.figure(figsize=(tile_size[0] / 150.0, tile_size[1] / 150.0), dpi=150)
    fig.patch.set_facecolor(background)
    ax = fig.add_subplot(1, 1, 1)
    ax.set_facecolor(background)
    ax.set_aspect("equal")
    ax.axis("off")

    extents: list[list[float]] = []
    if show_cell and np.linalg.norm(atoms.cell.array) > 0:
        for segment in _cell_segments(atoms.cell.array):
            projected_segment, _ = _project(segment - center, basis)
            extents.extend(projected_segment.tolist())
            ax.plot(
                projected_segment[:, 0],
                projected_segment[:, 1],
                color="#555555",
                lw=bond_width,
                alpha=0.9,
                zorder=1,
            )

    for idx in np.argsort(depth):
        x, y = projected[idx]
        radius = float(radii[idx])
        extents.extend([[float(x - radius), float(y - radius)], [float(x + radius), float(y + radius)]])
        ax.add_patch(
            patches.Circle(
                (float(x), float(y)),
                radius=radius,
                facecolor=legend_mapping.get(symbols[idx], "#999999"),
                edgecolor="#111111",
                linewidth=max(0.25, bond_width * 0.6),
                zorder=10 + int(idx),
            )
        )

    if not extents:
        extents = [[-1.0, -1.0], [1.0, 1.0]]
    extents_arr = np.array(extents, dtype=float)
    mins = extents_arr.min(axis=0)
    maxs = extents_arr.max(axis=0)
    center_xy = (mins + maxs) / 2.0
    span = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), 1.0)
    half = 0.5 * span * fit_scale + padding
    ax.set_xlim(center_xy[0] - half, center_xy[0] + half)
    ax.set_ylim(center_xy[1] - half, center_xy[1] + half)
    if title:
        ax.set_title(view.title, fontsize=10)

    tile_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(tile_path, dpi=150, bbox_inches="tight", pad_inches=0.02, facecolor=background)
    plt.close(fig)


def _compose_panel(
    *,
    views: list[ViewSpec],
    tile_paths: dict[str, Path],
    panel_path: Path,
    columns: int,
    tile_size: tuple[int, int],
    background: str,
    show_legend: bool,
    legend_mapping: dict[str, str],
) -> None:
    columns = max(1, int(columns))
    rows = int(np.ceil(len(views) / columns))
    legend_cols = 1 if show_legend else 0
    fig = plt.figure(
        figsize=((tile_size[0] * (columns + 0.55 * legend_cols)) / 150.0, (tile_size[1] * rows) / 150.0),
        dpi=150,
    )
    fig.patch.set_facecolor(background)
    gs = GridSpec(
        rows,
        columns + legend_cols,
        figure=fig,
        width_ratios=[1.0] * columns + ([0.55] if show_legend else []),
        wspace=0.04,
        hspace=0.08,
    )
    for idx, view in enumerate(views):
        ax = fig.add_subplot(gs[idx // columns, idx % columns])
        ax.imshow(plt.imread(tile_paths[view.name]))
        ax.axis("off")

    if show_legend:
        legend_ax = fig.add_subplot(gs[:, columns])
        legend_ax.set_facecolor(background)
        legend_ax.axis("off")
        legend_ax.text(0.0, 0.98, "Elements", fontsize=11, fontweight="bold", va="top", ha="left")
        y = 0.90
        for symbol, color in legend_mapping.items():
            legend_ax.add_patch(
                patches.Rectangle((0.02, y - 0.028), 0.12, 0.04, facecolor=color, edgecolor="#111111", lw=0.8)
            )
            legend_ax.text(0.19, y - 0.008, symbol, fontsize=10, va="center", ha="left")
            y -= 0.085

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_path, dpi=150, bbox_inches="tight", pad_inches=0.03, facecolor=background)
    plt.close(fig)


def render_structure_panel(
    *,
    structure_path: Path,
    output_path: Path,
    views: list[ViewSpec] | None = None,
    tile_dir: Path | None = None,
    supercell: tuple[int, int, int] = (1, 1, 1),
    tile_size: tuple[int, int] = (900, 900),
    columns: int = 2,
    fit_scale: float = 1.0,
    padding: float = 0.0,
    atom_scale: float = 0.55,
    bond_width: float = 1.0,
    background: str = "white",
    show_cell: bool = False,
    show_titles: bool = True,
    show_legend: bool = True,
) -> dict[str, Any]:
    views = list(views or DEFAULT_VIEWS)
    atoms = _load_atoms(structure_path, supercell)
    legend_mapping = _legend_mapping(atoms.get_chemical_symbols())
    tile_dir = tile_dir or output_path.parent / f"{output_path.stem}_tiles"
    tile_paths = {view.name: tile_dir / f"{view.name}.png" for view in views}

    for view in views:
        _render_view(
            atoms=atoms,
            view=view,
            tile_path=tile_paths[view.name],
            tile_size=tile_size,
            fit_scale=fit_scale,
            padding=padding,
            atom_scale=atom_scale,
            bond_width=bond_width,
            background=background,
            show_cell=show_cell,
            title=show_titles,
            legend_mapping=legend_mapping,
        )

    _compose_panel(
        views=views,
        tile_paths=tile_paths,
        panel_path=output_path,
        columns=columns,
        tile_size=tile_size,
        background=background,
        show_legend=show_legend,
        legend_mapping=legend_mapping,
    )
    return {
        "image_path": str(output_path),
        "tile_paths": {name: str(path) for name, path in tile_paths.items()},
        "views": [
            {
                "name": view.name,
                "title": view.title,
                "camera_dir": list(view.camera_dir),
                "camera_up": list(view.camera_up),
            }
            for view in views
        ],
        "supercell": list(supercell),
        "tile_size": list(tile_size),
        "columns": columns,
        "fit_scale": fit_scale,
        "padding": padding,
        "atom_scale": atom_scale,
        "show_cell": show_cell,
        "show_legend": show_legend,
        "legend_mapping": legend_mapping,
    }


def _parse_triplet(text: str, *, name: str) -> tuple[int, int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 3 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(f"{name} must look like 1,1,1 with positive integers")
    return (parts[0], parts[1], parts[2])


def _parse_pair(text: str, *, name: str) -> tuple[int, int]:
    parts = [int(part.strip()) for part in text.split(",") if part.strip()]
    if len(parts) != 2 or any(part <= 0 for part in parts):
        raise argparse.ArgumentTypeError(f"{name} must look like 900,900 with positive integers")
    return (parts[0], parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render a tunable atomistic structure panel from a structure file.")
    parser.add_argument("structure", type=Path, help="Input POSCAR/CIF/XYZ/etc.")
    parser.add_argument("-o", "--output", type=Path, required=True, help="Output panel PNG path.")
    parser.add_argument("--views-json", type=Path, help="JSON list of custom view specs.")
    parser.add_argument("--tile-dir", type=Path, help="Directory for individual view tiles.")
    parser.add_argument("--supercell", default="1,1,1", help="Supercell replication, e.g. 2,2,1.")
    parser.add_argument("--tile-size", default="900,900", help="Single tile size, e.g. 1200,900.")
    parser.add_argument("--columns", type=int, default=2, help="Panel columns before the legend column.")
    parser.add_argument("--fit-scale", type=float, default=1.0, help="Larger values zoom out; smaller values zoom in.")
    parser.add_argument("--padding", type=float, default=0.0, help="Extra projected-coordinate padding.")
    parser.add_argument("--atom-scale", type=float, default=0.55, help="Covalent-radius multiplier for atoms.")
    parser.add_argument("--bond-width", type=float, default=1.0, help="Cell-edge and atom-outline line width.")
    parser.add_argument("--background", default="white", help="Matplotlib background color.")
    parser.add_argument("--show-cell", action="store_true", help="Draw projected cell edges.")
    parser.add_argument("--hide-titles", action="store_true", help="Omit per-view titles.")
    parser.add_argument("--hide-legend", action="store_true", help="Omit element legend.")
    parser.add_argument("--metadata-json", type=Path, help="Optional output metadata JSON path.")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    views = _load_views(args.views_json)
    metadata = render_structure_panel(
        structure_path=args.structure,
        output_path=args.output,
        views=views,
        tile_dir=args.tile_dir,
        supercell=_parse_triplet(args.supercell, name="supercell"),
        tile_size=_parse_pair(args.tile_size, name="tile_size"),
        columns=args.columns,
        fit_scale=float(args.fit_scale),
        padding=float(args.padding),
        atom_scale=float(args.atom_scale),
        bond_width=float(args.bond_width),
        background=args.background,
        show_cell=bool(args.show_cell),
        show_titles=not bool(args.hide_titles),
        show_legend=not bool(args.hide_legend),
    )
    if args.metadata_json:
        args.metadata_json.parent.mkdir(parents=True, exist_ok=True)
        args.metadata_json.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    print(json.dumps(metadata, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
