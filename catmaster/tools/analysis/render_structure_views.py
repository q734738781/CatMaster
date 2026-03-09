from __future__ import annotations

import math
import warnings
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

import matplotlib
import numpy as np
from ase import Atoms
from ase.data import chemical_symbols, covalent_radii
from ase.data.colors import jmol_colors
from ase.io import read as ase_read
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib import patches
from matplotlib.gridspec import GridSpec
from pydantic import BaseModel, Field, field_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


_SUPPORTED_LABEL_MODES = {"elements", "none"}
_SUPPORTED_FIT_MODES = {"tight", "fit", "loose"}
_SUPPORTED_VIEWS_PRESETS = {"catmaster_four_views_v1"}
_SUPPORTED_STYLE_PRESETS = {"publication_atomistic_v1"}
_FIT_SCALE = {"tight": 0.88, "fit": 1.00, "loose": 1.15}
_BACKGROUND_RGB = {"white": (1.0, 1.0, 1.0)}


class RenderStructureViewsInput(BaseModel):
    """Render a structure into a stable four-view panel for visual inspection."""

    structure_path: str = Field(..., description="Workspace-relative input structure path (POSCAR/CIF/XYZ/etc.).")
    output_path: str | None = Field(
        None,
        description="Workspace-relative output panel PNG. Defaults to visualizations/<stem>_four_views.png.",
    )
    supercell: list[int] = Field(
        default_factory=lambda: [1, 1, 1],
        min_length=3,
        max_length=3,
        description="Explicit supercell replication [nx, ny, nz] before rendering.",
    )
    views_preset: str = Field(
        "catmaster_four_views_v1",
        description="View preset. Only catmaster_four_views_v1 is supported in V1.",
    )
    fit_mode: Literal["tight", "fit", "loose"] = Field(
        "fit",
        description="Panel fit mode controlling crop tightness.",
    )
    tile_size: list[int] = Field(
        default_factory=lambda: [900, 900],
        min_length=2,
        max_length=2,
        description="Single-tile render size [width, height] in pixels.",
    )
    background: str = Field("white", description="Background color. Only white is supported in V1.")
    show_cell: bool = Field(False, description="Show projected simulation cell edges.")
    show_legend: bool = Field(True, description="Show a shared legend of elements present in the structure.")
    label_mode: Literal["elements", "none"] = Field(
        "elements",
        description="Legend label mode. V1 supports element labels or no labels.",
    )
    style_preset: str = Field(
        "publication_atomistic_v1",
        description="Atom rendering style preset. Only publication_atomistic_v1 is supported in V1.",
    )

    @field_validator("supercell")
    @classmethod
    def _validate_supercell(cls, value: list[int]) -> list[int]:
        out = [int(item) for item in value]
        if any(item <= 0 for item in out):
            raise ValueError("supercell values must be positive integers")
        return out

    @field_validator("tile_size")
    @classmethod
    def _validate_tile_size(cls, value: list[int]) -> list[int]:
        out = [int(item) for item in value]
        if any(item <= 0 for item in out):
            raise ValueError("tile_size values must be positive integers")
        return out


@dataclass(frozen=True)
class _ViewSpec:
    title: str
    basis: np.ndarray


def _rotation_basis(view_name: str) -> np.ndarray:
    if view_name == "front":
        screen_x = np.array([0.0, 1.0, 0.0])
        screen_y = np.array([0.0, 0.0, 1.0])
        depth = np.array([1.0, 0.0, 0.0])
    elif view_name == "right":
        screen_x = np.array([1.0, 0.0, 0.0])
        screen_y = np.array([0.0, 0.0, 1.0])
        depth = np.array([0.0, 1.0, 0.0])
    elif view_name == "top":
        screen_x = np.array([1.0, 0.0, 0.0])
        screen_y = np.array([0.0, 1.0, 0.0])
        depth = np.array([0.0, 0.0, 1.0])
    elif view_name == "iso":
        depth = np.array([1.0, 1.0, 1.0], dtype=float)
        depth /= np.linalg.norm(depth)
        provisional_up = np.array([0.0, 0.0, 1.0], dtype=float)
        screen_x = np.cross(provisional_up, depth)
        if np.linalg.norm(screen_x) < 1e-8:
            provisional_up = np.array([0.0, 1.0, 0.0], dtype=float)
            screen_x = np.cross(provisional_up, depth)
        screen_x /= np.linalg.norm(screen_x)
        screen_y = np.cross(depth, screen_x)
        screen_y /= np.linalg.norm(screen_y)
    else:  # pragma: no cover - guarded by caller
        raise ValueError(f"Unsupported view name: {view_name}")
    return np.vstack([screen_x, screen_y, depth])


_VIEW_SPECS = {
    name: _ViewSpec(title=name.title(), basis=_rotation_basis(name))
    for name in ("front", "right", "top", "iso")
}


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
        if symbol in seen:
            continue
        seen.add(symbol)
        ordered.append(symbol)
    return {symbol: _symbol_color(symbol) for symbol in ordered}


def _default_output_path(structure_path: Path) -> Path:
    return resolve_workspace_path(f"visualizations/{structure_path.stem}_four_views.png")


def _tile_output_paths(panel_path: Path) -> dict[str, Path]:
    base_dir = panel_path.parent / f"{panel_path.stem}_tiles"
    return {name: base_dir / f"{name}.png" for name in _VIEW_SPECS}


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


def _project_points(points: np.ndarray, basis: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    projected = points @ basis.T
    return projected[:, :2], projected[:, 2]


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
        (0, 1), (0, 2), (0, 3),
        (1, 4), (1, 5),
        (2, 4), (2, 6),
        (3, 5), (3, 6),
        (4, 7), (5, 7), (6, 7),
    ]
    return np.array([[corners[i], corners[j]] for i, j in edge_ids], dtype=float)


def _render_matplotlib_view(
    *,
    atoms: Atoms,
    view_name: str,
    fit_mode: str,
    background: str,
    show_cell: bool,
    legend_mapping: dict[str, str],
    target_path: Path,
    tile_size: tuple[int, int],
) -> None:
    basis = _VIEW_SPECS[view_name].basis
    positions = atoms.get_positions()
    center = positions.mean(axis=0) if len(positions) else np.zeros(3)
    centered = positions - center
    projected, depth = _project_points(centered, basis)
    symbols = atoms.get_chemical_symbols()
    radii = np.array(
        [
            max(0.35, float(covalent_radii[chemical_symbols.index(symbol)]) * 0.55)
            if symbol in chemical_symbols
            else 0.7
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

    extents = []
    if show_cell and np.linalg.norm(atoms.cell.array) > 0:
        cell_segments = _cell_segments(atoms.cell.array)
        projected_segments = []
        for segment in cell_segments:
            proj_segment, _ = _project_points(segment - center, basis)
            projected_segments.append(proj_segment)
            extents.extend(proj_segment.tolist())
        for segment in projected_segments:
            ax.plot(segment[:, 0], segment[:, 1], color="#666666", lw=1.0, alpha=0.9, zorder=1)

    order = np.argsort(depth)
    for idx in order:
        x, y = projected[idx]
        extents.extend([[x - radii[idx], y - radii[idx]], [x + radii[idx], y + radii[idx]]])
        atom_patch = patches.Circle(
            (float(x), float(y)),
            radius=float(radii[idx]),
            facecolor=legend_mapping.get(symbols[idx], "#999999"),
            edgecolor="#111111",
            linewidth=0.55,
            zorder=10 + idx,
        )
        ax.add_patch(atom_patch)

    if not extents:
        extents = [[-1.0, -1.0], [1.0, 1.0]]
    extents_arr = np.array(extents, dtype=float)
    mins = extents_arr.min(axis=0)
    maxs = extents_arr.max(axis=0)
    center_xy = (mins + maxs) / 2.0
    span = max(float(maxs[0] - mins[0]), float(maxs[1] - mins[1]), 2.5)
    half = 0.5 * span * _FIT_SCALE[fit_mode]
    ax.set_xlim(center_xy[0] - half, center_xy[0] + half)
    ax.set_ylim(center_xy[1] - half, center_xy[1] + half)
    ax.set_title(_VIEW_SPECS[view_name].title, fontsize=10)

    target_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target_path, dpi=150, bbox_inches="tight", pad_inches=0.04, facecolor=background)
    plt.close(fig)


def _compose_panel(
    *,
    tile_paths: dict[str, Path],
    panel_path: Path,
    background: str,
    show_legend: bool,
    label_mode: str,
    legend_mapping: dict[str, str],
    tile_size: tuple[int, int],
    backend_name: str,
) -> None:
    fig = plt.figure(
        figsize=((tile_size[0] * 2.5) / 150.0, (tile_size[1] * 2.0) / 150.0),
        dpi=150,
    )
    fig.patch.set_facecolor(background)
    gs = GridSpec(
        2,
        3,
        figure=fig,
        width_ratios=[1.0, 1.0, 0.58],
        wspace=0.04,
        hspace=0.08,
    )

    for idx, view_name in enumerate(("front", "right", "top", "iso")):
        row = idx // 2
        col = idx % 2
        ax = fig.add_subplot(gs[row, col])
        image = plt.imread(tile_paths[view_name])
        ax.imshow(image)
        ax.axis("off")

    legend_ax = fig.add_subplot(gs[:, 2])
    legend_ax.set_facecolor(background)
    legend_ax.axis("off")
    legend_ax.text(0.0, 0.98, "Elements", fontsize=11, fontweight="bold", va="top", ha="left")
    if show_legend and legend_mapping:
        y = 0.90
        for symbol, color in legend_mapping.items():
            legend_ax.add_patch(
                patches.Rectangle((0.02, y - 0.028), 0.12, 0.04, facecolor=color, edgecolor="#111111", lw=0.8)
            )
            if label_mode == "elements":
                legend_ax.text(0.19, y - 0.008, symbol, fontsize=10, va="center", ha="left")
            y -= 0.085
    else:
        legend_ax.text(0.02, 0.88, "Legend hidden", fontsize=10, va="top", ha="left")
    legend_ax.text(0.0, 0.08, f"Backend: {backend_name}", fontsize=9, va="bottom", ha="left", color="#444444")

    panel_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(panel_path, dpi=150, bbox_inches="tight", pad_inches=0.04, facecolor=background)
    plt.close(fig)


def _try_render_with_ovito(
    *,
    structure_path: Path,
    tile_paths: dict[str, Path],
    supercell: tuple[int, int, int],
    fit_mode: str,
    tile_size: tuple[int, int],
    background: str,
    show_cell: bool,
) -> bool:
    try:
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*OVITO.*PyPI.*",
                category=UserWarning,
            )
            from ovito.io import import_file  # type: ignore
            from ovito.modifiers import ReplicateModifier  # type: ignore
            from ovito.vis import TachyonRenderer, Viewport  # type: ignore
    except Exception:
        return False

    bg_rgb = _BACKGROUND_RGB[background]
    pipeline = import_file(str(structure_path))
    try:
        if supercell != (1, 1, 1):
            pipeline.modifiers.append(
                ReplicateModifier(num_x=supercell[0], num_y=supercell[1], num_z=supercell[2])
            )
        pipeline.add_to_scene()

        scale = _FIT_SCALE[fit_mode]
        renderer = TachyonRenderer()
        view_params = {
            "front": {
                "camera_dir": (-1.0, 0.0, 0.0),
                "camera_up": (0.0, 0.0, 1.0),
                "type": Viewport.Type.Ortho,
            },
            "right": {
                "camera_dir": (0.0, -1.0, 0.0),
                "camera_up": (0.0, 0.0, 1.0),
                "type": Viewport.Type.Ortho,
            },
            "top": {
                "camera_dir": (0.0, 0.0, -1.0),
                "camera_up": (0.0, 1.0, 0.0),
                "type": Viewport.Type.Ortho,
            },
            "iso": {
                "camera_dir": (-1.0, -1.0, -1.0),
                "camera_up": (0.0, 0.0, 1.0),
                "type": Viewport.Type.Perspective,
            },
        }
        for view_name, params in view_params.items():
            vp = Viewport(type=params["type"])
            vp.camera_dir = params["camera_dir"]
            vp.camera_up = params["camera_up"]
            vp.zoom_all(size=tile_size)
            if hasattr(vp, "fov"):
                try:
                    vp.fov = float(vp.fov) * scale
                except Exception:
                    pass
            tile_paths[view_name].parent.mkdir(parents=True, exist_ok=True)
            vp.render_image(
                size=tile_size,
                filename=str(tile_paths[view_name]),
                background=bg_rgb,
                alpha=False,
                renderer=renderer,
            )
        return True
    finally:
        try:
            pipeline.remove_from_scene()
        except Exception:
            pass


def render_structure_views(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    tool_name = "render_structure_views"
    try:
        params = RenderStructureViewsInput(**payload)
        if params.views_preset not in _SUPPORTED_VIEWS_PRESETS:
            raise ValueError(f"Unsupported views_preset: {params.views_preset}")
        if params.fit_mode not in _SUPPORTED_FIT_MODES:
            raise ValueError(f"Unsupported fit_mode: {params.fit_mode}")
        if params.label_mode not in _SUPPORTED_LABEL_MODES:
            raise ValueError(f"Unsupported label_mode: {params.label_mode}")
        if params.style_preset not in _SUPPORTED_STYLE_PRESETS:
            raise ValueError(f"Unsupported style_preset: {params.style_preset}")
        if params.background not in _BACKGROUND_RGB:
            raise ValueError(f"Unsupported background: {params.background}")

        structure_path = resolve_workspace_path(params.structure_path, must_exist=True)
        panel_path = (
            resolve_workspace_path(params.output_path)
            if params.output_path
            else _default_output_path(structure_path)
        )
        supercell = tuple(int(v) for v in params.supercell)
        tile_size = (int(params.tile_size[0]), int(params.tile_size[1]))

        atoms = _load_atoms(structure_path, supercell)
        legend_mapping = _legend_mapping(atoms.get_chemical_symbols())
        tile_paths = _tile_output_paths(panel_path)

        backend_name = "OVITO TachyonRenderer"
        ovito_ok = _try_render_with_ovito(
            structure_path=structure_path,
            tile_paths=tile_paths,
            supercell=supercell,
            fit_mode=params.fit_mode,
            tile_size=tile_size,
            background=params.background,
            show_cell=params.show_cell,
        )
        if not ovito_ok:
            backend_name = "ASE/Matplotlib fallback"
            for view_name, tile_path in tile_paths.items():
                _render_matplotlib_view(
                    atoms=atoms,
                    view_name=view_name,
                    fit_mode=params.fit_mode,
                    background=params.background,
                    show_cell=params.show_cell,
                    legend_mapping=legend_mapping,
                    target_path=tile_path,
                    tile_size=tile_size,
                )

        _compose_panel(
            tile_paths=tile_paths,
            panel_path=panel_path,
            background=params.background,
            show_legend=params.show_legend,
            label_mode=params.label_mode,
            legend_mapping=legend_mapping,
            tile_size=tile_size,
            backend_name=backend_name,
        )

        data = {
            "image_path": workspace_relpath(panel_path),
            "tile_paths": {name: workspace_relpath(path) for name, path in tile_paths.items()},
            "legend_mapping": legend_mapping,
            "view_specs": {
                "preset": params.views_preset,
                "fit_mode": params.fit_mode,
                "supercell": list(supercell),
            },
            "notes": [
                "simulation cell hidden" if not params.show_cell else "simulation cell shown",
                f"rendered with {backend_name}",
                "label_mode=none only affects legend labels in V1",
            ],
        }
        content = "\n".join(
            [
                "render_structure_views completed.",
                f"image_path={data['image_path']}",
                f"views={', '.join(tile_paths.keys())}",
                f"elements={', '.join(legend_mapping.keys()) or '(none)'}",
            ]
        )
        return content, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={"tool_name": tool_name, "data": {"structure_path": payload.get("structure_path")}},
            error_code="render_structure_views_failed",
        ) from exc


__all__ = ["RenderStructureViewsInput", "render_structure_views"]
