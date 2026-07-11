from __future__ import annotations

import hashlib
import json
import os
import re
import shutil
import signal
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any, Literal

import matplotlib
from ase.io import read as ase_read
from ase.io import write as ase_write

matplotlib.use("Agg")

from matplotlib import image as mpimg
from matplotlib import pyplot as plt
from pydantic import BaseModel, ConfigDict, Field, field_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


_VESTA_CITATION = (
    "K. Momma and F. Izumi, VESTA 3 for three-dimensional visualization of crystal, "
    "volumetric and morphology data, J. Appl. Crystallogr. 44, 1272-1276 (2011)."
)
_VIEW_ROTATIONS: dict[str, tuple[tuple[str, str], ...]] = {
    "top": (),
    "side": (("-rotate_x", "90"),),
    "iso": (("-rotate_x", "55"), ("-rotate_z", "-45")),
}
_VIEW_TITLES = {"top": "Top (c-axis)", "side": "Side", "iso": "Isometric"}


class RenderVestaViewsInput(BaseModel):
    """[structure/viz] Render standardized VESTA structure views for multimodal inspection and reports."""

    model_config = ConfigDict(extra="forbid")

    structure_path: str = Field(..., description="Workspace-relative structure file to render with VESTA.")
    output_dir: str = Field(
        "structures/vesta",
        description="Workspace-relative output directory. Defaults to structures/vesta.",
    )
    basename: str = Field(
        "",
        description="Optional filename stem. Leave empty to derive it from structure_path.",
    )
    views: list[Literal["top", "side", "iso"]] = Field(
        default_factory=lambda: ["top", "side", "iso"],
        description="Views to export. Omit to render top, side, and isometric views.",
    )
    supercell: list[int] = Field(
        default_factory=lambda: [1, 1, 1],
        description="Three positive repeat counts [a, b, c]. Omit for the original cell.",
    )
    image_scale: int = Field(
        2,
        ge=1,
        le=4,
        description="VESTA raster export scale from 1 to 4. Use 2 for normal inspection and reports.",
    )
    display_width_angstrom: float = Field(
        0.0,
        ge=0.0,
        description="Optional fixed displayed width in angstrom. Leave 0 to use VESTA auto-fit.",
    )
    include_panel: bool = Field(
        True,
        description="If true, compose the exported views into one labeled PNG panel.",
    )

    @field_validator("structure_path", "output_dir")
    @classmethod
    def _require_path(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            raise ValueError("path must not be empty")
        return text

    @field_validator("basename")
    @classmethod
    def _validate_basename(cls, value: str) -> str:
        text = str(value or "").strip()
        if not text:
            return ""
        if Path(text).name != text or text in {".", ".."}:
            raise ValueError("basename must be a filename stem, not a path")
        return text

    @field_validator("views")
    @classmethod
    def _validate_views(cls, value: list[str]) -> list[str]:
        ordered = list(dict.fromkeys(value))
        if not ordered:
            raise ValueError("views must contain at least one of: top, side, iso")
        return ordered

    @field_validator("supercell")
    @classmethod
    def _validate_supercell(cls, value: list[int]) -> list[int]:
        if len(value) != 3 or any(int(item) < 1 for item in value):
            raise ValueError("supercell must contain exactly three positive integers")
        return [int(item) for item in value]


def _executable_candidate(raw: str, names: tuple[str, ...]) -> Path | None:
    text = str(raw or "").strip()
    if not text:
        return None
    expanded = Path(os.path.expandvars(text)).expanduser()
    candidates = [expanded]
    if expanded.is_dir():
        candidates = [expanded / name for name in names]
    for candidate in candidates:
        if candidate.is_file() and os.access(candidate, os.X_OK):
            return candidate.resolve()
    return None


def _resolve_vesta_executable() -> Path | None:
    configured = _executable_candidate(os.getenv("CATMASTER_VESTA_BIN", ""), ("VESTA", "vesta"))
    if configured is not None:
        return configured

    for name in ("VESTA", "vesta"):
        resolved = shutil.which(name)
        if resolved:
            candidate = _executable_candidate(resolved, (name,))
            if candidate is not None:
                return candidate

    home = Path.home()
    fixed = (
        home / "VESTA" / "VESTA",
        home / "vesta" / "VESTA",
        home / ".local" / "opt" / "VESTA-gtk3" / "VESTA",
    )
    for candidate in fixed:
        resolved = _executable_candidate(str(candidate), ("VESTA", "vesta"))
        if resolved is not None:
            return resolved
    return None


def _resolve_xvfb_run() -> Path | None:
    configured = _executable_candidate(os.getenv("CATMASTER_XVFB_RUN", ""), ("xvfb-run",))
    if configured is not None:
        return configured
    resolved = shutil.which("xvfb-run")
    return _executable_candidate(resolved or "", ("xvfb-run",))


def _safe_stem(value: str) -> str:
    cleaned = re.sub(r"[^A-Za-z0-9_.-]+", "_", value.strip()).strip("._")
    return cleaned or "structure"


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _infer_vesta_version(executable: Path) -> str:
    for part in executable.parts[::-1]:
        match = re.search(r"vesta[-_]?([0-9]+(?:\.[0-9A-Za-z]+)+)", part, re.IGNORECASE)
        if match:
            return match.group(1)
    return "unknown"


def _prepare_render_input(
    structure_path: Path,
    supercell: tuple[int, int, int],
    temp_dir: Path,
) -> tuple[Path, dict[str, Any]]:
    details: dict[str, Any] = {"natoms": None, "formula": ""}
    try:
        atoms = ase_read(str(structure_path))
        details.update({"natoms": len(atoms), "formula": atoms.get_chemical_formula()})
    except Exception as exc:
        if supercell != (1, 1, 1):
            raise ValueError(f"ASE could not read structure for supercell expansion: {exc}") from exc
        return structure_path, details

    if supercell == (1, 1, 1):
        return structure_path, details

    repeated = atoms.repeat(supercell)
    render_input = temp_dir / "vesta_render_input.vasp"
    ase_write(str(render_input), repeated, format="vasp", direct=True, sort=False, vasp5=True)
    details.update({"natoms": len(repeated), "formula": repeated.get_chemical_formula()})
    return render_input, details


def _terminate_process_group(process: subprocess.Popen[Any]) -> None:
    if process.poll() is not None:
        return
    try:
        os.killpg(process.pid, signal.SIGTERM)
        process.wait(timeout=3)
    except (ProcessLookupError, subprocess.TimeoutExpired):
        try:
            os.killpg(process.pid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            process.wait(timeout=2)
        except subprocess.TimeoutExpired:
            pass


def _render_one_view(
    *,
    executable: Path,
    structure_path: Path,
    output_path: Path,
    view: str,
    image_scale: int,
    display_width_angstrom: float,
    session_home: Path,
    timeout_seconds: float = 30.0,
) -> tuple[str, str]:
    output_path.unlink(missing_ok=True)
    command = [str(executable), "-open", str(structure_path)]
    for flag, angle in _VIEW_ROTATIONS[view]:
        command.extend([flag, angle])
    if display_width_angstrom > 0:
        command.extend(["-scale_width_to", f"{display_width_angstrom:g}"])
    command.extend(["-flush", "-export_img", f"scale={image_scale}", str(output_path), "-close"])

    display_mode = "display"
    xvfb_run = None
    if not str(os.getenv("DISPLAY") or "").strip():
        xvfb_run = _resolve_xvfb_run()
        if xvfb_run is None:
            raise CatMasterToolExecutionError(
                tool_name="render_vesta_views",
                public_message=(
                    "VESTA image export needs an X11 display. Set DISPLAY or install xvfb and make "
                    "xvfb-run available (CATMASTER_XVFB_RUN may point to it)."
                ),
                artifact={"tool_name": "render_vesta_views", "data": {"status": "error"}},
                error_code="missing_vesta_display",
            )
        display_mode = "xvfb"
        command = [
            str(xvfb_run),
            "-a",
            "-s",
            "-screen 0 1280x960x24 +extension GLX +render -noreset",
            *command,
        ]

    env = os.environ.copy()
    env.update(
        {
            "HOME": str(session_home),
            "XDG_CONFIG_HOME": str(session_home / ".config"),
            "XDG_CACHE_HOME": str(session_home / ".cache"),
            "GDK_BACKEND": "x11",
            "LIBGL_ALWAYS_SOFTWARE": env.get("LIBGL_ALWAYS_SOFTWARE", "1"),
        }
    )
    (session_home / ".config").mkdir(parents=True, exist_ok=True)
    (session_home / ".cache").mkdir(parents=True, exist_ok=True)

    log_path = session_home / f"vesta_{view}.log"
    with log_path.open("w", encoding="utf-8") as log_handle:
        try:
            process = subprocess.Popen(
                command,
                cwd=str(executable.parent),
                env=env,
                stdout=log_handle,
                stderr=subprocess.STDOUT,
                text=True,
                start_new_session=True,
            )
        except OSError as exc:
            raise RuntimeError(f"failed to launch VESTA: {exc}") from exc

        deadline = time.monotonic() + timeout_seconds
        stable_size = -1
        stable_polls = 0
        try:
            while time.monotonic() < deadline:
                if output_path.is_file() and output_path.stat().st_size > 512:
                    current_size = output_path.stat().st_size
                    stable_polls = stable_polls + 1 if current_size == stable_size else 0
                    stable_size = current_size
                    if stable_polls >= 2:
                        break
                if process.poll() is not None and not output_path.is_file():
                    break
                time.sleep(0.2)
        finally:
            _terminate_process_group(process)

    log_tail = log_path.read_text(encoding="utf-8", errors="replace")[-2000:]
    if not output_path.is_file() or output_path.stat().st_size <= 512:
        raise RuntimeError(f"VESTA did not produce {output_path.name}. Log tail: {log_tail}")
    try:
        image = mpimg.imread(output_path)
    except Exception as exc:
        raise RuntimeError(f"VESTA output is not a readable image: {exc}") from exc
    if image.ndim < 2 or min(image.shape[:2]) < 64:
        raise RuntimeError(f"VESTA output is unexpectedly small: shape={image.shape}")
    return display_mode, log_tail


def _compose_panel(view_paths: dict[str, Path], panel_path: Path) -> None:
    count = len(view_paths)
    fig, axes = plt.subplots(1, count, figsize=(5.0 * count, 4.2), squeeze=False)
    for axis, (view, image_path) in zip(axes[0], view_paths.items()):
        axis.imshow(mpimg.imread(image_path))
        axis.set_title(_VIEW_TITLES[view], fontsize=12)
        axis.set_axis_off()
    fig.patch.set_facecolor("white")
    fig.tight_layout(pad=0.8)
    fig.savefig(panel_path, dpi=180, facecolor="white", bbox_inches="tight")
    plt.close(fig)


def render_vesta_views(payload: dict[str, object]) -> tuple[str, dict[str, Any]]:
    """[structure/viz] Render standardized VESTA views and save image artifacts."""
    tool_name = "render_vesta_views"
    try:
        params = RenderVestaViewsInput(**payload)
        executable = _resolve_vesta_executable()
        if executable is None:
            raise CatMasterToolExecutionError(
                tool_name=tool_name,
                public_message=(
                    "VESTA was not found. Set CATMASTER_VESTA_BIN to the VESTA launcher or add "
                    "VESTA to PATH; see docs/readme/04-external-tools.*.md."
                ),
                artifact={"tool_name": tool_name, "data": {"status": "error"}},
                error_code="missing_vesta",
            )

        structure_path = resolve_workspace_path(params.structure_path, must_exist=True)
        output_dir = resolve_workspace_path(params.output_dir)
        output_dir.mkdir(parents=True, exist_ok=True)
        basename = _safe_stem(params.basename or structure_path.stem)
        supercell = tuple(params.supercell)

        with tempfile.TemporaryDirectory(prefix="catmaster-vesta-") as temp_name:
            temp_dir = Path(temp_name)
            render_input, structure_details = _prepare_render_input(structure_path, supercell, temp_dir)
            view_paths: dict[str, Path] = {}
            display_modes: set[str] = set()
            warnings: list[str] = []
            for view in params.views:
                output_path = output_dir / f"{basename}_vesta_{view}.png"
                display_mode, log_tail = _render_one_view(
                    executable=executable,
                    structure_path=render_input,
                    output_path=output_path,
                    view=view,
                    image_scale=params.image_scale,
                    display_width_angstrom=params.display_width_angstrom,
                    session_home=temp_dir / "home",
                )
                display_modes.add(display_mode)
                if "Gtk-WARNING" in log_tail or "Gtk-CRITICAL" in log_tail:
                    warnings.append(f"{view}: VESTA emitted GTK layout warnings; image export succeeded.")
                view_paths[view] = output_path

        panel_path = output_dir / f"{basename}_vesta_panel.png"
        if params.include_panel:
            _compose_panel(view_paths, panel_path)
        else:
            panel_path = Path()

        metadata_path = output_dir / f"{basename}_vesta_render.json"
        view_refs = {view: workspace_relpath(path) for view, path in view_paths.items()}
        data = {
            "backend": "vesta",
            "vesta_executable": str(executable),
            "vesta_version": _infer_vesta_version(executable),
            "display_mode": "+".join(sorted(display_modes)),
            "structure_path": workspace_relpath(structure_path),
            "structure_sha256": _sha256(structure_path),
            "formula": structure_details["formula"],
            "natoms": structure_details["natoms"],
            "supercell": list(supercell),
            "views": view_refs,
            "panel_path": workspace_relpath(panel_path) if params.include_panel else "",
            "metadata_path": workspace_relpath(metadata_path),
            "image_scale": params.image_scale,
            "display_width_angstrom": params.display_width_angstrom,
            "warnings": list(dict.fromkeys(warnings)),
            "publication_acknowledgement_required": True,
            "citation": _VESTA_CITATION,
        }
        metadata_path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")

        inspection_path = data["panel_path"] or next(iter(view_refs.values()))
        content = "\n".join(
            [
                "VESTA structure rendering completed.",
                f"inspection_image={inspection_path}",
                f"view_images={json.dumps(view_refs, ensure_ascii=False)}",
                f"metadata={data['metadata_path']}",
                "Use read_file on inspection_image for multimodal geometry inspection.",
                "For publication use, acknowledge and cite VESTA as recorded in metadata.",
            ]
        )
        return content, {"tool_name": tool_name, "data": data}
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={
                "tool_name": tool_name,
                "data": {
                    "structure_path": payload.get("structure_path"),
                    "output_dir": payload.get("output_dir", "structures/vesta"),
                },
            },
            error_code="vesta_render_failed",
        ) from exc


__all__ = [
    "RenderVestaViewsInput",
    "_resolve_vesta_executable",
    "_resolve_xvfb_run",
    "render_vesta_views",
]
