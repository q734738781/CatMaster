"""
Structured wrappers for slab generation and post-processing utilities.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict

from catmaster.core.structure import slab as slab_utils


def cut_slabs(payload: Dict[str, object]) -> Dict[str, object]:
    structure_file = Path(payload["structure_file"]).expanduser().resolve()
    compound_name = payload.get("compound_name") or structure_file.stem
    miller_list_raw = payload.get("miller_list") or []
    miller_list = [tuple(int(x) for x in item) for item in miller_list_raw]
    min_slab_size = float(payload.get("min_slab_size", 12.0))
    min_vacuum_size = float(payload.get("min_vacuum_size", 15.0))
    relax_thickness = float(payload.get("relax_thickness", 5.0))
    output_root = Path(payload["output_root"]).expanduser().resolve()
    get_symmetry_slab = bool(payload.get("get_symmetry_slab", False))
    fix_bottom = bool(payload.get("fix_bottom", True))

    if not miller_list:
        raise ValueError("miller_list must contain at least one facet.")

    slab_utils.generate_slabs(
        structure_file=structure_file,
        compound_name=compound_name,
        miller_list=miller_list,
        min_slab_size=min_slab_size,
        min_vacuum_size=min_vacuum_size,
        relax_thickness=relax_thickness,
        output_root=output_root,
        symmetrize=get_symmetry_slab,
        fix_bottom=fix_bottom,
    )

    generated = sorted(p.as_posix() for p in output_root.glob(f"{compound_name}/**/*.vasp"))
    return {
        "compound": compound_name,
        "facets": miller_list,
        "output_root": output_root.as_posix(),
        "generated": generated,
    }


def fix_slab(payload: Dict[str, object]) -> Dict[str, object]:
    input_path = Path(payload["input_path"]).expanduser().resolve()
    output_dir = Path(payload["output_dir"]).expanduser().resolve()
    relax_thickness = float(payload.get("relax_thickness", 5.0))
    fix_bottom = bool(payload.get("fix_bottom", True))
    centralize = bool(payload.get("centralize", False))

    output_dir.mkdir(parents=True, exist_ok=True)
    if input_path.is_dir():
        slab_utils.batch_fix_slabs(
            input_dir=input_path,
            output_dir=output_dir,
            relax_thickness=relax_thickness,
            fix_bottom=fix_bottom,
            centralize=centralize,
        )
    else:
        slab_utils.fix_slab_file(
            input_path,
            output_dir / input_path.name,
            relax_thickness=relax_thickness,
            fix_bottom=fix_bottom,
            centralize=centralize,
        )
    outputs = sorted(p.as_posix() for p in output_dir.glob("*.vasp"))
    return {
        "source": input_path.as_posix(),
        "output_dir": output_dir.as_posix(),
        "generated": outputs,
    }


