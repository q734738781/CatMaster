"""
Prepare normalized VASP input sets via StructWriter.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List
from ase.io import read as ase_read
from pymatgen.core import Structure
from pymatgen.io.ase import AseAtomsAdaptor

from .vasp_inputs import StructWriter
from catmaster.tools.base import create_tool_output


SUPPORTED_EXTS = {".vasp", ".cif", ".json", ".xyz"}


def _load_structure(path: Path) -> Structure:
    if path.suffix.lower() == ".xyz":
        atoms = ase_read(path.as_posix())
        return AseAtomsAdaptor.get_structure(atoms)
    return Structure.from_file(path)


def _discover_structures(input_path: Path) -> List[Path]:
    if input_path.is_file():
        return [input_path]
    files = []
    for ext in SUPPORTED_EXTS:
        files.extend(sorted(input_path.glob(f"*{ext}")))
    return files


def prepare_vasp_inputs(payload: Dict[str, object]) -> Dict[str, object]:
    input_path = Path(payload["input_path"])
    output_root = Path(payload["output_root"])
    calc_type = str(payload.get("calc_type", "bulk"))
    k_product = int(payload.get("k_product", 20))
    user_incar_settings = payload.get("user_incar_settings") or {}

    output_root.mkdir(parents=True, exist_ok=True)
    writer = StructWriter()

    structures = _discover_structures(input_path)
    emitted: List[Dict[str, object]] = []
    for struct_path in structures:
        structure = _load_structure(struct_path)
        out_dir = output_root / struct_path.stem
        writer.write_vasp_inputs(
            structure=structure,
            output_dir=out_dir,
            calc_type=calc_type,
            k_product=k_product,
            user_incar_overrides=user_incar_settings,
        )
        emitted.append(
            {
                "source": str(struct_path),
                "output_dir": str(out_dir),
            }
        )
    
    # Return standardized output
    return create_tool_output(
        tool_name="vasp_prepare",
        success=True,
        data={
            "calc_type": calc_type,
            "k_product": k_product,
            "structures_processed": len(emitted),
            "prepared_directories": [e["output_dir"] for e in emitted],
        }
    )


