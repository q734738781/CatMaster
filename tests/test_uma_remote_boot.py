from __future__ import annotations

import importlib.util
import sys
from pathlib import Path

from ase import Atoms


def _load_uma_common():
    repo_root = Path(__file__).resolve().parents[1]
    path = repo_root / "catmaster" / "remote" / "gpu" / "uma_common.py"
    spec = importlib.util.spec_from_file_location("catmaster_test_uma_common", path)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_uma_auto_task_uses_periodic_cell_only() -> None:
    uma_common = _load_uma_common()

    molecule = Atoms("H2O", positions=[[0, 0, 0], [0, 0, 1], [1, 0, 0]])
    assert uma_common.auto_uma_task(molecule) == "omol"

    periodic = Atoms(
        "Si2",
        positions=[[0, 0, 0], [1.35, 1.35, 1.35]],
        cell=[5.43, 5.43, 5.43],
        pbc=True,
    )
    assert uma_common.auto_uma_task(periodic) == "omat"


def test_uma_metadata_merges_defaults_and_item_overrides(tmp_path: Path) -> None:
    uma_common = _load_uma_common()

    input_root = tmp_path / "input"
    struct = input_root / "nested" / "mol.xyz"
    struct.parent.mkdir(parents=True)
    struct.write_text("1\nx\nH 0 0 0\n", encoding="utf-8")

    metadata = {
        "defaults": {"task": "omol", "charge": 1, "spin": 2},
        "items": {"nested/mol.xyz": {"charge": 0, "spin": 1}},
    }
    cfg = uma_common.resolve_item_config(
        structure_path=struct,
        input_root=input_root,
        metadata=metadata,
        default_task="auto",
        default_charge=0,
        default_spin=0,
    )

    assert cfg.uma_task == "omol"
    assert cfg.charge == 0
    assert cfg.spin == 1


def test_uma_rejects_nonzero_charge_spin_for_material_task() -> None:
    uma_common = _load_uma_common()

    atoms = Atoms("Si", positions=[[0, 0, 0]], cell=[5, 5, 5], pbc=True)
    cfg = uma_common.UmaItemConfig(uma_task="omat", charge=1, spin=1)

    try:
        uma_common.apply_charge_spin(atoms, cfg)
    except ValueError as exc:
        assert "expects charge=0 and spin=0" in str(exc)
    else:
        raise AssertionError("expected nonzero charge/spin rejection for omat")


def test_uma_model_load_hint_distinguishes_gated_hf_access() -> None:
    uma_common = _load_uma_common()

    hint = uma_common._model_load_hint(
        RuntimeError("GatedRepoError: Access to model facebook/UMA is restricted")
    )

    assert "gated on Hugging Face" in hint
    assert "facebook/UMA" in hint
    assert "HF_TOKEN" in hint


def test_uma_model_load_hint_distinguishes_missing_fairchem() -> None:
    uma_common = _load_uma_common()

    hint = uma_common._model_load_hint(ModuleNotFoundError("No module named 'fairchem'"))

    assert "FairChem is not importable" in hint
    assert "uma_gpu.source_list" in hint
