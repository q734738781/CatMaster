from __future__ import annotations

import pytest

pytest.importorskip("mp_api")
pytest.importorskip("pymatgen")

from pymatgen.core import Lattice, Structure

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import workspace_scope
from catmaster.tools.retrieval import matdb


class _FakeMPRester:
    def __init__(self, *, success_ids: set[str]):
        self._success_ids = success_ids

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc, tb):
        return False

    def get_structure_by_material_id(self, mp_id: str):
        if mp_id not in self._success_ids:
            raise RuntimeError(f"{mp_id} not found")
        return Structure(Lattice.cubic(3.2), ["Si"], [[0, 0, 0]])


def test_mp_download_structure_partial_failure_is_success(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(matdb, "_mpr", lambda **_: _FakeMPRester(success_ids={"mp-good"}))

    with workspace_scope(tmp_path):
        content, artifact = matdb.mp_download_structure(
            {
                "mp_ids": ["mp-good", "mp-bad"],
                "fmt": "poscar",
                "output_dir": "retrieval/mp",
            }
        )

    assert "mp_download_structure completed" in str(content)
    assert "retrieval/mp/mp-good.vasp" in str(content)
    assert artifact.get("warnings")
    data = (artifact or {}).get("data", {})
    assert len(data.get("results", [])) == 1
    assert len(data.get("errors", [])) == 1
    assert data.get("downloaded") == 1
    assert data.get("requested") == 2


def test_mp_download_structure_all_fail_raises(monkeypatch: pytest.MonkeyPatch, tmp_path) -> None:
    monkeypatch.setattr(matdb, "_mpr", lambda **_: _FakeMPRester(success_ids=set()))

    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            matdb.mp_download_structure(
                {
                    "mp_ids": ["mp-a", "mp-b"],
                    "fmt": "poscar",
                    "output_dir": "retrieval/mp",
                }
            )

    assert "Failed to download structures for all requested mp_ids." in str(excinfo.value)
