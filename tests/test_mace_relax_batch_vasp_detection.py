from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mace_dispatch import mace_relax_batch


def test_mace_relax_batch_detects_vasp_inputs(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def _fake_dispatch(req):
        captured["command"] = req.tasks[0].command
        return SimpleNamespace(
            task_states=["5"],
            submission_dir=str((Path(req.local_root) / "_fake_submission").resolve()),
            work_base=req.work_base,
            duration_s=0.01,
        )

    monkeypatch.setattr("catmaster.tools.execution.mace_dispatch._resolve_machine_for_resources", lambda _: "dummy")
    monkeypatch.setattr("catmaster.tools.execution.mace_dispatch.dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        files_root = tmp_path / "files"
        input_dir = files_root / "inputs"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "O2.vasp").write_text("dummy", encoding="utf-8")

        _content, artifact = mace_relax_batch(
            {
                "input_dir": "inputs",
                "output_root": "outputs",
            }
        )

    data = artifact["data"]
    assert "mace_relax.py" in str(captured["command"])
    assert data["structures_found"] == 1
