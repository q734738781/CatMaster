from __future__ import annotations

import importlib.util
import json
from types import SimpleNamespace
import subprocess
import sys
from pathlib import Path


def _load_smoke_module():
    script = Path(__file__).resolve().parents[1] / "scripts" / "remote_execution_smoke.py"
    spec = importlib.util.spec_from_file_location("catmaster_remote_execution_smoke", script)
    assert spec and spec.loader
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    spec.loader.exec_module(module)
    return module


def test_remote_execution_smoke_script_lists_cases_without_submitting() -> None:
    repo_root = Path(__file__).resolve().parents[1]
    proc = subprocess.run(
        [sys.executable, "scripts/remote_execution_smoke.py", "--list"],
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.PIPE,
        text=True,
        check=False,
    )

    assert proc.returncode == 0, proc.stderr
    assert "core: mace_sp, xtb_sp, orca_sp" in proc.stdout
    assert "uma: uma_mol_sp, uma_mol_relax, uma_mat_sp, uma_mat_relax" in proc.stdout
    assert "all: mace_sp, vasp_sp, xtb_sp, orca_sp, cp2k_sp, lammps_min, crest_quick" in proc.stdout
    assert "all: mace_sp, uma_mol_sp" not in proc.stdout
    assert "vasp_sp" in proc.stdout
    assert "uma_mol_sp" in proc.stdout
    assert "uma_mol_relax" in proc.stdout
    assert "cp2k_sp" in proc.stdout
    assert "mlff_si512: si512_mace_sp" in proc.stdout
    assert "si512_orb_md" in proc.stdout
    assert "orb_neb" in proc.stdout
    assert "no_cp2k" not in proc.stdout


def test_si512_acceptance_structure_is_deterministic(tmp_path: Path) -> None:
    from ase.io import read
    import numpy as np

    smoke = _load_smoke_module()
    first = tmp_path / "first.vasp"
    second = tmp_path / "second.vasp"
    smoke._write_si512_poscar(first, displacement_A=0.01)
    smoke._write_si512_poscar(second, displacement_A=0.01)

    atoms = read(first)
    repeated = read(second)
    assert len(atoms) == 512
    assert set(atoms.get_chemical_symbols()) == {"Si"}
    assert np.allclose(atoms.positions, repeated.positions)


def _smoke_context(smoke, tmp_path: Path):
    files_root = tmp_path / "files"
    run_dir = tmp_path / "metadata" / "runs" / "test"
    files_root.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    return smoke.SmokeContext(
        project_space=tmp_path,
        run_id="contract_test",
        files_root=files_root,
        metadata_root=tmp_path / "metadata",
        run_dir=run_dir,
    )


def test_orca_smoke_uses_stage_path_from_current_prepare_contract(monkeypatch, tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    ctx = _smoke_context(smoke, tmp_path)
    stage_rel = f"{ctx.case_root_rel}/orca_stage"
    stage_dir = ctx.files_root / stage_rel
    captured: dict[str, object] = {}

    def fake_invoke(_ctx, tool_name, payload, *, audience=""):
        captured["tool_name"] = tool_name
        captured["payload"] = payload
        stage_dir.mkdir(parents=True)
        return "", {"data": {"records": [{"stage_path": stage_rel}]}}

    def fake_submit(_ctx, **kwargs):
        captured["submit"] = kwargs
        (stage_dir / "status.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
        (stage_dir / "orca_summary.json").write_text(
            json.dumps({"execution_state": "completed", "normal_termination": True}),
            encoding="utf-8",
        )
        (stage_dir / "job.out").write_text("FINAL SINGLE POINT ENERGY -147.0\n", encoding="utf-8")
        return {"task_state_counts": {"finished": 1}}

    monkeypatch.setattr(smoke, "_invoke_tool", fake_invoke)
    monkeypatch.setattr(smoke, "_submit_remote", fake_submit)
    result = smoke.run_orca_sp(
        ctx,
        SimpleNamespace(orca_method="HF", orca_basis="STO-3G", orca_scf_maxiter=80, orca_check_interval=1),
    )

    assert result["stage_rel"] == stage_rel
    assert captured["submit"]["work_dir"] == stage_rel
    assert captured["payload"]["simple_keywords"] == ["HF", "STO-3G"]


def test_crest_smoke_passes_native_quick_argv_without_reconstructed_thresholds(monkeypatch, tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    ctx = _smoke_context(smoke, tmp_path)
    captured: dict[str, object] = {}

    def fake_invoke(_ctx, tool_name, payload, *, audience=""):
        captured["tool_name"] = tool_name
        captured["payload"] = payload
        stage_rel = str(payload["output_root"])
        (ctx.files_root / stage_rel).mkdir(parents=True)
        return "", {"data": {"stage_path": stage_rel}}

    def fake_submit(_ctx, **kwargs):
        captured["submit"] = kwargs
        attempt_rel = f"{kwargs['work_dir']}/attempts/attempt-1"
        attempt_dir = ctx.files_root / attempt_rel
        attempt_dir.mkdir(parents=True)
        (attempt_dir / "status.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
        (attempt_dir / "crest_summary.json").write_text(
            json.dumps({"execution_state": "completed"}),
            encoding="utf-8",
        )
        return {"attempt_paths": [attempt_rel], "task_state_counts": {"finished": 1}}

    monkeypatch.setattr(smoke, "_invoke_tool", fake_invoke)
    monkeypatch.setattr(smoke, "_submit_remote", fake_submit)
    smoke.run_crest_quick(ctx, SimpleNamespace(crest_method="gfn2", crest_check_interval=1))

    assert captured["tool_name"] == "crest_prepare"
    assert captured["payload"]["argv"] == ["input.xyz", "--gfn2", "--quick"]
    assert captured["submit"]["task_name"] == "crest_execute"
    assert "template_overrides" not in captured["submit"]


def test_cp2k_smoke_energy_pattern_matches_current_output(tmp_path: Path) -> None:
    smoke = _load_smoke_module()
    output = tmp_path / "job.out"
    output.write_text(
        " ENERGY| Total FORCE_EVAL ( QS ) energy [hartree]             -1.161622111559334\n",
        encoding="utf-8",
    )

    parsed = smoke._extract_last_float(
        r"ENERGY\|\s+Total\s+FORCE_EVAL.*?energy\s+\[hartree\]\s+([-+]?\d+(?:\.\d+)?(?:[Ee][-+]?\d+)?)",
        output,
    )
    assert parsed == -1.161622111559334
