from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

from catmaster.tools.analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.execution.orca_dispatch import orca_execute_batch
from catmaster.tools.execution.xtb_dispatch import xtb_run_batch
from catmaster.tools.geometry_inputs import orca_prepare


def _project_space(tmp_path: Path) -> Path:
    project = tmp_path / "project_space"
    ensure_project_space_layout(project, create=True)
    return project


def test_orca_prepare_single_structure_creates_input(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        xyz_path = files_root / "structures" / "h2o.xyz"
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        xyz_path.write_text(
            "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284\n",
            encoding="utf-8",
        )
        content, artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_opt",
                "task": "optfreq",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "charge": 0,
                "multiplicity": 1,
                "maxcore_mb": 512,
            }
        )
        assert "orca_prepare completed" in content
        run_dir = files_root / artifact["data"]["records"][0]["run_dir_rel"]
        inp_text = (run_dir / "job.inp").read_text(encoding="utf-8")
        assert "B3LYP" in inp_text
        assert "Opt" in inp_text
        assert "Freq" in inp_text
        assert "%pal" not in inp_text
        assert "* xyzfile 0 1 input.xyz" in inp_text


def test_orca_prepare_td_uses_tddft_block_without_simple_keyword(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        xyz_path = files_root / "structures" / "h2o.xyz"
        xyz_path.parent.mkdir(parents=True, exist_ok=True)
        xyz_path.write_text(
            "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284\n",
            encoding="utf-8",
        )
        _, artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_td",
                "task": "td",
                "method": "B3LYP",
                "basis": "def2-SVP",
                "maxcore_mb": 512,
                "safe_patch": {"nroots": 3},
            }
        )
        run_dir = files_root / artifact["data"]["records"][0]["run_dir_rel"]
        inp_text = (run_dir / "job.inp").read_text(encoding="utf-8")
        assert "! B3LYP def2-SVP TightSCF" in inp_text
        assert "TDDFT" not in inp_text
        assert "%tddft" in inp_text
        assert "NRoots 3" in inp_text


def test_analyze_xtb_results_parses_summary(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        run_dir = files_root / "results" / "xtb_case"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "xtb_summary.json").write_text(
            json.dumps({"completed": True, "returncode": 0, "log_file": "xtb_stdout.out"}),
            encoding="utf-8",
        )
        (run_dir / "xtb_stdout.out").write_text(
            "TOTAL ENERGY      -5.432100\nTOTAL FREE ENERGY -5.400000\n",
            encoding="utf-8",
        )
        (run_dir / "xtbopt.xyz").write_text(
            "3\nxtbopt\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )
        (run_dir / "g98.out").write_text("  1  -123.45 cm-1\n  2  345.67 cm-1\n", encoding="utf-8")
        _, artifact = analyze_xtb_results({"result_root": "results/xtb_case"})
        summary_path = files_root / artifact["data"]["summary_json_rel"]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["records"][0]["state"] == "completed"
        assert payload["records"][0]["energy_hartree"] == -5.4321
        assert payload["records"][0]["imaginary_frequency_count"] == 1


def test_analyze_orca_results_parses_output(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        run_dir = files_root / "results" / "orca_case"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "orca_summary.json").write_text(
            json.dumps({"completed": True, "returncode": 0}),
            encoding="utf-8",
        )
        (run_dir / "job.out").write_text(
            "\n".join(
                [
                    "FINAL SINGLE POINT ENERGY     -76.123456789",
                    "  1:      -321.00 cm**-1",
                    "  2:       456.70 cm**-1",
                    "Final Gibbs free energy  ...   -76.000000000",
                    "***  ORCA TERMINATED NORMALLY  ***",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "job.xyz").write_text(
            "3\norca\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )
        _, artifact = analyze_orca_results({"result_root": "results/orca_case"})
        summary_path = files_root / artifact["data"]["summary_json_rel"]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["records"][0]["state"] == "completed"
        assert payload["records"][0]["final_energy_hartree"] == -76.123456789
        assert payload["records"][0]["imaginary_frequency_count"] == 1


def test_xtb_run_batch_collects_outputs_with_mock_dispatch(tmp_path: Path, monkeypatch) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        input_dir = files_root / "molecules"
        input_dir.mkdir(parents=True, exist_ok=True)
        (input_dir / "h2o.xyz").write_text(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )

        def _fake_dispatch(batch):
            stage_root = Path(batch.local_root) / batch.work_base
            for task in batch.tasks:
                task_dir = stage_root / task.task_work_path
                (task_dir / "xtb_summary.json").write_text(
                    json.dumps({"completed": True, "returncode": 0, "log_file": "xtb_stdout.out"}),
                    encoding="utf-8",
                )
                (task_dir / "xtb_stdout.out").write_text("TOTAL ENERGY -5.0\n", encoding="utf-8")
                (task_dir / "xtbopt.xyz").write_text(
                    "3\nopt\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
                    encoding="utf-8",
                )
            return SimpleNamespace(
                task_states=["finished" for _ in batch.tasks],
                submission_dir=str(stage_root),
                work_base=batch.work_base,
                duration_s=0.1,
            )

        monkeypatch.setattr("catmaster.tools.execution.xtb_dispatch.dispatch_submission", _fake_dispatch)
        _, artifact = xtb_run_batch(
            {
                "input_path": "molecules",
                "output_root": "results/xtb_batch",
                "mode": "opt",
            }
        )
        output_dir = files_root / artifact["data"]["outputs"][0]["output_dir_rel"]
        assert (output_dir / "xtb_summary.json").is_file()
        assert (output_dir / "xtbopt.xyz").is_file()


def test_orca_execute_batch_collects_outputs_with_mock_dispatch(tmp_path: Path, monkeypatch) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        run_dir = files_root / "prepared" / "orca_job"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "job.inp").write_text("! B3LYP def2-SVP\n* xyzfile 0 1 input.xyz\n", encoding="utf-8")
        (run_dir / "input.xyz").write_text(
            "3\norca\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )

        def _fake_dispatch(batch):
            stage_root = Path(batch.local_root) / batch.work_base
            for task in batch.tasks:
                task_dir = stage_root / task.task_work_path
                (task_dir / "orca_summary.json").write_text(
                    json.dumps({"completed": True, "returncode": 0}),
                    encoding="utf-8",
                )
                (task_dir / "job.out").write_text(
                    "FINAL SINGLE POINT ENERGY -76.1\n*** ORCA TERMINATED NORMALLY ***\n",
                    encoding="utf-8",
                )
                (task_dir / "job.xyz").write_text(
                    "3\norca\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
                    encoding="utf-8",
                )
            return SimpleNamespace(
                task_states=["finished" for _ in batch.tasks],
                submission_dir=str(stage_root),
                work_base=batch.work_base,
                duration_s=0.1,
            )

        monkeypatch.setattr("catmaster.tools.execution.orca_dispatch.dispatch_submission", _fake_dispatch)
        _, artifact = orca_execute_batch(
            {
                "input_dir": "prepared/orca_job",
                "output_root": "results/orca_batch",
            }
        )
        output_dir = files_root / artifact["data"]["outputs"][0]["output_dir_rel"]
        assert (output_dir / "orca_summary.json").is_file()
        assert (output_dir / "job.out").is_file()
