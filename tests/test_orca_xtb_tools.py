from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.analysis import analyze_orca_results, analyze_xtb_results
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.execution.orca_dispatch import orca_execute_batch
from catmaster.tools.geometry_inputs import orca_prepare, xtb_prepare
from catmaster.tools.registry import ToolRegistry


def _project_space(tmp_path: Path) -> Path:
    project = tmp_path / "project_space"
    ensure_project_space_layout(project, create=True)
    return project


def _write_water_xyz(files_root: Path) -> None:
    xyz_path = files_root / "structures" / "h2o.xyz"
    xyz_path.parent.mkdir(parents=True, exist_ok=True)
    xyz_path.write_text(
        "3\nwater\nO 0.000000 0.000000 0.000000\nH 0.758602 0.000000 0.504284\nH -0.758602 0.000000 0.504284\n",
        encoding="utf-8",
    )


def test_xtb_prepare_agent_schemas_keep_optional_controls_non_nullable() -> None:
    registry = ToolRegistry()
    openai_tool = next(item for item in registry.as_openai_tools() if item["name"] == "xtb_prepare")
    langchain_tool = registry.as_langchain_tools(allowlist=["xtb_prepare"])[0]
    schemas = [openai_tool["parameters"], langchain_tool.args_schema]

    for schema in schemas:
        properties = schema["properties"]
        assert properties["xcontrol_path"]["type"] == "string"
        assert "anyOf" not in properties["xcontrol_path"]
        for name in (
            "fixed_atom_indices",
            "constrained_atom_indices",
            "distance_constraints",
            "angle_constraints",
            "dihedral_constraints",
        ):
            assert properties[name]["type"] == "array"
            assert "anyOf" not in properties[name]
        assert properties["fixed_atom_indices"]["items"]["minimum"] == 0
        assert properties["constrained_atom_indices"]["items"]["minimum"] == 0


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


def test_orca_prepare_auto_optfreq_uses_r2scan3c_structure_level(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_water_xyz(files_root)
        _, artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_auto_optfreq",
                "task": "optfreq",
                "maxcore_mb": 512,
            }
        )
        run_dir = files_root / artifact["data"]["records"][0]["run_dir_rel"]
        inp_text = (run_dir / "job.inp").read_text(encoding="utf-8")
        assert "! r2SCAN-3c TightSCF Opt Freq" in inp_text
        assert artifact["data"]["method"] == "r2SCAN-3c"
        assert artifact["data"]["basis"] == ""


def test_orca_prepare_auto_sp_uses_larger_basis(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_water_xyz(files_root)
        _, artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_auto_sp",
                "task": "sp",
                "maxcore_mb": 512,
            }
        )
        run_dir = files_root / artifact["data"]["records"][0]["run_dir_rel"]
        inp_text = (run_dir / "job.inp").read_text(encoding="utf-8")
        assert "! WB97X-D4 def2-TZVP TightSCF" in inp_text
        assert artifact["data"]["basis"] == "def2-TZVP"


def test_orca_prepare_auto_basis_stays_blank_for_internal_basis_methods(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_water_xyz(files_root)
        _, r2scan_artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_r2scan",
                "task": "opt",
                "method": "r2SCAN-3c",
                "maxcore_mb": 512,
            }
        )
        r2scan_dir = files_root / r2scan_artifact["data"]["records"][0]["run_dir_rel"]
        r2scan_line = (r2scan_dir / "job.inp").read_text(encoding="utf-8").splitlines()[0]
        assert r2scan_line == "! r2SCAN-3c TightSCF Opt"
        assert r2scan_artifact["data"]["basis"] == ""

        _, xtb_artifact = orca_prepare(
            {
                "input_path": "structures/h2o.xyz",
                "output_root": "calculations/orca/h2o_xtb2",
                "task": "opt",
                "method": "XTB2",
                "maxcore_mb": 512,
            }
        )
        xtb_dir = files_root / xtb_artifact["data"]["records"][0]["run_dir_rel"]
        xtb_line = (xtb_dir / "job.inp").read_text(encoding="utf-8").splitlines()[0]
        assert xtb_line == "! XTB2 TightSCF Opt"
        assert xtb_artifact["data"]["basis"] == ""


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
        content, artifact = analyze_xtb_results({"result_root": "results/xtb_case"})
        summary_path = files_root / artifact["data"]["summary_json_rel"]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["records"][0]["state"] == "completed"
        assert payload["records"][0]["energy_hartree"] == -5.4321
        assert payload["records"][0]["imaginary_frequency_count"] == 1
        assert artifact["data"]["summary_json_rel"] in content
        assert artifact["data"]["summary_csv_rel"] in content


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
        content, artifact = analyze_orca_results({"result_root": "results/orca_case"})
        summary_path = files_root / artifact["data"]["summary_json_rel"]
        payload = json.loads(summary_path.read_text(encoding="utf-8"))
        assert payload["records"][0]["state"] == "completed"
        assert payload["records"][0]["final_energy_hartree"] == -76.123456789
        assert payload["records"][0]["imaginary_frequency_count"] == 1
        assert artifact["data"]["summary_json_rel"] in content
        assert artifact["data"]["summary_csv_rel"] in content


def test_xtb_prepare_writes_manifest_and_constraint_input(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        input_path = files_root / "molecules" / "h2o.xyz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )

        content, artifact = xtb_prepare(
            {
                "input_path": "molecules/h2o.xyz",
                "output_root": "prepared/xtb_h2o",
                "mode": "opt",
                "gfn": "gfn1",
                "solvent_model": "alpb",
                "solvent": "water",
                "charge": -1,
                "uhf": 1,
                "opt_level": "tight",
                "fixed_atom_indices": [0],
                "constrained_atom_indices": [1],
                "distance_constraints": [{"atom1": 1, "atom2": 2, "value_angstrom": 0.96}],
                "constraint_force_constant": 0.5,
            }
        )

        assert "xtb_prepare completed" in content
        stage_dir = files_root / "prepared" / "xtb_h2o"
        manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest == {
            "schema_version": 1,
            "program": "xtb",
            "coordinate_file": "coord.xyz",
            "xcontrol_file": "xtb.inp",
            "mode": "opt",
            "gfn": "gfn1",
            "solvent_model": "alpb",
            "solvent": "water",
            "charge": -1,
            "uhf": 1,
            "opt_level": "tight",
            "source_rel": "molecules/h2o.xyz",
            "xcontrol_source_rel": "",
            "atom_count": 3,
        }
        xcontrol = (stage_dir / "xtb.inp").read_text(encoding="utf-8")
        assert "$fix\n  atoms: 1\n$end" in xcontrol
        assert "$constrain" in xcontrol
        assert "force constant=0.5" in xcontrol
        assert "atoms: 2" in xcontrol
        assert "distance: 2, 3, 0.96" in xcontrol
        assert artifact["data"]["prepared_count"] == 1


def test_xtb_prepare_copies_complete_xcontrol_verbatim(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        input_path = files_root / "molecules" / "h2o.xyz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text(
            "3\nwater\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
            encoding="utf-8",
        )
        custom_text = "$constrain\n  distance: 1, 2, 1.25\n$end\n"
        custom_path = files_root / "controls" / "custom.inp"
        custom_path.parent.mkdir(parents=True, exist_ok=True)
        custom_path.write_text(custom_text, encoding="utf-8")

        xtb_prepare(
            {
                "input_path": "molecules/h2o.xyz",
                "output_root": "prepared/xtb_custom",
                "mode": "opt",
                "xcontrol_path": "controls/custom.inp",
            }
        )

        stage_dir = files_root / "prepared" / "xtb_custom"
        assert (stage_dir / "xtb.inp").read_text(encoding="utf-8") == custom_text
        manifest = json.loads((stage_dir / "manifest.json").read_text(encoding="utf-8"))
        assert manifest["xcontrol_file"] == "xtb.inp"
        assert manifest["xcontrol_source_rel"] == "controls/custom.inp"


def test_xtb_prepare_flattens_nested_batch_inputs_to_first_level_stages(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        for subdir in ("set_a", "set_b"):
            input_path = files_root / "molecules" / subdir / "h2o.xyz"
            input_path.parent.mkdir(parents=True, exist_ok=True)
            input_path.write_text(
                "3\nwater\nO 0.0 0.0 0.0\nH 0.7 0.0 0.5\nH -0.7 0.0 0.5\n",
                encoding="utf-8",
            )

        _, artifact = xtb_prepare(
            {
                "input_path": "molecules",
                "output_root": "prepared/xtb_batch",
                "mode": "sp",
            }
        )

        stage_root = files_root / "prepared" / "xtb_batch"
        stage_names = {path.name for path in stage_root.iterdir() if path.is_dir()}
        assert stage_names == {"set_a_h2o", "set_b_h2o"}
        assert all((stage_root / name / "manifest.json").is_file() for name in stage_names)
        assert artifact["data"]["prepared_count"] == 2


def test_xtb_prepare_rejects_out_of_range_constraint_index(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        input_path = files_root / "molecules" / "h2.xyz"
        input_path.parent.mkdir(parents=True, exist_ok=True)
        input_path.write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")

        with pytest.raises(CatMasterToolExecutionError) as exc_info:
            xtb_prepare(
                {
                    "input_path": "molecules/h2.xyz",
                    "output_root": "prepared/xtb_h2",
                    "distance_constraints": [{"atom1": 0, "atom2": 2, "value_angstrom": 0.7}],
                }
            )

        assert exc_info.value.error_code == "constraint_index_out_of_range"


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
