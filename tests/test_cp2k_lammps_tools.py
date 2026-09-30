from __future__ import annotations

import json
from pathlib import Path

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.analysis import analyze_trajectory
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.dynamics import cp2k_output_summary, lammps_log_summary, lammps_prepare, md_trajectory_summary
from catmaster.tools.geometry_inputs import cp2k_prepare
from catmaster.tools.registry import ToolRegistry


def _project_space(tmp_path: Path) -> Path:
    project = tmp_path / "project_space"
    ensure_project_space_layout(project, create=True)
    return project


def _schema_pair(name: str) -> tuple[dict, dict]:
    registry = ToolRegistry()
    openai_schema = next(item for item in registry.as_openai_tools() if item["name"] == name)["parameters"]
    langchain_schema = registry.as_langchain_tools(allowlist=[name])[0].args_schema
    return openai_schema, langchain_schema


def test_cp2k_lammps_prepare_schemas_expose_complete_native_files() -> None:
    for name in ("cp2k_prepare", "lammps_prepare"):
        for schema in _schema_pair(name):
            assert set(schema["properties"]) == {"input_path", "output_root", "asset_mappings"}
            assert {"input_path", "output_root"} <= set(schema["required"])
            assert schema["properties"]["asset_mappings"]["type"] == "array"
            assert "anyOf" not in schema["properties"]["asset_mappings"]

    names = {item["name"] for item in ToolRegistry().as_openai_tools()}
    assert "cp2k_aimd_prepare" not in names
    assert "lammps_forcefield_validate" not in names


def test_cp2k_prepare_copies_complete_input_and_explicit_assets_without_science_injection(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        native = (
            "&GLOBAL\n  RUN_TYPE ENERGY_FORCE\n&END GLOBAL\n"
            "&FORCE_EVAL\n  METHOD QS\n  &DFT\n    UKS TRUE\n    &SCF\n      SCF_GUESS RESTART\n    &END SCF\n"
            "    &KPOINTS\n      SCHEME MONKHORST-PACK 2 2 2\n    &END KPOINTS\n"
            "  &END DFT\n&END FORCE_EVAL\n"
        )
        source = files / "inputs" / "cp2k.inp"
        source.parent.mkdir(parents=True)
        source.write_text(native, encoding="utf-8")
        restart = files / "inputs" / "state" / "restart.wfn"
        restart.parent.mkdir(parents=True)
        restart.write_bytes(b"native-wavefunction")

        content, artifact = cp2k_prepare(
            {
                "input_path": "inputs/cp2k.inp",
                "output_root": "prepared/cp2k",
                "asset_mappings": [
                    {"source_path": "inputs/state/restart.wfn", "stage_path": "state/restart.wfn"}
                ],
            }
        )
        stage = files / artifact["data"]["stage_path"]
        assert (stage / "job.inp").read_text(encoding="utf-8") == native
        assert (stage / "state" / "restart.wfn").read_bytes() == b"native-wavefunction"
        assert json.loads((stage / "manifest.json").read_text(encoding="utf-8")) == {
            "input_file": "job.inp",
            "input_assets": ["state/restart.wfn"],
        }
        assert "completed" in content


def test_cp2k_prepare_rejects_recipe_surface_and_existing_stage(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        source = files / "inputs" / "cp2k.inp"
        source.parent.mkdir(parents=True)
        source.write_text("&GLOBAL\n RUN_TYPE MD\n&END GLOBAL\n", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError):
            cp2k_prepare(
                {
                    "input_path": "inputs/cp2k.inp",
                    "output_root": "prepared/old-recipe",
                    "recipe": "nvt",
                }
            )
        occupied = files / "prepared" / "occupied"
        occupied.mkdir(parents=True)
        (occupied / "old").write_text("old", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError):
            cp2k_prepare({"input_path": "inputs/cp2k.inp", "output_root": "prepared/occupied"})


def test_cp2k_summary_uses_formal_termination_final_metrics_and_frequency_units(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "cp2k"
        run.mkdir(parents=True)
        (run / "job.inp").write_text("&GLOBAL\n RUN_TYPE VIBRATIONAL_ANALYSIS\n&END GLOBAL\n", encoding="utf-8")
        (run / "cp2k_summary.json").write_text(json.dumps({"returncode": 0}), encoding="utf-8")
        (run / "job.out").write_text(
            "ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -10.0\n"
            "Max. gradient = 0.20\n"
            "Max. gradient = 0.01\n"
            "VIB| Frequency (cm^-1) -123.4 456.7\n"
            "PROGRAM ENDED AT\n",
            encoding="utf-8",
        )
        _, artifact = cp2k_output_summary({"result_root": "results/cp2k"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        assert record["execution_state"] == "completed"
        assert record["task_state"] == "not_applicable"
        assert record["optimization"]["convergence_metrics"]["max_gradient"] == 0.01
        assert record["frequencies"]["values_cm-1"] == [-123.4, 456.7]
        assert record["frequencies"]["imaginary_count"] == 1


def test_cp2k_geo_opt_normal_exit_without_convergence_is_not_converged(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "cp2k-opt"
        run.mkdir(parents=True)
        (run / "job.inp").write_text("&GLOBAL\n RUN_TYPE GEO_OPT\n&END GLOBAL\n", encoding="utf-8")
        (run / "job.out").write_text("ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -1.0\nPROGRAM ENDED AT\n", encoding="utf-8")
        _, artifact = cp2k_output_summary({"result_root": "results/cp2k-opt"})
        payload = json.loads((project / "files" / artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        assert record["execution_state"] == "completed"
        assert record["task_state"] == "not_converged"
        assert record["frequencies"]["count"] is None


def test_lammps_prepare_copies_complete_script_data_restart_and_potential(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files = project / "files"
        script_text = (
            "units real\natom_style full\nboundary p p f\nread_restart state/restart.bin\n"
            "pair_style table linear 1000\npair_coeff 1 1 potentials/pair.table PAIR\n"
            "timestep 0.25\nfix thermostat all nvt temp 300 300 100\nrun 1000\n"
        )
        source = files / "inputs" / "in.custom"
        source.parent.mkdir(parents=True)
        source.write_text(script_text, encoding="utf-8")
        restart = files / "inputs" / "restart.bin"
        restart.write_bytes(b"restart")
        table = files / "inputs" / "pair.table"
        table.write_text("PAIR\nN 1\n\n1 1.0 0.0 0.0\n", encoding="utf-8")
        _, artifact = lammps_prepare(
            {
                "input_path": "inputs/in.custom",
                "output_root": "prepared/lammps",
                "asset_mappings": [
                    {"source_path": "inputs/restart.bin", "stage_path": "state/restart.bin"},
                    {"source_path": "inputs/pair.table", "stage_path": "potentials/pair.table"},
                ],
            }
        )
        stage = files / artifact["data"]["stage_path"]
        assert (stage / "in.lammps").read_text(encoding="utf-8") == script_text
        assert (stage / "state" / "restart.bin").read_bytes() == b"restart"
        assert (stage / "potentials" / "pair.table").is_file()


def test_lammps_log_separates_process_task_and_thermo_states(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "lammps"
        run.mkdir(parents=True)
        (run / "log.lammps").write_text(
            "Step Temp PotEng TotEng\n"
            "0 300 -1.0 -0.5\n"
            "10 305 -1.1 -0.6\n"
            "Loop time of 0.01 on 1 procs for 10 steps with 2 atoms\n"
            "Stopping criterion = max iterations\n"
            "Iterations, force evaluations = 10 20\n",
            encoding="utf-8",
        )
        _, artifact = lammps_log_summary({"result_root": "results/lammps"})
        payload = json.loads((project / "files" / artifact["data"]["summary_path"]).read_text(encoding="utf-8"))
        record = payload["records"][0]
        assert record["execution_state"] == "completed"
        assert record["task_state"] == "not_converged"
        assert record["thermo_state"] == "calculated"
        assert record["thermo_rows"] == 2
        assert record["thermo_drift"]["temperature"] == 5.0


def _write_lammps_dump(path: Path) -> None:
    path.write_text(
        "ITEM: TIMESTEP\n0\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type element x y z ix iy iz\n"
        "1 1 Li 9.8 0 0 0 0 0\n"
        "2 1 Li 5.0 0 0 0 0 0\n"
        "ITEM: TIMESTEP\n10\n"
        "ITEM: NUMBER OF ATOMS\n2\n"
        "ITEM: BOX BOUNDS pp pp pp\n0 10\n0 10\n0 10\n"
        "ITEM: ATOMS id type element x y z ix iy iz\n"
        "1 1 Li 0.2 0 0 1 0 0\n"
        "2 1 Li 5.1 0 0 0 0 0\n",
        encoding="utf-8",
    )


def test_lammps_dump_analysis_uses_image_flags_explicit_time_and_normalized_rdf(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "trajectory"
        run.mkdir(parents=True)
        dump = run / "trajectory.lammpstrj"
        _write_lammps_dump(dump)
        _, artifact = analyze_trajectory(
            {
                "path": "results/trajectory/trajectory.lammpstrj",
                "frame_interval_fs": 100.0,
                "species": "Li",
                "rdf_species": "Li",
                "fit_start_frame": 0,
                "fit_end_frame": 2,
                "rdf_bins": 10,
                "rdf_max_angstrom": 5.0,
            }
        )
        payload = json.loads((project / "files" / artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        assert payload["coordinate_mode"] == "wrapped_cartesian+image_flags"
        assert payload["positions_are_wrapped"] is False
        assert payload["frame_interval_fs"] == 100.0
        assert payload["final_msd_a2"] == pytest.approx(0.085)
        assert payload["diffusion_coefficient_a2_per_ps"] == pytest.approx(0.85 / 6.0)
        assert payload["rdf_state"] == "calculated"
        rdf_lines = (project / "files" / payload["rdf_csv_rel"]).read_text(encoding="utf-8").splitlines()
        assert len(rdf_lines) == 11
        expected_last_bin = 1000.0 / ((4.0 * 3.141592653589793 / 3.0) * (5.0**3 - 4.5**3))
        assert float(rdf_lines[-1].split(",")[1]) == pytest.approx(expected_last_bin)


def test_trajectory_analysis_requires_explicit_sampling_and_fit_window() -> None:
    for schema in _schema_pair("analyze_trajectory"):
        assert {"path", "frame_interval_fs"} <= set(schema["required"])
        assert "timestep_fs" not in schema["properties"]
        assert "fit_fraction" not in schema["properties"]
    from catmaster.tools.analysis.results_analysis import AnalyzeTrajectoryInput
    with pytest.raises(ValueError, match="fit_start_frame"):
        AnalyzeTrajectoryInput(path="run.extxyz", frame_interval_fs=1)
    assert AnalyzeTrajectoryInput(path="run.extxyz", frame_interval_fs=1, compute_msd=False).fit_start_frame == -1


def test_periodic_generic_trajectory_requires_explicit_coordinate_semantics(tmp_path: Path) -> None:
    from ase import Atoms
    from ase.io import write as ase_write

    project = _project_space(tmp_path)
    with workspace_scope(project):
        trajectory = project / "files" / "periodic.extxyz"
        frames = [
            Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
            Atoms("Li", positions=[[3.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
        ]
        ase_write(str(trajectory), frames, format="extxyz")
        with pytest.raises(CatMasterToolExecutionError, match="coordinate_semantics"):
            analyze_trajectory(
                {
                    "path": "periodic.extxyz",
                    "frame_interval_fs": 1.0,
                    "fit_start_frame": 0,
                }
            )
        _, artifact = analyze_trajectory(
            {
                "path": "periodic.extxyz",
                "frame_interval_fs": 1.0,
                "fit_start_frame": 0,
                "coordinate_semantics": "unwrapped",
            }
        )
        payload = json.loads((project / "files" / artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        assert payload["final_msd_a2"] == pytest.approx(9.0)


def test_lammps_dump_rejects_unstable_atom_ids(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        dump = project / "files" / "unstable.lammpstrj"
        _write_lammps_dump(dump)
        text = dump.read_text(encoding="utf-8")
        split = text.index("ITEM: TIMESTEP\n10")
        dump.write_text(text[:split] + text[split:].replace("2 1 Li 5.1", "3 1 Li 5.1", 1), encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError, match="atom IDs"):
            analyze_trajectory(
                {
                    "path": "unstable.lammpstrj",
                    "frame_interval_fs": 100.0,
                    "fit_start_frame": 0,
                }
            )


def test_md_trajectory_summary_is_inventory_only_and_requires_unambiguous_source(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "summary"
        run.mkdir(parents=True)
        dump = run / "trajectory.lammpstrj"
        _write_lammps_dump(dump)
        (run / "msd.dat").write_text("0 0\n10 0.085\n", encoding="utf-8")
        content, artifact = md_trajectory_summary({"path": "results/summary/trajectory.lammpstrj"})
        payload = json.loads((project / "files" / artifact["data"]["summary_path"]).read_text(encoding="utf-8"))
        assert payload["nframes"] == 2
        assert payload["native_observables"]["msd.dat"]["last_row"] == [10.0, 0.085]
        assert "diffusion" not in json.dumps(payload).lower()
        assert payload["final_frame"] in content

        (run / "other.xyz").write_text("1\nother\nH 0 0 0\n", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError, match="Multiple trajectory"):
            md_trajectory_summary({"path": "results/summary"})


def test_md_trajectory_summary_supports_ase_traj_and_exports_final_frame(tmp_path: Path) -> None:
    from ase import Atoms
    from ase.io.trajectory import Trajectory

    project = _project_space(tmp_path)
    with workspace_scope(project):
        run = project / "files" / "results" / "ase_summary"
        run.mkdir(parents=True)
        trajectory_path = run / "md.traj"
        frames = [
            Atoms("Li2", positions=[[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
            Atoms("Li2", positions=[[0.2, 0.0, 0.0], [2.1, 0.0, 0.0]], cell=[5.0, 5.0, 5.0], pbc=True),
        ]
        with Trajectory(str(trajectory_path), mode="w") as trajectory:
            for frame in frames:
                trajectory.write(frame)

        content, artifact = md_trajectory_summary({"path": "results/ase_summary"})
        payload = json.loads((project / "files" / artifact["data"]["summary_path"]).read_text(encoding="utf-8"))
        assert payload["trajectory"].endswith("results/ase_summary/md.traj")
        assert payload["format"] == "ase-trajectory"
        assert payload["nframes"] == 2
        assert payload["natoms"] == 2
        assert payload["final_frame"].endswith("final_frame.traj")
        assert payload["final_frame"] in content

        with Trajectory(str(project / "files" / payload["final_frame"]), mode="r") as final_trajectory:
            assert len(final_trajectory) == 1
            assert final_trajectory[0].get_positions() == pytest.approx(frames[-1].get_positions())
