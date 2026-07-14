from __future__ import annotations

import json
from pathlib import Path

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import ensure_project_space_layout, workspace_scope
from catmaster.tools.dynamics import (
    cp2k_aimd_prepare,
    cp2k_output_summary,
    lammps_forcefield_validate,
    lammps_log_summary,
    lammps_prepare,
    md_trajectory_summary,
)
from catmaster.tools.execution import get_avail_remote_task
from catmaster.tools.geometry_inputs import cp2k_prepare


def _project_space(tmp_path: Path) -> Path:
    project = tmp_path / "project_space"
    ensure_project_space_layout(project, create=True)
    return project


def _write_o2(files_root: Path) -> None:
    path = files_root / "structures" / "O2.xyz"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("2\nO2\nO 0.000000 0.000000 0.000000\nO 0.000000 0.000000 1.210000\n", encoding="utf-8")


def test_cp2k_prepare_writes_conventional_dft_stage(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_o2(files_root)
        content, artifact = cp2k_prepare(
            {
                "input_path": "structures/O2.xyz",
                "output_root": "calculations/cp2k/o2_sp",
                "recipe": "sp",
                "settings": {"periodic": "none", "cell_abc": [12, 12, 12], "xc": "PBE"},
            }
        )
        assert "cp2k_prepare completed" in content
        assert artifact["data"]["manifest_rel"] in content
        stage = files_root / artifact["data"]["records"][0]["stage_dir_rel"]
        inp = (stage / "job.inp").read_text(encoding="utf-8")
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))

    assert "RUN_TYPE ENERGY_FORCE" in inp
    assert "PERIODIC NONE" in inp
    assert "POTENTIAL GTH-PBE" in inp
    assert manifest["program"] == "cp2k"
    assert manifest["recipe"] == "sp"
    assert "https://manual.cp2k.org/" in manifest["references"]


def test_cp2k_prepare_writes_path_refinement_stages(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        path_root = files_root / "paths" / "o2_neb"
        path_root.mkdir(parents=True, exist_ok=True)
        (path_root / "00.xyz").write_text(
            "2\nO2 reactant\nO 0 0 0\nO 0 0 1.20\n",
            encoding="utf-8",
        )
        (path_root / "01.xyz").write_text(
            "2\nO2 product\nO 0 0 0\nO 0 0 1.50\n",
            encoding="utf-8",
        )
        _, neb_artifact = cp2k_prepare(
            {
                "input_path": "paths/o2_neb",
                "output_root": "calculations/cp2k/o2_neb",
                "recipe": "neb",
                "settings": {"periodic": "none", "cell_abc": [12, 12, 12], "band_max_iter": 3},
            }
        )
        neb_stage = files_root / neb_artifact["data"]["records"][0]["stage_dir_rel"]
        neb_inp = (neb_stage / "job.inp").read_text(encoding="utf-8")

        _, dimer_artifact = cp2k_prepare(
            {
                "input_path": "paths/o2_neb/00.xyz",
                "output_root": "calculations/cp2k/o2_dimer",
                "recipe": "dimer",
                "settings": {"periodic": "none", "cell_abc": [12, 12, 12], "max_iter": 3},
            }
        )
        dimer_stage = files_root / dimer_artifact["data"]["records"][0]["stage_dir_rel"]
        dimer_inp = (dimer_stage / "job.inp").read_text(encoding="utf-8")

    assert "RUN_TYPE BAND" in neb_inp
    assert "&BAND" in neb_inp
    assert "NUMBER_OF_REPLICA 2" in neb_inp
    assert (neb_stage / "replica_000.xyz").is_file()
    assert (neb_stage / "replica_001.xyz").is_file()
    assert "TYPE TRANSITION_STATE" in dimer_inp
    assert "METHOD DIMER" in dimer_inp


def test_cp2k_aimd_prepare_requires_plumed_for_metadynamics(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_o2(files_root)
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            cp2k_aimd_prepare(
                {
                    "input_path": "structures/O2.xyz",
                    "output_root": "calculations/cp2k/o2_meta",
                    "recipe": "metadynamics_user_plumed",
                }
            )
    assert "plumed_input_path" in str(excinfo.value)


def test_cp2k_aimd_prepare_writes_md_stage(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_o2(files_root)
        content, artifact = cp2k_aimd_prepare(
            {
                "input_path": "structures/O2.xyz",
                "output_root": "calculations/cp2k/o2_md",
                "recipe": "nvt",
                "settings": {
                    "periodic": "none",
                    "cell_abc": [12, 12, 12],
                    "steps": 5,
                    "temperature": 300,
                    "energy_stride": 1,
                    "scf_guess": "atomic",
                    "extrapolation": "aspc",
                    "extrapolation_order": 3,
                },
            }
        )
        stage = files_root / artifact["data"]["records"][0]["stage_dir_rel"]
        inp = (stage / "job.inp").read_text(encoding="utf-8")
        assert artifact["data"]["manifest_rel"] in content

    assert "RUN_TYPE MD" in inp
    assert "ENSEMBLE NVT" in inp
    assert "STEPS 5" in inp
    assert "TEMPERATURE 300" in inp
    assert "&ENERGY" in inp
    assert "MD 1" in inp
    assert "EXTRAPOLATION ASPC" in inp
    assert "EXTRAPOLATION_ORDER 3" in inp


def test_cp2k_output_summary_parses_common_run_evidence(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        run_dir = files_root / "results" / "cp2k_case"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "job.inp").write_text("&GLOBAL\n  RUN_TYPE GEO_OPT\n&END GLOBAL\n", encoding="utf-8")
        (run_dir / "cp2k_summary.json").write_text(
            json.dumps({"completed": True, "returncode": 0, "mpi_ranks": 32, "omp_num_threads": 1}) + "\n",
            encoding="utf-8",
        )
        (run_dir / "job.out").write_text(
            "\n".join(
                [
                    " CP2K| version string: CP2K test",
                    " ENERGY| Total FORCE_EVAL ( QS ) energy (a.u.): -31.2345",
                    " SCF run converged in 6 steps",
                    " GEOMETRY OPTIMIZATION COMPLETED",
                    " PROGRAM ENDED AT",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "o2-1.ener").write_text(
            "# Step Time[fs] Kin Temp Pot Conserved UsedTime\n"
            "0 0.0 0.1 300.0 -31.0 -30.9 0.1\n"
            "2 1.0 0.2 310.0 -31.1 -30.9 0.2\n",
            encoding="utf-8",
        )
        _, artifact = cp2k_output_summary({"result_root": "results/cp2k_case"})
        payload = json.loads((files_root / artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        record = payload["records"][0]

    assert record["completed"] is True
    assert record["run_type"] == "GEO_OPT"
    assert record["energies"]["last"]["hartree"] == -31.2345
    assert record["scf"]["converged_count"] == 1
    assert record["optimization"]["completed"] is True
    assert record["energy_files"][0]["temperature_drift"] == 10.0


def test_lammps_forcefield_validate_and_prepare_minimize(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_o2(files_root)
        _, ff_artifact = lammps_forcefield_validate(
            {
                "forcefield_card": {
                    "units": "metal",
                    "atom_style": "atomic",
                    "pair_style": "lj/cut 8.5",
                    "pair_coeff": ["* * 0.0103 3.0"],
                }
            }
        )
        prep_content, prep_artifact = lammps_prepare(
            {
                "input_path": "structures/O2.xyz",
                "output_root": "calculations/lammps/o2_min",
                "recipe": "minimize",
                "forcefield_card_path": ff_artifact["data"]["output_path_rel"],
            }
        )
        stage = files_root / prep_artifact["data"]["records"][0]["stage_dir_rel"]
        script = (stage / "in.lammps").read_text(encoding="utf-8")
        data = (stage / "system.data").read_text(encoding="utf-8")
        manifest = json.loads((stage / "manifest.json").read_text(encoding="utf-8"))
        assert prep_artifact["data"]["manifest_rel"] in prep_content

    assert "pair_style lj/cut 8.5" in script
    assert "minimize 1e-06 1e-08 1000 10000" in script
    assert "2 atoms" in data
    assert manifest["recipe"] == "minimize"
    assert "https://docs.lammps.org/minimize.html" in manifest["references"]


def test_lammps_prepare_nvt_with_restart_and_observables(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        _write_o2(files_root)
        _, ff_artifact = lammps_forcefield_validate(
            {
                "forcefield_card": {
                    "units": "metal",
                    "atom_style": "atomic",
                    "pair_style": "lj/cut 8.5",
                    "pair_coeff": ["* * 0.0103 3.0"],
                }
            }
        )
        _, prep_artifact = lammps_prepare(
            {
                "input_path": "structures/O2.xyz",
                "output_root": "calculations/lammps/o2_nvt",
                "recipe": "nvt",
                "forcefield_card_path": ff_artifact["data"]["output_path_rel"],
                "settings": {"steps": 10, "thermo": 2, "dump_stride": 2, "rdf": True, "msd": True},
            }
        )
        stage = files_root / prep_artifact["data"]["records"][0]["stage_dir_rel"]
        script = (stage / "in.lammps").read_text(encoding="utf-8")

    assert "fix int all nvt temp" in script
    assert "run 10" in script
    assert "write_restart restart.final" in script
    assert "compute rdf_all all rdf" in script
    assert "compute msd_all all msd" in script


def test_lammps_log_and_trajectory_summaries(tmp_path: Path) -> None:
    project = _project_space(tmp_path)
    with workspace_scope(project):
        files_root = project / "files"
        run_dir = files_root / "results" / "lammps_case"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "log.lammps").write_text(
            "\n".join(
                [
                    "Step Temp PotEng TotEng Press Vol",
                    "0 300 -1.0 -0.5 1.0 1000",
                    "10 305 -1.1 -0.6 1.1 1000",
                    "Loop time of 0.01 on 1 procs for 10 steps with 2 atoms",
                    "Minimization stats:",
                    "  Stopping criterion = energy tolerance",
                    "  Energy initial, next-to-last, final =",
                    "       -1.0 -1.05 -1.1",
                    "  Iterations, force evaluations = 4 8",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        (run_dir / "trajectory.lammpstrj").write_text(
            "\n".join(
                [
                    "ITEM: TIMESTEP",
                    "0",
                    "ITEM: NUMBER OF ATOMS",
                    "2",
                    "ITEM: BOX BOUNDS pp pp pp",
                    "0 10",
                    "0 10",
                    "0 10",
                    "ITEM: ATOMS id type x y z",
                    "1 1 0 0 0",
                    "2 1 0 0 1",
                    "ITEM: TIMESTEP",
                    "1",
                    "ITEM: NUMBER OF ATOMS",
                    "2",
                    "ITEM: BOX BOUNDS pp pp pp",
                    "0 10",
                    "0 10",
                    "0 10",
                    "ITEM: ATOMS id type x y z",
                    "1 1 0 0 0.1",
                    "2 1 0 0 1.1",
                ]
            )
            + "\n",
            encoding="utf-8",
        )
        _, log_artifact = lammps_log_summary({"result_root": "results/lammps_case"})
        traj_content, traj_artifact = md_trajectory_summary({"path": "results/lammps_case"})
        log_payload = json.loads((files_root / log_artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        traj_payload = json.loads((files_root / traj_artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))

    assert log_payload["records"][0]["thermo_rows"] == 2
    assert log_payload["records"][0]["final_thermo"]["Step"] == 10
    assert log_payload["records"][0]["thermo_drift"]["temperature"] == 5
    assert log_payload["records"][0]["minimization"]["stopping_criterion"] == "energy tolerance"
    assert traj_payload["format"] == "lammps-dump"
    assert traj_payload["nframes"] == 2
    assert traj_payload["natoms"] == 2
    assert traj_payload["final_frame_rel"].endswith("final_frame.lammpstrj")
    assert traj_payload["final_frame_rel"] in traj_content


def test_cp2k_lammps_remote_task_visibility_by_worker() -> None:
    with toolcall_context("catalog", audience="materials_worker"):
        _, artifact = get_avail_remote_task({"return_resource": False})
    material_tasks = {item["task_name"] for item in artifact["data"]["tasks"]}
    assert "cp2k_execute" in material_tasks
    assert "lammps_execute" not in material_tasks

    with toolcall_context("catalog", audience="dynamics_worker"):
        _, artifact = get_avail_remote_task({"return_resource": True})
    dynamics_items = {item["task_name"]: item for item in artifact["data"]["tasks"]}
    dynamics_tasks = set(dynamics_items)
    assert {"cp2k_execute", "lammps_execute"}.issubset(dynamics_tasks)
    assert "vasp_execute" not in dynamics_tasks
    assert not any(name.startswith("cp2k_") and name != "cp2k_execute" for name in dynamics_tasks)
    assert not any(name.startswith("lammps_") and name != "lammps_execute" for name in dynamics_tasks)
    assert dynamics_items["cp2k_execute"]["resources"]["resources"] == "cp2k_cpu"
    assert dynamics_items["lammps_execute"]["resources"]["resources"] == "lammps_cpu"
