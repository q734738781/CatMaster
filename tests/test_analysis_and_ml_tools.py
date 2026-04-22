from __future__ import annotations

import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.calculators.singlepoint import SinglePointCalculator
from ase.io import read as ase_read
from ase.io import write as ase_write
from pymatgen.core import Lattice, Structure
from pymatgen.io.vasp import Poscar

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.analysis import results_analysis
from catmaster.tools.base import workspace_scope
from catmaster.tools.machine_learning.dataset_tools import build_dataset_from_runs
from catmaster.tools.machine_learning.mace_ml import (
    calculate_al_candidates,
    mace_evaluate,
    mace_train,
)


def _write_poscar(path: Path, structure: Structure) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    Poscar(structure).write_file(str(path))


def test_analyze_vasp_results_with_fake_vasprun(monkeypatch, tmp_path: Path) -> None:
    class _FakeVasprun:
        def __init__(self, path: str, parse_projected_eigen: bool = False):
            _ = parse_projected_eigen
            self.path = path
            self.converged_electronic = True
            self.converged_ionic = not path.endswith("run_b/vasprun.xml")
            self.final_energy = -10.0 if "run_a" in path else -9.5
            self.eigenvalue_band_properties = (0.8, 0.0, 0.0)
            self.ionic_steps = [{"forces": [[0.01, 0.0, 0.0], [0.0, 0.02, 0.0]]}]

    monkeypatch.setattr(results_analysis, "Vasprun", _FakeVasprun)
    monkeypatch.setattr(results_analysis, "Oszicar", lambda path: SimpleNamespace(final_energy=-9.8))
    monkeypatch.setattr(results_analysis, "Outcar", lambda path: SimpleNamespace(total_mag=1.2))

    with workspace_scope(tmp_path):
        root = tmp_path / "files" / "runs"
        for name in ("run_a", "run_b"):
            run_dir = root / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "vasprun.xml").write_text("<xml />", encoding="utf-8")
            (run_dir / "CONTCAR").write_text("dummy\n", encoding="utf-8")

        _, artifact = results_analysis.analyze_vasp_results({"result_root": "runs"})

    data = artifact["data"]
    summary = json.loads((tmp_path / "files" / data["summary_json_rel"]).read_text(encoding="utf-8"))
    assert summary["runs_analyzed"] == 2
    assert data["failed_or_incomplete"] == 1


def test_analyze_vasp_results_excludes_frozen_atom_forces(monkeypatch, tmp_path: Path) -> None:
    class _FakeVasprun:
        def __init__(self, path: str, parse_projected_eigen: bool = False):
            _ = (path, parse_projected_eigen)
            self.converged_electronic = True
            self.converged_ionic = True
            self.final_energy = -10.0
            self.eigenvalue_band_properties = (0.8, 0.0, 0.0)
            self.ionic_steps = [{"forces": [[0.0, 0.0, 0.55], [0.02, 0.0, 0.0]]}]

    class _FakePoscar:
        @staticmethod
        def from_file(path: str):
            _ = path
            return SimpleNamespace(selective_dynamics=[[False, False, False], [True, True, True]])

    monkeypatch.setattr(results_analysis, "Vasprun", _FakeVasprun)
    monkeypatch.setattr(results_analysis, "Poscar", _FakePoscar)
    monkeypatch.setattr(results_analysis, "Oszicar", lambda path: SimpleNamespace(final_energy=-10.0))
    monkeypatch.setattr(results_analysis, "Outcar", lambda path: SimpleNamespace(total_mag=0.0))

    with workspace_scope(tmp_path):
        run_dir = tmp_path / "files" / "run"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "vasprun.xml").write_text("<xml />", encoding="utf-8")
        (run_dir / "CONTCAR").write_text("dummy\n", encoding="utf-8")
        _, artifact = results_analysis.analyze_vasp_results({"result_root": "run"})

    data = artifact["data"]
    summary = json.loads((tmp_path / "files" / data["summary_json_rel"]).read_text(encoding="utf-8"))
    assert summary["records"][0]["max_force_ev_per_a"] == pytest.approx(0.02)


def test_analyze_vasp_neb_and_trajectory(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        results_analysis,
        "_scan_neb_images",
        lambda result_dir: ([
            {"image": 0, "image_dir_rel": "neb/00", "energy_ev": -20.0},
            {"image": 1, "image_dir_rel": "neb/01", "energy_ev": -19.2},
            {"image": 2, "image_dir_rel": "neb/02", "energy_ev": -19.8},
        ], []),
    )
    with workspace_scope(tmp_path):
        neb_dir = tmp_path / "files" / "neb"
        neb_dir.mkdir(parents=True, exist_ok=True)
        _, neb_artifact = results_analysis.analyze_vasp_neb_results({"result_dir": "neb"})
        assert neb_artifact["data"]["ts_image"] == 1

        traj_dir = tmp_path / "files" / "md"
        traj_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for step in range(4):
            atoms = Atoms("Li2", positions=[[0, 0, 0], [1.0 + 0.1 * step, 0, 0]], cell=[5, 5, 5], pbc=True)
            frames.append(atoms)
        ase_write(str(traj_dir / "traj.extxyz"), frames, format="extxyz")
        (traj_dir / "OSZICAR").write_text(
            " 1 T= 300.0 E0= -10.0 F= -10.1\n 2 T= 305.0 E0= -9.9 F= -10.0\n",
            encoding="utf-8",
        )
        _, traj_artifact = results_analysis.analyze_trajectory(
            {"path": "md", "timestep_fs": 2.0, "species": "Li", "diffusion_dimension": "xy"}
        )
        assert traj_artifact["data"]["nframes"] == 4
        traj_summary = json.loads((tmp_path / "files" / traj_artifact["data"]["summary_json_rel"]).read_text(encoding="utf-8"))
        assert traj_summary["rdf_species"] == "Li"
        assert traj_summary["diffusion_dimension"] == "xy"


def test_build_dataset_from_runs_and_calculate_al_candidates(monkeypatch, tmp_path: Path) -> None:
    import catmaster.tools.machine_learning.dataset_tools as dataset_tools
    frames = []
    for energy, force in [(-1.0, 0.0), (-0.9, 0.1)]:
        atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[3.5, 3.5, 3.5], pbc=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=[[force, 0.0, 0.0]],
            stress=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        frames.append(atoms)

    monkeypatch.setattr(dataset_tools, "read_vasp_xml", lambda path, index=slice(None): list(frames))
    monkeypatch.setattr(dataset_tools, "_parse_vasp_step_metadata", lambda path: (120, [20, 30], [None, None], ""))

    with workspace_scope(tmp_path):
        run_dir = tmp_path / "files" / "runs" / "job1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "vasprun.xml").write_text("<xml />", encoding="utf-8")

        _, artifact = build_dataset_from_runs(
            {
                "result_root": "runs",
                "output_dir": "dataset",
                "head_label": "omat_pbe",
                "config_type": "dft",
            }
        )
        data = artifact["data"]
        assert data["frames_written"] == 2
        dataset_dir = tmp_path / "files" / "dataset"
        dataset_frames = ase_read(str(dataset_dir / "dataset.extxyz"), index=":")
        assert dataset_frames[0].info["head"] == "omat_pbe"
        assert dataset_frames[0].info["config_type"] == "dft"
        assert dataset_frames[0].info["step_electronic_converged_guess"] is True

        candidates_dir = tmp_path / "files" / "candidates"
        for idx, shift in enumerate([0.0, 0.1, 0.2]):
            atoms = Structure(Lattice.cubic(3.0 + shift), ["Li"], [[0.0, 0.0, 0.0]])
            _write_poscar(candidates_dir / f"cand_{idx}.vasp", atoms)
        _, al_artifact = calculate_al_candidates(
            {
                "structure_dir": "candidates",
                "output_dir": "al_out",
                "selection_size": 2,
            }
        )
        assert al_artifact["data"]["selected_count"] == 2
        assert dataset_dir.joinpath("dataset.extxyz").is_file()


def test_build_dataset_from_runs_filters_unconverged_and_groups_splits(monkeypatch, tmp_path: Path) -> None:
    import catmaster.tools.machine_learning.dataset_tools as dataset_tools

    def _make_frame(energy: float) -> Atoms:
        atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[3.5, 3.5, 3.5], pbc=True)
        atoms.calc = SinglePointCalculator(
            atoms,
            energy=energy,
            forces=[[0.0, 0.0, 0.0]],
            stress=[1.0, 1.0, 1.0, 0.0, 0.0, 0.0],
        )
        return atoms

    run_frames = {
        "job1": [_make_frame(-1.0), _make_frame(-0.9)],
        "job2": [_make_frame(-0.8), _make_frame(-0.7)],
    }

    def _fake_read_vasp_xml(path, index=slice(None)):
        _ = index
        return list(run_frames[Path(path).parent.name])

    def _fake_parse_metadata(path):
        name = Path(path).parent.name
        if name == "job1":
            return 100, [20, 100], [None, None], ""
        return 100, [10, 15], [None, None], ""

    monkeypatch.setattr(dataset_tools, "read_vasp_xml", _fake_read_vasp_xml)
    monkeypatch.setattr(dataset_tools, "_parse_vasp_step_metadata", _fake_parse_metadata)

    with workspace_scope(tmp_path):
        for name in ("job1", "job2"):
            run_dir = tmp_path / "files" / "runs" / name
            run_dir.mkdir(parents=True, exist_ok=True)
            (run_dir / "vasprun.xml").write_text("<xml />", encoding="utf-8")

        _, artifact = build_dataset_from_runs(
            {
                "result_root": "runs",
                "output_dir": "dataset",
                "train_fraction": 0.5,
                "valid_fraction": 0.5,
                "test_fraction": 0.0,
                "shuffle": False,
                "split_unit": "source_run",
                "require_converged": True,
            }
        )

        dataset_dir = tmp_path / "files" / "dataset"
        train_frames = ase_read(str(dataset_dir / "train.extxyz"), index=":")
        valid_frames = ase_read(str(dataset_dir / "valid.extxyz"), index=":")
        summary = json.loads((dataset_dir / "dataset_summary.json").read_text(encoding="utf-8"))

    assert artifact["data"]["frames_written"] == 3
    assert summary["filtered_frames"] == 1
    assert summary["split_unit"] == "source_run"
    assert len(train_frames) == 1
    assert len(valid_frames) == 2
    assert {frame.info["source_run_rel"] for frame in train_frames} == {"runs/job1"}
    assert {frame.info["source_run_rel"] for frame in valid_frames} == {"runs/job2"}


def test_build_dataset_from_runs_skips_alignment_mismatch(monkeypatch, tmp_path: Path) -> None:
    import catmaster.tools.machine_learning.dataset_tools as dataset_tools

    atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[3.5, 3.5, 3.5], pbc=True)
    calc = SinglePointCalculator(atoms, energy=-1.0, forces=[[0.0, 0.0, 0.0]], stress=[1, 1, 1, 0, 0, 0])
    calc.results["free_energy"] = -1.0
    atoms.calc = calc

    monkeypatch.setattr(dataset_tools, "read_vasp_xml", lambda path, index=slice(None): [atoms.copy()])
    monkeypatch.setattr(
        dataset_tools,
        "_parse_vasp_step_metadata",
        lambda path: (120, [20], [-0.5], ""),
    )

    with workspace_scope(tmp_path):
        run_dir = tmp_path / "files" / "runs" / "job1"
        run_dir.mkdir(parents=True, exist_ok=True)
        (run_dir / "vasprun.xml").write_text("<xml />", encoding="utf-8")
        with pytest.raises(CatMasterToolExecutionError, match="No frames were extracted"):
            build_dataset_from_runs({"result_root": "runs", "output_dir": "dataset"})


def test_mace_train_and_evaluate_stage_and_collect(monkeypatch, tmp_path: Path) -> None:
    import catmaster.tools.machine_learning.mace_ml as mace_ml

    monkeypatch.setattr(mace_ml, "_resolve_machine_for_resources", lambda _: "dummy_machine")

    def _fake_dispatch(req):
        stage_root = Path(req.local_root) / req.work_base
        train_params = stage_root / "params" / "train_params.json"
        if train_params.exists():
            payload = json.loads(train_params.read_text(encoding="utf-8"))
            assert payload["foundation_head"] == "omat_pbe"
            assert payload["e0s"] == "assets/e0s/e0s/e0s.json"
            assert payload["multiheads_finetuning"] is True
            assert payload["pt_train_file"] == "assets/replay/replay/replay.pt"
            assert payload["forces_weight"] == 10.0
            assert payload["stress_weight"] == 1.0
            assert payload["restart_latest"] is True
            assert payload["weight_decay"] == pytest.approx(1.0e-6)
            assert payload["scheduler"] == "ReduceLROnPlateau"
            assert payload["patience"] == 5
            assert payload["eval_interval"] == 2
            assert payload["valid_batch_size"] == 8
            assert payload["save_all_checkpoints"] is True
            assert payload["keep_checkpoints"] is True
            assert payload["foundation_model"] == "assets/models/finetune-model/checkpoints/best.model"
            assert payload["cli_args"]["statistics_file"] == "assets/cli_args/stats/train.json"
            assert payload["cli_args"]["energy_key"] == "REF_energy"
            assert payload["cli_args"]["stage_two"] is False
        eval_params = stage_root / "params" / "eval_params.json"
        if eval_params.exists():
            payload = json.loads(eval_params.read_text(encoding="utf-8"))
            assert payload["device"] == "cpu"
        output = stage_root / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "batch_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str((Path(req.local_root) / "_submission").resolve()),
            work_base=req.work_base,
            duration_s=0.01,
        )

    monkeypatch.setattr(mace_ml, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        dataset_dir = tmp_path / "files" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        for name in ("train.extxyz", "valid.extxyz", "test.extxyz"):
            (dataset_dir / name).write_text("", encoding="utf-8")
        model_dir = tmp_path / "files" / "finetune-model"
        (model_dir / "checkpoints").mkdir(parents=True, exist_ok=True)
        (model_dir / "checkpoints" / "best.model").write_text("weights", encoding="utf-8")
        e0_dir = tmp_path / "files" / "e0s"
        e0_dir.mkdir(parents=True, exist_ok=True)
        (e0_dir / "e0s.json").write_text("{}", encoding="utf-8")
        replay_dir = tmp_path / "files" / "replay"
        replay_dir.mkdir(parents=True, exist_ok=True)
        (replay_dir / "replay.pt").write_text("replay", encoding="utf-8")
        stats_dir = tmp_path / "files" / "stats"
        stats_dir.mkdir(parents=True, exist_ok=True)
        (stats_dir / "train.json").write_text("{}", encoding="utf-8")

        _, train_artifact = mace_train(
            {
                "dataset_dir": "dataset",
                "output_root": "train_out",
                "foundation_model": "finetune-model",
                "foundation_head": "omat_pbe",
                "e0s": "e0s/e0s.json",
                "pt_train_file": "replay/replay.pt",
                "weight_decay": 1.0e-6,
                "scheduler": "ReduceLROnPlateau",
                "patience": 5,
                "eval_interval": 2,
                "valid_batch_size": 8,
                "save_all_checkpoints": True,
                "keep_checkpoints": True,
                "atomic_numbers": [1, 8, 26],
                "cli_args": {
                    "statistics_file": "stats/train.json",
                    "energy_key": "REF_energy",
                    "stage_two": False,
                },
            }
        )
        assert train_artifact["data"]["batch_summary_rel"]

        _, eval_artifact = mace_evaluate(
            {"dataset_dir": "dataset", "output_root": "eval_out", "model": "mh-1", "device": "cpu"}
        )
        assert eval_artifact["data"]["batch_summary_rel"]


def test_remote_mace_train_resolves_staged_paths(tmp_path: Path, monkeypatch) -> None:
    from catmaster.remote.gpu import mace_train as remote_train

    stage_root = tmp_path / "stage"
    dataset_root = stage_root / "dataset"
    output_root = stage_root / "output"
    params_root = stage_root / "params"
    assets_root = stage_root / "assets" / "models"
    e0_root = stage_root / "assets" / "e0s"
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    params_root.mkdir(parents=True, exist_ok=True)
    assets_root.mkdir(parents=True, exist_ok=True)
    e0_root.mkdir(parents=True, exist_ok=True)
    (dataset_root / "train.extxyz").write_text("", encoding="utf-8")
    (assets_root / "best.model").write_text("x", encoding="utf-8")
    (e0_root / "e0s.json").write_text("{}", encoding="utf-8")
    params_path = params_root / "train_params.json"
    params_path.write_text(
        json.dumps(
            {
                "model_name": "x",
                "train_file": "train.extxyz",
                "valid_file": None,
                "test_file": None,
                "foundation_model": "assets/models/best.model",
                "foundation_head": "omat_pbe",
                "e0s": "assets/e0s/e0s.json",
                "multiheads_finetuning": False,
                "pt_train_file": "omat",
                "num_samples_pt": 0,
                "filter_type_pt": "combinations",
                "subselect_pt": "fps",
                "weight_pt": 1.0,
                "atomic_numbers": [],
                "compute_stress": True,
                "energy_weight": 1.0,
                "forces_weight": 1.0,
                "stress_weight": 1.0,
                "max_num_epochs": 1,
                "batch_size": 1,
                "learning_rate": 1e-4,
                "weight_decay": 1.0e-6,
                "scheduler": "ReduceLROnPlateau",
                "patience": 5,
                "eval_interval": 2,
                "valid_batch_size": 8,
                "save_all_checkpoints": True,
                "keep_checkpoints": True,
                "default_dtype": "float32",
                "device": "cpu",
                "seed": 1,
                "restart_latest": False,
                "cli_args": {"statistics_file": "assets/e0s/e0s.json", "stage_two": False},
            }
        ),
        encoding="utf-8",
    )

    seen: dict[str, object] = {}

    class _Done:
        returncode = 0

    def _fake_run(cmd, check, cwd):
        seen["cmd"] = cmd
        seen["cwd"] = cwd
        return _Done()

    monkeypatch.setattr(remote_train.subprocess, "run", _fake_run)
    monkeypatch.chdir(stage_root)
    remote_train.run_training(Path("dataset"), Path("output"), Path("params/train_params.json"))
    cmd = list(seen["cmd"])
    assert seen["cwd"] == str(output_root)
    foundation_idx = cmd.index("--foundation_model") + 1
    e0_idx = cmd.index("--E0s") + 1
    statistics_idx = cmd.index("--statistics_file") + 1
    assert cmd[foundation_idx] == str((assets_root / "best.model").resolve())
    assert cmd[e0_idx] == str((e0_root / "e0s.json").resolve())
    assert cmd[statistics_idx] == str((e0_root / "e0s.json").resolve())
    assert "--weight_decay" in cmd
    assert "--scheduler" in cmd
    assert "--patience" in cmd
    assert "--eval_interval" in cmd
    assert "--valid_batch_size" in cmd
    assert "--save_all_checkpoints" in cmd
    assert "--keep_checkpoints" in cmd


def test_mace_train_auto_infers_atomic_numbers_for_replay(monkeypatch, tmp_path: Path) -> None:
    import catmaster.tools.machine_learning.mace_ml as mace_ml

    monkeypatch.setattr(mace_ml, "_resolve_machine_for_resources", lambda _: "dummy_machine")

    seen: dict[str, object] = {}

    def _fake_dispatch(req):
        stage_root = Path(req.local_root) / req.work_base
        payload = json.loads((stage_root / "params" / "train_params.json").read_text(encoding="utf-8"))
        seen["atomic_numbers"] = payload["atomic_numbers"]
        output = stage_root / "output"
        output.mkdir(parents=True, exist_ok=True)
        (output / "batch_summary.json").write_text(json.dumps({"ok": True}), encoding="utf-8")
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str((Path(req.local_root) / "_submission").resolve()),
            work_base=req.work_base,
            duration_s=0.01,
        )

    monkeypatch.setattr(mace_ml, "dispatch_submission", _fake_dispatch)

    with workspace_scope(tmp_path):
        dataset_dir = tmp_path / "files" / "dataset"
        dataset_dir.mkdir(parents=True, exist_ok=True)
        frames = []
        for symbols in ("CH", "FeO"):
            atoms = Atoms(symbols)
            atoms.positions = np.zeros((len(atoms), 3))
            atoms.calc = SinglePointCalculator(
                atoms,
                energy=-1.0,
                forces=np.zeros((len(atoms), 3)),
                stress=[0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            )
            frames.append(atoms)
        ase_write(str(dataset_dir / "train.extxyz"), frames, format="extxyz")

        _, artifact = mace_train(
            {
                "dataset_dir": "dataset",
                "output_root": "train_out",
                "valid_file": None,
                "test_file": None,
                "foundation_model": "mh-1",
                "foundation_head": "omat_pbe",
                "atomic_numbers": [],
            }
        )

    assert artifact["data"]["atomic_numbers"] == [1, 6, 8, 26]
    assert seen["atomic_numbers"] == [1, 6, 8, 26]


def test_remote_mace_evaluate_reports_stress_metrics(monkeypatch, tmp_path: Path) -> None:
    from catmaster.remote.gpu import mace_evaluate as remote_eval

    dataset_root = tmp_path / "dataset"
    output_root = tmp_path / "output"
    dataset_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)

    atoms = Atoms("Li", positions=[[0.0, 0.0, 0.0]], cell=[3.0, 3.0, 3.0], pbc=True)
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=-1.0,
        forces=[[0.1, 0.0, 0.0]],
        stress=[1.0, 2.0, 3.0, 0.0, 0.0, 0.0],
    )
    ase_write(str(dataset_root / "test.extxyz"), [atoms], format="extxyz")

    class _FakeCalc:
        pass

    monkeypatch.setattr(remote_eval, "_load_calculator", lambda model, head, default_dtype, device: _FakeCalc())

    original_get_potential_energy = Atoms.get_potential_energy
    original_get_forces = Atoms.get_forces
    original_get_stress = Atoms.get_stress

    def _fake_get_potential_energy(self, *args, **kwargs):
        if isinstance(getattr(self, "calc", None), _FakeCalc):
            return -0.9
        return original_get_potential_energy(self, *args, **kwargs)

    def _fake_get_forces(self, *args, **kwargs):
        if isinstance(getattr(self, "calc", None), _FakeCalc):
            return np.array([[0.2, 0.0, 0.0]])
        return original_get_forces(self, *args, **kwargs)

    def _fake_get_stress(self, *args, **kwargs):
        if isinstance(getattr(self, "calc", None), _FakeCalc):
            return np.array([1.5, 2.5, 3.5, 0.0, 0.0, 0.0])
        return original_get_stress(self, *args, **kwargs)

    monkeypatch.setattr(Atoms, "get_potential_energy", _fake_get_potential_energy)
    monkeypatch.setattr(Atoms, "get_forces", _fake_get_forces)
    monkeypatch.setattr(Atoms, "get_stress", _fake_get_stress)

    params_path = tmp_path / "params.json"
    params_path.write_text(
        json.dumps(
            {
                "dataset_file": "test.extxyz",
                "model": "dummy.model",
                "head": "omat_pbe",
                "default_dtype": "float32",
                "device": "cpu",
            }
        ),
        encoding="utf-8",
    )

    summary = remote_eval.run_evaluation(dataset_root, output_root, params_path)
    metrics = summary["metrics"]
    assert metrics["stress_rmse_eVA3"] is not None
    assert metrics["stress_mae_eVA3"] is not None


def test_analyze_vasp_neb_results_rejects_partial_profiles(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(
        results_analysis,
        "_scan_neb_images",
        lambda result_dir: ([{"image": 0, "image_dir_rel": "neb/00", "energy_ev": -20.0}], ["missing image directories: 01"]),
    )
    with workspace_scope(tmp_path):
        neb_dir = tmp_path / "files" / "neb"
        neb_dir.mkdir(parents=True, exist_ok=True)
        with pytest.raises(CatMasterToolExecutionError, match="Incomplete NEB image energies"):
            results_analysis.analyze_vasp_neb_results({"result_dir": "neb"})


def test_scan_neb_images_parses_outcar_toten_only(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        neb_dir = tmp_path / "files" / "neb"
        for idx, energy in enumerate((-20.0, -19.2, -19.8)):
            image_dir = neb_dir / f"{idx:02d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "OUTCAR").write_text(
                "\n".join(
                    [
                        " some header",
                        f"  free energy    TOTEN  =      {energy - 0.1: .8f} eV",
                        f"  free  energy   TOTEN  =      {energy: .8f} eV",
                    ]
                )
                + "\n",
                encoding="utf-8",
            )
            (image_dir / "OSZICAR").write_text(" 1 F= 999.0 E0= 999.0\n", encoding="utf-8")
        records, issues = results_analysis._scan_neb_images(neb_dir)
    assert issues == []
    assert [record["image"] for record in records] == [0, 1, 2]
    assert [record["energy_ev"] for record in records] == [-20.0, -19.2, -19.8]


def test_analyze_vasp_neb_results_requires_endpoint_outcar(tmp_path: Path) -> None:
    with workspace_scope(tmp_path):
        neb_dir = tmp_path / "files" / "neb"
        for idx, energy in ((1, -19.2), (2, -19.8), (3, -19.5)):
            image_dir = neb_dir / f"{idx:02d}"
            image_dir.mkdir(parents=True, exist_ok=True)
            (image_dir / "OUTCAR").write_text(
                f"  free  energy   TOTEN  =      {energy: .8f} eV\n",
                encoding="utf-8",
            )
        for idx in (0, 4):
            (neb_dir / f"{idx:02d}").mkdir(parents=True, exist_ok=True)
        with pytest.raises(CatMasterToolExecutionError) as exc_info:
            results_analysis.analyze_vasp_neb_results({"result_dir": "neb"})
        message = str(exc_info.value)
        assert "image 00: no energy parsed from OUTCAR" in message
        assert "VASP NEB endpoint images do not produce their own OUTCAR energies" in message
        assert "Copy the original relax OUTCAR files into 00/OUTCAR and 04/OUTCAR" in message
