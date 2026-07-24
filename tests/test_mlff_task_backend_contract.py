from __future__ import annotations

import importlib
import hashlib
import json
from pathlib import Path
import subprocess
import sys
from types import ModuleType
from types import SimpleNamespace

import numpy as np
import pytest
from ase import Atoms
from ase.build import bulk, molecule
from ase.calculators.calculator import Calculator, all_changes
from ase.calculators.singlepoint import SinglePointCalculator
from ase.constraints import FixAtoms, FixCartesian, FixScaled
from ase.io import read, write

from catmaster.runtime.tool_runtime import toolcall_context
from catmaster.tools.base import workspace_scope
from catmaster.tools.execution.mlff_specs import resolve_mlff_template
from catmaster.tools.execution.mlff_stage import materialize_mlff_run_config
from catmaster.tools.execution.remote_submission import (
    GetRemoteTaskSpecInput,
    RemoteSubmissionInput,
    get_remote_task_spec,
    remote_submission,
    remote_submission_batch,
)
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.registry import ToolRegistry


remote_submission_module = importlib.import_module("catmaster.tools.execution.remote_submission")
mlff_common = importlib.import_module("catmaster.remote.mlff.mlff_common")
mlff_dynamics = importlib.import_module("catmaster.remote.mlff.mlff_dynamics")
mlff_md = importlib.import_module("catmaster.remote.mlff.mlff_md")
mlff_neb = importlib.import_module("catmaster.remote.mlff.mlff_neb")


def test_registry_cutover_exposes_only_operation_named_mlff_tasks() -> None:
    registry = TaskRegistry()
    assert {name for name, cfg in registry.tasks.items() if cfg.operation} == {
        "mlff_sp",
        "mlff_relax",
        "mlff_md",
        "mlff_neb",
    }
    assert not {
        "mace_sp_dir",
        "mace_relax_dir",
        "uma_sp_dir",
        "uma_relax_dir",
        "mace_md_dir",
        "mace_neb_dir",
        "mace_train_dir",
        "mace_eval_dir",
    } & set(registry.tasks)
    assert {"mace_train", "mace_eval"}.issubset(registry.tasks)


@pytest.mark.parametrize(
    "task_name",
    [
        "mace_sp_dir",
        "mace_relax_dir",
        "uma_sp_dir",
        "uma_relax_dir",
        "mace_md_dir",
        "mace_neb_dir",
        "mace_train_dir",
        "mace_eval_dir",
    ],
)
def test_removed_provider_named_task_specs_are_rejected(task_name: str) -> None:
    with toolcall_context("spec", audience="materials_worker"):
        with pytest.raises(KeyError, match="not found in task configs"):
            get_remote_task_spec({"task_name": task_name})


def test_public_mlff_environment_templates_are_shell_syntax_valid() -> None:
    root = Path(__file__).resolve().parents[1] / "configs" / "dpdispatcher" / "env_templates"
    expected = {
        "catmaster_env_proxy.sh",
        "catmaster_env_mace.sh",
        "catmaster_env_uma.sh",
        "catmaster_env_mattersim.sh",
        "catmaster_env_orb.sh",
    }
    scripts = {path.name: path for path in root.glob("*.sh")}
    assert set(scripts) == expected
    for path in scripts.values():
        subprocess.run(["bash", "-n", str(path)], check=True, capture_output=True, text=True)


def test_public_gpu_resources_source_network_environment_before_provider() -> None:
    import yaml

    path = Path(__file__).resolve().parents[1] / "configs" / "dpdispatcher" / "resources_template.yaml"
    resources = yaml.safe_load(path.read_text(encoding="utf-8"))
    for resource_name in ("general_gpu", "mace_gpu", "uma_gpu", "mattersim_gpu", "orb_gpu"):
        source_list = resources[resource_name]["source_list"]
        assert source_list[0] == "<REMOTE_GPU_NETWORK_ENV_SCRIPT>"


@pytest.mark.parametrize("task_name", ["mlff_sp", "mlff_relax", "mlff_md", "mlff_neb"])
@pytest.mark.parametrize("backend", ["mace", "fairchem_uma", "mattersim", "orb_v3"])
def test_every_enabled_backend_supports_every_managed_mlff_operation(task_name: str, backend: str) -> None:
    audience = "dynamics_worker" if task_name == "mlff_md" else "materials_worker"
    resolved = resolve_mlff_template(task_name, {"backend": backend}, audience=audience)
    assert resolved["resolved_backend"] == backend
    assert set(resolved["available_backends"]) == {"mace", "fairchem_uma", "mattersim", "orb_v3"}


def test_mace_registered_models_expose_strict_model_specific_heads() -> None:
    mh1 = resolve_mlff_template(
        "mlff_relax",
        {"backend": "mace", "backend_config": {"model": "mh-1", "head": "omol"}},
        audience="materials_worker",
    )
    assert mh1["enabled_models"] == ["mh-1", "omol-0"]
    assert mh1["model_capabilities"]["mh-1"] == {
        "provider_model": "mh-1",
        "tasks": [],
        "default_task": "",
        "supports_charge_spin": False,
        "loader": "mace_mp",
        "heads": [
            "matpes_r2scan",
            "mp_pbe_refit_add",
            "spice_wB97M",
            "oc20_usemppbe",
            "omol",
            "omat_pbe",
        ],
        "default_head": "omat_pbe",
    }
    mh1_schema = mh1["template_schema"]["properties"]["backend_config"]["properties"]
    assert mh1_schema["model"]["enum"] == ["", "mh-1", "omol-0"]
    assert mh1_schema["head"]["enum"] == mh1["model_capabilities"]["mh-1"]["heads"]
    assert "defaults" not in mh1_schema
    assert "items" not in mh1_schema

    omol = resolve_mlff_template(
        "mlff_relax",
        {"backend": "mace", "backend_config": {"model": "omol-0"}},
        audience="materials_worker",
    )
    omol_config = omol["normalized_template_overrides"]["backend_config"]
    assert omol_config["loader"] == "mace_omol"
    assert omol_config["provider_model"] == "extra_large"
    assert omol_config["head"] == "omol"
    assert omol_config["defaults"] == {"charge": 0, "spin": 1}
    omol_schema = omol["template_schema"]["properties"]["backend_config"]["properties"]
    assert omol_schema["head"]["enum"] == ["omol"]
    assert {"defaults", "items"}.issubset(omol_schema)

    with pytest.raises(ValueError, match="Head 'not-a-head'.*Allowed"):
        resolve_mlff_template(
            "mlff_relax",
            {"backend": "mace", "backend_config": {"model": "mh-1", "head": "not-a-head"}},
            audience="materials_worker",
        )
    with pytest.raises(ValueError, match="does not support.*dispersion"):
        resolve_mlff_template(
            "mlff_relax",
            {"backend": "mace", "backend_config": {"model": "omol-0", "dispersion": True}},
            audience="materials_worker",
        )


def test_non_mace_profiles_use_exact_official_names_and_model_specific_tasks() -> None:
    uma = resolve_mlff_template(
        "mlff_relax",
        {"backend": "fairchem_uma", "backend_config": {"model": "uma-s-1p1"}},
        audience="materials_worker",
    )
    assert uma["enabled_models"] == ["uma-m-1p1", "uma-s-1p1", "uma-s-1p2"]
    assert uma["model_capabilities"]["uma-s-1p2"]["tasks"] == [
        "oc20",
        "oc22",
        "oc25",
        "omat",
        "omol",
        "odac",
        "omc",
    ]
    assert uma["model_capabilities"]["uma-s-1p1"]["tasks"] == ["oc20", "omat", "omol", "odac", "omc"]
    assert uma["model_capabilities"]["uma-m-1p1"]["provider_model"] == "uma-m-1p1"
    uma_schema = uma["template_schema"]["properties"]["backend_config"]["properties"]
    assert uma_schema["model"]["enum"] == uma["enabled_models"]
    assert uma_schema["defaults"]["properties"]["uma_task"]["enum"] == ["oc20", "omat", "omol", "odac", "omc"]
    assert uma_schema["items"]["additionalProperties"]["properties"]["uma_task"]["enum"] == [
        "oc20",
        "omat",
        "omol",
        "odac",
        "omc",
    ]
    with pytest.raises(ValueError, match="UMA task 'oc22'.*uma-s-1p1.*Allowed"):
        resolve_mlff_template(
            "mlff_relax",
            {
                "backend": "fairchem_uma",
                "backend_config": {
                    "model": "uma-s-1p1",
                    "defaults": {"uma_task": "oc22", "charge": 0, "spin": 0},
                },
            },
            audience="materials_worker",
        )
    with pytest.raises(ValueError, match="Input should be"):
        resolve_mlff_template(
            "mlff_relax",
            {
                "backend": "fairchem_uma",
                "backend_config": {
                    "model": "uma-s-1p2",
                    "defaults": {"uma_task": "auto", "charge": 0, "spin": 0},
                },
            },
            audience="materials_worker",
        )

    mattersim = resolve_mlff_template(
        "mlff_relax",
        {"backend": "mattersim"},
        audience="materials_worker",
    )
    assert mattersim["enabled_models"] == ["MatterSim-v1.0.0-1M", "MatterSim-v1.0.0-5M"]
    assert mattersim["normalized_template_overrides"]["backend_config"]["provider_model"] == (
        "MatterSim-v1.0.0-1M"
    )
    with pytest.raises(ValueError, match="not enabled"):
        resolve_mlff_template(
            "mlff_relax",
            {"backend": "mattersim", "backend_config": {"model": "mattersim-v1-1m"}},
            audience="materials_worker",
        )

    orb = resolve_mlff_template(
        "mlff_relax",
        {"backend": "orb_v3"},
        audience="materials_worker",
    )
    assert orb["enabled_models"] == [
        "orb-v3-conservative-20-omat",
        "orb-v3-conservative-inf-omat",
        "orb-v3-direct-20-omat",
        "orb-v3-direct-inf-omat",
    ]
    assert all(capability["tasks"] == ["omat"] for capability in orb["model_capabilities"].values())


def test_md_and_neb_tasks_stage_shared_operation_dependencies() -> None:
    registry = TaskRegistry()
    md_files = registry.get("mlff_md").forward_files
    neb_files = registry.get("mlff_neb").forward_files
    assert "task_script/mlff_common.py" in md_files
    assert "task_script/mlff_dynamics.py" in md_files
    assert "task_script/mace_md.py" not in md_files
    assert "task_script/mlff_common.py" in neb_files
    assert "task_script/mace_neb.py" not in neb_files


def test_md_backend_performance_controls_are_schema_visible_and_validated() -> None:
    uma = resolve_mlff_template(
        "mlff_md",
        {"backend": "fairchem_uma", "backend_config": {"inference_settings": "turbo"}},
        audience="dynamics_worker",
    )
    assert uma["normalized_template_overrides"]["backend_config"]["inference_settings"] == "turbo"

    mattersim = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mattersim",
            "backend_config": {
                "dtype": "float32",
                "compute_stress": False,
                "direct_graph": False,
                "compile": False,
            },
        },
        audience="dynamics_worker",
    )
    assert mattersim["normalized_template_overrides"]["backend_config"] == {
        "model": "MatterSim-v1.0.0-1M",
        "device": "auto",
        "dtype": "float32",
        "compute_stress": False,
        "direct_graph": False,
        "compile": False,
        "provider_model": "MatterSim-v1.0.0-1M",
    }

    with pytest.raises(ValueError, match="direct_graph/compile are disabled"):
        resolve_mlff_template(
            "mlff_md",
            {"backend": "mattersim", "backend_config": {"direct_graph": True}},
            audience="dynamics_worker",
        )

    orb = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "orb_v3",
            "backend_config": {
                "precision": "float32-high",
                "compile_mode": "on",
                "edge_method": "knn_alchemi",
                "half_supercell": "off",
            },
        },
        audience="dynamics_worker",
    )
    orb_config = orb["normalized_template_overrides"]["backend_config"]
    assert orb_config["compile_mode"] == "on"
    assert orb_config["edge_method"] == "knn_alchemi"
    assert orb_config["half_supercell"] == "off"

    with pytest.raises(ValueError, match="MatterSim NPT requires"):
        resolve_mlff_template(
            "mlff_md",
            {
                "backend": "mattersim",
                "backend_config": {"compute_stress": False},
                "task_config": {
                    "dynamics": {"ensemble": "npt"},
                    "barostat": {"type": "isotropic_mtk"},
                },
            },
            audience="dynamics_worker",
        )


def test_md_temperature_schedule_schema_and_supported_combinations() -> None:
    defaults = resolve_mlff_template("mlff_md", {"backend": "mace"}, audience="dynamics_worker")
    dynamics_schema = defaults["template_schema"]["properties"]["task_config"]["properties"]["dynamics"]
    end_temperature = dynamics_schema["properties"]["temperature_end_K"]
    assert end_temperature["type"] == "number"
    assert end_temperature["default"] == 0.0
    assert "anyOf" not in json.dumps(end_temperature)
    assert defaults["normalized_template_overrides"]["task_config"]["dynamics"]["temperature_end_K"] == 0.0
    assert any("per-step linear" in item for item in defaults["constraints"])

    langevin = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {
                "dynamics": {"temperature_K": 300.0, "temperature_end_K": 900.0, "steps": 3},
                "thermostat": {"type": "langevin"},
            },
        },
        audience="dynamics_worker",
    )
    assert langevin["normalized_template_overrides"]["task_config"]["dynamics"]["temperature_end_K"] == 900.0

    npt_berendsen = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {
                "dynamics": {
                    "ensemble": "npt",
                    "temperature_K": 300.0,
                    "temperature_end_K": 600.0,
                    "steps": 3,
                },
                "barostat": {"type": "berendsen", "compressibility_bar_inv": 1.0e-5},
            },
        },
        audience="dynamics_worker",
    )
    assert npt_berendsen["normalized_template_overrides"]["task_config"]["barostat"]["type"] == "berendsen"

    # Equal endpoints are a constant-temperature request and retain the default
    # Bussi path instead of unnecessarily requiring a schedule-capable method.
    constant_bussi = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {"dynamics": {"temperature_K": 300.0, "temperature_end_K": 300.0}},
        },
        audience="dynamics_worker",
    )
    assert constant_bussi["normalized_template_overrides"]["task_config"]["thermostat"]["type"] == "bussi"


@pytest.mark.parametrize(
    ("task_config", "message"),
    [
        (
            {"dynamics": {"temperature_K": 300.0, "temperature_end_K": 600.0}},
            "Variable-temperature NVT requires",
        ),
        (
            {
                "dynamics": {"temperature_K": 300.0, "temperature_end_K": 600.0},
                "thermostat": {"type": "nhc"},
            },
            "Variable-temperature NVT requires",
        ),
        (
            {
                "dynamics": {"ensemble": "nve", "temperature_K": 300.0, "temperature_end_K": 600.0},
            },
            "NVE does not support",
        ),
        (
            {
                "dynamics": {
                    "ensemble": "npt",
                    "temperature_K": 300.0,
                    "temperature_end_K": 600.0,
                },
                "barostat": {"type": "isotropic_mtk"},
            },
            "Variable-temperature NPT requires",
        ),
        (
            {
                "dynamics": {"temperature_K": 300.0, "temperature_end_K": 600.0, "steps": 1},
                "thermostat": {"type": "langevin"},
            },
            "steps >= 2",
        ),
    ],
)
def test_md_temperature_schedule_rejects_unsupported_integrators(
    task_config: dict[str, object],
    message: str,
) -> None:
    with pytest.raises(ValueError, match=message):
        resolve_mlff_template(
            "mlff_md",
            {"backend": "mace", "task_config": task_config},
            audience="dynamics_worker",
        )


def test_remote_task_spec_returns_temperature_schedule_validation_error() -> None:
    with toolcall_context("spec", audience="dynamics_worker"):
        content, artifact = get_remote_task_spec(
            {
                "task_name": "mlff_md",
                "template_overrides": {
                    "backend": "mace",
                    "task_config": {
                        "dynamics": {"temperature_K": 300.0, "temperature_end_K": 600.0},
                    },
                },
                "detail": "full",
            }
        )
    assert "validation=failed" in content
    assert artifact["data"]["errors"]
    assert any("Variable-temperature NVT requires" in item["message"] for item in artifact["data"]["errors"])


def test_md_temperature_schedule_uses_public_setter_before_each_step() -> None:
    class RecordingDynamics:
        def __init__(self) -> None:
            self.nsteps = 0
            self.targets: list[float] = []

        def set_temperature(self, *, temperature_K: float) -> None:
            self.targets.append(temperature_K)

        def irun(self, steps: int):
            yield False
            for _ in range(steps):
                self.nsteps += 1
                yield False

        def run(self, steps: int) -> None:  # pragma: no cover - ramp must not use this path
            raise AssertionError(f"Unexpected constant-temperature run({steps})")

    dynamics = RecordingDynamics()
    schedule = mlff_dynamics._run_dynamics(
        dynamics,
        {"temperature_K": 300.0, "temperature_end_K": 900.0, "steps": 4},
    )
    assert dynamics.targets == pytest.approx([300.0, 500.0, 700.0, 900.0])
    assert dynamics.nsteps == 4
    assert schedule == {
        "mode": "linear",
        "start_K": 300.0,
        "end_K": 900.0,
        "steps": 4,
        "update_interval_steps": 1,
        "temperature_api": "set_temperature",
    }


def test_provider_adapters_forward_supported_performance_controls(monkeypatch: pytest.MonkeyPatch) -> None:
    mace_metadata = mlff_common.MaceAdapter().provider_metadata(
        None,
        {
            "head": "omat_pbe",
            "dispersion": False,
            "default_dtype": "float32",
            "enable_cueq": True,
            "compile_mode": "",
        },
        None,
    )
    assert mace_metadata["default_dtype"] == "float32"
    assert mace_metadata["enable_cueq"] is True
    assert mace_metadata["compile_mode"] == ""

    uma_capture: dict[str, object] = {}
    fairchem = ModuleType("fairchem")
    fairchem_core = ModuleType("fairchem.core")

    class FakePretrained:
        @staticmethod
        def get_predict_unit(model, **kwargs):
            uma_capture.update({"model": model, **kwargs})
            return object()

    class FakeUmaCalculator:
        def __init__(self, predictor, task_name):
            self.predictor = predictor
            self.task_name = task_name

    fairchem_core.pretrained_mlip = FakePretrained
    fairchem_core.FAIRChemCalculator = FakeUmaCalculator
    monkeypatch.setitem(sys.modules, "fairchem", fairchem)
    monkeypatch.setitem(sys.modules, "fairchem.core", fairchem_core)
    uma_adapter = mlff_common.FairChemUmaAdapter()
    uma_atoms = Atoms("Si", cell=[5, 5, 5], pbc=True)
    uma_adapter.calculator_for(
        uma_atoms,
        {
            "model": "uma-s-1p2",
            "provider_model": "uma-s-1p2",
            "device": "cpu",
            "uma_task": "omat",
            "charge": 0,
            "spin": 0,
            "inference_settings": "turbo",
        },
    )
    assert uma_capture["inference_settings"] == "turbo"

    mattersim_capture: dict[str, object] = {}
    mattersim = ModuleType("mattersim")
    mattersim_forcefield = ModuleType("mattersim.forcefield")

    class FakeMatterSimCalculator:
        def __init__(self, **kwargs):
            mattersim_capture.update(kwargs)

    mattersim_forcefield.MatterSimCalculator = FakeMatterSimCalculator
    monkeypatch.setitem(sys.modules, "mattersim", mattersim)
    monkeypatch.setitem(sys.modules, "mattersim.forcefield", mattersim_forcefield)
    mlff_common.MatterSimAdapter().calculator_for(
        None,
        {
            "model": "MatterSim-v1.0.0-1M",
            "provider_model": "MatterSim-v1.0.0-1M",
            "device": "cpu",
            "dtype": "float32",
            "compute_stress": False,
            "direct_graph": True,
            "compile": True,
        },
    )
    assert mattersim_capture == {
        "device": "cpu",
        "dtype": "float32",
        "compute_stress": False,
        "direct_graph": True,
        "compile": True,
        "load_path": "MatterSim-v1.0.0-1M",
    }

    orb_loader_capture: dict[str, object] = {}
    orb_calculator_capture: dict[str, object] = {}
    orb_models = ModuleType("orb_models")
    orb_forcefield = ModuleType("orb_models.forcefield")
    orb_pretrained = ModuleType("orb_models.forcefield.pretrained")
    orb_inference = ModuleType("orb_models.forcefield.inference")
    orb_calculator = ModuleType("orb_models.forcefield.inference.calculator")

    def fake_loader(**kwargs):
        orb_loader_capture.update(kwargs)
        return object(), object()

    class FakeOrbCalculator:
        results: dict[str, object] = {}

        def __init__(self, model, **kwargs):
            del model
            orb_calculator_capture.update(kwargs)

    orb_pretrained.orb_v3_conservative_inf_omat = fake_loader
    orb_forcefield.pretrained = orb_pretrained
    orb_calculator.ORBCalculator = FakeOrbCalculator
    monkeypatch.setitem(sys.modules, "orb_models", orb_models)
    monkeypatch.setitem(sys.modules, "orb_models.forcefield", orb_forcefield)
    monkeypatch.setitem(sys.modules, "orb_models.forcefield.pretrained", orb_pretrained)
    monkeypatch.setitem(sys.modules, "orb_models.forcefield.inference", orb_inference)
    monkeypatch.setitem(sys.modules, "orb_models.forcefield.inference.calculator", orb_calculator)
    mlff_common.OrbV3Adapter().calculator_for(
        None,
        {
            "model": "orb-v3-conservative-inf-omat",
            "provider_model": "orb-v3-conservative-inf-omat",
            "device": "cpu",
            "precision": "float32-high",
            "compile_mode": "on",
            "edge_method": "knn_alchemi",
            "half_supercell": "off",
        },
    )
    assert orb_loader_capture["compile"] is True
    assert orb_calculator_capture["edge_method"] == "knn_alchemi"
    assert orb_calculator_capture["half_supercell"] is False


def test_mace_adapter_dispatches_registered_loaders_and_reuses_omol_model(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    calls: list[tuple[str, dict[str, object]]] = []
    mace_module = ModuleType("mace")
    calculators_module = ModuleType("mace.calculators")

    class FakeMaceCalculator:
        def __init__(self, **kwargs):
            calls.append(("checkpoint", dict(kwargs)))

    def fake_mace_mp(**kwargs):
        calls.append(("mace_mp", dict(kwargs)))
        return object()

    omol_calculator = object()

    def fake_mace_omol(**kwargs):
        calls.append(("mace_omol", dict(kwargs)))
        return omol_calculator

    calculators_module.MACECalculator = FakeMaceCalculator
    calculators_module.mace_mp = fake_mace_mp
    calculators_module.mace_omol = fake_mace_omol
    mace_module.calculators = calculators_module
    monkeypatch.setitem(sys.modules, "mace", mace_module)
    monkeypatch.setitem(sys.modules, "mace.calculators", calculators_module)

    adapter = mlff_common.MaceAdapter()
    singlet = Atoms("H2")
    triplet = Atoms("O2")
    common = {
        "model": "omol-0",
        "loader": "mace_omol",
        "provider_model": "extra_large",
        "head": "omol",
        "supports_charge_spin": True,
        "device": "cpu",
        "default_dtype": "float64",
        "dispersion": False,
        "enable_cueq": False,
        "compile_mode": "",
        "checkpoint_artifact": "",
    }
    first = adapter.calculator_for(singlet, {**common, "charge": 0, "spin": 1})
    second = adapter.calculator_for(triplet, {**common, "charge": 0, "spin": 3})
    assert first is omol_calculator
    assert second is omol_calculator
    assert singlet.info == {"charge": 0, "spin": 1}
    assert triplet.info == {"charge": 0, "spin": 3}
    assert calls == [
        (
            "mace_omol",
            {
                "model": "extra_large",
                "device": "cpu",
                "default_dtype": "float64",
            },
        )
    ]

    mh1_adapter = mlff_common.MaceAdapter()
    mh1_adapter.calculator_for(
        Atoms("H2"),
        {
            "model": "mh-1",
            "loader": "mace_mp",
            "provider_model": "mh-1",
            "head": "omol",
            "supports_charge_spin": False,
            "device": "cpu",
            "default_dtype": "float64",
            "dispersion": False,
            "enable_cueq": False,
            "compile_mode": "",
            "checkpoint_artifact": "",
        },
    )
    assert calls[-1] == (
        "mace_mp",
        {
            "device": "cpu",
            "default_dtype": "float64",
            "head": "omol",
            "model": "mh-1",
            "dispersion": False,
        },
    )


def test_remote_task_spec_returns_one_concrete_non_union_mlff_schema() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, artifact = get_remote_task_spec(
            {
                "task_name": "mlff_relax",
                "template_overrides": {"backend": "fairchem_uma"},
                "detail": "full",
            }
        )
    data = artifact["data"]
    assert data["resolved_backend"] == "fairchem_uma"
    assert data["errors"] == []
    schema = data["template_schema"]
    encoded = json.dumps(schema)
    assert "anyOf" not in encoded
    assert "oneOf" not in encoded
    assert schema["additionalProperties"] is False
    assert schema["properties"]["backend"]["const"] == "fairchem_uma"
    assert schema["properties"]["backend_config"]["additionalProperties"] is False
    assert schema["properties"]["task_config"]["additionalProperties"] is False
    assert data["enabled_models"]
    assert data["default_backend"] == "mace"
    assert data["template_defaults"]["backend"] == "mace"
    assert data["resolved_template_defaults"]["backend"] == "fairchem_uma"
    assert "registered_default_backend=mace" in content
    assert "Registered task defaults (used when template_overrides is empty):" in content
    assert "Resolved defaults for the selected backend:" in content
    assert '"backend": "fairchem_uma"' in content
    assert schema["properties"]["backend_config"]["properties"]["model"]["enum"] == data["enabled_models"]
    assert any(item["path"] == "backend_config.defaults.uma_task" for item in data["backend_fields"])
    assert any(item["path"] == "task_config.fmax" for item in data["task_fields"])
    assert "Accepted fields:" in content
    assert "backend_config.defaults.uma_task" in content
    assert "task_config.fmax" in content
    assert "Constraints:" in content
    assert any("move_mask" in item for item in data["constraints"])
    assert "Minimal template_overrides example:" in content
    assert "Concrete template JSON Schema:" in content
    assert '"additionalProperties": false' in content
    assert '"const": "fairchem_uma"' in content
    assert "resource" not in content


def test_remote_task_spec_surfaces_mace_model_capabilities_and_selected_head_enum() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, artifact = get_remote_task_spec(
            {
                "task_name": "mlff_relax",
                "template_overrides": {
                    "backend": "mace",
                    "backend_config": {"model": "omol-0"},
                },
                "detail": "full",
            }
        )
    data = artifact["data"]
    assert data["selected_model"] == "omol-0"
    assert data["model_capabilities"]["omol-0"]["loader"] == "mace_omol"
    assert "selected_model=omol-0" in content
    assert '"supports_charge_spin": true' in content
    head = data["template_schema"]["properties"]["backend_config"]["properties"]["head"]
    assert head["enum"] == ["omol"]
    assert head["default"] == "omol"


def test_remote_task_spec_preserves_selected_uma_model_schema_after_invalid_task() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, artifact = get_remote_task_spec(
            {
                "task_name": "mlff_relax",
                "template_overrides": {
                    "backend": "fairchem_uma",
                    "backend_config": {
                        "model": "uma-s-1p1",
                        "defaults": {"uma_task": "oc22", "charge": 0, "spin": 0},
                    },
                },
                "detail": "full",
            }
        )
    data = artifact["data"]
    assert data["errors"]
    assert data["selected_model"] == "uma-s-1p1"
    assert "validation=failed" in content
    task_field = data["template_schema"]["properties"]["backend_config"]["properties"]["defaults"]["properties"][
        "uma_task"
    ]
    assert task_field["enum"] == ["oc20", "omat", "omol", "odac", "omc"]
    assert "oc22" not in task_field["enum"]


def test_remote_task_spec_compact_content_keeps_field_table_without_full_schema() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, _ = get_remote_task_spec(
            {
                "task_name": "mlff_sp",
                "template_overrides": {"backend": "mattersim"},
                "detail": "compact",
            }
        )
    assert "Accepted fields:" in content
    assert "backend_config.model" in content
    assert "enabled_models=MatterSim-v1.0.0-1M, MatterSim-v1.0.0-5M" in content
    assert 'allowed=["MatterSim-v1.0.0-1M", "MatterSim-v1.0.0-5M"]' in content
    assert "Concrete template JSON Schema:" not in content


def test_backend_switch_rejects_provider_field_leak_without_dispatch() -> None:
    with toolcall_context("spec", audience="materials_worker"):
        content, artifact = get_remote_task_spec(
            {
                "task_name": "mlff_sp",
                "template_overrides": {
                    "backend": "fairchem_uma",
                    "backend_config": {"head": "omat_pbe"},
                },
            }
        )
    assert "validation=failed" in content
    assert artifact["data"]["errors"]
    assert "normalized_template_overrides" not in artifact["data"]


def test_agent_visible_spec_and_submission_schemas_are_non_nullable() -> None:
    registry = ToolRegistry()
    tools = {item["name"]: item for item in registry.as_openai_tools()}
    exported = {name: item["parameters"] for name, item in tools.items()}
    assert "get_remote_task_spec" in exported
    assert "sufficient infrastructure provenance" in tools["get_avail_remote_task"]["description"]
    assert "execution_binding.status=configured is sufficient platform preflight" in tools["get_remote_task_spec"]["description"]
    assert "their absence is not a blocker" in tools["get_avail_resources"]["description"]
    for name in ("get_remote_task_spec", "remote_submission", "remote_submission_batch"):
        properties = exported[name]["properties"]
        assert properties["template_overrides"]["type"] == "object"
        assert "anyOf" not in json.dumps(properties["template_overrides"])
    for name in ("remote_submission", "remote_submission_batch"):
        description = exported[name]["properties"]["submission_config"]["description"]
        assert "With task_name, do not pass resources or machine" in description
        assert "blocks until" in tools[name]["description"]
        assert "terminal" in tools[name]["description"]
        assert "licensed-executable metadata" in tools[name]["description"]
    parsed = GetRemoteTaskSpecInput(task_name="mlff_sp", template_overrides=None)
    assert parsed.template_overrides == {}
    submitted = RemoteSubmissionInput(
        work_dir="stage",
        task_name="mlff_sp",
        template_overrides=None,
        submission_config=None,
    )
    assert submitted.template_overrides == {}
    assert submitted.submission_config == {}


def test_materialized_uma_item_metadata_is_resolved_per_structure(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(stage / "input" / "water.xyz", molecule("H2O"))
    write(stage / "input" / "silicon.vasp", bulk("Si", "diamond", a=5.43))
    resolved = resolve_mlff_template(
        "mlff_sp",
        {
            "backend": "fairchem_uma",
            "backend_config": {
                "defaults": {"uma_task": "omat", "charge": 0, "spin": 0},
                "items": {"water.xyz": {"uma_task": "omol", "spin": 1}},
            },
        },
    )
    config_path = materialize_mlff_run_config(
        stage_dir=stage,
        task_name="mlff_sp",
        resolved=resolved,
        explicit_overrides={"backend": "fairchem_uma"},
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["items"]["water.xyz"]["uma_task"] == "omol"
    assert data["items"]["water.xyz"]["spin"] == 1
    assert data["items"]["silicon.vasp"]["uma_task"] == "omat"
    assert "resource" not in data
    assert len(data["config_digest"]) == 64


def test_materialized_mace_omol_metadata_is_resolved_per_structure(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(stage / "input" / "hydrogen.xyz", molecule("H2"))
    write(stage / "input" / "oxygen.xyz", molecule("O2"))
    resolved = resolve_mlff_template(
        "mlff_relax",
        {
            "backend": "mace",
            "backend_config": {
                "model": "omol-0",
                "defaults": {"charge": 0, "spin": 1},
                "items": {"oxygen.xyz": {"charge": 0, "spin": 3}},
            },
        },
    )
    config_path = materialize_mlff_run_config(
        stage_dir=stage,
        task_name="mlff_relax",
        resolved=resolved,
        explicit_overrides={"backend_config": {"model": "omol-0"}},
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    assert data["backend_config"]["loader"] == "mace_omol"
    assert data["backend_config"]["provider_model"] == "extra_large"
    assert data["items"]["hydrogen.xyz"]["charge"] == 0
    assert data["items"]["hydrogen.xyz"]["spin"] == 1
    assert data["items"]["oxygen.xyz"]["charge"] == 0
    assert data["items"]["oxygen.xyz"]["spin"] == 3


def test_mace_checkpoint_is_staged_under_models_and_content_hashed(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(stage / "input" / "silicon.vasp", bulk("Si", "diamond", a=5.43))
    checkpoint = stage / "models" / "trained.model"
    checkpoint.parent.mkdir(parents=True)
    checkpoint.write_bytes(b"synthetic checkpoint bytes")
    resolved = resolve_mlff_template(
        "mlff_sp",
        {
            "backend": "mace",
            "backend_config": {"checkpoint_artifact": "models/trained.model"},
        },
    )
    config_path = materialize_mlff_run_config(
        stage_dir=stage,
        task_name="mlff_sp",
        resolved=resolved,
        explicit_overrides={"backend_config": {"checkpoint_artifact": "models/trained.model"}},
    )
    data = json.loads(config_path.read_text(encoding="utf-8"))
    expected_hash = hashlib.sha256(b"synthetic checkpoint bytes").hexdigest()
    assert data["backend_config"]["checkpoint_sha256"] == expected_hash
    assert data["backend_config"]["checkpoint_size_bytes"] == len(b"synthetic checkpoint bytes")
    assert data["items"]["silicon.vasp"]["checkpoint_sha256"] == expected_hash


def test_mace_checkpoint_outside_models_is_rejected_before_dispatch(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(stage / "input" / "silicon.vasp", bulk("Si", "diamond", a=5.43))
    (stage / "trained.model").write_bytes(b"wrong location")
    resolved = resolve_mlff_template(
        "mlff_sp",
        {
            "backend": "mace",
            "backend_config": {"checkpoint_artifact": "trained.model"},
        },
    )
    with pytest.raises(ValueError, match="under models"):
        materialize_mlff_run_config(stage_dir=stage, task_name="mlff_sp", resolved=resolved)


def test_remote_submission_resolves_backend_resource_and_fixed_command(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_dispatch(req, *, register=None, config_path=None):
        del register, config_path
        dispatch_stage = Path(req.local_root) / req.work_base
        (dispatch_stage / "output").mkdir(parents=True)
        (dispatch_stage / "output" / "batch_summary.json").write_text(
            json.dumps(
                {
                    "results": [
                        {
                            "summary": {
                                "provider_version": "2.21.0",
                            }
                        }
                    ]
                }
            ),
            encoding="utf-8",
        )
        captured.update(
            {
                "resources": req.resources,
                "command": req.tasks[0].command,
                "forward_files": list(req.tasks[0].forward_files),
                "run_config": json.loads(
                    (dispatch_stage / ".catmaster" / "generated" / "run_config.json").read_text(encoding="utf-8")
                ),
            }
        )
        return SimpleNamespace(
            task_states=["finished"],
            submission_dir=str(dispatch_stage),
            work_base=req.work_base,
            duration_s=0.01,
            remote_context={"remote_context_id": "dp_mlff", "submission_hash": "hash", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_module, "dispatch_submission", fake_dispatch)
    with workspace_scope(tmp_path):
        stage = tmp_path / "files" / "stage"
        (stage / "input").mkdir(parents=True)
        write(stage / "input" / "water.xyz", molecule("H2O"))
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission(
                {
                    "work_dir": "stage",
                    "task_name": "mlff_sp",
                    "template_overrides": {
                        "backend": "fairchem_uma",
                        "backend_config": {"defaults": {"uma_task": "omol", "spin": 1}},
                    },
                }
            )
    assert artifact["data"]["resources"] == "uma_gpu"
    assert artifact["data"]["backend"] == "fairchem_uma"
    assert artifact["data"]["provider_version"] == "2.21.0"
    assert artifact["data"]["provider_versions"] == ["2.21.0"]
    assert len(artifact["data"]["config_digest"]) == 64
    assert captured["resources"] == "uma_gpu"
    assert captured["command"] == "python task_script/mlff_sp.py --run_config .catmaster/generated/run_config.json"
    assert "task_script/mlff_common.py" in captured["forward_files"]
    assert captured["run_config"]["backend"] == "fairchem_uma"


def test_remote_submission_batch_materializes_each_first_level_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    captured: dict[str, object] = {}

    def fake_dispatch(req, *, register=None, config_path=None):
        del register, config_path
        dispatch_root = Path(req.local_root) / req.work_base
        captured["task_paths"] = [task.task_work_path for task in req.tasks]
        captured["digests"] = [
            json.loads(
                (dispatch_root / task.task_work_path / ".catmaster" / "generated" / "run_config.json").read_text(
                    encoding="utf-8"
                )
            )["config_digest"]
            for task in req.tasks
        ]
        return SimpleNamespace(
            task_states=["finished", "finished"],
            submission_dir=str(dispatch_root),
            work_base=req.work_base,
            duration_s=0.01,
            remote_context={"remote_context_id": "dp_batch", "submission_hash": "hash", "receipt_rel": "receipt.json"},
        )

    monkeypatch.setattr(remote_submission_module, "dispatch_submission", fake_dispatch)
    with workspace_scope(tmp_path):
        batch = tmp_path / "files" / "batch"
        for name, atoms in (("stage_a", molecule("H2")), ("stage_b", molecule("H2O"))):
            (batch / name / "input").mkdir(parents=True)
            write(batch / name / "input" / f"{name}.xyz", atoms)
        with toolcall_context("submit", audience="materials_worker"):
            _, artifact = remote_submission_batch(
                {
                    "work_dir": "batch",
                    "task_name": "mlff_sp",
                    "template_overrides": {"backend": "mace"},
                }
            )
    assert artifact["data"]["task_count"] == 2
    assert len(artifact["data"]["config_digests"]) == 2
    assert captured["task_paths"] == ["stage_a", "stage_b"]
    assert len(set(captured["digests"])) == 2


class _HarmonicCalculator(Calculator):
    implemented_properties = ["energy", "forces", "stress"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        positions = np.asarray(atoms.positions, dtype=float)
        self.results = {
            "energy": float(0.5 * np.sum(positions**2)),
            "forces": -positions,
            "stress": np.zeros(6),
        }


class _NonFiniteCalculator(Calculator):
    implemented_properties = ["energy", "forces"]

    def calculate(self, atoms=None, properties=("energy",), system_changes=all_changes):
        super().calculate(atoms, properties, system_changes)
        self.results = {
            "energy": float("nan"),
            "forces": np.full((len(atoms), 3), np.nan),
        }


def test_common_runner_reuses_adapter_initialization_within_one_stage(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(stage / "input" / "a.xyz", molecule("H2"))
    write(stage / "input" / "b.xyz", molecule("H2O"))
    resolved = resolve_mlff_template("mlff_sp", {"backend": "mace"})
    materialize_mlff_run_config(stage_dir=stage, task_name="mlff_sp", resolved=resolved)

    instances: list[Any] = []

    class FakeAdapter:
        provider_version = "test"

        def __init__(self):
            self.calculator = _HarmonicCalculator()
            self.calls = 0
            instances.append(self)

        def calculator_for(self, atoms, config):
            del atoms, config
            self.calls += 1
            return self.calculator

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    monkeypatch.setitem(mlff_common._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    summary = mlff_common.run("sp", ".catmaster/generated/run_config.json")
    assert len(instances) == 1
    assert instances[0].calls == 2
    assert len(summary["results"]) == 2
    assert summary["errors"] == []


def test_common_relax_runner_uses_ase_force_norm_convergence(tmp_path: Path) -> None:
    atoms = Atoms("H", positions=[[0.04, 0.04, 0.04]])

    class FakeAdapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            del atoms, config
            return _HarmonicCalculator()

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    summary = mlff_common._run_relax(
        atoms=atoms,
        output_dir=tmp_path,
        config={
            "backend": "mace",
            "operation": "relax",
            "config_digest": "test",
            "task_config": {"fmax": 0.05, "steps": 0, "optimizer": "FIRE", "relax_cell": False},
        },
        item_config={"model": "test", "device": "cpu"},
        adapter=FakeAdapter(),
    )

    assert summary["max_force_eVA"] == pytest.approx(np.sqrt(3.0) * 0.04)
    assert "max_force_abs_eVA" not in summary
    assert "max_force_norm_eVA" not in summary
    assert summary["converged"] is False


def test_common_relax_runner_preserves_extxyz_fixatoms_and_output_format(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    input_dir = stage / "input"
    input_dir.mkdir(parents=True)
    atoms = Atoms(
        "Cu2",
        positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[0]))
    atoms.calc = SinglePointCalculator(
        atoms,
        energy=123.0,
        forces=np.ones((len(atoms), 3)),
        stress=np.zeros(6),
    )
    write(input_dir / "constrained.extxyz", atoms, format="extxyz")
    resolved = resolve_mlff_template(
        "mlff_relax",
        {
            "backend": "mace",
            "task_config": {"fmax": 1.0e-3, "steps": 500},
        },
        audience="materials_worker",
    )
    materialize_mlff_run_config(
        stage_dir=stage,
        task_name="mlff_relax",
        resolved=resolved,
    )

    class FakeAdapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            del atoms, config
            return _HarmonicCalculator()

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    monkeypatch.setitem(mlff_common._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    batch = mlff_common.run("relax", ".catmaster/generated/run_config.json")

    assert batch["errors"] == []
    summary = batch["results"][0]["summary"]
    assert summary["output_structure"] == "opt.extxyz"
    output_path = stage / "output" / "constrained" / "opt.extxyz"
    assert output_path.is_file()
    assert not (output_path.parent / "opt.vasp").exists()
    restored = read(output_path, index=-1)
    assert restored.positions[0] == pytest.approx([1.0, 0.0, 0.0])
    assert restored.positions[1] == pytest.approx([0.0, 0.0, 0.0], abs=1.0e-3)
    assert restored.get_potential_energy() == pytest.approx(summary["final_energy_eV"])
    assert restored.get_potential_energy() != pytest.approx(123.0)
    assert len(restored.constraints) == 1
    assert isinstance(restored.constraints[0], FixAtoms)
    assert restored.constraints[0].get_indices().tolist() == [0]
    assert "move_mask:L:1" in output_path.read_text(encoding="utf-8").splitlines()[1]


def test_common_sp_runner_preserves_extxyz_fixatoms_and_output_format(
    tmp_path: Path,
) -> None:
    atoms = Atoms(
        "Cu2",
        positions=[[1.0, 0.0, 0.0], [2.0, 0.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_constraint(FixAtoms(indices=[0]))

    class FakeAdapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            del atoms, config
            return _HarmonicCalculator()

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    summary = mlff_common._run_sp(
        atoms=atoms,
        output_dir=tmp_path,
        config={
            "backend": "mace",
            "operation": "sp",
            "config_digest": "test",
            "task_config": {},
        },
        item_config={"model": "test", "device": "cpu"},
        adapter=FakeAdapter(),
        source_name="input.extxyz",
    )

    assert summary["output_structure"] == "sp.extxyz"
    restored = read(tmp_path / "sp.extxyz", index=-1)
    assert len(restored.constraints) == 1
    assert isinstance(restored.constraints[0], FixAtoms)
    assert restored.constraints[0].get_indices().tolist() == [0]


def test_extxyz_output_preserves_cartesian_mask_and_rejects_scaled_constraints(
    tmp_path: Path,
) -> None:
    atoms = Atoms(
        "Cu2",
        positions=[[1.0, 0.0, 0.0], [2.0, 1.0, 0.0]],
        cell=[10.0, 10.0, 10.0],
        pbc=True,
    )
    atoms.set_constraint(FixCartesian(1, mask=[True, False, True]))
    output_name = mlff_common._output_structure(
        tmp_path,
        atoms,
        stem="opt",
        source_name="input.extxyz",
    )
    assert output_name == "opt.extxyz"
    output_path = tmp_path / output_name
    assert "move_mask:L:3" in output_path.read_text(encoding="utf-8").splitlines()[1]
    restored = read(output_path, index=-1)
    constrained = next(
        constraint
        for constraint in restored.constraints
        if 1 in constraint.get_indices()
    )
    assert isinstance(constrained, FixCartesian)
    assert constrained.mask.tolist() == [True, False, True]

    atoms.set_constraint(FixScaled(1, mask=[True, False, True]))
    with pytest.raises(ValueError, match="FixAtoms and FixCartesian.*FixScaled"):
        mlff_common._output_structure(
            tmp_path,
            atoms,
            stem="scaled",
            source_name="input.extxyz",
        )


def test_generic_md_runner_uses_adapter_calculator_and_resultant_force(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    atoms = Atoms("Ar2", positions=[[0.2, 0.1, 0.0], [3.5, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    write(stage / "input" / "start.vasp", atoms)
    resolved = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {
                "dynamics": {"ensemble": "nve", "steps": 1, "timestep_fs": 0.5},
                "output": {"traj_interval": 1, "log_interval": 1, "overwrite": True},
            },
        },
    )
    materialize_mlff_run_config(stage_dir=stage, task_name="mlff_md", resolved=resolved)

    instances: list[Any] = []

    class FakeAdapter:
        provider_version = "test"

        def __init__(self):
            self.calculator = _HarmonicCalculator()
            self.calls = 0
            instances.append(self)

        def calculator_for(self, atoms, config):
            del atoms, config
            self.calls += 1
            return self.calculator

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {"adapter": "fake"}

    monkeypatch.setitem(mlff_md._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    batch = mlff_md.run(".catmaster/generated/run_config.json")
    summary = batch["results"][0]["summary"]
    assert len(instances) == 1
    assert instances[0].calls == 1
    assert summary["completed"] is True
    assert summary["max_force_eVA"] > 0
    assert "max_force_abs_eVA" not in summary
    assert summary["provider_metadata"] == {"adapter": "fake"}


def test_generic_md_runner_completes_langevin_temperature_schedule(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    atoms = Atoms("Ar2", positions=[[0.2, 0.1, 0.0], [3.5, 0.0, 0.0]], cell=[10, 10, 10], pbc=True)
    write(stage / "input" / "start.vasp", atoms)
    resolved = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {
                "dynamics": {
                    "temperature_K": 300.0,
                    "temperature_end_K": 600.0,
                    "steps": 3,
                    "timestep_fs": 0.5,
                },
                "thermostat": {"type": "langevin", "friction_per_fs": 0.01},
                "output": {"traj_interval": 1, "log_interval": 1, "overwrite": True},
            },
        },
        audience="dynamics_worker",
    )
    materialize_mlff_run_config(stage_dir=stage, task_name="mlff_md", resolved=resolved)

    class FakeAdapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            del atoms, config
            return _HarmonicCalculator()

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    monkeypatch.setitem(mlff_md._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    batch = mlff_md.run(".catmaster/generated/run_config.json")
    summary = batch["results"][0]["summary"]
    assert summary["completed"] is True
    assert summary["temperature_schedule"] == {
        "mode": "linear",
        "start_K": 300.0,
        "end_K": 600.0,
        "steps": 3,
        "update_interval_steps": 1,
        "temperature_api": "set_temperature",
    }
    assert summary["dynamics"]["temperature_end_K"] == 600.0
    assert summary["step_timing_statistics_s"]["all_steps"]["count"] == 3


def test_generic_md_runner_rejects_non_finite_calculator_state(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    (stage / "input").mkdir(parents=True)
    write(
        stage / "input" / "start.vasp",
        Atoms("Ar2", positions=[[0.0, 0.0, 0.0], [3.0, 0.0, 0.0]], cell=[10, 10, 10], pbc=True),
    )
    resolved = resolve_mlff_template(
        "mlff_md",
        {
            "backend": "mace",
            "task_config": {
                "dynamics": {"ensemble": "nve", "steps": 1},
                "output": {"traj_interval": 1, "log_interval": 1, "overwrite": True},
            },
        },
    )
    materialize_mlff_run_config(stage_dir=stage, task_name="mlff_md", resolved=resolved)

    class FakeAdapter:
        provider_version = "test"

        def calculator_for(self, atoms, config):
            del atoms, config
            return _NonFiniteCalculator()

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    monkeypatch.setitem(mlff_md._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    with pytest.raises(RuntimeError, match="see output/batch_summary.json"):
        mlff_md.run(".catmaster/generated/run_config.json")
    summary = json.loads((stage / "output" / "start" / "summary.json").read_text(encoding="utf-8"))
    assert summary["completed"] is False
    assert "non-finite potential energy" in summary["error"]


def test_generic_neb_runner_reuses_adapter_calculator_across_images(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    stage = tmp_path / "stage"
    path = stage / "input" / "path"
    path.mkdir(parents=True)
    for index, x in enumerate((0.1, 0.2, 0.3)):
        atoms = Atoms("H", positions=[[x, 0.1, 0.1]], cell=[10, 10, 10], pbc=True)
        write(path / f"{index:02d}.vasp", atoms)
    resolved = resolve_mlff_template(
        "mlff_neb",
        {"backend": "mace", "task_config": {"fmax": 1.0, "steps": 1}},
    )
    materialize_mlff_run_config(stage_dir=stage, task_name="mlff_neb", resolved=resolved)

    instances: list[Any] = []

    class FakeAdapter:
        provider_version = "test"

        def __init__(self):
            self.calculator = _HarmonicCalculator()
            self.calls = 0
            instances.append(self)

        def calculator_for(self, atoms, config):
            del atoms, config
            self.calls += 1
            return self.calculator

        def provider_metadata(self, atoms, config, calculator):
            del atoms, config, calculator
            return {}

    monkeypatch.setitem(mlff_neb._ADAPTER_TYPES, "mace", FakeAdapter)
    monkeypatch.chdir(stage)
    batch = mlff_neb.run(".catmaster/generated/run_config.json")
    summary = json.loads((stage / "output" / "path" / "summary.json").read_text(encoding="utf-8"))
    assert len(instances) == 1
    assert instances[0].calls == 3
    assert batch["tasks"][0]["status"] == "completed"
    assert summary["results"]["max_force_eVA"] >= 0
    assert all("max_force_eVA" in row for row in summary["image_profile"])


def test_neb_stage_requires_local_intermediate_images(tmp_path: Path) -> None:
    stage = tmp_path / "stage"
    path = stage / "input" / "path"
    path.mkdir(parents=True)
    atoms = bulk("Cu", cubic=True)
    write(path / "00.vasp", atoms)
    write(path / "01.vasp", atoms)
    resolved = resolve_mlff_template("mlff_neb", {})
    with pytest.raises(ValueError, match="intermediate"):
        materialize_mlff_run_config(stage_dir=stage, task_name="mlff_neb", resolved=resolved)
