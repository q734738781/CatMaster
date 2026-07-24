from __future__ import annotations

"""Dependency-light MLFF task/backend specifications and deployment profiles."""

import copy
import json
import math
from pathlib import Path
from typing import Any, Dict, Iterable, Literal, Mapping, Optional, Type

import yaml
from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.tools.execution.machine_registry import MachineRegister


MlffOperation = Literal["sp", "relax", "md", "neb"]
_MLFF_TASK_OPERATIONS: dict[str, MlffOperation] = {
    "mlff_sp": "sp",
    "mlff_relax": "relax",
    "mlff_md": "md",
    "mlff_neb": "neb",
}
_BACKEND_CAPABILITIES: dict[str, frozenset[str]] = {
    "mace": frozenset({"sp", "relax", "md", "neb"}),
    "fairchem_uma": frozenset({"sp", "relax", "md", "neb"}),
    "mattersim": frozenset({"sp", "relax", "md", "neb"}),
    "orb_v3": frozenset({"sp", "relax", "md", "neb"}),
}
_DEFAULT_PROFILE_PATHS = [
    Path(__file__).resolve().parents[3] / "configs" / "dpdispatcher",
]


class _StrictModel(BaseModel):
    model_config = ConfigDict(extra="forbid")


class MlffSpecValidationError(ValueError):
    def __init__(self, errors: list[dict[str, Any]]):
        self._errors = errors
        super().__init__("; ".join(str(item.get("msg") or item) for item in errors))

    def errors(self) -> list[dict[str, Any]]:
        return list(self._errors)


class MaceMolecularMetadata(_StrictModel):
    charge: int = Field(0, description="Total molecular charge for charge/spin-aware MACE models.")
    spin: int = Field(
        0,
        ge=0,
        description=(
            "Spin multiplicity for charge/spin-aware MACE models: 1 singlet, 2 doublet, 3 triplet. "
            "Zero means not applicable to the selected model."
        ),
    )


class MaceBackendConfig(_StrictModel):
    model: str = Field(
        "mh-1",
        description="Enabled registered MACE model name. Leave empty only when checkpoint_artifact is supplied.",
    )
    checkpoint_artifact: str = Field(
        "",
        description="Stage-relative checkpoint under models/. Leave empty when using a registered model.",
    )
    head: str = Field(
        "omat_pbe",
        description=(
            "MACE model head. Query the concrete model spec and use one of its enabled heads; "
            "leave empty only for a staged checkpoint whose head is selected internally."
        ),
    )
    defaults: MaceMolecularMetadata = Field(
        default_factory=MaceMolecularMetadata,
        description=(
            "Stage-wide charge and multiplicity-style spin defaults. These are active only for a registered "
            "MACE model that declares charge/spin support, such as omol-0."
        ),
    )
    items: dict[str, MaceMolecularMetadata] = Field(
        default_factory=dict,
        description=(
            "Per-input charge/spin overrides keyed by exact paths relative to input/. "
            "Each item should state both charge and spin when it differs from defaults."
        ),
    )
    dispersion: bool = Field(False, description="Enable the MACE dispersion wrapper.")
    default_dtype: Literal["float32", "float64"] = Field("float64", description="Calculator precision.")
    enable_cueq: bool = Field(False, description="Enable cuEquivariance when the remote environment supports it.")
    compile_mode: Literal["", "default", "reduce-overhead", "max-autotune"] = Field(
        "",
        description="Optional torch compilation mode; leave empty to disable.",
    )
    device: str = Field("auto", min_length=1, description="Device request such as auto, cpu, cuda, or cuda:0.")

    @model_validator(mode="after")
    def _model_or_checkpoint(self) -> "MaceBackendConfig":
        if self.checkpoint_artifact and self.model:
            raise ValueError("model and checkpoint_artifact are mutually exclusive.")
        if not self.checkpoint_artifact and not self.model:
            raise ValueError("model is required when checkpoint_artifact is empty.")
        return self


class UmaItemConfig(_StrictModel):
    uma_task: Literal["omat", "omol", "oc20", "oc22", "oc25", "odac", "omc"] = Field(
        "omat",
        description="Official FairChem UMA task name supported by the selected UMA model.",
    )
    charge: int = Field(0, description="Molecular charge for omol; keep zero for non-omol tasks.")
    spin: int = Field(0, ge=0, description="FairChem OMOL multiplicity-style spin value.")


class UmaBackendConfig(_StrictModel):
    model: str = Field(
        "uma-s-1p2",
        min_length=1,
        description="Exact official FairChem pretrained model name enabled by this deployment.",
    )
    device: str = Field("auto", min_length=1, description="Device request such as auto, cpu, cuda, or cuda:0.")
    inference_settings: Literal["default", "turbo"] = Field(
        "default",
        description=(
            "UMA inference preset. Use turbo for repeated inference when atomic composition stays fixed, "
            "such as one MD trajectory."
        ),
    )
    defaults: UmaItemConfig = Field(default_factory=UmaItemConfig, description="Stage-wide UMA metadata defaults.")
    items: dict[str, UmaItemConfig] = Field(
        default_factory=dict,
        description="Per-input UMA metadata keyed by paths relative to input/.",
    )


class MatterSimBackendConfig(_StrictModel):
    model: str = Field(
        "MatterSim-v1.0.0-1M",
        min_length=1,
        description="Exact official MatterSim checkpoint identity enabled by this deployment.",
    )
    device: str = Field("auto", min_length=1, description="Device request such as auto, cpu, cuda, or cuda:0.")
    dtype: Literal["float32", "float64"] = Field(
        "float32",
        description="MatterSim inference precision; float32 is the normal speed/accuracy choice.",
    )
    compute_stress: bool = Field(
        True,
        description="Compute stress. Disable for fixed-cell SP/MD when stress is not needed.",
    )
    direct_graph: bool = Field(
        False,
        description=(
            "MatterSim direct tensor graph construction. Keep false in the pinned 1.2.5 deployment because "
            "finite-output regression tests fail for periodic Si."
        ),
    )
    compile: bool = Field(
        False,
        description=(
            "Compile the MatterSim model forward pass. Keep false in the pinned 1.2.5 deployment because it "
            "implies the currently disabled direct-graph path."
        ),
    )

    @model_validator(mode="after")
    def _pinned_runtime_safe_graph_path(self) -> "MatterSimBackendConfig":
        if self.direct_graph or self.compile:
            raise ValueError(
                "MatterSim 1.2.5 direct_graph/compile are disabled in managed tasks: remote periodic-Si "
                "regression tests returned non-finite energy. Keep both false."
            )
        return self


class OrbV3BackendConfig(_StrictModel):
    model: str = Field(
        "orb-v3-conservative-inf-omat",
        min_length=1,
        description="Exact official ORB-v3 pretrained model name enabled by this deployment.",
    )
    device: str = Field("auto", min_length=1, description="Device request such as auto, cpu, cuda, or cuda:0.")
    precision: Literal["float32-high", "float32-highest", "float64"] = Field(
        "float32-high",
        description="ORB inference precision accepted by the official pretrained loader.",
    )
    compile_mode: Literal["auto", "on", "off"] = Field(
        "auto",
        description="ORB model compilation policy. Auto compiles on CUDA and disables compilation on MPS.",
    )
    edge_method: Literal[
        "knn_alchemi",
        "knn_scipy",
        "knn_brute_force",
        "knn_cuml_brute",
        "knn_cuml_rbc",
    ] = Field(
        "knn_alchemi",
        description="ORB neighbor-graph implementation; knn_alchemi is the recommended GPU default.",
    )
    half_supercell: Literal["auto", "on", "off"] = Field(
        "auto",
        description="ORB large-cell graph optimization; auto enables it only when the provider recommends it.",
    )


class SinglePointTaskConfig(_StrictModel):
    pass


class RelaxTaskConfig(_StrictModel):
    fmax: float = Field(0.02, gt=0, description="Force convergence threshold in eV/Angstrom.")
    steps: int = Field(500, ge=1, description="Maximum optimizer steps.")
    optimizer: Literal["FIRE", "BFGS", "LBFGS"] = Field("FIRE", description="ASE optimizer.")
    relax_cell: bool = Field(False, description="Relax the periodic cell as well as atomic positions.")


class MdDynamicsConfig(_StrictModel):
    ensemble: Literal["nve", "nvt", "npt"] = Field("nvt", description="MD statistical ensemble.")
    temperature_K: float = Field(
        300.0,
        gt=0,
        description="Constant target temperature, or the start of a linear target-temperature schedule, in kelvin.",
    )
    temperature_end_K: float = Field(
        0.0,
        ge=0,
        description=(
            "End of a linear target-temperature schedule in kelvin. Leave at 0 or set equal to temperature_K "
            "for constant temperature."
        ),
    )
    initial_temperature_K: float = Field(
        0.0,
        ge=0,
        description="Initial velocity temperature; zero means use temperature_K.",
    )
    timestep_fs: float = Field(1.0, gt=0, description="MD timestep in femtoseconds.")
    steps: int = Field(1000, ge=1, description="Number of MD steps.")
    seed: int = Field(2026, ge=0, description="Random seed for velocity generation and stochastic thermostats.")
    zero_rotation: bool = Field(False, description="Remove net angular momentum when velocities are generated.")
    force_temp: bool = Field(False, description="Force exact initial kinetic temperature.")
    reinitialize_velocities: bool = Field(False, description="Replace existing momenta before starting MD.")


class MdThermostatConfig(_StrictModel):
    type: Literal["bussi", "nhc", "langevin", "berendsen"] = Field("bussi", description="NVT thermostat.")
    tau_fs: float = Field(100.0, gt=0, description="Thermostat coupling time in femtoseconds.")
    friction_per_fs: float = Field(0.0, ge=0, description="Langevin friction; zero uses the runner default.")
    tchain: int = Field(3, ge=1, description="Nose-Hoover thermostat chain length.")
    tloop: int = Field(1, ge=1, description="Thermostat chain integration loops.")


class MdBarostatConfig(_StrictModel):
    type: Literal["none", "isotropic_mtk", "inhomogeneous_mtk", "berendsen"] = Field(
        "none",
        description="NPT barostat; keep none outside NPT.",
    )
    pressure_bar: float = Field(1.01325, gt=0, description="Target pressure in bar.")
    taup_fs: float = Field(1000.0, gt=0, description="Pressure coupling time in femtoseconds.")
    pdamp_fs: float = Field(1000.0, gt=0, description="MTK pressure damping time in femtoseconds.")
    compressibility_bar_inv: float = Field(
        0.0,
        ge=0,
        description="Berendsen compressibility in 1/bar; zero is invalid when Berendsen is selected.",
    )
    pchain: int = Field(3, ge=1, description="MTK barostat chain length.")
    ploop: int = Field(1, ge=1, description="MTK barostat chain integration loops.")


class MdOutputConfig(_StrictModel):
    traj_interval: int = Field(10, ge=1, description="Trajectory write interval in steps.")
    log_interval: int = Field(10, ge=1, description="Log write interval in steps.")
    log_stress: bool = Field(False, description="Record stress in the MD log.")
    overwrite: bool = Field(False, description="Allow replacement of an existing output directory.")


class MolecularDynamicsTaskConfig(_StrictModel):
    dynamics: MdDynamicsConfig = Field(default_factory=MdDynamicsConfig)
    thermostat: MdThermostatConfig = Field(default_factory=MdThermostatConfig)
    barostat: MdBarostatConfig = Field(default_factory=MdBarostatConfig)
    output: MdOutputConfig = Field(default_factory=MdOutputConfig)

    @model_validator(mode="after")
    def _ensemble_controls(self) -> "MolecularDynamicsTaskConfig":
        ensemble = self.dynamics.ensemble
        if ensemble != "npt" and self.barostat.type != "none":
            raise ValueError("barostat.type must be none unless dynamics.ensemble is npt.")
        if ensemble == "npt" and self.barostat.type == "none":
            raise ValueError("NPT requires a non-none barostat.type.")
        if self.barostat.type == "berendsen" and self.barostat.compressibility_bar_inv <= 0:
            raise ValueError("Berendsen NPT requires compressibility_bar_inv > 0.")

        end_temperature = self.dynamics.temperature_end_K
        variable_temperature = end_temperature > 0 and not math.isclose(
            end_temperature,
            self.dynamics.temperature_K,
        )
        if not variable_temperature:
            return self
        if self.dynamics.steps < 2:
            raise ValueError("A variable-temperature schedule requires dynamics.steps >= 2.")
        if ensemble == "nve":
            raise ValueError("NVE does not support a target-temperature schedule.")
        if ensemble == "nvt" and self.thermostat.type not in {"langevin", "berendsen"}:
            raise ValueError(
                "Variable-temperature NVT requires thermostat.type=langevin or berendsen; "
                "ASE Bussi and NHC do not expose set_temperature()."
            )
        if ensemble == "npt" and self.barostat.type != "berendsen":
            raise ValueError(
                "Variable-temperature NPT requires barostat.type=berendsen; "
                "ASE MTK integrators do not expose set_temperature()."
            )
        return self


class NebTaskConfig(_StrictModel):
    fmax: float = Field(0.05, gt=0, description="NEB force convergence threshold in eV/Angstrom.")
    steps: int = Field(300, ge=1, description="Maximum FIRE optimizer steps.")
    optimizer: Literal["FIRE"] = Field("FIRE", description="Initial generic NEB optimizer.")
    mode: Literal["plain"] = Field("plain", description="Fixed-image plain NEB mode.")
    climb: bool = Field(False, description="Enable climbing-image refinement.")


_BACKEND_MODELS: dict[str, Type[_StrictModel]] = {
    "mace": MaceBackendConfig,
    "fairchem_uma": UmaBackendConfig,
    "mattersim": MatterSimBackendConfig,
    "orb_v3": OrbV3BackendConfig,
}
_TASK_MODELS: dict[MlffOperation, Type[_StrictModel]] = {
    "sp": SinglePointTaskConfig,
    "relax": RelaxTaskConfig,
    "md": MolecularDynamicsTaskConfig,
    "neb": NebTaskConfig,
}


class MlffModelProfile(_StrictModel):
    enabled: bool = True
    loader: Literal["", "mace_mp", "mace_omol"] = ""
    provider_model: str = ""
    heads: list[str] = Field(default_factory=list)
    default_head: str = ""
    tasks: list[str] = Field(default_factory=list)
    default_task: str = ""
    supports_charge_spin: bool = False


class MlffBackendProfile(_StrictModel):
    enabled: bool = True
    default: bool = False
    resource: str
    operations: list[MlffOperation]
    default_model: str
    models: dict[str, MlffModelProfile]
    audiences: list[str] = Field(default_factory=list)


_OFFICIAL_MODEL_CONTRACTS: dict[str, dict[str, dict[str, Any]]] = {
    "mace": {
        "mh-1": {
            "loader": "mace_mp",
            "provider_model": "mh-1",
            "heads": [
                "matpes_r2scan",
                "mp_pbe_refit_add",
                "spice_wB97M",
                "oc20_usemppbe",
                "omol",
                "omat_pbe",
            ],
            "default_head": "omat_pbe",
            "tasks": [],
            "default_task": "",
            "supports_charge_spin": False,
        },
        "omol-0": {
            "loader": "mace_omol",
            "provider_model": "extra_large",
            "heads": ["omol"],
            "default_head": "omol",
            "tasks": ["omol"],
            "default_task": "omol",
            "supports_charge_spin": True,
        },
    },
    "fairchem_uma": {
        "uma-s-1p2": {
            "provider_model": "uma-s-1p2",
            "tasks": ["oc20", "oc22", "oc25", "omat", "omol", "odac", "omc"],
            "default_task": "omat",
            "supports_charge_spin": True,
        },
        "uma-s-1p1": {
            "provider_model": "uma-s-1p1",
            "tasks": ["oc20", "omat", "omol", "odac", "omc"],
            "default_task": "omat",
            "supports_charge_spin": True,
        },
        "uma-m-1p1": {
            "provider_model": "uma-m-1p1",
            "tasks": ["oc20", "omat", "omol", "odac", "omc"],
            "default_task": "omat",
            "supports_charge_spin": True,
        },
    },
    "mattersim": {
        "MatterSim-v1.0.0-1M": {
            "provider_model": "MatterSim-v1.0.0-1M",
            "tasks": ["omat"],
            "default_task": "omat",
        },
        "MatterSim-v1.0.0-5M": {
            "provider_model": "MatterSim-v1.0.0-5M",
            "tasks": ["omat"],
            "default_task": "omat",
        },
    },
    "orb_v3": {
        name: {
            "provider_model": name,
            "tasks": ["omat"],
            "default_task": "omat",
        }
        for name in (
            "orb-v3-conservative-inf-omat",
            "orb-v3-conservative-20-omat",
            "orb-v3-direct-inf-omat",
            "orb-v3-direct-20-omat",
        )
    },
}


def _load_yaml(path: Path) -> dict[str, Any]:
    data = yaml.safe_load(path.read_text(encoding="utf-8")) or {}
    if not isinstance(data, dict):
        raise ValueError(f"MLFF backend config must be an object: {path}")
    return data


def _iter_profile_files(base: Path) -> Iterable[Path]:
    if not base.exists():
        return []
    if base.is_file():
        return [base]
    files: list[Path] = []
    for pattern in ("mlff_backends*.yaml", "mlff_backends*.yml", "mlff_backends*.json"):
        files.extend(sorted(path for path in base.glob(pattern) if "template" not in path.stem.lower()))
    return files


class MlffBackendRegistry:
    """Load administrator-enabled MLFF backend deployment profiles."""

    def __init__(self, extra_paths: Optional[Iterable[Path]] = None):
        paths = list(_DEFAULT_PROFILE_PATHS)
        if extra_paths:
            paths.extend(extra_paths)
        self.profiles: dict[str, MlffBackendProfile] = {}
        for base in paths:
            for path in _iter_profile_files(base):
                raw = _load_yaml(path)
                section = raw.get("mlff_backends", raw)
                if not isinstance(section, dict):
                    raise ValueError(f"mlff_backends must be an object: {path}")
                for name, value in section.items():
                    backend_name = str(name)
                    if backend_name not in _BACKEND_MODELS:
                        raise ValueError(f"Unknown MLFF backend profile: {backend_name}")
                    profile = MlffBackendProfile.model_validate(value)
                    unsupported = sorted(set(profile.operations) - set(_BACKEND_CAPABILITIES[backend_name]))
                    if unsupported:
                        raise ValueError(
                            f"Backend {backend_name!r} configures unsupported operations: {', '.join(unsupported)}"
                        )
                    default_model = profile.models.get(profile.default_model)
                    if default_model is None or not default_model.enabled:
                        raise ValueError(
                            f"Backend {backend_name!r} default_model {profile.default_model!r} must be enabled in models."
                        )
                    contracts = _OFFICIAL_MODEL_CONTRACTS[backend_name]
                    for model_name, model in profile.models.items():
                        if not model.enabled:
                            continue
                        contract = contracts.get(model_name)
                        if contract is None:
                            allowed = ", ".join(sorted(contracts))
                            raise ValueError(
                                f"Backend {backend_name!r} model {model_name!r} is not an exact supported "
                                f"official model name. Allowed: {allowed}"
                            )
                        for field_name, expected in contract.items():
                            actual = getattr(model, field_name)
                            if actual != expected:
                                raise ValueError(
                                    f"Backend {backend_name!r} model {model_name!r} must declare "
                                    f"{field_name}={expected!r}; got {actual!r}."
                                )
                    self.profiles[backend_name] = profile
        defaults = [name for name, profile in self.profiles.items() if profile.enabled and profile.default]
        if len(defaults) > 1:
            raise ValueError("Exactly zero or one enabled MLFF backend profile may set default=true.")

    def get(self, name: str) -> MlffBackendProfile:
        if name not in self.profiles:
            raise KeyError(f"MLFF backend '{name}' is not configured.")
        return self.profiles[name]

    def effective_names(
        self,
        operation: MlffOperation,
        *,
        audience: str = "",
        machines: MachineRegister | None = None,
    ) -> list[str]:
        register = machines or MachineRegister()
        names: list[str] = []
        for name, profile in sorted(self.profiles.items()):
            if not profile.enabled or operation not in profile.operations:
                continue
            if operation not in _BACKEND_CAPABILITIES.get(name, frozenset()):
                continue
            if profile.audiences and audience and audience not in profile.audiences:
                continue
            try:
                resource = register.get_resources(profile.resource)
                machine_name = str(resource.get("machine") or "")
                if not machine_name:
                    continue
                register.get_machine(machine_name)
            except Exception:
                continue
            resource_audiences = resource.get("audiences")
            if isinstance(resource_audiences, list) and audience and audience not in {str(x) for x in resource_audiences}:
                continue
            if resource.get("enabled") is False:
                continue
            if register.get_machine(machine_name).get("enabled") is False:
                continue
            model = profile.models.get(profile.default_model)
            if model is None or not model.enabled:
                continue
            names.append(name)
        return names

    def default_name(self, operation: MlffOperation, *, audience: str = "") -> str:
        effective = set(self.effective_names(operation, audience=audience))
        defaults = [
            name
            for name, profile in self.profiles.items()
            if profile.enabled and profile.default and name in effective
        ]
        return defaults[0] if len(defaults) == 1 else ""


def mlff_operation_for_task(task_name: str) -> MlffOperation | None:
    return _MLFF_TASK_OPERATIONS.get(str(task_name))


def is_mlff_task(task_name: str) -> bool:
    return mlff_operation_for_task(task_name) is not None


def _deep_merge(base: Mapping[str, Any], override: Mapping[str, Any]) -> dict[str, Any]:
    out = copy.deepcopy(dict(base))
    for key, value in override.items():
        if isinstance(value, Mapping) and isinstance(out.get(key), Mapping):
            out[str(key)] = _deep_merge(out[str(key)], value)
        else:
            out[str(key)] = copy.deepcopy(value)
    return out


def _backend_defaults(name: str, operation: MlffOperation, profile: MlffBackendProfile) -> dict[str, Any]:
    model_type = _BACKEND_MODELS[name]
    defaults = model_type().model_dump(mode="json")
    defaults["model"] = profile.default_model
    model = profile.models[profile.default_model]
    if name == "mace":
        defaults["head"] = model.default_head
        defaults["defaults"] = {
            "charge": 0,
            "spin": 1 if model.supports_charge_spin else 0,
        }
        if operation == "md":
            defaults["default_dtype"] = "float32"
    elif name == "fairchem_uma":
        defaults["defaults"] = {
            "uma_task": model.default_task,
            "charge": 0,
            "spin": 1 if model.default_task == "omol" else 0,
        }
    return defaults


def _task_defaults(operation: MlffOperation) -> dict[str, Any]:
    return _TASK_MODELS[operation]().model_dump(mode="json")


def _inline_schema(schema: dict[str, Any]) -> dict[str, Any]:
    source = copy.deepcopy(schema)
    definitions = source.pop("$defs", {})

    def visit(value: Any) -> Any:
        if isinstance(value, list):
            return [visit(item) for item in value]
        if not isinstance(value, dict):
            return value
        ref = value.get("$ref")
        if isinstance(ref, str) and ref.startswith("#/$defs/"):
            name = ref.rsplit("/", 1)[-1]
            resolved = copy.deepcopy(definitions[name])
            resolved.update({key: item for key, item in value.items() if key != "$ref"})
            return visit(resolved)
        return {key: visit(item) for key, item in value.items() if key != "$defs"}

    return visit(source)


def _apply_defaults(schema: dict[str, Any], defaults: Any) -> None:
    if not isinstance(defaults, dict):
        schema["default"] = defaults
        return
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        schema["default"] = defaults
        return
    for key, value in defaults.items():
        child = properties.get(key)
        if isinstance(child, dict):
            _apply_defaults(child, value)


def _flatten_schema(schema: dict[str, Any], *, prefix: str = "") -> list[dict[str, Any]]:
    fields: list[dict[str, Any]] = []
    properties = schema.get("properties")
    if not isinstance(properties, dict):
        return fields
    required = set(schema.get("required") or [])
    for name, child in properties.items():
        if not isinstance(child, dict):
            continue
        path = f"{prefix}.{name}" if prefix else str(name)
        if child.get("type") == "object" and isinstance(child.get("properties"), dict):
            fields.extend(_flatten_schema(child, prefix=path))
            continue
        item: dict[str, Any] = {
            "path": path,
            "type": "enum" if "enum" in child else child.get("type", "value"),
            "required": name in required,
        }
        for source, target in (
            ("default", "default"),
            ("enum", "allowed"),
            ("const", "const"),
            ("minimum", "minimum"),
            ("maximum", "maximum"),
            ("exclusiveMinimum", "exclusive_minimum"),
            ("exclusiveMaximum", "exclusive_maximum"),
            ("description", "description"),
        ):
            if source in child:
                item[target] = child[source]
        fields.append(item)
    return fields


def _validate_profile_model(
    *,
    backend: str,
    config: _StrictModel,
    profile: MlffBackendProfile,
) -> None:
    checkpoint = str(getattr(config, "checkpoint_artifact", "") or "")
    if checkpoint:
        if backend != "mace":
            raise ValueError(f"backend {backend!r} does not accept checkpoint_artifact.")
        return
    model = str(getattr(config, "model", "") or "")
    configured = profile.models.get(model)
    if configured is None or not configured.enabled:
        allowed = sorted(name for name, item in profile.models.items() if item.enabled)
        raise ValueError(f"Model {model!r} is not enabled for backend {backend!r}. Allowed: {', '.join(allowed)}")
    if backend == "mace" and isinstance(config, MaceBackendConfig):
        if config.head not in configured.heads:
            raise ValueError(
                f"Head {config.head!r} is not enabled for MACE model {model!r}. "
                f"Allowed: {', '.join(configured.heads)}"
            )
        if configured.loader == "mace_omol" and config.dispersion:
            raise ValueError("MACE omol-0 does not support the mace_mp dispersion wrapper.")
        defaults = config.defaults
        if configured.supports_charge_spin:
            if defaults.spin < 1:
                raise ValueError(
                    f"MACE model {model!r} requires a positive multiplicity-style defaults.spin."
                )
            for rel, metadata in config.items.items():
                if "spin" in metadata.model_fields_set and metadata.spin < 1:
                    raise ValueError(
                        f"MACE model {model!r} item {rel!r} requires a positive multiplicity-style spin."
                    )
        elif config.items or defaults.charge != 0 or defaults.spin != 0:
            raise ValueError(
                f"MACE model {model!r} does not accept charge/spin metadata. "
                "Use omol-0 for charge/spin-conditioned molecular inference."
            )
    if backend == "fairchem_uma" and isinstance(config, UmaBackendConfig):
        metadata_items: list[tuple[str, dict[str, Any]]] = [
            ("defaults", config.defaults.model_dump(mode="json"))
        ]
        for item_name, metadata in sorted(config.items.items()):
            merged = config.defaults.model_dump(mode="json")
            merged.update(metadata.model_dump(mode="json", exclude_unset=True))
            metadata_items.append((item_name, merged))
        for item_name, metadata in metadata_items:
            task = str(metadata["uma_task"])
            charge = int(metadata["charge"])
            spin = int(metadata["spin"])
            if task not in configured.tasks:
                raise ValueError(
                    f"UMA task {task!r} is not supported by model {model!r} "
                    f"for {item_name!r}. Allowed: {', '.join(configured.tasks)}"
                )
            if task == "omol" and spin < 1:
                raise ValueError(
                    f"UMA task 'omol' for {item_name!r} requires multiplicity-style spin >= 1."
                )
            if task != "omol" and (charge != 0 or spin != 0):
                raise ValueError(
                    f"UMA task {task!r} for {item_name!r} requires charge=0 and spin=0."
                )


def _validate_cross_constraints(
    *,
    backend: str,
    operation: MlffOperation,
    backend_config: _StrictModel,
    task_config: _StrictModel,
) -> None:
    if operation not in _BACKEND_CAPABILITIES.get(backend, frozenset()):
        raise ValueError(f"Backend {backend!r} does not support operation {operation!r}.")
    if operation == "relax" and isinstance(task_config, RelaxTaskConfig):
        if task_config.relax_cell and backend in {"mattersim", "orb_v3"}:
            raise ValueError(f"Backend {backend!r} is initially enabled only for fixed-cell relaxation.")
    if isinstance(backend_config, MaceBackendConfig):
        if operation not in {"relax", "md"} and backend_config.enable_cueq:
            raise ValueError("enable_cueq is currently supported only for MACE relax and MD.")
        if operation != "md" and backend_config.compile_mode:
            raise ValueError("compile_mode is currently supported only for MACE MD.")
    if (
        isinstance(backend_config, MatterSimBackendConfig)
        and isinstance(task_config, MolecularDynamicsTaskConfig)
        and task_config.dynamics.ensemble == "npt"
        and not backend_config.compute_stress
    ):
        raise ValueError("MatterSim NPT requires backend_config.compute_stress=true.")


def _prefixed_validation_errors(exc: BaseException, prefix: str) -> MlffSpecValidationError:
    errors_method = getattr(exc, "errors", None)
    if not callable(errors_method):
        return MlffSpecValidationError(
            [{"loc": (prefix,), "msg": str(exc), "type": "validation_error"}]
        )
    errors: list[dict[str, Any]] = []
    for item in errors_method():
        copied = dict(item)
        copied["loc"] = (prefix, *(item.get("loc") or ()))
        errors.append(copied)
    return MlffSpecValidationError(errors)


def _constraints(backend: str, operation: MlffOperation) -> list[str]:
    out: list[str] = []
    if backend == "mace":
        out.append("backend_config.model and backend_config.checkpoint_artifact are mutually exclusive.")
        out.append("Registered MACE models accept only their model-specific enabled head allowlist.")
        out.append("MACE omol-0 requires multiplicity-style spin metadata and does not use the mace_mp dispersion wrapper.")
        if operation != "md":
            out.append("backend_config.compile_mode must be empty outside MLFF MD.")
    if backend == "fairchem_uma":
        out.append("UMA accepts only the selected official model's task allowlist; auto is not a provider task.")
        out.append("UMA omol requires multiplicity-style spin >= 1; non-omol tasks require charge=0 and spin=0.")
        out.append("UMA turbo inference requires fixed atomic composition throughout the task.")
    if backend == "mattersim":
        out.append("Pinned MatterSim 1.2.5 requires backend_config.direct_graph=false and compile=false.")
        if operation == "md":
            out.append("MatterSim NPT requires backend_config.compute_stress=true.")
    if backend == "orb_v3":
        out.append("ORB knn_alchemi is recommended; legacy edge methods are compatibility controls.")
    if operation in {"sp", "relax"}:
        out.append(
            "An .extxyz input is returned as the same output format; ASE FixAtoms and FixCartesian "
            "constraints are preserved through the standard move_mask property. Use POSCAR/VASP "
            "Selective Dynamics when a FixScaled constraint is required."
        )
    if operation == "relax":
        out.append("task_config.relax_cell=true requires a fully periodic structure with a valid cell.")
    if operation == "md":
        out.append("NPT requires a non-none barostat; non-NPT ensembles require barostat.type=none.")
        out.append(
            "task_config.dynamics.temperature_end_K=0 or temperature_K keeps a constant target; a different "
            "positive value requests a per-step linear target-temperature schedule."
        )
        out.append(
            "Variable-temperature NVT supports thermostat.type=langevin or berendsen; variable-temperature "
            "NPT supports barostat.type=berendsen; NVE, Bussi, NHC, and MTK schedules are rejected."
        )
    if operation == "neb":
        out.append("MLFF NEB accepts a complete locally prepared fixed-image path with at least one intermediate image.")
    return out


def _enabled_model_capabilities(
    backend: str,
    profile: MlffBackendProfile,
) -> dict[str, dict[str, Any]]:
    capabilities: dict[str, dict[str, Any]] = {}
    for model_name, model in sorted(profile.models.items()):
        if not model.enabled:
            continue
        item: dict[str, Any] = {
            "provider_model": model.provider_model,
            "tasks": list(model.tasks),
            "default_task": model.default_task,
            "supports_charge_spin": model.supports_charge_spin,
        }
        if model.loader:
            item["loader"] = model.loader
        if model.heads:
            item["heads"] = list(model.heads)
            item["default_head"] = model.default_head
        capabilities[model_name] = item
    return capabilities


def _normalized_backend_config(
    backend: str,
    config: _StrictModel,
    *,
    backend_overrides: Mapping[str, Any],
    profile: MlffBackendProfile,
) -> dict[str, Any]:
    normalized = config.model_dump(mode="json")
    raw_items = backend_overrides.get("items") or {}
    if backend == "mace" and isinstance(config, MaceBackendConfig) and isinstance(raw_items, Mapping):
        normalized["items"] = {
            str(rel): MaceMolecularMetadata.model_validate(metadata).model_dump(
                mode="json",
                exclude_unset=True,
            )
            for rel, metadata in raw_items.items()
        }
    if backend == "fairchem_uma" and isinstance(config, UmaBackendConfig) and isinstance(raw_items, Mapping):
        normalized["items"] = {
            str(rel): UmaItemConfig.model_validate(metadata).model_dump(
                mode="json",
                exclude_unset=True,
            )
            for rel, metadata in raw_items.items()
        }
    if backend == "mace" and isinstance(config, MaceBackendConfig) and config.checkpoint_artifact:
        normalized.update(
            {
                "loader": "checkpoint",
                "provider_model": "",
                "supports_charge_spin": False,
            }
        )
        return normalized
    model_name = str(getattr(config, "model"))
    model = profile.models[model_name]
    normalized["provider_model"] = model.provider_model
    if backend == "mace":
        normalized.update(
            {
                "loader": model.loader,
                "supports_charge_spin": model.supports_charge_spin,
            }
        )
    return normalized


def resolve_mlff_template(
    task_name: str,
    template_overrides: Mapping[str, Any] | None = None,
    *,
    audience: str = "",
    registry: MlffBackendRegistry | None = None,
) -> dict[str, Any]:
    """Resolve and validate one concrete MLFF task/backend template."""

    operation = mlff_operation_for_task(task_name)
    if operation is None:
        raise KeyError(f"Task {task_name!r} is not a structured MLFF task.")
    overrides = dict(template_overrides or {})
    unknown_top = sorted(str(key) for key in overrides if key not in {"backend", "backend_config", "task_config"})
    if unknown_top:
        raise ValueError(
            "Unknown MLFF template_overrides key(s): "
            + ", ".join(unknown_top)
            + ". Accepted keys: backend, backend_config, task_config."
        )
    backend_overrides = overrides.get("backend_config") or {}
    task_overrides = overrides.get("task_config") or {}
    if not isinstance(backend_overrides, dict):
        raise ValueError("template_overrides.backend_config must be an object.")
    if not isinstance(task_overrides, dict):
        raise ValueError("template_overrides.task_config must be an object.")

    profile_registry = registry or MlffBackendRegistry()
    available = profile_registry.effective_names(operation, audience=audience)
    default_backend = profile_registry.default_name(operation, audience=audience)
    requested = str(overrides.get("backend") or default_backend).strip()
    if not requested:
        raise ValueError(
            f"No effective default backend for {task_name}. Available backends: {', '.join(available) or 'none'}."
        )
    if requested not in available:
        raise ValueError(
            f"Backend {requested!r} is not available for {task_name}. "
            f"Available backends: {', '.join(available) or 'none'}."
        )
    profile = profile_registry.get(requested)
    backend_model_type = _BACKEND_MODELS[requested]
    task_model_type = _TASK_MODELS[operation]
    backend_defaults = _backend_defaults(requested, operation, profile)
    task_defaults = _task_defaults(operation)
    # A staged checkpoint is an explicit replacement for the deployment's
    # default MACE model. Callers should not need to know or clear that inherited
    # default themselves.
    if requested == "mace":
        if backend_overrides.get("checkpoint_artifact"):
            backend_defaults["model"] = ""
            if "head" not in backend_overrides:
                backend_defaults["head"] = ""
            if "defaults" not in backend_overrides:
                backend_defaults["defaults"] = {"charge": 0, "spin": 0}
        else:
            selected_model = (
                str(backend_overrides.get("model") or "")
                if "model" in backend_overrides
                else profile.default_model
            )
            backend_defaults["model"] = selected_model
            selected_profile = profile.models.get(selected_model)
            if selected_profile is not None:
                if "head" not in backend_overrides:
                    backend_defaults["head"] = selected_profile.default_head
                if "defaults" not in backend_overrides:
                    backend_defaults["defaults"] = {
                        "charge": 0,
                        "spin": 1 if selected_profile.supports_charge_spin else 0,
                    }
    elif requested == "fairchem_uma":
        selected_model = (
            str(backend_overrides.get("model") or "")
            if "model" in backend_overrides
            else profile.default_model
        )
        backend_defaults["model"] = selected_model
        selected_profile = profile.models.get(selected_model)
        if selected_profile is not None and "defaults" not in backend_overrides:
            backend_defaults["defaults"] = {
                "uma_task": selected_profile.default_task,
                "charge": 0,
                "spin": 1 if selected_profile.default_task == "omol" else 0,
            }
    elif "model" in backend_overrides:
        backend_defaults["model"] = str(backend_overrides.get("model") or "")
    try:
        backend_config = backend_model_type.model_validate(_deep_merge(backend_defaults, backend_overrides))
    except Exception as exc:
        raise _prefixed_validation_errors(exc, "backend_config") from exc
    try:
        task_config = task_model_type.model_validate(_deep_merge(task_defaults, task_overrides))
    except Exception as exc:
        raise _prefixed_validation_errors(exc, "task_config") from exc
    try:
        _validate_profile_model(backend=requested, config=backend_config, profile=profile)
    except Exception as exc:
        raise _prefixed_validation_errors(exc, "backend_config.model") from exc
    try:
        _validate_cross_constraints(
            backend=requested,
            operation=operation,
            backend_config=backend_config,
            task_config=task_config,
        )
    except Exception as exc:
        raise _prefixed_validation_errors(exc, "template_overrides") from exc

    backend_schema = _inline_schema(backend_model_type.model_json_schema())
    task_schema = _inline_schema(task_model_type.model_json_schema())
    enabled_models = sorted(name for name, item in profile.models.items() if item.enabled)
    model_capabilities = _enabled_model_capabilities(requested, profile)
    public_backend_defaults = copy.deepcopy(backend_defaults)
    selected_model = str(getattr(backend_config, "model", "") or "")
    selected_model_profile = profile.models.get(selected_model)
    backend_properties = backend_schema.get("properties") or {}
    if requested == "mace" and isinstance(backend_properties, dict):
        if selected_model_profile is None or not selected_model_profile.supports_charge_spin:
            public_backend_defaults.pop("defaults", None)
            public_backend_defaults.pop("items", None)
            backend_properties.pop("defaults", None)
            backend_properties.pop("items", None)
        head_field = backend_properties.get("head")
        if isinstance(head_field, dict) and selected_model_profile is not None:
            head_field["enum"] = list(selected_model_profile.heads)
            head_field["default"] = selected_model_profile.default_head
            head_field["description"] = (
                f"Enabled head for MACE model {selected_model!r}. "
                "Choose exactly one value from this model-specific allowlist."
            )
    if requested == "fairchem_uma" and isinstance(backend_properties, dict) and selected_model_profile is not None:
        task_schemas = [
            ((backend_properties.get("defaults") or {}).get("properties") or {}).get("uma_task"),
            (
                (((backend_properties.get("items") or {}).get("additionalProperties") or {}).get("properties") or {})
            ).get("uma_task"),
        ]
        for task_field in task_schemas:
            if not isinstance(task_field, dict):
                continue
            task_field["enum"] = list(selected_model_profile.tasks)
            task_field["default"] = selected_model_profile.default_task
            task_field["description"] = (
                f"Official FairChem task supported by UMA model {selected_model!r}. "
                "Choose exactly one value from this model-specific allowlist."
            )
    _apply_defaults(backend_schema, public_backend_defaults)
    _apply_defaults(task_schema, task_defaults)
    model_field = (backend_schema.get("properties") or {}).get("model")
    if enabled_models and isinstance(model_field, dict):
        model_field["enum"] = ([""] + enabled_models) if requested == "mace" else enabled_models
    template_schema: dict[str, Any] = {
        "type": "object",
        "additionalProperties": False,
        "properties": {
            "backend": {
                "type": "string",
                "const": requested,
                "default": requested,
                "description": "Concrete enabled backend for this task-spec query.",
            },
            "backend_config": backend_schema,
            "task_config": task_schema,
        },
    }
    resolved_template_defaults = {
        "backend": requested,
        "backend_config": public_backend_defaults,
        "task_config": task_defaults,
    }
    registered_backend = default_backend or requested
    registered_profile = profile_registry.get(registered_backend)
    template_defaults = {
        "backend": registered_backend,
        "backend_config": _backend_defaults(registered_backend, operation, registered_profile),
        "task_config": task_defaults,
    }
    if registered_backend == "mace":
        registered_model = registered_profile.models[registered_profile.default_model]
        if not registered_model.supports_charge_spin:
            template_defaults["backend_config"].pop("defaults", None)
            template_defaults["backend_config"].pop("items", None)
    normalized_backend_config = _normalized_backend_config(
        requested,
        backend_config,
        backend_overrides=backend_overrides,
        profile=profile,
    )
    normalized = {
        "backend": requested,
        "backend_config": normalized_backend_config,
        "task_config": task_config.model_dump(mode="json"),
    }
    return {
        "task_name": task_name,
        "operation": operation,
        "resolved_backend": requested,
        "available_backends": available,
        "enabled_models": enabled_models,
        "model_capabilities": model_capabilities,
        "selected_model": selected_model,
        "default_backend": default_backend,
        "resource": profile.resource,
        "template_override_keys": ["backend", "backend_config", "task_config"],
        "template_defaults": template_defaults,
        "resolved_template_defaults": resolved_template_defaults,
        "template_schema": template_schema,
        "fields": _flatten_schema(template_schema),
        "constraints": _constraints(requested, operation),
        "errors": [],
        "warnings": [],
        "normalized_template_overrides": normalized,
        "example": {
            "backend": requested,
            **(
                {
                    "backend_config": {
                        "model": selected_model,
                        "head": str(getattr(backend_config, "head", "") or ""),
                    }
                }
                if requested == "mace" and selected_model
                else {}
            ),
            **(
                {
                    "backend_config": {
                        "model": selected_model,
                        "defaults": {
                            "uma_task": str(getattr(backend_config, "defaults").uma_task),
                            "charge": int(getattr(backend_config, "defaults").charge),
                            "spin": int(getattr(backend_config, "defaults").spin),
                        },
                    }
                }
                if requested == "fairchem_uma" and selected_model
                else {}
            ),
        },
    }


def format_spec_error(exc: BaseException) -> dict[str, Any]:
    """Normalize Pydantic and resolver errors for the agent-facing spec tool."""

    errors_method = getattr(exc, "errors", None)
    if callable(errors_method):
        items: list[dict[str, Any]] = []
        for item in errors_method():
            loc = item.get("loc") or ()
            items.append(
                {
                    "path": ".".join(str(part) for part in loc),
                    "message": str(item.get("msg") or item),
                    "type": str(item.get("type") or "validation_error"),
                }
            )
        return {"errors": items}
    return {
        "errors": [
            {
                "path": "",
                "message": f"{type(exc).__name__}: {exc}",
                "type": "validation_error",
            }
        ]
    }


__all__ = [
    "MatterSimBackendConfig",
    "MaceBackendConfig",
    "MlffSpecValidationError",
    "MlffBackendProfile",
    "MlffBackendRegistry",
    "MolecularDynamicsTaskConfig",
    "NebTaskConfig",
    "OrbV3BackendConfig",
    "RelaxTaskConfig",
    "SinglePointTaskConfig",
    "UmaBackendConfig",
    "format_spec_error",
    "is_mlff_task",
    "mlff_operation_for_task",
    "resolve_mlff_template",
]
