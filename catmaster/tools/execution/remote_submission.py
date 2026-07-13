from __future__ import annotations

import json
import re
import shlex
import shutil
import time
from datetime import datetime
from pathlib import Path
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.machine_time_stats import append_machine_time_record, build_machine_time_record
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import current_run_dir, current_tool_audience, current_toolcall_key
from catmaster.tools.base import resolve_workspace_path, system_root, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    BatchDispatchRequest,
    TaskSpec,
    cleanup_dpdispatcher_transfer_archives,
    dispatch_submission,
    remote_context_from_receipt,
    remote_context_from_exception,
    remote_context_from_result,
    task_state_counts,
    write_dispatch_attempt_receipt,
)
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskConfig, TaskRegistry, format_template

_REPO_ROOT = Path(__file__).resolve().parents[3]
_PLACEHOLDER_RE = re.compile(r"\{([A-Za-z_][A-Za-z0-9_]*)\}")

_CONTROL_CONFIG_FIELDS = {"resources", "machine", "check_interval", "clean_remote", "overrides", "audience"}
_SAFE_RESOURCE_OVERRIDE_FIELDS = {
    "number_node",
    "cpu_per_node",
    "gpu_per_node",
    "queue_name",
    "group_size",
    "custom_flags",
    "source_list",
    "prepend_script",
}
_WORKER_RESOURCE_OVERRIDE_FIELDS = {
    "cpu_per_node",
    "gpu_per_node",
}
_FORBIDDEN_CONFIG_FIELDS = {
    "machines",
    "remote_profile",
    "remote_root",
    "local_root",
    "context_type",
    "batch_type",
    "hostname",
    "host",
    "port",
    "username",
    "password",
    "key_filename",
    "private_key",
    "token",
    "credential",
    "credentials",
}

_REMOTE_SUBMISSION_GUIDANCE: dict[str, Any] = {
    "remote_submission": (
        "Use for one prepared stage directory. `work_dir` is the stage itself, and the boot script "
        "runs directly in that stage cwd. A stage may contain internal batch inputs when the layout says so, "
        "for example MACE `input/` with many structures. If you have two or more independent prepared "
        "stages for the same task_name/template_overrides/config, prefer one remote_submission_batch parent root instead "
        "of multiple parallel remote_submission calls."
    ),
    "remote_submission_batch": (
        "Preferred when there are multiple independent prepared stages for the same task_name/template_overrides/config. "
        "Use a parent batch root whose first-level children are independent stages. "
        "The boot script runs once inside each first-level child directory. Each child is submitted "
        "as one DPDispatcher task with the same task_name, template_overrides, and config. No recursive nested "
        "discovery is performed."
    ),
    "registered_task_config": (
        "For registered task_name templates, normally omit config.resources and config.machine. "
        "The task's resource card owns the machine and environment; override only explicit sizing fields "
        "such as cpu_per_node or gpu_per_node when intentionally requested."
    ),
    "template_overrides": (
        "For registered task_name templates, pass command-template overrides through template_overrides "
        "using only keys declared by the task catalog. Use this for method-critical values such as MACE head/fmax or UMA "
        "uma_task/spin. Do not patch copied task_script files or sitecustomize to change template defaults."
    ),
}


def _catalog_task_hint(task_name: str) -> str:
    if task_name.startswith("mace_"):
        return (
            "MACE *_dir layouts are usually one stage with input/ and output/; use remote_submission "
            "for one such stage even when input/ contains many structures. Use remote_submission_batch "
            "only for a parent root containing multiple independent MACE stage directories."
        )
    if task_name.startswith("uma_"):
        return (
            "UMA *_dir layouts are usually one stage with input/ and output/; use remote_submission "
            "for one such stage even when input/ contains many structures. Use remote_submission_batch "
            "only for a parent root containing multiple independent UMA stage directories."
        )
    if task_name.startswith("vasp_"):
        return (
            "One VASP stage is one complete calculation folder. For multiple scheduler jobs, make a "
            "parent root with one complete VASP folder per first-level child and use remote_submission_batch."
        )
    if task_name in {"xtb_run", "crest_run", "orca_execute", "cp2k_execute", "lammps_execute"}:
        return (
            "One stage is one prepared job directory for this executable. Use remote_submission_batch "
            "only when the parent root contains one prepared job directory per first-level child."
        )
    return "Use remote_submission for one matching stage; use remote_submission_batch for first-level child stages."


def _catalog_content(*, tasks: list[dict[str, Any]]) -> str:
    lines = [
        f"Available remote tasks: {len(tasks)}",
        "Submission decision:",
        "- remote_submission: work_dir is one prepared stage; the boot script runs directly in that directory.",
        "- remote_submission_batch: work_dir is a parent root; the boot script runs once inside each first-level child directory.",
        "- If two or more independent stages share the same task_name, template_overrides, and config, prefer one remote_submission_batch call over multiple parallel remote_submission calls.",
        "- Do not use remote_submission_batch just because a single stage contains many inputs; MACE *_dir stages commonly batch internally under input/.",
        "- For registered task_name templates, omit config.resources/config.machine unless an explicit sizing override is needed.",
        "- For registered task_name templates, use only catalog-declared template_overrides keys for command defaults such as MACE head/fmax or UMA spin; do not patch copied task_script/sitecustomize files.",
        "Tasks:",
    ]
    for item in tasks:
        details = [str(item["description"]).rstrip(".")]
        resources = item.get("resources")
        if isinstance(resources, dict) and resources.get("resources"):
            details.append(f"resources={resources['resources']}")
        defaults = item.get("template_defaults")
        if isinstance(defaults, dict) and defaults:
            details.append(f"template_defaults={_compact_template_defaults(defaults)}")
        if item.get("layout_ref"):
            details.append(f"layout_ref={item['layout_ref']}")
        lines.append(f"- {item['task_name']}: " + "; ".join(details))
    return "\n".join(lines)


def _compact_template_defaults(defaults: dict[str, Any]) -> str:
    parts: list[str] = []
    for key, value in defaults.items():
        if isinstance(value, str):
            rendered = value
        else:
            rendered = json.dumps(value, ensure_ascii=False)
        parts.append(f"{key}={rendered}")
    return "{" + ", ".join(parts) + "}"


def _parse_bool(value: Any, *, field: str) -> bool:
    if isinstance(value, bool):
        return value
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"1", "true", "t", "yes", "y", "on"}:
            return True
        if text in {"0", "false", "f", "no", "n", "off"}:
            return False
    raise ValueError(f"config.{field} must be a boolean.")


def _parse_positive_int(value: Any, *, field: str) -> int:
    try:
        parsed = int(value)
    except Exception as exc:
        raise ValueError(f"config.{field} must be a positive integer.") from exc
    if parsed <= 0:
        raise ValueError(f"config.{field} must be a positive integer.")
    return parsed


class RemoteSubmissionInput(BaseModel):
    """[remote/execute] Submit exactly one prepared stage directory as one DPDispatcher task.

    Use this when `work_dir` itself matches the selected task layout. The boot
    script runs directly with `work_dir` as cwd. If that layout internally accepts
    many inputs, such as a MACE stage containing `input/` with many structures,
    it is still one stage and should use this tool. For two or more independent
    prepared stages that share the same task_name/template_overrides/config, prefer
    remote_submission_batch instead of parallel remote_submission calls.
    """

    model_config = ConfigDict(extra="forbid")

    work_dir: str = Field(
        ...,
        description=(
            "Workspace-relative prepared stage directory. For remote_submission this is the stage itself, "
            "not a parent batch root. The remote task runs with this directory as cwd."
        ),
    )
    task_name: str = Field(
        "",
        description="Registered remote task template name. Leave empty when using boot_script. Mutually exclusive with boot_script.",
    )
    boot_script: str = Field(
        "",
        description=(
            "Workspace-relative custom boot script path. Leave empty when using task_name. Uses the default general CPU resource card "
            "unless config.resources selects another visible card such as general_gpu."
        ),
    )
    template_overrides: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Registered task command-template overrides using only keys listed by get_avail_remote_task. "
            "For example, {'head': 'omol'} for mace_relax_dir or "
            "{'uma_task': 'omol', 'spin': 3} for uma_relax_dir. "
            "This does not change resource cards; use config only for allowed resource/submission controls."
        ),
    )
    config: dict[str, Any] = Field(
        default_factory=dict,
        description=(
            "Optional resource-card override such as {'resources': 'general_gpu'}, plus safe resource "
            "overrides and submission controls."
        ),
    )

    @model_validator(mode="before")
    @classmethod
    def _coerce_legacy_null_objects(cls, data: Any) -> Any:
        if not isinstance(data, dict):
            return data
        normalized = dict(data)
        for key in ("template_overrides", "config"):
            if normalized.get(key) is None:
                normalized[key] = {}
        for key in ("task_name", "boot_script"):
            if normalized.get(key) is None:
                normalized[key] = ""
        return normalized

    @model_validator(mode="after")
    def _exactly_one_task_source(self) -> "RemoteSubmissionInput":
        has_task = bool(str(self.task_name or "").strip())
        has_script = bool(str(self.boot_script or "").strip())
        if has_task == has_script:
            raise ValueError("Exactly one of task_name or boot_script is required.")
        if has_script and self.template_overrides:
            raise ValueError("template_overrides is only valid with a registered task_name.")
        return self


def _template_overrides(payload: RemoteSubmissionInput) -> dict[str, Any] | None:
    return dict(payload.template_overrides) if payload.template_overrides else None


class RemoteSubmissionBatchInput(RemoteSubmissionInput):
    """[remote/execute] Submit a parent batch root as multiple DPDispatcher tasks.

    Prefer this when there are two or more independent prepared stages for the
    same task_name/template_overrides/config. Use only when `work_dir` contains one
    first-level child directory per task and every child independently matches
    the same selected task layout. The boot script runs once inside each
    first-level child directory, not once in the parent. No recursive nested
    discovery is performed; the same `task_name`, template_overrides, and config are applied
    to every child.
    """

    work_dir: str = Field(
        ...,
        description=(
            "Workspace-relative parent batch root. Each first-level child directory is submitted as one "
            "independent stage for the same task_name/boot_script. Do not pass a single prepared stage here."
        ),
    )


class GetAvailRemoteTaskInput(BaseModel):
    """[remote/catalog] List remote task templates and single-vs-batch submission rules visible to the current worker."""

    model_config = ConfigDict(extra="forbid")

    return_resource: bool = Field(False, description="When true, include each task's default resource summary.")


class GetAvailResourcesInput(BaseModel):
    """[remote/catalog] List general custom-boot resource cards visible to the current worker."""

    model_config = ConfigDict(extra="forbid")


def _success(tool_name: str, *, content: str, data: dict[str, Any], warnings: list[str] | None = None, execution_time: float | None = None) -> tuple[str, dict[str, Any]]:
    artifact: dict[str, Any] = {"tool_name": tool_name, "data": data}
    if warnings:
        artifact["warnings"] = warnings
    if execution_time is not None:
        artifact["execution_time"] = execution_time
    return content, artifact


def _fail(tool_name: str, *, message: str, data: dict[str, Any] | None = None, error_code: str = "") -> None:
    lines = [str(message).strip()]
    for key in (
        "task_name",
        "work_dir_rel",
        "work_base",
        "resources",
        "duration_s",
        "remote_context_id",
        "submission_hash",
        "submitted_at",
        "updated_at",
        "receipt_rel",
    ):
        value = (data or {}).get(key)
        if value not in (None, "", [], {}):
            lines.append(f"{key}={value}")
    jobs = (data or {}).get("jobs")
    if isinstance(jobs, list):
        lines.append(f"jobs={len(jobs)}")
    job_status_counts = (data or {}).get("job_status_counts")
    if isinstance(job_status_counts, dict) and job_status_counts:
        lines.append(f"job_status_counts={json.dumps(job_status_counts, ensure_ascii=False, sort_keys=True)}")
    raise CatMasterToolExecutionError(
        tool_name=tool_name,
        public_message="\n".join(lines),
        artifact={"tool_name": tool_name, "data": data or {}},
        error_code=error_code,
    )


def _current_audience(config: dict[str, Any] | None = None) -> str:
    audience = current_tool_audience()
    if audience:
        return audience
    if isinstance(config, dict):
        return str(config.get("audience") or "").strip()
    return ""


def _render_template_values(
    cfg: TaskConfig | None,
    template_overrides: dict[str, Any] | None,
) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if cfg is not None:
        merged.update(dict(cfg.defaults or {}))
    if template_overrides:
        accepted = set(cfg.defaults or {}) if cfg is not None else set()
        unknown = sorted(str(key) for key in template_overrides if key not in accepted)
        if unknown:
            accepted_text = ", ".join(sorted(accepted)) or "none"
            raise ValueError(
                f"Unknown template_overrides key(s): {', '.join(unknown)}. "
                f"Accepted keys: {accepted_text}."
            )
        merged.update(dict(template_overrides))
    out: dict[str, Any] = {}
    for key, value in merged.items():
        if isinstance(value, bool):
            out[str(key)] = "true" if value else "false"
        elif value is None:
            out[str(key)] = ""
        elif isinstance(value, (dict, list)):
            out[str(key)] = json.dumps(value, ensure_ascii=False, separators=(",", ":"))
        else:
            out[str(key)] = value
    return out


def _shell_quote_params(ctx: dict[str, Any]) -> dict[str, str]:
    return {str(key): shlex.quote(str(value)) for key, value in ctx.items()}


def _unresolved_placeholders(command: str) -> list[str]:
    return sorted(set(_PLACEHOLDER_RE.findall(command or "")))


def _resolve_builtin_boot_script(cfg: TaskConfig) -> Path | None:
    raw = str(cfg.boot_script or "").strip()
    if not raw:
        return None
    path = Path(raw)
    if path.is_absolute():
        candidate = path
    else:
        candidate = _REPO_ROOT / path
    if not candidate.is_file():
        raise FileNotFoundError(f"Missing configured boot_script for task: {raw}")
    return candidate


def _copy_boot_script(script_src: Path | None, stage_dir: Path) -> str:
    if script_src is None:
        return ""
    dst = stage_dir / "task_script" / script_src.name
    dst.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(script_src, dst)
    return dst.relative_to(stage_dir).as_posix()


def _copy_task_script_forward_dependencies(
    *,
    script_src: Path | None,
    stage_dir: Path,
    forward_files: list[str],
) -> None:
    if script_src is None:
        return
    for rel in forward_files:
        path = Path(str(rel))
        if path.is_absolute() or len(path.parts) != 2 or path.parts[0] != "task_script":
            continue
        dst = stage_dir / path
        if dst.exists():
            continue
        src = script_src.parent / path.name
        if src.is_file():
            dst.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(src, dst)


def _custom_boot_command(script_rel: str) -> str:
    if script_rel.endswith(".py"):
        return f"python {script_rel}"
    return f"bash {script_rel}"


def _task_work_path(stage_name: str | None, cfg_task_work_path: str) -> str:
    suffix = str(cfg_task_work_path or ".").strip() or "."
    if stage_name is None:
        return suffix
    if suffix in {".", "./"}:
        return stage_name
    return f"{stage_name}/{suffix}"


def _task_and_script(
    *,
    task_name: str | None,
    boot_script: str | None,
    audience: str,
) -> tuple[TaskConfig | None, str, Path | None]:
    if task_name:
        registry = TaskRegistry()
        cfg = registry.get(task_name)
        if not registry.task_visible_to(task_name, audience=audience):
            raise PermissionError(f"Remote task '{task_name}' is not visible to audience '{audience}'.")
        return cfg, task_name, _resolve_builtin_boot_script(cfg)
    script_src = resolve_workspace_path(str(boot_script or ""), must_exist=True)
    if not script_src.is_file():
        raise FileNotFoundError(f"boot_script is not a file: {script_src}")
    return None, "", script_src


def _visible_resource_names(*, audience: str, registry: TaskRegistry, register: MachineRegister) -> set[str]:
    if not audience:
        return set(register.resources.keys())
    names: set[str] = set()
    for name, cfg in register.resources.items():
        audiences = cfg.get("audiences")
        if isinstance(audiences, list) and audience in {str(item) for item in audiences}:
            names.add(name)
    for cfg in registry.list_tasks(audience=audience).values():
        if cfg.resources:
            names.add(str(cfg.resources))
    return names


def _visible_machine_names(*, audience: str, registry: TaskRegistry, register: MachineRegister) -> set[str]:
    if not audience:
        return set(register.machines.keys())
    names: set[str] = set()
    for resource_name in _visible_resource_names(audience=audience, registry=registry, register=register):
        try:
            machine_name = str(register.get_resources(resource_name).get("machine") or "").strip()
        except Exception:
            continue
        if machine_name:
            names.add(machine_name)
    return names


def _resource_allowed(resources_key: str, *, audience: str, registry: TaskRegistry, register: MachineRegister) -> bool:
    return resources_key in _visible_resource_names(audience=audience, registry=registry, register=register)


def _machine_allowed(machine_key: str, *, audience: str, registry: TaskRegistry, register: MachineRegister) -> bool:
    return machine_key in _visible_machine_names(audience=audience, registry=registry, register=register)


def _default_custom_boot_resources(*, audience: str, registry: TaskRegistry, register: MachineRegister) -> str:
    candidates: list[str] = []
    for name, cfg in register.resources.items():
        if not cfg.get("default_for_custom_boot"):
            continue
        if _resource_allowed(str(name), audience=audience, registry=registry, register=register):
            candidates.append(str(name))
    if len(candidates) == 1:
        return candidates[0]
    if len(candidates) > 1:
        raise ValueError(
            "config.resources is required when multiple default custom boot resources are visible: "
            + ", ".join(sorted(candidates))
        )
    return ""


def _extract_submission_config(config: dict[str, Any] | None, *, audience: str = "") -> tuple[dict[str, Any], int, bool]:
    raw = dict(config or {})
    raw.pop("audience", None)
    resources = raw.pop("resources", None)
    machine = raw.pop("machine", None)
    forbidden = sorted(key for key in raw if key in _FORBIDDEN_CONFIG_FIELDS)
    if forbidden:
        raise ValueError(f"Forbidden remote config field(s): {', '.join(forbidden)}")

    nested = raw.pop("overrides", None)
    overrides: dict[str, Any] = {}
    if nested is not None:
        if not isinstance(nested, dict):
            raise ValueError("config.overrides must be an object when provided.")
        nested_forbidden = sorted(key for key in nested if key in _FORBIDDEN_CONFIG_FIELDS)
        if nested_forbidden:
            raise ValueError(f"Forbidden remote config override field(s): {', '.join(nested_forbidden)}")
        overrides.update(nested)

    check_interval = _parse_positive_int(raw.pop("check_interval", 30), field="check_interval")
    clean_remote = _parse_bool(raw.pop("clean_remote", False), field="clean_remote")
    safe_fields = _WORKER_RESOURCE_OVERRIDE_FIELDS if audience else _SAFE_RESOURCE_OVERRIDE_FIELDS
    for key, value in raw.items():
        if key not in safe_fields:
            raise ValueError(f"Unsupported remote config field: {key}")
        overrides[key] = value
    for key in overrides:
        if key not in safe_fields:
            raise ValueError(f"Unsupported remote resource override field: {key}")
    if resources not in (None, ""):
        overrides["_resources_key"] = str(resources)
    if machine not in (None, ""):
        overrides["_machine_key"] = str(machine)
    return overrides, check_interval, clean_remote


def _register_with_resource_override(
    *,
    resources_key: str,
    overrides: dict[str, Any],
    base_resource_cfg: dict[str, Any] | None = None,
) -> MachineRegister:
    register = MachineRegister()
    register.resources = {name: dict(cfg) for name, cfg in register.resources.items()}
    register.machines = {name: dict(cfg) for name, cfg in register.machines.items()}
    resource_cfg = dict(base_resource_cfg if base_resource_cfg is not None else register.get_resources(resources_key))
    for key, value in overrides.items():
        if key in {"_resources_key", "_machine_key"}:
            continue
        resource_cfg[key] = value
    register.resources[resources_key] = resource_cfg
    return register


def _resource_kind(resource_cfg: dict[str, Any]) -> str:
    return str(resource_cfg.get("kind") or "").strip()


def _resource_capabilities(resource_cfg: dict[str, Any]) -> set[str]:
    raw = resource_cfg.get("capabilities") or []
    if isinstance(raw, str):
        raw = [raw]
    return {str(item).strip() for item in raw if str(item).strip()}


def _task_requires(cfg: TaskConfig | None) -> set[str]:
    if cfg is None:
        return set()
    raw = getattr(cfg, "requires", []) or []
    return {str(item).strip() for item in raw if str(item).strip()}


def _resource_allows_custom_boot(resource_cfg: dict[str, Any]) -> bool:
    kind = _resource_kind(resource_cfg)
    return bool(resource_cfg.get("allow_custom_boot")) or kind.startswith("general")


def _assert_resource_matches_task(*, cfg: TaskConfig | None, resource_name: str, resource_cfg: dict[str, Any]) -> None:
    required = _task_requires(cfg)
    if not required:
        return
    available = _resource_capabilities(resource_cfg)
    missing = sorted(required - available)
    if missing:
        raise ValueError(
            f"Remote resources '{resource_name}' do not satisfy task requirement(s): {', '.join(missing)}"
        )


def _resolve_resources_spec(
    *,
    cfg: TaskConfig | None,
    config_overrides: dict[str, Any],
    audience: str,
    registry: TaskRegistry,
    register: MachineRegister,
) -> tuple[str, dict[str, Any]]:
    override_key = str(config_overrides.get("_resources_key") or "").strip()
    machine_key = str(config_overrides.get("_machine_key") or "").strip()
    if machine_key and audience:
        raise PermissionError("config.machine is not available to worker tools; use config.resources resource cards.")
    default_key = str(cfg.resources or "").strip() if cfg is not None else ""
    if cfg is not None and audience and override_key and override_key != default_key:
        raise PermissionError(
            "Registered remote tasks use their task-bound resource card; override cpu_per_node/gpu_per_node only."
        )
    resources_key = override_key or default_key
    if resources_key:
        resource_cfg = dict(register.get_resources(resources_key))
        if not _resource_allowed(resources_key, audience=audience, registry=registry, register=register):
            raise PermissionError(f"Remote resources '{resources_key}' are not visible to audience '{audience}'.")
    elif machine_key:
        safe_machine = re.sub(r"[^A-Za-z0-9_.-]+", "_", machine_key).strip("._") or "machine"
        resources_key = f"custom_{safe_machine}"
        resource_cfg = {
            "machine": machine_key,
            "number_node": 1,
            "cpu_per_node": 1,
            "group_size": 1,
            "allow_custom_boot": True,
        }
    elif cfg is None:
        resources_key = _default_custom_boot_resources(audience=audience, registry=registry, register=register)
        if not resources_key:
            raise ValueError("config.resources is required when using a custom boot_script without a visible default resource card.")
        resource_cfg = dict(register.get_resources(resources_key))
    else:
        raise ValueError("config.resources is required because this task has no default resource card.")

    if machine_key:
        register.get_machine(machine_key)
        if not _machine_allowed(machine_key, audience=audience, registry=registry, register=register):
            raise PermissionError(f"Remote machine '{machine_key}' is not visible to audience '{audience}'.")
        resource_cfg["machine"] = machine_key
    if cfg is None and not _resource_allows_custom_boot(resource_cfg):
        raise PermissionError(f"Remote resources '{resources_key}' are not available for custom boot_script submissions.")
    _assert_resource_matches_task(cfg=cfg, resource_name=resources_key, resource_cfg=resource_cfg)
    return resources_key, resource_cfg


def _build_task_spec(
    *,
    cfg: TaskConfig | None,
    task_name: str,
    boot_script_src: Path | None,
    stage_dir: Path,
    stage_name: str | None,
    template_overrides: dict[str, Any] | None,
) -> TaskSpec:
    script_rel = _copy_boot_script(boot_script_src, stage_dir)
    if cfg is None:
        command = _custom_boot_command(script_rel)
        forward_files = ["*"]
        backward_files = ["*"]
        task_work_path = _task_work_path(stage_name, ".")
    else:
        ctx = _render_template_values(cfg, template_overrides)
        rendered = render_task_fields(cfg, ctx, stage_dir)
        command = format_template(cfg.command, _shell_quote_params(ctx))
        missing = _unresolved_placeholders(command)
        if missing:
            raise ValueError(f"Missing template values for task '{task_name}': {', '.join(missing)}")
        forward_files = list(rendered["forward_files"])
        _copy_task_script_forward_dependencies(
            script_src=boot_script_src,
            stage_dir=stage_dir,
            forward_files=forward_files,
        )
        if script_rel and script_rel not in forward_files and "*" not in forward_files:
            forward_files.append(script_rel)
        backward_files = list(rendered["backward_files"])
        task_work_path = _task_work_path(stage_name, str(rendered["task_work_path"]))
    return TaskSpec(
        command=command,
        task_work_path=task_work_path,
        forward_files=forward_files,
        backward_files=backward_files,
    )


def _now_iso() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _safe_submission_token(work_dir: Path) -> str:
    rel = workspace_relpath(work_dir)
    token = re.sub(r"[^A-Za-z0-9_.-]+", "_", rel).strip("._")
    return (token or work_dir.name or "stage")[:96]


def _prepare_dispatch_workspace(work_dir: Path, *, tool_name: str) -> tuple[Path, str, Path]:
    staging_root = system_root() / "dpdispatcher" / "staging"
    staging_root.mkdir(parents=True, exist_ok=True)
    work_base = f"{tool_name}_{_safe_submission_token(work_dir)}_{uuid4().hex[:10]}"
    local_root = staging_root / work_base
    dispatch_dir = local_root / work_base
    if local_root.exists():
        shutil.rmtree(local_root)
    shutil.copytree(work_dir, dispatch_dir, symlinks=True)
    return local_root, work_base, dispatch_dir


def _sync_dispatch_workspace_back(dispatch_dir: Path, work_dir: Path) -> None:
    if not dispatch_dir.is_dir():
        return
    shutil.copytree(dispatch_dir, work_dir, dirs_exist_ok=True, symlinks=True)


def _record_machine_time(
    *,
    status: str,
    tool_name: str,
    work_dir: Path,
    task_name: str,
    work_base: str,
    tasks: list[TaskSpec],
    resources_key: str,
    register: MachineRegister,
    result: Any = None,
    remote_context: dict[str, Any] | None = None,
    error: str = "",
) -> None:
    run_dir = current_run_dir()
    if not run_dir:
        return
    try:
        resource_cfg = dict(register.get_resources(resources_key))
        record = build_machine_time_record(
            status=status,
            tool_name=tool_name,
            task_name=task_name,
            work_dir_rel=workspace_relpath(work_dir),
            work_base=work_base,
            resources_key=resources_key,
            resource_cfg=resource_cfg,
            task_count=len(tasks),
            toolcall_id=current_toolcall_key(),
            result=result,
            remote_context=remote_context,
            error=error,
        )
        append_machine_time_record(run_dir, record)
    except Exception:
        return


def _submit(
    *,
    tool_name: str,
    work_dir: Path,
    task_name: str,
    cfg: TaskConfig | None,
    tasks: list[TaskSpec],
    resources_key: str,
    register: MachineRegister,
    check_interval: int,
    clean_remote: bool,
) -> tuple[str, dict[str, Any]]:
    local_root, work_base, dispatch_dir = _prepare_dispatch_workspace(work_dir, tool_name=tool_name)
    req = BatchDispatchRequest(
        machine=str(register.get_resources(resources_key).get("machine") or ""),
        resources=resources_key,
        work_base=work_base,
        local_root=str(local_root),
        tasks=tasks,
        forward_common_files=list(cfg.forward_common_files if cfg is not None else []),
        backward_common_files=list(cfg.backward_common_files if cfg is not None else []),
        clean_remote=clean_remote,
        check_interval=check_interval,
        tool_name=tool_name,
    )
    dispatch_error: Exception | None = None
    result = None
    submitted_at = _now_iso()
    dispatch_started = time.time()
    try:
        result = dispatch_submission(req, register=register)
    except Exception as exc:
        dispatch_error = exc
    try:
        cleanup_dpdispatcher_transfer_archives(dispatch_dir)
        _sync_dispatch_workspace_back(dispatch_dir, work_dir)
    except Exception as sync_exc:
        remote_context = remote_context_from_exception(dispatch_error)
        if dispatch_error is None:
            sync_error = RuntimeError(f"local result sync failed: {sync_exc}")
        else:
            sync_error = RuntimeError(f"{dispatch_error}; local result sync failed: {sync_exc}")
        if remote_context:
            setattr(sync_error, "remote_context", remote_context)
        sync_error.__cause__ = sync_exc
        dispatch_error = sync_error
    if dispatch_error is not None:
        remote_context = remote_context_from_exception(dispatch_error)
        if remote_context and "duration_s" not in remote_context:
            remote_context["duration_s"] = round(max(0.0, time.time() - dispatch_started), 3)
        if not remote_context:
            receipt = write_dispatch_attempt_receipt(
                tool_name=tool_name,
                work_base=work_base,
                task_name=task_name,
                work_dir_rel=workspace_relpath(work_dir),
                resources=resources_key,
                submitted_at=submitted_at,
                error=f"{type(dispatch_error).__name__}: {dispatch_error}",
                duration_s=time.time() - dispatch_started,
            )
            remote_context = remote_context_from_receipt(receipt, include_jobs=True)
        data = {
            "task_name": task_name,
            "work_dir_rel": workspace_relpath(work_dir),
            "work_base": work_base,
            "resources": resources_key,
            **remote_context,
        }
        _record_machine_time(
            status="failed",
            tool_name=tool_name,
            work_dir=work_dir,
            task_name=task_name,
            work_base=work_base,
            tasks=tasks,
            resources_key=resources_key,
            register=register,
            remote_context=remote_context,
            error=f"{type(dispatch_error).__name__}: {dispatch_error}",
        )
        _fail(tool_name, message=f"DPDispatcher submission failed: {dispatch_error}", data=data, error_code="dispatch_failed")

    states = result.task_states if result else []
    state_counts = dict(getattr(result, "task_state_counts", None) or task_state_counts(states))
    data = {
        "task_name": task_name,
        "work_dir_rel": workspace_relpath(work_dir),
        "work_base": result.work_base if result else work_dir.name,
        "resources": resources_key,
        "task_count": len(tasks),
        "task_state_counts": state_counts,
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        **remote_context_from_result(result),
    }
    _record_machine_time(
        status="success",
        tool_name=tool_name,
        work_dir=work_dir,
        task_name=task_name,
        work_base=data["work_base"],
        tasks=tasks,
        resources_key=resources_key,
        register=register,
        result=result,
        remote_context=remote_context_from_result(result),
    )
    content = (
        f"{tool_name} completed.\n"
        f"task_name={task_name or 'custom_boot_script'} tasks={len(tasks)} resources={resources_key}\n"
        f"task_state_counts={json.dumps(state_counts, ensure_ascii=False, sort_keys=True)}\n"
        f"work_dir_rel={data['work_dir_rel']} remote_context_id={data.get('remote_context_id', '')} "
        f"duration_s={data.get('duration_s', '')}"
    )
    return _success(tool_name, content=content, data=data, execution_time=result.duration_s if result else None)


def _prepare_common(payload: RemoteSubmissionInput) -> tuple[Path, str, TaskConfig | None, Path | None, str, MachineRegister, int, bool]:
    config = dict(payload.config or {})
    audience = _current_audience(config)
    cfg, resolved_task_name, boot_script_src = _task_and_script(
        task_name=str(payload.task_name or "").strip(),
        boot_script=str(payload.boot_script or "").strip(),
        audience=audience,
    )
    registry = TaskRegistry()
    base_register = MachineRegister()
    overrides, check_interval, clean_remote = _extract_submission_config(config, audience=audience)
    resources_key, resource_cfg = _resolve_resources_spec(
        cfg=cfg,
        config_overrides=overrides,
        audience=audience,
        registry=registry,
        register=base_register,
    )
    register = _register_with_resource_override(resources_key=resources_key, overrides=overrides, base_resource_cfg=resource_cfg)
    work_dir = resolve_workspace_path(payload.work_dir, must_exist=True)
    if not work_dir.is_dir():
        raise NotADirectoryError(f"work_dir is not a directory: {work_dir}")
    return work_dir, resolved_task_name, cfg, boot_script_src, resources_key, register, check_interval, clean_remote


def remote_submission(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    request = RemoteSubmissionInput(**payload)
    try:
        work_dir, task_name, cfg, boot_script_src, resources_key, register, check_interval, clean_remote = _prepare_common(request)
        task = _build_task_spec(
            cfg=cfg,
            task_name=task_name,
            boot_script_src=boot_script_src,
            stage_dir=work_dir,
            stage_name=None,
            template_overrides=_template_overrides(request),
        )
        return _submit(
            tool_name="remote_submission",
            work_dir=work_dir,
            task_name=task_name,
            cfg=cfg,
            tasks=[task],
            resources_key=resources_key,
            register=register,
            check_interval=check_interval,
            clean_remote=clean_remote,
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail("remote_submission", message=f"{type(exc).__name__}: {exc}", data={"work_dir_rel": str(payload.get("work_dir") or "")}, error_code="remote_submission_failed")


def remote_submission_batch(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    request = RemoteSubmissionBatchInput(**payload)
    try:
        work_dir, task_name, cfg, boot_script_src, resources_key, register, check_interval, clean_remote = _prepare_common(request)
        children = sorted(path for path in work_dir.iterdir() if path.is_dir())
        if not children:
            raise ValueError("remote_submission_batch requires first-level child task directories under work_dir.")
        tasks = [
            _build_task_spec(
                cfg=cfg,
                task_name=task_name,
                boot_script_src=boot_script_src,
                stage_dir=child,
                stage_name=child.name,
                template_overrides=_template_overrides(request),
            )
            for child in children
        ]
        return _submit(
            tool_name="remote_submission_batch",
            work_dir=work_dir,
            task_name=task_name,
            cfg=cfg,
            tasks=tasks,
            resources_key=resources_key,
            register=register,
            check_interval=check_interval,
            clean_remote=clean_remote,
        )
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        _fail("remote_submission_batch", message=f"{type(exc).__name__}: {exc}", data={"work_dir_rel": str(payload.get("work_dir") or "")}, error_code="remote_submission_batch_failed")


def _resource_summary(name: str, cfg: dict[str, Any], *, register: MachineRegister) -> dict[str, Any]:
    out: dict[str, Any] = {
        "resources": name,
        "kind": cfg.get("kind") or ("general" if str(name).startswith("general_") else "domain"),
    }
    if cfg.get("description"):
        out["description"] = cfg.get("description")
    for key in ("cpu_per_node", "gpu_per_node", "default_for_custom_boot"):
        if key in cfg:
            out[key] = cfg.get(key)
    return out


def _resource_visible_in_general_catalog(cfg: dict[str, Any]) -> bool:
    return _resource_allows_custom_boot(cfg)


def get_avail_remote_task(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    request = GetAvailRemoteTaskInput(**payload)
    audience = _current_audience()
    registry = TaskRegistry()
    register = MachineRegister()
    tasks: list[dict[str, Any]] = []
    for name, cfg in sorted(registry.list_tasks(audience=audience).items()):
        item: dict[str, Any] = {
            "task_name": name,
            "description": cfg.description,
            "layout_ref": cfg.layout_ref,
            "submission_hint": _catalog_task_hint(name),
            "template_defaults": dict(cfg.defaults),
            "template_override_keys": list(cfg.defaults.keys()),
        }
        if request.return_resource and cfg.resources:
            resources_cfg = register.get_resources(cfg.resources)
            item["resources"] = _resource_summary(cfg.resources, resources_cfg, register=register)
        tasks.append(item)
    data = {"audience": audience, "submission_guidance": dict(_REMOTE_SUBMISSION_GUIDANCE), "tasks": tasks}
    content = _catalog_content(tasks=tasks)
    return _success("get_avail_remote_task", content=content, data=data)


def get_avail_resources(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    _ = GetAvailResourcesInput(**payload)
    audience = _current_audience()
    registry = TaskRegistry()
    register = MachineRegister()
    visible = _visible_resource_names(audience=audience, registry=registry, register=register)
    resources = [
        _resource_summary(name, register.get_resources(name), register=register)
        for name in sorted(visible)
        if _resource_visible_in_general_catalog(register.get_resources(name))
    ]
    data = {"audience": audience, "resources": resources}
    if resources:
        content_lines = [
            "Available general remote resources: "
            + ", ".join(str(item["resources"]) for item in resources)
        ]
        for item in resources:
            details: list[str] = []
            if item.get("description"):
                details.append(str(item["description"]).rstrip("."))
            for key in ("kind", "cpu_per_node", "gpu_per_node", "default_for_custom_boot"):
                if key in item:
                    value = item[key]
                    if isinstance(value, bool):
                        value = str(value).lower()
                    details.append(f"{key}={value}")
            content_lines.append(f"- {item['resources']}: " + "; ".join(details))
        content = "\n".join(content_lines)
    else:
        content = "Available general remote resources: none"
    return _success("get_avail_resources", content=content, data=data)


__all__ = [
    "RemoteSubmissionInput",
    "RemoteSubmissionBatchInput",
    "GetAvailRemoteTaskInput",
    "GetAvailResourcesInput",
    "remote_submission",
    "remote_submission_batch",
    "get_avail_remote_task",
    "get_avail_resources",
]
