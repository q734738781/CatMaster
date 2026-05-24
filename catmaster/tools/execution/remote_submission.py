from __future__ import annotations

import json
import re
import shutil
from collections import Counter
from pathlib import Path
from typing import Any

from pydantic import BaseModel, ConfigDict, Field, model_validator

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.runtime.tool_runtime import current_tool_audience
from catmaster.tools.base import resolve_workspace_path, workspace_relpath
from catmaster.tools.execution.dpdispatcher_runner import (
    BatchDispatchRequest,
    TaskSpec,
    dispatch_submission,
    remote_context_from_exception,
    remote_context_from_result,
)
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_payloads import render_task_fields
from catmaster.tools.execution.task_registry import TaskConfig, TaskRegistry

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
    """[remote/execute] Submit one prepared stage directory through DPDispatcher."""

    model_config = ConfigDict(extra="forbid")

    work_dir: str = Field(..., description="Workspace-relative prepared stage directory. The remote task runs with this directory as cwd.")
    task_name: str | None = Field(None, description="Registered remote task template name. Mutually exclusive with boot_script.")
    boot_script: str | None = Field(None, description="Workspace-relative custom boot script path. Requires config.resources or config.machine.")
    params: dict[str, Any] | None = Field(None, description="Task-template parameters used only for command placeholders.")
    config: dict[str, Any] | None = Field(None, description="Optional resource preset or machine plus safe resource overrides and submission controls.")

    @model_validator(mode="after")
    def _exactly_one_task_source(self) -> "RemoteSubmissionInput":
        has_task = bool(str(self.task_name or "").strip())
        has_script = bool(str(self.boot_script or "").strip())
        if has_task == has_script:
            raise ValueError("Exactly one of task_name or boot_script is required.")
        return self


class RemoteSubmissionBatchInput(RemoteSubmissionInput):
    """[remote/execute] Submit each first-level child of a prepared batch stage directory as one DPDispatcher task."""


class GetAvailRemoteTaskInput(BaseModel):
    """[remote/catalog] List remote task templates visible to the current worker."""

    model_config = ConfigDict(extra="forbid")

    return_resource: bool = Field(False, description="When true, include each task's default resource summary.")


class GetAvailResourcesInput(BaseModel):
    """[remote/catalog] List remote resources visible to the current worker."""

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


def _normalize_params(cfg: TaskConfig | None, params: dict[str, Any] | None) -> dict[str, Any]:
    merged: dict[str, Any] = {}
    if cfg is not None:
        merged.update(dict(cfg.defaults or {}))
    if params:
        merged.update(dict(params))
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


def _extract_submission_config(config: dict[str, Any] | None) -> tuple[dict[str, Any], int, bool]:
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
    for key, value in raw.items():
        if key not in _SAFE_RESOURCE_OVERRIDE_FIELDS:
            raise ValueError(f"Unsupported remote config field: {key}")
        overrides[key] = value
    for key in overrides:
        if key not in _SAFE_RESOURCE_OVERRIDE_FIELDS:
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
    default_key = str(cfg.resources or "").strip() if cfg is not None else ""
    resources_key = override_key or default_key
    if resources_key:
        resource_cfg = dict(register.get_resources(resources_key))
        if not _resource_allowed(resources_key, audience=audience, registry=registry, register=register):
            raise PermissionError(f"Remote resources '{resources_key}' are not visible to audience '{audience}'.")
    elif machine_key:
        safe_machine = re.sub(r"[^A-Za-z0-9_.-]+", "_", machine_key).strip("._") or "machine"
        resources_key = f"custom_{safe_machine}"
        resource_cfg = {"machine": machine_key, "number_node": 1, "cpu_per_node": 1, "group_size": 1}
    else:
        raise ValueError("config.resources or config.machine is required when using a custom boot_script.")

    if machine_key:
        register.get_machine(machine_key)
        if not _machine_allowed(machine_key, audience=audience, registry=registry, register=register):
            raise PermissionError(f"Remote machine '{machine_key}' is not visible to audience '{audience}'.")
        resource_cfg["machine"] = machine_key
    return resources_key, resource_cfg


def _build_task_spec(
    *,
    cfg: TaskConfig | None,
    task_name: str,
    boot_script_src: Path | None,
    stage_dir: Path,
    stage_name: str | None,
    params: dict[str, Any] | None,
) -> TaskSpec:
    script_rel = _copy_boot_script(boot_script_src, stage_dir)
    if cfg is None:
        command = _custom_boot_command(script_rel)
        forward_files = ["*"]
        backward_files = ["*"]
        task_work_path = _task_work_path(stage_name, ".")
    else:
        ctx = _normalize_params(cfg, params)
        rendered = render_task_fields(cfg, ctx, stage_dir)
        command = str(rendered["command"])
        missing = _unresolved_placeholders(command)
        if missing:
            raise ValueError(f"Missing params for task '{task_name}': {', '.join(missing)}")
        forward_files = list(rendered["forward_files"])
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


def _state_counts(states: list[str]) -> dict[str, int]:
    return dict(Counter(str(item) for item in states))


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
    req = BatchDispatchRequest(
        machine=str(register.get_resources(resources_key).get("machine") or ""),
        resources=resources_key,
        work_base=work_dir.name,
        local_root=str(work_dir.parent),
        tasks=tasks,
        forward_common_files=list(cfg.forward_common_files if cfg is not None else []),
        backward_common_files=list(cfg.backward_common_files if cfg is not None else []),
        clean_remote=clean_remote,
        check_interval=check_interval,
        tool_name=tool_name,
    )
    dispatch_error: Exception | None = None
    result = None
    try:
        result = dispatch_submission(req, register=register)
    except Exception as exc:
        dispatch_error = exc
    if dispatch_error is not None:
        data = {
            "task_name": task_name,
            "work_dir_rel": workspace_relpath(work_dir),
            "work_base": work_dir.name,
            "resources": resources_key,
            **remote_context_from_exception(dispatch_error),
        }
        _fail(tool_name, message=f"DPDispatcher submission failed: {dispatch_error}", data=data, error_code="dispatch_failed")

    states = result.task_states if result else []
    data = {
        "task_name": task_name,
        "work_dir_rel": workspace_relpath(work_dir),
        "work_base": result.work_base if result else work_dir.name,
        "resources": resources_key,
        "task_count": len(tasks),
        "task_state_counts": _state_counts(states),
        "submission_dir": workspace_relpath(Path(result.submission_dir)) if result and result.submission_dir else "",
        **remote_context_from_result(result),
    }
    content = (
        f"{tool_name} completed.\n"
        f"task_name={task_name or 'custom_boot_script'} tasks={len(tasks)} resources={resources_key}\n"
        f"work_dir_rel={data['work_dir_rel']} remote_context_id={data.get('remote_context_id', '')}"
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
    overrides, check_interval, clean_remote = _extract_submission_config(config)
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
    params = RemoteSubmissionInput(**payload)
    try:
        work_dir, task_name, cfg, boot_script_src, resources_key, register, check_interval, clean_remote = _prepare_common(params)
        task = _build_task_spec(
            cfg=cfg,
            task_name=task_name,
            boot_script_src=boot_script_src,
            stage_dir=work_dir,
            stage_name=None,
            params=params.params,
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
    params = RemoteSubmissionBatchInput(**payload)
    try:
        work_dir, task_name, cfg, boot_script_src, resources_key, register, check_interval, clean_remote = _prepare_common(params)
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
                params=params.params,
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
    machine_name = str(cfg.get("machine") or "")
    machine_cfg = register.machines.get(machine_name, {})
    out: dict[str, Any] = {
        "resources": name,
        "machine": machine_name,
        "batch_type": machine_cfg.get("batch_type") or "",
        "context_type": machine_cfg.get("context_type") or "",
    }
    for key in ("number_node", "cpu_per_node", "gpu_per_node", "queue_name", "group_size", "custom_flags"):
        if key in cfg:
            out[key] = cfg.get(key)
    if cfg.get("source_list"):
        out["source_list_count"] = len(list(cfg.get("source_list") or []))
    return out


def get_avail_remote_task(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = GetAvailRemoteTaskInput(**payload)
    audience = _current_audience()
    registry = TaskRegistry()
    register = MachineRegister()
    tasks: list[dict[str, Any]] = []
    for name, cfg in sorted(registry.list_tasks(audience=audience).items()):
        item: dict[str, Any] = {
            "task_name": name,
            "description": cfg.description,
            "layout_ref": cfg.layout_ref,
        }
        if params.return_resource and cfg.resources:
            resources_cfg = register.get_resources(cfg.resources)
            item["resources"] = _resource_summary(cfg.resources, resources_cfg, register=register)
        tasks.append(item)
    data = {"audience": audience, "tasks": tasks}
    content = f"Available remote tasks: {len(tasks)}"
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
    ]
    data = {"audience": audience, "resources": resources}
    content = f"Available remote resources: {len(resources)}"
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
