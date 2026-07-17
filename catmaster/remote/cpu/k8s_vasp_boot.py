#!/usr/bin/env python3
# Code writing date: 2026-07-17
# Responsible agent: Codex, for CatMaster managed execution
# Implementation principle: keep DPDispatcher/SSH as the transfer and receipt layer while a blocking,
# idempotent kubectl bridge owns Kubernetes Job rendering, terminal-state detection, and diagnostics.
# Purpose: submit one prepared VASP stage through a remote Kubernetes Job and block until it completes
# or fails. Inputs come from the current stage and CATMASTER_K8S_* environment variables; diagnostics
# are written back into the stage for normal DPDispatcher result collection.

from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import time
from typing import Any, Iterable


BRIDGE_STATUS_FILE = "k8s_bridge_status.json"
JOB_STATUS_FILE = "k8s_job_status.json"
POD_STATUS_FILE = "k8s_pod_status.json"
POD_LOG_FILE = "k8s_pod.log"
JOB_DESCRIBE_FILE = "k8s_job_describe.txt"
EVENTS_FILE = "k8s_events.txt"
MANIFEST_FILE = "k8s_job_manifest.yaml"
APPLY_LOG_FILE = "k8s_apply.log"

_PLACEHOLDER_RE = re.compile(r"__[A-Z][A-Z0-9_]*__")
_DNS_LABEL_RE = re.compile(r"[^a-z0-9-]+")
_SAFE_EXECUTABLE_RE = re.compile(r"^[A-Za-z0-9_./+-]+$")
_K8S_QUANTITY_RE = re.compile(r"^[0-9]+(?:\.[0-9]+)?(?:m|Ki|Mi|Gi|Ti|Pi|Ei|k|K|M|G|T|P|E)?$")


class BridgeError(RuntimeError):
    """Configuration or kubectl failure that should fail the managed task cleanly."""


def _env(name: str, default: str = "") -> str:
    return str(os.environ.get(name, default)).strip()


def _positive_int(value: Any, *, name: str) -> int:
    try:
        parsed = int(str(value).strip())
    except Exception as exc:
        raise BridgeError(f"{name} must be a positive integer, got {value!r}") from exc
    if parsed <= 0:
        raise BridgeError(f"{name} must be a positive integer, got {value!r}")
    return parsed


def _positive_float(value: Any, *, name: str) -> float:
    try:
        parsed = float(str(value).strip())
    except Exception as exc:
        raise BridgeError(f"{name} must be positive, got {value!r}") from exc
    if parsed <= 0:
        raise BridgeError(f"{name} must be positive, got {value!r}")
    return parsed


def _safe_k8s_quantity(value: str, *, name: str) -> str:
    text = str(value).strip()
    if not _K8S_QUANTITY_RE.fullmatch(text):
        raise BridgeError(f"{name} is not a supported Kubernetes quantity: {value!r}")
    return text


def _safe_vasp_executable(value: str) -> str:
    text = str(value).strip()
    if not text or not _SAFE_EXECUTABLE_RE.fullmatch(text):
        raise BridgeError(f"VASP executable contains unsafe characters: {value!r}")
    return text


def build_job_name(*, prefix: str, token: str, work_dir: Path) -> str:
    """Build a stable DNS-label Job name for one DPDispatcher submission attempt."""

    safe_prefix = _DNS_LABEL_RE.sub("-", prefix.lower()).strip("-") or "cm-vasp"
    raw_hint = token.split(":")[-2] if token.count(":") >= 2 else work_dir.name
    safe_hint = _DNS_LABEL_RE.sub("-", raw_hint.lower()).strip("-")
    safe_hint = safe_hint[:28].strip("-") or "stage"
    digest_source = token or str(work_dir.resolve())
    digest = hashlib.sha256(digest_source.encode("utf-8")).hexdigest()[:12]
    suffix = f"-{safe_hint}-{digest}"
    max_prefix = max(1, 63 - len(suffix))
    name = f"{safe_prefix[:max_prefix].rstrip('-')}{suffix}".strip("-")
    if not name or len(name) > 63:
        raise BridgeError(f"Unable to build a valid Kubernetes Job name from prefix={prefix!r}")
    return name


def render_manifest(template_text: str, replacements: dict[str, str]) -> str:
    """Render a strict remote template and reject unresolved placeholders."""

    rendered = str(template_text)
    required = {"__JOB_NAME__", "__STRUCTURE_DIR__", "__MPI_RANKS__", "__VASP_EXE__"}
    missing_required = sorted(key for key in required if key not in rendered)
    if missing_required:
        raise BridgeError("Kubernetes template is missing required placeholders: " + ", ".join(missing_required))
    for key, value in replacements.items():
        rendered = rendered.replace(key, value)
    unresolved = sorted(set(_PLACEHOLDER_RE.findall(rendered)))
    if unresolved:
        raise BridgeError("Kubernetes template has unresolved placeholders: " + ", ".join(unresolved))
    return rendered


def terminal_condition(job_payload: dict[str, Any]) -> tuple[str, str]:
    """Return (terminal_state, message), where state is complete, failed, or empty."""

    conditions = (job_payload.get("status") or {}).get("conditions") or []
    for wanted, state in (("Failed", "failed"), ("Complete", "complete")):
        for condition in conditions:
            if str(condition.get("type")) != wanted or str(condition.get("status")).lower() != "true":
                continue
            message = str(condition.get("message") or condition.get("reason") or "").strip()
            return state, message
    return "", ""


class KubectlClient:
    def __init__(self, *, executable: str, namespace: str, context: str = "") -> None:
        self.executable = executable
        self.namespace = namespace
        self.context = context

    def command(self, args: Iterable[str]) -> list[str]:
        cmd = [self.executable]
        if self.context:
            cmd.extend(["--context", self.context])
        if self.namespace:
            cmd.extend(["--namespace", self.namespace])
        cmd.extend(str(arg) for arg in args)
        return cmd

    def run(self, args: Iterable[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        cmd = self.command(args)
        try:
            return subprocess.run(cmd, text=True, capture_output=True, check=False, timeout=timeout)
        except FileNotFoundError as exc:
            raise BridgeError(f"kubectl executable not found: {self.executable}") from exc
        except subprocess.TimeoutExpired as exc:
            raise BridgeError(f"kubectl command timed out: {shlex.join(cmd)}") from exc

    def require(self, args: Iterable[str], *, timeout: float = 60.0) -> subprocess.CompletedProcess[str]:
        proc = self.run(args, timeout=timeout)
        if proc.returncode != 0:
            detail = (proc.stderr or proc.stdout or "").strip()
            raise BridgeError(f"kubectl command failed ({proc.returncode}): {shlex.join(proc.args)}\n{detail}")
        return proc


def _write_text(path: Path, text: str) -> None:
    path.write_text(str(text), encoding="utf-8")


def _write_json(path: Path, value: Any) -> None:
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def _parse_json_output(proc: subprocess.CompletedProcess[str], *, action: str) -> dict[str, Any]:
    if proc.returncode != 0:
        detail = (proc.stderr or proc.stdout or "").strip()
        raise BridgeError(f"{action} failed ({proc.returncode}): {detail}")
    try:
        payload = json.loads(proc.stdout)
    except Exception as exc:
        raise BridgeError(f"{action} returned invalid JSON") from exc
    if not isinstance(payload, dict):
        raise BridgeError(f"{action} returned a non-object JSON payload")
    return payload


def collect_diagnostics(client: KubectlClient, *, job_name: str, work_dir: Path) -> None:
    """Best-effort collection; never hide the original terminal state."""

    commands = [
        (["get", "job", job_name, "-o", "json"], JOB_STATUS_FILE),
        (["get", "pods", "-l", f"job-name={job_name}", "-o", "json"], POD_STATUS_FILE),
        (["logs", f"job/{job_name}", "--all-containers=true", "--prefix=true"], POD_LOG_FILE),
        (["describe", "job", job_name], JOB_DESCRIBE_FILE),
        (
            [
                "get",
                "events",
                "--field-selector",
                f"involvedObject.kind=Job,involvedObject.name={job_name}",
                "--sort-by=.lastTimestamp",
            ],
            EVENTS_FILE,
        ),
    ]
    for args, filename in commands:
        try:
            proc = client.run(args, timeout=60)
            content = proc.stdout
            if proc.returncode != 0:
                content += f"\n[kubectl returncode={proc.returncode}]\n{proc.stderr}"
            _write_text(work_dir / filename, content)
        except Exception as exc:
            _write_text(work_dir / filename, f"diagnostic collection failed: {type(exc).__name__}: {exc}\n")


def _bridge_status(
    *,
    job_name: str,
    namespace: str,
    phase: str,
    started_at: float,
    detail: str = "",
) -> dict[str, Any]:
    return {
        "job_name": job_name,
        "namespace": namespace,
        "phase": phase,
        "detail": detail,
        "started_at": started_at,
        "updated_at": time.time(),
        "elapsed_seconds": round(max(0.0, time.time() - started_at), 3),
    }


def run_bridge(args: argparse.Namespace) -> int:
    work_dir = Path.cwd().resolve()
    template_path = Path(args.template).expanduser()
    if not template_path.is_absolute():
        template_path = (work_dir / template_path).resolve()
    if not template_path.is_file():
        raise BridgeError(f"Kubernetes template does not exist: {template_path}")

    mpi_ranks = _positive_int(args.mpi_ranks, name="mpi_ranks")
    cpu_limit = _positive_int(args.cpu_limit or mpi_ranks + 1, name="cpu_limit")
    if cpu_limit < mpi_ranks:
        raise BridgeError(f"cpu_limit ({cpu_limit}) cannot be smaller than mpi_ranks ({mpi_ranks})")
    memory_request = _safe_k8s_quantity(args.memory_request, name="memory_request")
    memory_limit = _safe_k8s_quantity(args.memory_limit, name="memory_limit")
    poll_interval = _positive_float(args.poll_interval, name="poll_interval")
    timeout_seconds = _positive_float(args.timeout_seconds, name="timeout_seconds")
    vasp_exe = _safe_vasp_executable(args.vasp_exe)
    namespace = str(args.namespace).strip() or "default"
    token = _env("CM_DPDISPATCHER_SUBMISSION_TOKEN")
    job_name = build_job_name(prefix=args.job_prefix, token=token, work_dir=work_dir)

    replacements = {
        "__JOB_NAME__": job_name,
        "__STRUCTURE_DIR__": shlex.quote(str(work_dir)),
        "__MPI_RANKS__": str(mpi_ranks),
        "__CPU_REQUEST__": str(mpi_ranks),
        "__CPU_LIMIT__": str(cpu_limit),
        "__MEMORY_REQUEST__": memory_request,
        "__MEMORY_LIMIT__": memory_limit,
        "__VASP_EXE__": shlex.quote(vasp_exe),
    }
    rendered = render_manifest(template_path.read_text(encoding="utf-8"), replacements)
    manifest_path = work_dir / MANIFEST_FILE
    _write_text(manifest_path, rendered)

    client = KubectlClient(executable=args.kubectl, namespace=namespace, context=args.context)
    started_at = time.time()
    _write_json(
        work_dir / BRIDGE_STATUS_FILE,
        _bridge_status(job_name=job_name, namespace=namespace, phase="rendered", started_at=started_at),
    )

    if args.render_only:
        return 0

    apply_proc = client.run(["apply", "-f", str(manifest_path)], timeout=120)
    _write_text(work_dir / APPLY_LOG_FILE, (apply_proc.stdout or "") + (apply_proc.stderr or ""))
    if apply_proc.returncode != 0:
        raise BridgeError(f"kubectl apply failed ({apply_proc.returncode}): {(apply_proc.stderr or apply_proc.stdout).strip()}")

    _write_json(
        work_dir / BRIDGE_STATUS_FILE,
        _bridge_status(job_name=job_name, namespace=namespace, phase="submitted", started_at=started_at),
    )

    consecutive_errors = 0
    while True:
        elapsed = time.time() - started_at
        if elapsed > timeout_seconds:
            detail = f"Kubernetes Job exceeded bridge timeout of {timeout_seconds:g} seconds"
            _write_json(
                work_dir / BRIDGE_STATUS_FILE,
                _bridge_status(job_name=job_name, namespace=namespace, phase="timeout", started_at=started_at, detail=detail),
            )
            collect_diagnostics(client, job_name=job_name, work_dir=work_dir)
            raise BridgeError(detail)

        proc = client.run(["get", "job", job_name, "-o", "json"], timeout=60)
        if proc.returncode != 0:
            consecutive_errors += 1
            detail = (proc.stderr or proc.stdout or "").strip()
            _write_json(
                work_dir / BRIDGE_STATUS_FILE,
                _bridge_status(
                    job_name=job_name,
                    namespace=namespace,
                    phase="query_error",
                    started_at=started_at,
                    detail=detail,
                ),
            )
            if consecutive_errors >= 10:
                collect_diagnostics(client, job_name=job_name, work_dir=work_dir)
                raise BridgeError(f"kubectl get job failed {consecutive_errors} consecutive times: {detail}")
            time.sleep(poll_interval)
            continue

        consecutive_errors = 0
        payload = _parse_json_output(proc, action=f"kubectl get job {job_name}")
        state, message = terminal_condition(payload)
        status = payload.get("status") or {}
        if not state:
            if status.get("active"):
                phase = "running"
            elif status.get("ready"):
                phase = "ready"
            else:
                phase = "waiting"
            _write_json(
                work_dir / BRIDGE_STATUS_FILE,
                _bridge_status(job_name=job_name, namespace=namespace, phase=phase, started_at=started_at),
            )
            time.sleep(poll_interval)
            continue

        _write_json(work_dir / JOB_STATUS_FILE, payload)
        _write_json(
            work_dir / BRIDGE_STATUS_FILE,
            _bridge_status(job_name=job_name, namespace=namespace, phase=state, started_at=started_at, detail=message),
        )
        collect_diagnostics(client, job_name=job_name, work_dir=work_dir)
        if state == "complete":
            return 0
        raise BridgeError(f"Kubernetes Job failed: {message or job_name}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Blocking Kubernetes Job bridge for one prepared VASP stage")
    parser.add_argument("--template", default=_env("CATMASTER_K8S_VASP_TEMPLATE"), help="Remote Kubernetes Job template path")
    parser.add_argument("--namespace", default=_env("CATMASTER_K8S_NAMESPACE", "default"))
    parser.add_argument("--context", default=_env("CATMASTER_K8S_CONTEXT"))
    parser.add_argument("--kubectl", default=_env("CATMASTER_K8S_KUBECTL", "kubectl"))
    parser.add_argument(
        "--mpi_ranks",
        "--ncores",
        dest="mpi_ranks",
        default=_env(
            "CATMASTER_K8S_MPI_RANKS",
            _env("CATMASTER_K8S_VASP_NCORES", _env("DPDISPATCHER_CPU_PER_NODE", "1")),
        ),
        help="MPI rank count; --ncores and CATMASTER_K8S_VASP_NCORES remain accepted for compatibility",
    )
    parser.add_argument("--cpu_limit", default=_env("CATMASTER_K8S_CPU_LIMIT"))
    parser.add_argument("--memory_request", default=_env("CATMASTER_K8S_MEMORY_REQUEST", "8Gi"))
    parser.add_argument("--memory_limit", default=_env("CATMASTER_K8S_MEMORY_LIMIT", "9Gi"))
    parser.add_argument("--vasp_exe", default=_env("CATMASTER_K8S_VASP_EXE", "vasp_std"))
    parser.add_argument("--poll_interval", default=_env("CATMASTER_K8S_POLL_INTERVAL", "20"))
    parser.add_argument("--timeout_seconds", default=_env("CATMASTER_K8S_TIMEOUT_SECONDS", "864000"))
    parser.add_argument("--job_prefix", default=_env("CATMASTER_K8S_JOB_PREFIX", "cm-vasp"))
    parser.add_argument("--render_only", action="store_true", help="Render the manifest without submitting it")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        return run_bridge(args)
    except Exception as exc:
        payload = {
            "phase": "bridge_error",
            "error_type": type(exc).__name__,
            "error": str(exc),
            "updated_at": time.time(),
        }
        try:
            _write_json(Path.cwd() / BRIDGE_STATUS_FILE, payload)
        except Exception:
            pass
        sys.stderr.write(f"[k8s_vasp_boot] {type(exc).__name__}: {exc}\n")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
