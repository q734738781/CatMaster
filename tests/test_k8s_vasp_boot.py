from __future__ import annotations

import json
from pathlib import Path
import re
import subprocess
from types import SimpleNamespace

import pytest

from catmaster.remote.cpu import k8s_vasp_boot


def _template() -> str:
    return """\
apiVersion: batch/v1
kind: Job
metadata:
  name: __JOB_NAME__
spec:
  template:
    spec:
      containers:
      - name: vasp
        image: example/vasp:test
        args:
        - cd __STRUCTURE_DIR__ && mpirun -np __MPI_RANKS__ __VASP_EXE__
        resources:
          requests:
            cpu: __CPU_REQUEST__
            memory: __MEMORY_REQUEST__
          limits:
            cpu: __CPU_LIMIT__
            memory: __MEMORY_LIMIT__
      restartPolicy: Never
"""


def test_build_job_name_is_stable_and_dns_safe(tmp_path: Path) -> None:
    token = "dispatch:id With Spaces:remote_submission_stage_o2:0"
    first = k8s_vasp_boot.build_job_name(prefix="CM_VASP", token=token, work_dir=tmp_path)
    second = k8s_vasp_boot.build_job_name(prefix="CM_VASP", token=token, work_dir=tmp_path)

    assert first == second
    assert len(first) <= 63
    assert re.fullmatch(r"[a-z0-9](?:[-a-z0-9]*[a-z0-9])?", first)


def test_render_manifest_is_strict() -> None:
    replacements = {
        "__JOB_NAME__": "cm-vasp-o2",
        "__STRUCTURE_DIR__": "/shared/o2",
        "__MPI_RANKS__": "4",
        "__CPU_REQUEST__": "4",
        "__CPU_LIMIT__": "5",
        "__MEMORY_REQUEST__": "8Gi",
        "__MEMORY_LIMIT__": "9Gi",
        "__VASP_EXE__": "vasp_std",
    }
    rendered = k8s_vasp_boot.render_manifest(_template(), replacements)

    assert "__" not in rendered
    assert "name: cm-vasp-o2" in rendered
    assert "mpirun -np 4 vasp_std" in rendered

    with pytest.raises(k8s_vasp_boot.BridgeError, match="unresolved placeholders"):
        k8s_vasp_boot.render_manifest(_template() + "\nextra: __UNKNOWN__\n", replacements)


def test_terminal_condition_recognizes_complete_and_failed() -> None:
    complete = {"status": {"conditions": [{"type": "Complete", "status": "True", "reason": "Done"}]}}
    failed = {"status": {"conditions": [{"type": "Failed", "status": "True", "message": "Pod failed"}]}}

    assert k8s_vasp_boot.terminal_condition(complete) == ("complete", "Done")
    assert k8s_vasp_boot.terminal_condition(failed) == ("failed", "Pod failed")
    assert k8s_vasp_boot.terminal_condition({"status": {"active": 1}}) == ("", "")


def test_run_bridge_blocks_until_complete_and_writes_diagnostics(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    template = tmp_path / "remote-template.yaml"
    template.write_text(_template(), encoding="utf-8")
    monkeypatch.chdir(tmp_path)
    monkeypatch.setenv("CM_DPDISPATCHER_SUBMISSION_TOKEN", "abc:remote_submission_o2:0")
    monkeypatch.setattr(k8s_vasp_boot.time, "sleep", lambda _: None)

    job_queries = 0

    def fake_run(self, args, *, timeout=60.0):
        nonlocal job_queries
        args = list(args)
        _ = (self, timeout)
        if args[0] == "apply":
            return subprocess.CompletedProcess(args, 0, "job.batch/cm-vasp configured\n", "")
        if args[:2] == ["get", "job"]:
            job_queries += 1
            if job_queries == 1:
                payload = {"status": {"active": 1}}
            else:
                payload = {"status": {"succeeded": 1, "conditions": [{"type": "Complete", "status": "True"}]}}
            return subprocess.CompletedProcess(args, 0, json.dumps(payload), "")
        if args[:2] == ["get", "pods"]:
            return subprocess.CompletedProcess(args, 0, '{"items": []}', "")
        return subprocess.CompletedProcess(args, 0, "diagnostic output\n", "")

    monkeypatch.setattr(k8s_vasp_boot.KubectlClient, "run", fake_run)
    args = SimpleNamespace(
        template=str(template),
        namespace="default",
        context="",
        kubectl="kubectl",
        mpi_ranks="4",
        cpu_limit="5",
        memory_request="8Gi",
        memory_limit="9Gi",
        vasp_exe="vasp_std",
        poll_interval="1",
        timeout_seconds="60",
        job_prefix="cm-vasp",
        render_only=False,
    )

    assert k8s_vasp_boot.run_bridge(args) == 0
    assert job_queries >= 2
    assert "cpu: 4" in (tmp_path / k8s_vasp_boot.MANIFEST_FILE).read_text(encoding="utf-8")
    bridge_status = json.loads((tmp_path / k8s_vasp_boot.BRIDGE_STATUS_FILE).read_text(encoding="utf-8"))
    assert bridge_status["phase"] == "complete"
    assert (tmp_path / k8s_vasp_boot.POD_LOG_FILE).is_file()


def test_parser_prefers_mpi_rank_name_and_accepts_legacy_alias(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("CATMASTER_K8S_MPI_RANKS", "8")
    assert k8s_vasp_boot.build_parser().parse_args([]).mpi_ranks == "8"
    assert k8s_vasp_boot.build_parser().parse_args(["--mpi_ranks", "6"]).mpi_ranks == "6"
    assert k8s_vasp_boot.build_parser().parse_args(["--ncores", "4"]).mpi_ranks == "4"
