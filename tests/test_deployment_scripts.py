from __future__ import annotations

import os
from pathlib import Path
import shutil
import subprocess
import tarfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
RUNTIME_FILES = (
    "catmaster/runtime/execution.py",
    "catmaster/webui/local_execution.py",
    "catmaster/tools/execution/dpdispatcher_runner.py",
    "catmaster/tools/execution/remote_submission.py",
    "catmaster/tools/execution/mlff_specs.py",
    "catmaster/tools/execution/mlff_stage.py",
    "catmaster/remote/cpu/k8s_vasp_boot.py",
    "catmaster/remote/mlff/mlff_common.py",
    "catmaster/webui/static/app.js",
    "catmaster/webui/static/app.css",
    "scripts/remote_execution_smoke.py",
    "requirements/mace.txt",
    "requirements/uma.txt",
    "requirements/mattersim.txt",
    "requirements/orb.txt",
    "third_party/easyslides/scripts/easyslides.py",
    "third_party/easyslides/templates/layouts/example/slide.svg",
    "third_party/easyslides/references/native.md",
)
PRIVATE_FILES = (
    "configs/llm.yaml",
    "configs/dpdispatcher/machines.yaml",
    ".env",
    ".env.local",
    ".sesskey",
    ".runtime/webui.pid",
    ".catmaster/execution.sqlite",
    ".webui_auth/auth.sqlite",
    ".langgraph_api/checkpoints.pckl",
    "project_space/project/metadata/deepagent_threads.sqlite",
)
LAUNCHER = """#!/usr/bin/env bash
set -eu
printf '%s\\n' "$PWD" "${CATMASTER_PROJECT_SPACE_ROOT:-launcher-default}" "$@" > launched.txt
"""


def _write(root: Path, relative: str, content: str) -> Path:
    path = root / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")
    return path


@pytest.fixture
def source_repo(tmp_path: Path) -> Path:
    if shutil.which("rsync") is None:
        pytest.skip("deployment scripts require rsync")
    source = tmp_path / "source checkout"
    for relative in RUNTIME_FILES:
        _write(source, relative, f"runtime fixture: {relative}\n")
    for relative in PRIVATE_FILES:
        _write(source, relative, "source-private-value\n")
    for name in ("deploy_runtime.sh", "package_remote_deploy.sh"):
        shutil.copy2(REPO_ROOT / "scripts" / name, source / "scripts" / name)
    _write(
        source,
        "scripts/install_easyslides.py",
        "from pathlib import Path\nPath(__file__).with_name('easyslides-prepared').touch()\n",
    )
    for name in ("README.md", "LICENSE", "AGENTS.md", ".env.example", "main.py"):
        _write(source, name, "public runtime fixture\n")
    _write(source, "docs/operations.md", "public operations\n")
    _write(source, "skills/example/SKILL.md", "public skill fixture\n")
    for relative in (
        "configs/llm.template.yaml",
        "configs/llm.full.template.yaml",
        "configs/llm_codex_oauth.template.yaml",
        "configs/tool_output.yaml",
        "configs/tool_policy.yaml",
        "configs/dpdispatcher/machines_template.yaml",
        "configs/dpdispatcher/resources_template.yaml",
        "configs/dpdispatcher/tasks_template.yaml",
        "configs/dpdispatcher/mlff_backends_template.yaml",
    ):
        shutil.copy2(REPO_ROOT / relative, source / relative)
    shutil.copytree(
        REPO_ROOT / "configs/dpdispatcher/env_templates",
        source / "configs/dpdispatcher/env_templates",
    )
    _write(source, "start_webui.sh", LAUNCHER).chmod(0o755)
    return source


def _run(source: Path, script: str, *args: str) -> subprocess.CompletedProcess[str]:
    env = os.environ.copy()
    env.pop("CATMASTER_AGENT_SERVER_URL", None)
    env.pop("CATMASTER_PROJECT_SPACE_ROOT", None)
    result = subprocess.run(
        ["bash", str(source / "scripts" / script), "--skip-frontend-build", *args],
        cwd=source,
        env=env,
        text=True,
        capture_output=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr
    assert (source / "scripts/easyslides-prepared").is_file()
    return result


def test_new_runtime_deployment_installs_local_sql_without_legacy_server(
    source_repo: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "new deployment"
    result = _run(source_repo, "deploy_runtime.sh", "--target", str(target), "--no-autorun")

    for relative in RUNTIME_FILES:
        assert (target / relative).read_bytes() == (source_repo / relative).read_bytes()
    assert (target / "start_webui.sh").read_text() == LAUNCHER
    assert os.access(target / "start_webui.sh", os.X_OK)
    assert not (target / "start_agent_server.sh").exists()
    assert not (target / "langgraph.json").exists()
    assert not (target / "launched.txt").exists()
    assert "CATMASTER_AGENT_SERVER_URL" not in result.stdout
    assert "./start_webui.sh --start" in result.stdout


@pytest.mark.parametrize("full_repo", [False, True])
def test_runtime_sync_preserves_private_launcher_and_sql_state(
    source_repo: Path, tmp_path: Path, full_repo: bool,
) -> None:
    target = tmp_path / "existing deployment"
    for relative in PRIVATE_FILES:
        _write(target, relative, "target-private-value\n")
    private_launcher = LAUNCHER.replace("launcher-default", "private-workspace-root")
    _write(target, "start_webui.sh", private_launcher).chmod(0o644)
    _write(target, "configs/dpdispatcher/tasks_template.yaml", "stale public template\n")
    _write(target, "catmaster/obsolete_runtime.py", "obsolete code\n")
    args = ["--target", str(target)]
    if full_repo:
        args.append("--full-repo")
    _run(source_repo, "deploy_runtime.sh", *args)

    for relative in PRIVATE_FILES:
        assert (target / relative).read_text() == "target-private-value\n"
    assert (target / "start_webui.sh").read_text() == private_launcher
    assert (target / "launched.txt").read_text().splitlines() == [
        str(target), "private-workspace-root", "--start",
    ]
    assert not (target / "catmaster/obsolete_runtime.py").exists()
    template = "configs/dpdispatcher/tasks_template.yaml"
    assert (target / template).read_bytes() == (source_repo / template).read_bytes()


def test_autorun_passes_explicit_project_root_to_new_launcher(
    source_repo: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "new deployment"
    project_root = tmp_path / "external projects"
    _run(
        source_repo, "deploy_runtime.sh", "--target", str(target),
        "--project-space-root", str(project_root),
    )
    assert project_root.is_dir()
    assert (target / "launched.txt").read_text().splitlines() == [
        str(target), str(project_root), "--start",
    ]


def test_explicit_config_and_launcher_sync_remain_supported(
    source_repo: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "existing deployment"
    _write(target, "configs/llm.yaml", "private override\n")
    _write(target, "start_webui.sh", "exit 1\n")
    _run(
        source_repo, "deploy_runtime.sh", "--target", str(target),
        "--sync-configs", "--sync-start-webui", "--no-autorun",
    )
    assert (target / "configs/llm.yaml").read_bytes() == (source_repo / "configs/llm.yaml").read_bytes()
    assert (target / "start_webui.sh").read_text() == LAUNCHER
    assert not (target / "launched.txt").exists()


def test_dry_run_does_not_replace_existing_runtime_or_start_it(
    source_repo: Path, tmp_path: Path,
) -> None:
    target = tmp_path / "existing deployment"
    _run(source_repo, "deploy_runtime.sh", "--target", str(target), "--no-autorun")
    _write(target, "catmaster/runtime/execution.py", "existing execution code\n")
    previous_info = (target / ".deploy_info").read_bytes()
    _run(source_repo, "deploy_runtime.sh", "--target", str(target), "--dry-run")
    assert (target / "catmaster/runtime/execution.py").read_text() == "existing execution code\n"
    assert (target / ".deploy_info").read_bytes() == previous_info
    assert not (target / "launched.txt").exists()


def test_remote_archive_contains_local_sql_and_excludes_private_state(
    source_repo: Path, tmp_path: Path,
) -> None:
    # State nested in a copied runtime subtree must be excluded too.
    for directory in (".catmaster", ".webui_auth", ".langgraph_api"):
        _write(source_repo, f"catmaster/{directory}/private-state", "private data\n")
    output = tmp_path / "archives"
    _run(
        source_repo, "package_remote_deploy.sh", "--output-dir", str(output),
        "--archive-name", "local-sql.tar.gz",
    )
    with tarfile.open(output / "local-sql.tar.gz", "r:gz") as archive:
        names = set(archive.getnames())
        for relative in RUNTIME_FILES:
            assert f"CatMaster_Deploy/{relative}" in names
        for relative in PRIVATE_FILES:
            assert f"CatMaster_Deploy/{relative}" not in names
        assert not any(
            part in {".catmaster", ".webui_auth", ".langgraph_api"}
            for name in names for part in Path(name).parts
        )
        assert "CatMaster_Deploy/start_agent_server.sh" not in names
        assert "CatMaster_Deploy/langgraph.json" not in names
        assert "CatMaster_Deploy/start_webui.sh" in names
        assert "CatMaster_Deploy/configs/llm.template.yaml" in names
        readme_file = archive.extractfile("CatMaster_Deploy/DEPLOY_REMOTE.md")
        assert readme_file is not None
        readme = readme_file.read().decode()
        assert "CATMASTER_AGENT_SERVER_URL" not in readme
        assert "start_agent_server.sh" not in readme
        assert ".catmaster/execution.sqlite" in readme
