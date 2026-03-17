from __future__ import annotations

import sys

import pytest

from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, system_root, workspace_root, workspace_scope
from catmaster.tools.misc.bash_exec import bash_exec


@pytest.mark.parametrize(
    ("script", "expected_flag"),
    [
        ("ln -s src dst", "ln symbolic options"),
        ("cp -s src dst", "cp -s"),
        ("python - <<'PY'\nimport os\nos.symlink('a', 'b')\nPY", "os.symlink()"),
        ("python - <<'PY'\nfrom pathlib import Path\nPath('b').symlink_to('a')\nPY", "Path.symlink_to()"),
    ],
)
def test_bash_exec_blocks_symlink_usage(tmp_path, script: str, expected_flag: str) -> None:
    with workspace_scope(tmp_path):
        with pytest.raises(CatMasterToolExecutionError) as excinfo:
            bash_exec(
                {
                    "script": script,
                    "cwd": ".",
                    "no_network": False,
                    "timeout_s": 5.0,
                }
            )
    assert "Symbolic link operations are disabled" in str(excinfo.value)
    assert expected_flag in str(excinfo.value.artifact.get("data", {}).get("blocked_reason", ""))


def test_bash_exec_allows_regular_copy(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp a.txt b.txt && cat b.txt",
                "cwd": ".",
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    _, artifact = out
    assert "ok" in str(artifact.get("data", {}).get("stdout", ""))
    assert (files_root / "b.txt").exists()


def test_bash_exec_does_not_block_cp_long_options_without_symlink(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp --preserve=mode a.txt c.txt && cat c.txt",
                "cwd": ".",
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    _, artifact = out
    assert "ok" in str(artifact.get("data", {}).get("stdout", ""))
    assert (files_root / "c.txt").exists()


def test_bash_exec_persists_full_stream_logs_without_pre_truncation(tmp_path) -> None:
    with workspace_scope(tmp_path):
        out = bash_exec(
            {
                "script": (
                    "python - <<'PY'\n"
                    "import sys\n"
                    "sys.stdout.write('A' * 12000)\n"
                    "sys.stderr.write('B' * 9000)\n"
                    "PY\n"
                ),
                "cwd": ".",
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
        audit_dir = system_root(tmp_path) / "audit" / "bash"

    _, artifact = out
    data = artifact.get("data", {})
    assert "stdout_path" not in data
    assert "stderr_path" not in data
    stdout_logs = sorted(audit_dir.glob("*.stdout.txt"))
    stderr_logs = sorted(audit_dir.glob("*.stderr.txt"))
    assert stdout_logs
    assert stderr_logs
    stdout_file = stdout_logs[-1]
    stderr_file = stderr_logs[-1]
    assert len(stdout_file.read_text(encoding="utf-8")) >= 12000
    assert len(stderr_file.read_text(encoding="utf-8")) >= 9000
    assert str(data.get("stdout") or "").startswith("A")
    assert str(data.get("stderr") or "").startswith("B")
    assert len(str(data.get("stdout") or "")) >= 12000
    assert len(str(data.get("stderr") or "")) >= 9000


def test_bash_exec_python3_uses_current_interpreter(tmp_path) -> None:
    with workspace_scope(tmp_path):
        content, artifact = bash_exec(
            {
                "script": (
                    "python3 - <<'PY'\n"
                    "import sys\n"
                    "print(sys.executable)\n"
                    "PY\n"
                ),
                "cwd": ".",
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    _ = content
    stdout = str((artifact or {}).get("data", {}).get("stdout") or "").strip()
    assert stdout == sys.executable


def test_resolve_workspace_path_accepts_deepagent_virtual_root(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "nested").mkdir()
        assert resolve_workspace_path("/") == files_root.resolve()
        assert resolve_workspace_path("/nested") == (files_root / "nested").resolve()


def test_resolve_workspace_path_rejects_host_absolute_path(tmp_path) -> None:
    with workspace_scope(tmp_path):
        host_abs = str((tmp_path / "outside" / "report.md").resolve())
        with pytest.raises(ValueError, match="Absolute host path outside project files root"):
            resolve_workspace_path(host_abs)


def test_bash_exec_accepts_virtual_root_cwd(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "hello.txt").write_text("ok", encoding="utf-8")
        _, artifact = bash_exec(
            {
                "script": "pwd && cat hello.txt",
                "cwd": "/",
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    data = artifact.get("data", {})
    stdout = str(data.get("stdout") or "")
    assert str(files_root.resolve()) in stdout
    assert "ok" in stdout
