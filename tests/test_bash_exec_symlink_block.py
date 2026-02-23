from __future__ import annotations

import pytest

from catmaster.tools.base import system_root, workspace_root, workspace_scope
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
        out = bash_exec(
            {
                "script": script,
                "cwd": ".",
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    assert out["status"] == "failed"
    assert "Symbolic link operations are disabled" in (out.get("error") or "")
    assert expected_flag in str(out.get("data", {}).get("blocked_reason", ""))


def test_bash_exec_allows_regular_copy(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp a.txt b.txt && cat b.txt",
                "cwd": ".",
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    assert out["status"] == "success"
    assert "ok" in str(out.get("data", {}).get("stdout", ""))
    assert (files_root / "b.txt").exists()


def test_bash_exec_does_not_block_cp_long_options_without_symlink(tmp_path) -> None:
    with workspace_scope(tmp_path):
        files_root = workspace_root(tmp_path)
        (files_root / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp --preserve=mode a.txt c.txt && cat c.txt",
                "cwd": ".",
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    assert out["status"] == "success"
    assert "ok" in str(out.get("data", {}).get("stdout", ""))
    assert (files_root / "c.txt").exists()


def test_bash_exec_persists_full_stream_logs_and_returns_tails(tmp_path) -> None:
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
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
        audit_dir = system_root(tmp_path) / "audit" / "bash_exec"

    assert out["status"] == "success"
    data = out.get("data", {})
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
    assert "stdout_tail" not in data
    assert "stderr_tail" not in data
    assert str(data.get("stdout") or "").startswith("\n...[output truncated]...\n")
    assert str(data.get("stderr") or "").startswith("\n...[output truncated]...\n")
    assert len(str(data.get("stdout") or "")) <= 3000
    assert len(str(data.get("stderr") or "")) <= 3000
