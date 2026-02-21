from __future__ import annotations

import pytest

from catmaster.tools.base import workspace_root, workspace_scope
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
        files_root = workspace_root(tmp_path)

    assert out["status"] == "success"
    data = out.get("data", {})
    stdout_path = str(data.get("stdout_path") or "")
    stderr_path = str(data.get("stderr_path") or "")
    assert stdout_path
    assert stderr_path
    stdout_file = files_root / stdout_path
    stderr_file = files_root / stderr_path
    assert stdout_file.exists()
    assert stderr_file.exists()
    assert len(stdout_file.read_text(encoding="utf-8")) >= 12000
    assert len(stderr_file.read_text(encoding="utf-8")) >= 9000
    assert len(str(data.get("stdout_tail") or "")) <= 3000
    assert len(str(data.get("stderr_tail") or "")) <= 3000
