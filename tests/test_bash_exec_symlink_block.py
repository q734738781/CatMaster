from __future__ import annotations

import pytest

from catmaster.tools.base import workspace_scope
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
                "view": "user",
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
        (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp a.txt b.txt && cat b.txt",
                "cwd": ".",
                "view": "user",
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    assert out["status"] == "success"
    assert "ok" in str(out.get("data", {}).get("stdout", ""))
    assert (tmp_path / "b.txt").exists()


def test_bash_exec_does_not_block_cp_long_options_without_symlink(tmp_path) -> None:
    with workspace_scope(tmp_path):
        (tmp_path / "a.txt").write_text("ok", encoding="utf-8")
        out = bash_exec(
            {
                "script": "cp --preserve=mode a.txt c.txt && cat c.txt",
                "cwd": ".",
                "view": "user",
                "strict": True,
                "no_network": False,
                "timeout_s": 5.0,
            }
        )
    assert out["status"] == "success"
    assert "ok" in str(out.get("data", {}).get("stdout", ""))
    assert (tmp_path / "c.txt").exists()
