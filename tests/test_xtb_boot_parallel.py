from __future__ import annotations

import json
import sys
import tarfile
from pathlib import Path
from types import SimpleNamespace

import pytest

from catmaster.remote.cpu import crest_boot, native_argv_runtime, xtb_boot


def _write_manifest(stage: Path, *, argv: list[str] | None = None, assets: list[str] | None = None) -> Path:
    (stage / "inputs").mkdir(parents=True, exist_ok=True)
    (stage / "inputs" / "coord.xyz").write_text("2\nH2\nH 0 0 0\nH 0 0 0.7\n", encoding="utf-8")
    (stage / ".xcontrol").write_text("$constrain\n distance: 1, 2, 0.8\n$end\n", encoding="utf-8")
    path = stage / "manifest.json"
    path.write_text(
        json.dumps(
            {
                "argv": argv if argv is not None else ["inputs/coord.xyz"],
                "input_assets": assets if assets is not None else ["inputs/coord.xyz", ".xcontrol"],
            }
        ),
        encoding="utf-8",
    )
    return path


def test_native_manifest_keeps_nested_files_dotfiles_and_exact_argv(tmp_path: Path) -> None:
    argv = ["inputs/coord.xyz", "--input", ".xcontrol", "literal;token", "$(not-shell)", "*.xyz"]
    path = _write_manifest(tmp_path, argv=argv)

    loaded_argv, assets = native_argv_runtime.load_manifest(path)

    assert loaded_argv == argv
    assert assets == ["inputs/coord.xyz", ".xcontrol"]


def test_native_runtime_executes_literal_list_without_shell_parsing(tmp_path: Path, monkeypatch) -> None:
    argv = ["inputs/coord.xyz", "--input", ".xcontrol", "literal;token", "$(not-shell)", "*.xyz"]
    manifest = _write_manifest(tmp_path, argv=argv)
    captured: dict[str, object] = {}

    def _fake_run(command, **kwargs):
        captured["command"] = command
        captured["kwargs"] = kwargs
        kwargs["stdout"].write("normal termination of xtb\n")
        kwargs["stdout"].flush()
        return SimpleNamespace(returncode=0)

    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(native_argv_runtime.subprocess, "run", _fake_run)
    returncode = native_argv_runtime.run_native_argv(
        manifest_path=manifest,
        program="/opt/xtb",
        summary_name="xtb_summary.json",
        log_name="xtb_stdout.out",
        normal_markers=("normal termination of xtb",),
    )

    assert returncode == 0
    assert captured["command"] == ["/opt/xtb", *argv]
    assert captured["kwargs"]["shell"] is False
    summary = json.loads((tmp_path / "xtb_summary.json").read_text(encoding="utf-8"))
    assert summary["execution_state"] == "completed"
    assert summary["task_state"] == "unknown"
    assert summary["argv"] == argv


def test_result_archive_keeps_hidden_and_nested_engine_outputs(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.chdir(tmp_path)
    (tmp_path / ".hidden-engine-output").write_text("hidden", encoding="utf-8")
    (tmp_path / "nested").mkdir()
    (tmp_path / "nested" / "result.dat").write_text("result", encoding="utf-8")

    native_argv_runtime.write_result_archive()

    with tarfile.open(tmp_path / native_argv_runtime.RESULT_ARCHIVE, "r:gz") as archive:
        names = set(archive.getnames())
    assert ".hidden-engine-output" in names
    assert "nested/result.dat" in names


@pytest.mark.parametrize(
    ("module", "summary_name"),
    [
        (xtb_boot, "xtb_summary.json"),
        (crest_boot, "crest_summary.json"),
    ],
)
def test_boot_main_uses_minimal_manifest_and_writes_process_state(
    tmp_path: Path,
    monkeypatch,
    module,
    summary_name: str,
) -> None:
    monkeypatch.chdir(tmp_path)
    _write_manifest(tmp_path)
    monkeypatch.setattr(sys, "argv", [module.__file__, "--manifest", "manifest.json", "--program", "/bin/true"])

    assert module.main() == 0

    summary = json.loads((tmp_path / summary_name).read_text(encoding="utf-8"))
    assert summary["execution_state"] == "completed"
    assert summary["returncode"] == 0
    assert summary["normal_termination"] is False
    assert summary["task_state"] == "unknown"
    assert (tmp_path / native_argv_runtime.RESULT_ARCHIVE).is_file()


def test_native_manifest_rejects_undeclared_fields(tmp_path: Path) -> None:
    path = _write_manifest(tmp_path)
    path.write_text(
        json.dumps({"argv": ["inputs/coord.xyz"], "input_assets": ["inputs/coord.xyz"], "mode": "opt"}),
        encoding="utf-8",
    )

    with pytest.raises(ValueError, match="only argv and input_assets"):
        native_argv_runtime.load_manifest(path)
