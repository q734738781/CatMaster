import asyncio
import shlex
import sys
from pathlib import Path
from types import SimpleNamespace

from langgraph.store.memory import InMemoryStore

from catmaster.specialists.runtime import SpecialistRunner


def test_workspace_receipt_read_list_grep_and_skill_mount(tmp_path: Path):
    files = tmp_path / "files"
    receipts = files / ".deepagents/dpdispatcher/receipts"
    receipts.mkdir(parents=True)
    (receipts / "historical.json").write_text('{"context_id":"existing-job"}')
    snapshot = tmp_path / "metadata/snapshot"
    (snapshot / "skills/example").mkdir(parents=True)
    (snapshot / "skills/example/SKILL.md").write_text("# reachable skill")
    runner = object.__new__(SpecialistRunner)
    runner._skill_snapshot_root = snapshot
    runner.run_context = SimpleNamespace(workspace=tmp_path)
    runner._memory_namespace = lambda: ("test",)
    backend = runner._make_backend(files_root=files, store=InMemoryStore())
    path = "/.deepagents/dpdispatcher/receipts/historical.json"
    read = backend.read(path)
    assert read.error is None
    assert "existing-job" in read.file_data["content"]
    skill = backend.read("/.deepagents/skills/example/SKILL.md")
    assert skill.error is None
    assert "reachable skill" in skill.file_data["content"]
    listing = backend.ls("/.deepagents/dpdispatcher/receipts/")
    assert listing.error is None
    assert path in [item["path"] for item in listing.entries]
    matches = backend.grep("existing-job", path="/.deepagents/dpdispatcher/receipts/")
    assert matches.error is None and matches.matches[0]["path"] == path
    async_read = asyncio.run(backend.aread(path))
    assert async_read.error is None
    assert "existing-job" in async_read.file_data["content"]


def test_shell_executes_its_own_skill_snapshot_without_copying(tmp_path: Path):
    files = tmp_path / "files"
    files.mkdir()
    backends = []
    for version in ("stable", "canary"):
        snapshot = tmp_path / f"metadata/{version} snapshot"
        resource = snapshot / "skills/example/references"
        resource.mkdir(parents=True)
        (resource / "value.txt").write_text(version)
        (resource / "helper.py").write_text(f"VALUE = {version!r}\n")
        (resource / "run.py").write_text(
            "from pathlib import Path\n"
            "from helper import VALUE\n"
            "value = Path(__file__).with_name('value.txt').read_text()\n"
            "assert value == VALUE\n"
            "Path(value + '.txt').write_text(value)\n"
            "print(value)\n"
        )
        runner = object.__new__(SpecialistRunner)
        runner._skill_snapshot_root = snapshot
        runner.run_context = SimpleNamespace(workspace=tmp_path)
        runner._memory_namespace = lambda: ("test",)
        backends.append(runner._make_backend(files_root=files, store=InMemoryStore()))
    # Creating another backend must not replace an earlier run's shell binding.
    for version, backend in zip(("stable", "canary"), backends):
        result = backend.execute(
            f'{shlex.quote(sys.executable)} "$CATMASTER_SKILLS_ROOT/example/references/run.py"'
        )
        assert result.exit_code == 0, result.output
        assert result.output.strip() == version
        assert (files / f"{version}.txt").read_text() == version
        resource = backend.read("/.deepagents/skills/example/references/value.txt")
        assert version in resource.file_data["content"]
    assert not list(files.rglob("*.py"))
    for version in ("stable", "canary"):
        assert not list((tmp_path / f"metadata/{version} snapshot").rglob("__pycache__"))
