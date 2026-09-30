from __future__ import annotations

import importlib.util
import io
from pathlib import Path
import subprocess
import zipfile

import pytest


REPO_ROOT = Path(__file__).resolve().parents[1]
spec = importlib.util.spec_from_file_location("install_easyslides", REPO_ROOT / "scripts/install_easyslides.py")
installer = importlib.util.module_from_spec(spec)
spec.loader.exec_module(installer)


def _source_archive(monkeypatch: pytest.MonkeyPatch, *, fail: bool = False) -> list[str]:
    calls: list[str] = []
    archive = io.BytesIO()
    builder = """from pathlib import Path
import sys
out = Path(sys.argv[sys.argv.index('--out') + 1])
for name in ['scripts/easyslides.py', 'templates/example/slide.svg', 'references/guide.md', 'skills/example/SKILL.md']:
    path = out / name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(name)
"""
    if fail:
        builder = "raise RuntimeError('build failed')\n"
    with zipfile.ZipFile(archive, "w") as output:
        output.writestr(f"easyslides-{installer.REVISION}/scripts/build_plugin_bundle.py", builder)

    def download(request, **kwargs):
        calls.append(request.full_url)
        return io.BytesIO(archive.getvalue())

    monkeypatch.setattr(installer.urllib.request, "urlopen", download)
    return calls


def test_install_keeps_runtime_assets_and_reuses_installed_bundle(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    calls = _source_archive(monkeypatch)
    destination = tmp_path / "third party/easyslides"
    installer.install(destination, quiet=True)
    for path in ("scripts/easyslides.py", "templates/example/slide.svg", "references/guide.md", "skills/example/SKILL.md"):
        assert (destination / path).read_text() == path
    installer.install(destination, quiet=True)
    assert calls == [installer.SOURCE_URL]


def test_failed_update_preserves_previous_install(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    _source_archive(monkeypatch, fail=True)
    destination = tmp_path / "easyslides"
    destination.mkdir()
    (destination / installer.REVISION_FILE).write_text("previous")
    old = destination / "existing.py"
    old.write_text("previous runtime")
    with pytest.raises(subprocess.CalledProcessError):
        installer.install(destination, quiet=True)
    assert old.read_text() == "previous runtime"
    assert (destination / installer.REVISION_FILE).read_text() == "previous"
