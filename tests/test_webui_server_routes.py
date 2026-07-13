from __future__ import annotations

import json
import shutil
import sqlite3
import zipfile
from io import BytesIO
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
from fastapi.testclient import TestClient
from starlette.routing import Match

from catmaster.webui import server
from catmaster.webui.server import _candidate_change_preview, _discover_project_spaces, create_app
from catmaster.runtime.self_evolution import LearningCandidate, SelfEvolutionStore
from catmaster.runtime.self_evolution.storage import hash_tree, utc_now
from catmaster.webui.session import WebSession
from catmaster.webui.web_reporter import WebReporter
from catmaster.runtime.machine_time_stats import append_machine_time_record
from catmaster.runtime.observability_store import ObservabilityStore
from catmaster.ui.events import make_event


def _scope(path: str) -> dict:
    return {"type": "http", "path": path, "method": "GET", "root_path": ""}


def test_self_evolution_worker_discovers_authenticated_user_workspaces(tmp_path: Path) -> None:
    direct = tmp_path / "direct"
    authenticated = tmp_path / "users" / "alice" / "project-one"
    incomplete = tmp_path / "users" / "bob" / "not-a-workspace"
    for workspace in (direct, authenticated):
        (workspace / "files").mkdir(parents=True)
        (workspace / "metadata").mkdir()
    (incomplete / "files").mkdir(parents=True)

    discovered = _discover_project_spaces(tmp_path)

    assert discovered == [direct.resolve(), authenticated.resolve()]


def test_self_evolution_human_review_preview_includes_complete_bundle_diff(tmp_path: Path) -> None:
    workspace = tmp_path / "workspace"
    (workspace / "files").mkdir(parents=True)
    (workspace / "metadata").mkdir()
    store = SelfEvolutionStore(workspace, project_id="demo")
    candidate = LearningCandidate(
        candidate_id="sec_preview",
        project_id="demo",
        run_id="run-one",
        thread_id="thread-one",
        action="skill",
        status="approved",
        group="materials_worker",
        name="demo-skill",
        created_at=utc_now(),
    )
    root = store.reset_candidate_dir(candidate.candidate_id)
    before = root / "current" / "target"
    after = root / "proposed" / candidate.group / candidate.name
    (before / "scripts").mkdir(parents=True)
    (after / "scripts").mkdir(parents=True)
    (before / "SKILL.md").write_text("# demo\n\nold rule\n", encoding="utf-8")
    (after / "SKILL.md").write_text("# demo\n\nnew rule\n", encoding="utf-8")
    (after / "scripts" / "helper.py").write_text("VALUE = 1\n", encoding="utf-8")
    candidate.bundle_hash = hash_tree(after)
    store.write_candidate(candidate)

    preview, truncated = _candidate_change_preview(store, candidate)

    assert truncated is False
    assert "-old rule" in preview
    assert "+new rule" in preview
    assert "scripts/helper.py" in preview
    assert "+VALUE = 1" in preview


def test_monitor_path_redirect_route_precedes_root_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor"


def test_monitor_path_with_slash_hits_monitor_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor/"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor/"


def test_default_page_loads_react_static_bundle_and_legacy_pages_redirect(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert 'rel="icon"' in home.text
    assert '/static/app.css' in home.text
    assert '/static/app.js' in home.text

    monitor = client.get("/monitor/", follow_redirects=False)
    assert monitor.status_code == 307
    assert monitor.headers["location"] == "/#tab=monitor"

    files_page = client.get("/files/", follow_redirects=False)
    assert files_page.status_code == 307
    assert files_page.headers["location"] == "/#tab=files"


def test_legacy_pages_are_debug_only(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setenv("CATMASTER_WEBUI_LEGACY_ROUTES", "1")
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    monitor = client.get("/monitor/")
    assert monitor.status_code == 200
    assert '/static/app.css' in monitor.text
    assert '/static/app.js' in monitor.text

    files_page = client.get("/files/")
    assert files_page.status_code == 200
    assert '/static/app.css' in files_page.text
    assert '/static/app.js' in files_page.text


def test_bootstrap_recovers_from_stale_missing_project_space(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    response = client.get(
        "/api/bootstrap",
        params={"ctx": "ctx_stale_001", "project_space": "missing_space", "lane": "experiment"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["ctx"] == "ctx_stale_001"
    assert payload["workspace_root"] == str(tmp_path.resolve())
    assert payload["workspace_name"] == ""
    assert "Project space does not exist: missing_space" in payload["status_message"]


def test_workspace_delete_requires_confirmation_and_non_active_workspace(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    boot = client.get("/api/bootstrap").json()
    ctx = boot["ctx"]
    created = client.post(f"/api/session/{ctx}/workspace/create", json={"workspace": "scratch"})
    assert created.status_code == 200
    assert (tmp_path / "scratch").is_dir()

    wrong_confirm = client.request(
        "DELETE",
        f"/api/session/{ctx}/workspace/delete",
        json={"workspace": "scratch", "confirm_name": "wrong"},
    )
    assert wrong_confirm.status_code == 400
    assert (tmp_path / "scratch").is_dir()

    opened_scratch = client.post(f"/api/session/{ctx}/workspace/open", json={"workspace": "scratch"})
    assert opened_scratch.status_code == 200
    active_delete = client.request(
        "DELETE",
        f"/api/session/{ctx}/workspace/delete",
        json={"workspace": "scratch", "confirm_name": "scratch", "active_workspace": "scratch"},
    )
    assert active_delete.status_code == 400
    assert "Switch away" in active_delete.json()["detail"]
    assert (tmp_path / "scratch").is_dir()

    opened = client.post(f"/api/session/{ctx}/workspace/open", json={"workspace": "admin"})
    assert opened.status_code == 200
    deleted = client.request(
        "DELETE",
        f"/api/session/{ctx}/workspace/delete",
        json={"workspace": "scratch", "confirm_name": "scratch", "active_workspace": "admin"},
    )
    assert deleted.status_code == 200
    assert deleted.json()["ok"] is True
    assert not (tmp_path / "scratch").exists()


def test_files_routes_list_preview_and_download(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "notes.md").write_text("# Demo\n\nHello files view.\n", encoding="utf-8")
    shutil.copyfile("tests/assets/Fe.cif", ws / "files" / "Fe.cif")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    tree = client.get(f"/api/session/{ctx}/files/tree")
    assert tree.status_code == 200
    tree_payload = tree.json()
    assert [item["name"] for item in tree_payload["children"][:2]] == ["files", "metadata"]

    files_branch = client.get(f"/api/session/{ctx}/files/tree", params={"path": "files"})
    assert files_branch.status_code == 200
    files_payload = files_branch.json()
    assert {item["name"] for item in files_payload["children"]} >= {"notes.md", "Fe.cif"}

    preview = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/notes.md"})
    assert preview.status_code == 200
    preview_payload = preview.json()
    assert preview_payload["kind"] == "markdown"
    assert "Hello files view" in preview_payload["preview_text"]

    structure = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/Fe.cif"})
    assert structure.status_code == 200
    structure_payload = structure.json()
    assert structure_payload["kind"] == "structure"
    assert structure_payload["structure"]["viewer_format"] in {"cif", "xyz"}
    assert structure_payload["structure"]["atom_count"] >= 1
    assert structure_payload["structure"]["elements"]
    assert isinstance(structure_payload["structure"]["element_counts"], dict)

    download = client.get(f"/api/session/{ctx}/files/download", params={"path": "files/notes.md"})
    assert download.status_code == 200
    assert "Hello files view" in download.text


def test_files_routes_upload_and_archive(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files" / "nested").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "nested" / "existing.txt").write_text("old", encoding="utf-8")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    upload = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/nested", "filename": "new.txt"},
        content=b"uploaded content",
        headers={"content-type": "text/plain"},
    )
    assert upload.status_code == 200
    assert upload.json()["path"] == "files/nested/new.txt"
    assert (ws / "files" / "nested" / "new.txt").read_text(encoding="utf-8") == "uploaded content"

    conflict = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/nested", "filename": "new.txt"},
        content=b"replacement",
    )
    assert conflict.status_code == 409

    overwrite = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/nested", "filename": "new.txt", "overwrite": "true"},
        content=b"replacement",
    )
    assert overwrite.status_code == 200
    assert (ws / "files" / "nested" / "new.txt").read_text(encoding="utf-8") == "replacement"

    escaped = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "../outside", "filename": "bad.txt"},
        content=b"bad",
    )
    assert escaped.status_code == 400

    archive = client.get(f"/api/session/{ctx}/files/archive", params={"path": "files/nested"})
    assert archive.status_code == 200
    assert archive.headers["content-type"].startswith("application/zip")
    with zipfile.ZipFile(BytesIO(archive.content)) as zip_handle:
        names = set(zip_handle.namelist())
        assert "nested/new.txt" in names
        assert zip_handle.read("nested/new.txt") == b"replacement"


def test_files_routes_upload_unzip_and_delete(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files" / "incoming").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_handle:
        zip_handle.writestr("folder/a.txt", "alpha")
        zip_handle.writestr("folder/b.txt", "beta")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    unzip = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/incoming", "filename": "bundle.zip", "unzip": "true"},
        content=zip_buffer.getvalue(),
        headers={"content-type": "application/zip"},
    )
    assert unzip.status_code == 200
    payload = unzip.json()
    assert payload["unzipped"] is True
    assert payload["extracted_count"] == 2
    assert not (ws / "files" / "incoming" / "bundle.zip").exists()
    assert (ws / "files" / "incoming" / "folder" / "a.txt").read_text(encoding="utf-8") == "alpha"

    conflict = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/incoming", "filename": "bundle.zip", "unzip": "true"},
        content=zip_buffer.getvalue(),
        headers={"content-type": "application/zip"},
    )
    assert conflict.status_code == 409

    overwrite = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files/incoming", "filename": "bundle.zip", "unzip": "true", "overwrite": "true"},
        content=zip_buffer.getvalue(),
        headers={"content-type": "application/zip"},
    )
    assert overwrite.status_code == 200

    delete_file = client.delete(f"/api/session/{ctx}/files/delete", params={"path": "files/incoming/folder/a.txt"})
    assert delete_file.status_code == 200
    assert not (ws / "files" / "incoming" / "folder" / "a.txt").exists()

    delete_dir = client.delete(f"/api/session/{ctx}/files/delete", params={"path": "files/incoming/folder"})
    assert delete_dir.status_code == 200
    assert not (ws / "files" / "incoming" / "folder").exists()

    delete_root = client.delete(f"/api/session/{ctx}/files/delete", params={"path": ""})
    assert delete_root.status_code == 400


def test_files_upload_unzip_rejects_unsafe_paths(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w") as zip_handle:
        zip_handle.writestr("../escape.txt", "bad")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    response = client.post(
        f"/api/session/{ctx}/files/upload",
        params={"path": "files", "filename": "bad.zip", "unzip": "true"},
        content=zip_buffer.getvalue(),
        headers={"content-type": "application/zip"},
    )
    assert response.status_code == 400
    assert not (ws / "escape.txt").exists()


def test_files_archive_skips_symlinks_that_escape_workspace(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    outside = tmp_path / "outside.txt"
    outside.write_text("outside secret", encoding="utf-8")
    (ws / "files" / "safe.txt").write_text("safe", encoding="utf-8")
    try:
        (ws / "files" / "escape.txt").symlink_to(outside)
    except OSError:
        return

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    archive = client.get(f"/api/session/{ctx}/files/archive", params={"path": "files"})
    assert archive.status_code == 200
    with zipfile.ZipFile(BytesIO(archive.content)) as zip_handle:
        names = set(zip_handle.namelist())
        assert "files/safe.txt" in names
        assert "files/escape.txt" not in names


def test_structure_view_and_animation_routes(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    frames = [
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]]),
        Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.85]]),
    ]
    ase_write(ws / "files" / "md.traj", frames, format="traj")
    (ws / "files" / "OUTCAR").write_text(
        "\n".join(
            [
                "  1 f/i=   23.224372 THz   145.923033 2PiTHz   774.681641 cm-1   96.048317 meV",
                " X         Y         Z           dx          dy          dz",
                " 0.000000 0.000000 0.000000   0.100000   0.000000   0.000000",
                " 0.000000 0.000000 0.740000  -0.100000   0.000000   0.000000",
            ]
        ),
        encoding="utf-8",
    )
    original_read_structure_frames = server._read_structure_frames

    def _patched_read_structure_frames(path: Path, *, limit: int = server.STRUCTURE_ANIMATION_FRAME_LIMIT):
        if path.name == "OUTCAR":
            return ([Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])], 1, False)
        return original_read_structure_frames(path, limit=limit)

    monkeypatch.setattr(server, "_read_structure_frames", _patched_read_structure_frames)

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    structure = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/md.traj"})
    assert structure.status_code == 200
    structure_payload = structure.json()["structure"]
    assert structure_payload["supports_animation"] is True
    assert structure_payload["viewer_source_mode"] == "url"
    assert structure_payload["frame_count"] == 2

    animation = client.get(f"/api/session/{ctx}/files/structure-animation", params={"path": "files/md.traj"})
    assert animation.status_code == 200
    assert "H" in animation.text

    vibration = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/OUTCAR"})
    assert vibration.status_code == 200
    vibration_payload = vibration.json()["structure"]
    assert vibration_payload["supports_vibration"] is True
    assert vibration_payload["vibration_modes"]
    assert vibration_payload["viewer_source_mode"] == "url"
    assert vibration_payload["viewer_source_file_type"] == "Xyz"

    vibration_xyz = client.get(f"/api/session/{ctx}/files/structure-vibration", params={"path": "files/OUTCAR"})
    assert vibration_xyz.status_code == 200
    assert "frequency_cm-1=" in vibration_xyz.text


def test_plain_outcar_uses_native_view_without_vibration_controls(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "OUTCAR").write_text("plain outcar without vibration block\n", encoding="utf-8")
    original_read_structure_frames = server._read_structure_frames

    def _patched_read_structure_frames(path: Path, *, limit: int = server.STRUCTURE_ANIMATION_FRAME_LIMIT):
        if path.name == "OUTCAR":
            return ([Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])], 1, False)
        return original_read_structure_frames(path, limit=limit)

    monkeypatch.setattr(server, "_read_structure_frames", _patched_read_structure_frames)

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/OUTCAR"})
    assert response.status_code == 200
    structure_payload = response.json()["structure"]
    assert structure_payload["viewer_source_mode"] == "url"
    assert structure_payload["viewer_source_file_type"] == "VaspOutcar"
    assert structure_payload["supports_vibration"] is False


def test_outcar_with_vibration_header_without_equals_uses_compatibility_view(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "OUTCAR").write_text(
        "\n".join(
            [
                "  1 f    32.464781 THz   203.982238 2PiTHz   1082.908545 cm-1   134.263487 meV",
                " X         Y         Z           dx          dy          dz",
                " 0.000000 0.000000 0.000000   0.100000   0.000000   0.000000",
                " 0.000000 0.000000 0.740000  -0.100000   0.000000   0.000000",
            ]
        ),
        encoding="utf-8",
    )
    original_read_structure_frames = server._read_structure_frames

    def _patched_read_structure_frames(path: Path, *, limit: int = server.STRUCTURE_ANIMATION_FRAME_LIMIT):
        if path.name == "OUTCAR":
            return ([Atoms("H2", positions=[[0.0, 0.0, 0.0], [0.0, 0.0, 0.74]])], 1, False)
        return original_read_structure_frames(path, limit=limit)

    monkeypatch.setattr(server, "_read_structure_frames", _patched_read_structure_frames)

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/OUTCAR"})
    assert response.status_code == 200
    structure_payload = response.json()["structure"]
    assert structure_payload["supports_vibration"] is True
    assert structure_payload["viewer_source_file_type"] == "Xyz"

    vibration_xyz = client.get(f"/api/session/{ctx}/files/structure-vibration", params={"path": "files/OUTCAR"})
    assert vibration_xyz.status_code == 200
    assert "frequency_cm-1=" in vibration_xyz.text


def test_root_asset_route_serves_static_font_files(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    response = client.get("/asset-KaTeX_Main-Regular.woff")
    assert response.status_code == 200
    assert response.content


def test_named_vasp_outputs_are_classified_as_structure(tmp_path: Path) -> None:
    assert server._entry_preview_kind(tmp_path / "OUTCAR") == "structure"
    assert server._entry_preview_kind(tmp_path / "XDATCAR") == "structure"


def test_unknown_suffix_small_text_files_are_previewed(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "job.sub.run").write_text("#!/bin/bash\necho running\n", encoding="utf-8")
    (ws / "files" / "job.gbw").write_bytes(b"\x00\x01\x02binary")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    tree = client.get(f"/api/session/{ctx}/files/tree", params={"path": "files"})
    assert tree.status_code == 200
    kinds = {item["name"]: item["preview_kind"] for item in tree.json()["children"]}
    assert kinds["job.sub.run"] == "text"
    assert kinds["job.gbw"] == "binary"

    preview = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/job.sub.run"})
    assert preview.status_code == 200
    payload = preview.json()
    assert payload["kind"] == "text"
    assert "echo running" in payload["preview_text"]


def test_poscar_uses_native_vasp_poscar_view(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "CONTCAR").write_text(
        "\n".join(
            [
                "Si",
                "1.0",
                "5.430000 0.000000 0.000000",
                "0.000000 5.430000 0.000000",
                "0.000000 0.000000 5.430000",
                "Si",
                "2",
                "Direct",
                "0.000000 0.000000 0.000000",
                "0.250000 0.250000 0.250000",
            ]
        ),
        encoding="utf-8",
    )
    original_read_structure_frames = server._read_structure_frames

    def _patched_read_structure_frames(path: Path, *, limit: int = server.STRUCTURE_ANIMATION_FRAME_LIMIT):
        if path.name == "CONTCAR":
            return ([Atoms("Si2", cell=[5.43, 5.43, 5.43], pbc=True, scaled_positions=[[0, 0, 0], [0.25, 0.25, 0.25]])], 1, False)
        return original_read_structure_frames(path, limit=limit)

    monkeypatch.setattr(server, "_read_structure_frames", _patched_read_structure_frames)

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/CONTCAR"})
    assert response.status_code == 200
    structure_payload = response.json()["structure"]
    assert structure_payload["periodic"] is True
    assert structure_payload["viewer_source_mode"] == "url"
    assert structure_payload["viewer_source_file_type"] == "VaspPoscar"


def test_cif_uses_native_cif_view(tmp_path: Path, monkeypatch) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "sample.cif").write_text(
        "data_sample\n_cell_length_a 5.0\n_cell_length_b 5.0\n_cell_length_c 5.0\n",
        encoding="utf-8",
    )
    original_read_structure_frames = server._read_structure_frames

    def _patched_read_structure_frames(path: Path, *, limit: int = server.STRUCTURE_ANIMATION_FRAME_LIMIT):
        if path.name == "sample.cif":
            return ([Atoms("Si2", cell=[5.0, 5.0, 5.0], pbc=True, scaled_positions=[[0, 0, 0], [0.5, 0.5, 0.5]])], 1, False)
        return original_read_structure_frames(path, limit=limit)

    monkeypatch.setattr(server, "_read_structure_frames", _patched_read_structure_frames)

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/files/content", params={"path": "files/sample.cif"})
    assert response.status_code == 200
    structure_payload = response.json()["structure"]
    assert structure_payload["periodic"] is True
    assert structure_payload["viewer_source_mode"] == "url"
    assert structure_payload["viewer_source_file_type"] == "Cif"


def test_favicon_route_does_not_404(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)

    response = client.get("/favicon.ico")
    assert response.status_code in {200, 204}


def test_coerce_int_treats_empty_string_as_default() -> None:
    assert server._coerce_int("", 0) == 0
    assert server._coerce_int("7", 0) == 7


def test_memory_route_returns_workspace_memory(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    db_path = ws / "metadata" / "deepagent_memory.sqlite"
    conn = sqlite3.connect(str(db_path))
    conn.execute("CREATE TABLE store (prefix TEXT NOT NULL, key TEXT NOT NULL, value BLOB NOT NULL)")
    prefix = ".".join(("catmaster", WebSession._project_id_for_workspace(ws), "filesystem"))
    payload = {"content": ["Stable preference: prefer compact reports."]}
    conn.execute(
        "INSERT INTO store(prefix, key, value) VALUES (?, ?, ?)",
        (prefix, "/AGENTS.md", json.dumps(payload).encode("utf-8")),
    )
    conn.commit()
    conn.close()

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    response = client.get(f"/api/session/{ctx}/memory")
    assert response.status_code == 200
    payload = response.json()
    assert "prefer compact reports" in payload["memory"]


def test_active_run_name_falls_back_to_run_info_when_runtime_has_no_run_name() -> None:
    class _DummyRunDir:
        name = "run_old"

    class _DummySession:
        run_info = {"run_id": "run_new"}

        @staticmethod
        def get_selected_run_dir():
            return _DummyRunDir()

    assert server._active_run_name(_DummySession(), {"run_name": ""}) == "run_new"


def test_merge_usage_summary_preserves_cost_fields_for_active_runs() -> None:
    merged = server._merge_usage_summary(
        {"output_tokens": 42, "reasoning_tokens": 5, "total_tokens": 142},
        {
            "cost_usd": 0.1234,
            "exact_cost_usd": 0.1,
            "estimated_cost_usd": 0.0234,
            "cost_source": "mixed",
            "missing_cost_calls": 0,
            "output_tokens": 30,
        },
    )

    assert merged["cost_usd"] == 0.1234
    assert merged["cost_source"] == "mixed"
    assert merged["output_tokens"] == 42
    assert merged["reasoning_tokens"] == 5


def test_merge_usage_summary_prefers_runtime_cost_when_runtime_is_newer() -> None:
    merged = server._merge_usage_summary(
        {
            "cost_usd": 0.25,
            "exact_cost_usd": 0.2,
            "estimated_cost_usd": 0.05,
            "cost_source": "mixed",
            "breakdown_usd": {"completion": 0.05},
            "output_tokens": 50,
        },
        {
            "cost_usd": 0.1234,
            "exact_cost_usd": 0.1,
            "estimated_cost_usd": 0.0234,
            "cost_source": "estimated",
            "output_tokens": 30,
        },
    )

    assert merged["cost_usd"] == 0.25
    assert merged["cost_source"] == "mixed"
    assert merged["breakdown_usd"] == {"completion": 0.05}
    assert merged["output_tokens"] == 50


def test_web_reporter_runtime_usage_totals_include_cost_summary(monkeypatch) -> None:
    monkeypatch.setattr(
        "catmaster.webui.web_reporter.summarize_usage_from_metadata",
        lambda usage_metadata, **kwargs: {
            "calls": 1,
            "input_tokens": 120,
            "input_cached_tokens": 80,
            "output_tokens": 30,
            "reasoning_tokens": 7,
            "total_tokens": 150,
            "cost_usd": 0.1234,
            "exact_cost_usd": 0.1,
            "estimated_cost_usd": 0.0234,
            "cost_source": "mixed",
            "by_model": [{"name": "openai/gpt-5.4", "cost_usd": 0.1234}],
            "by_role": [{"name": "materials_worker", "cost_usd": 0.1234}],
        },
    )
    reporter = WebReporter()
    reporter.emit(
        make_event(
            "LLM_CALL_END",
            category="llm",
            payload={
                "model": "openai/gpt-5.4",
                "agent_name": "materials_worker",
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                    "input_token_details": {"cache_read": 80},
                    "output_token_details": {"reasoning": 7},
                    "cost": 0.1,
                    "cost_details": {"provider": "openrouter"},
                },
            },
            run_id="run_demo",
        )
    )

    snapshot = reporter.get_snapshot()
    assert snapshot["usage_totals"]["cost_usd"] == 0.1234
    assert snapshot["usage_totals"]["input_tokens"] == 120
    assert snapshot["usage_totals"]["input_cached_tokens"] == 80
    assert snapshot["usage_totals"]["output_tokens"] == 30
    assert snapshot["usage_totals"]["reasoning_tokens"] == 7


def test_web_reporter_persists_ui_events_and_usage_summary(tmp_path: Path) -> None:
    reporter = WebReporter()
    reporter.set_run_dir(tmp_path)

    reporter.emit(
        make_event(
            "LLM_CALL_END",
            category="llm",
            payload={
                "model": "openai/gpt-5.4",
                "agent_name": "materials_worker",
                "usage": {
                    "input_tokens": 120,
                    "output_tokens": 30,
                    "total_tokens": 150,
                    "cost": 0.42,
                },
                "text_preview": "latest answer",
            },
            run_id="run_demo",
        )
    )

    assert not (tmp_path / "ui_events.jsonl").exists()

    usage = json.loads((tmp_path / "usage_summary.json").read_text(encoding="utf-8"))
    assert usage["cost_usd"] == 0.42
    assert usage["cost_source"] == "exact"
    assert usage["calls"] == 1

    observability = ObservabilityStore(tmp_path).read_snapshot()
    assert observability["metrics"]["llm_calls"] == 1
    assert observability["events"][-1]["name"] == "LLM_CALL_END"
    event_page = WebSession().read_ui_events(tmp_path, limit=2)
    assert event_page["events"][-1]["seq"] == 1
    assert event_page["events"][-1]["name"] == "LLM_CALL_END"

    resumed = WebReporter()
    resumed.set_run_dir(tmp_path)
    resumed.emit(make_event("RUN_END", category="run", payload={"status": "done"}, run_id="run_demo"))
    assert not (tmp_path / "ui_events.jsonl").exists()
    resumed_page = WebSession().read_ui_events(tmp_path, limit=2)
    assert [event["seq"] for event in resumed_page["events"]] == [1, 2]


def test_web_reporter_does_not_backfill_legacy_ui_events_before_sqlite_write(tmp_path: Path) -> None:
    (tmp_path / "ui_events.jsonl").write_text(
        "\n".join(
            json.dumps({"seq": seq, "name": "RUN_EVENT", "payload": {"status": str(seq)}})
            for seq in (1, 2)
        )
        + "\n",
        encoding="utf-8",
    )

    reporter = WebReporter()
    reporter.set_run_dir(tmp_path)
    reporter.emit(make_event("RUN_END", category="run", payload={"status": "done"}, run_id="run_demo"))

    page = WebSession().read_ui_events(tmp_path, limit=5)
    assert [event["seq"] for event in page["events"]] == [1]


def test_observability_route_returns_metrics_from_observability_store_only(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    run_dir = ws / "metadata" / "runs" / "run_obs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ws / "files").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps({"status": "done", "entrypoint": "experiment", "summary": "Observed run"}),
        encoding="utf-8",
    )
    (run_dir / "usage_summary.json").write_text(
        json.dumps({"source": "legacy_export", "input_tokens": 999, "output_tokens": 999, "calls": 999}),
        encoding="utf-8",
    )
    obs = ObservabilityStore(run_dir)
    obs.record_ui_event(
        {
            "seq": 1,
            "ts": 1.0,
            "name": "LLM_CALL_END",
            "category": "llm",
            "payload": {
                "model": "gpt-test",
                "elapsed_ms": 1000,
                "text_preview": "Done",
                "usage": {"input_tokens": 2, "output_tokens": 3},
            },
        }
    )
    obs.record_ui_event(
        {
            "seq": 2,
            "ts": 2.0,
            "name": "TOOL_CALL_END",
            "category": "tool",
            "payload": {"tool": "bash", "status": "success"},
        }
    )
    append_machine_time_record(
        run_dir,
        {
            "status": "success",
            "tool_name": "remote_submission",
            "task_name": "vasp_execute",
            "resources": "vasp_cpu",
            "machine": "cpu_server_2",
            "task_count": 1,
            "number_node": 1,
            "cpu_per_node": 32,
            "elapsed_seconds": 3600,
            "core_hours": 32.0,
            "node_hours": 1.0,
        },
    )

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo", "lane": "experiment"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    response = client.get(
        f"/api/session/{ctx}/observability",
        params={"project_space": "demo", "lane": "experiment", "run": "run_obs"},
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_run"] == "run_obs"
    assert payload["metrics"]["llm_calls"] == 1
    assert payload["metrics"]["tool_calls"] == 1
    assert payload["raw_logs"]["total_events"] == 3
    assert payload["usage_summary"]["input_tokens"] == 2
    assert payload["usage_summary"]["output_tokens"] == 3
    assert payload["usage_summary"]["calls"] == 1
    assert payload["usage_summary"]["source"] == "observability_store"
    assert "MACHINE_TIME_RECORD" in payload["metrics"]["event_counts"]
    assert payload["machine_time_summary"]["requests"] == 1
    assert payload["machine_time_summary"]["core_hours"] == 32.0


def test_details_route_hides_legacy_traces_unless_requested(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    run_dir = ws / "metadata" / "runs" / "run_trace"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ws / "files").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(json.dumps({"status": "done", "entrypoint": "experiment"}), encoding="utf-8")
    (run_dir / "event_trace.jsonl").write_text('{"event":"LLM_RAW_RESPONSE","payload":{"text":"legacy"}}\n', encoding="utf-8")
    (run_dir / "tool_trace.jsonl").write_text('{"tool_name":"legacy_tool","status":"success"}\n', encoding="utf-8")

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo", "lane": "experiment"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    response = client.get(
        f"/api/session/{ctx}/details",
        params={"project_space": "demo", "run": "run_trace"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert "trace_event" not in payload
    assert "trace_tool" not in payload
    assert "trace_patch" not in payload

    legacy_response = client.get(
        f"/api/session/{ctx}/details",
        params={"project_space": "demo", "run": "run_trace", "include_legacy_traces": "true"},
    )
    assert legacy_response.status_code == 200
    legacy_payload = legacy_response.json()
    assert "LLM_RAW_RESPONSE" in legacy_payload["trace_event"]
    assert "legacy_tool" in legacy_payload["trace_tool"]


def test_events_route_reads_unified_observability_events_with_filters(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    run_dir = ws / "metadata" / "runs" / "run_obs"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ws / "files").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(json.dumps({"status": "done", "entrypoint": "experiment"}), encoding="utf-8")

    store = ObservabilityStore(run_dir)
    store.record_event(
        source="langchain_callback",
        channel="callback",
        name="LLM_CALL_END",
        category="llm",
        ts=1.0,
        seq=None,
        run_id="run_obs",
        task_id="",
        step_id=None,
        payload={"model": "model-a", "agent_name": "experiment_specialist", "callback_run_id": "llm_1"},
    )
    store.record_event(
        source="thread_event",
        channel="thread",
        name="tool_call.completed",
        category="thread",
        ts=2.0,
        seq=7,
        run_id="run_obs",
        task_id="",
        step_id=None,
        thread_id="thread_1",
        message_id="msg_1",
        payload={
            "seq": 7,
            "event": "tool_call.completed",
            "thread_id": "thread_1",
            "message_id": "msg_1",
            "status": "completed",
            "data": {"tool": "mace_relax_dir", "run_id": "run_obs"},
        },
    )

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo", "lane": "experiment"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    response = client.get(
        f"/api/session/{ctx}/events",
        params={"project_space": "demo", "run": "run_obs", "limit": 20},
    )
    assert response.status_code == 200
    names = [event["name"] for event in response.json()["events"]]
    assert "LLM_CALL_END" in names
    assert "tool_call.completed" in names

    filtered = client.get(
        f"/api/session/{ctx}/events",
        params={"project_space": "demo", "run": "run_obs", "channel": "thread", "thread_id": "thread_1"},
    )
    assert filtered.status_code == 200
    payload = filtered.json()
    assert [event["name"] for event in payload["events"]] == ["tool_call.completed"]
    assert payload["events"][0]["thread_id"] == "thread_1"


def test_websession_read_ui_events_paginates_sqlite_by_sequence(tmp_path: Path) -> None:
    store = ObservabilityStore(tmp_path)
    for seq in range(1, 6):
        store.record_ui_event({"seq": seq, "name": "RUN_EVENT", "payload": {"status": str(seq)}})

    session = WebSession()

    latest = session.read_ui_events(tmp_path, limit=2)
    assert [event["seq"] for event in latest["events"]] == [4, 5]
    assert latest["has_more"] is True
    assert latest["min_seq"] == 4

    older = session.read_ui_events(tmp_path, limit=2, before_seq=4)
    assert [event["seq"] for event in older["events"]] == [2, 3]
    assert older["has_more"] is True

    newer = session.read_ui_events(tmp_path, limit=2, after_seq=3)
    assert [event["seq"] for event in newer["events"]] == [4, 5]


def test_websession_read_ui_events_does_not_fallback_to_legacy_jsonl(tmp_path: Path) -> None:
    for seq in range(1, 6):
        with (tmp_path / "ui_events.jsonl").open("a", encoding="utf-8") as handle:
            handle.write(json.dumps({"seq": seq, "name": "RUN_EVENT", "payload": {"status": str(seq)}}) + "\n")

    session = WebSession()

    latest = session.read_ui_events(tmp_path, limit=2)
    assert latest == {"events": [], "has_more": False, "min_seq": 0, "max_seq": 0}

    assert ObservabilityStore(tmp_path).import_legacy_jsonl(include_ui_events=True) == 5
    imported = session.read_ui_events(tmp_path, limit=2)
    assert [event["seq"] for event in imported["events"]] == [4, 5]
    assert imported["has_more"] is True


def test_runtime_snapshot_omits_removed_prompt_payload() -> None:
    class _DummyReporter:
        @staticmethod
        def get_snapshot():
            return {
                "run_name": "run_live",
                "seq": 7,
                "live_state": {},
                "llm": {},
                "graph": {},
                "usage_totals": {},
                "recent_events": [],
            }

    class _DummySession:
        reporter = _DummyReporter()

    snapshot = server._runtime_snapshot(_DummySession())
    assert "prompt" not in snapshot


def test_chat_create_clears_selected_run_view_when_no_active_run(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    runs_root = ws / "metadata" / "runs"
    run_dir = runs_root / "run_old"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ws / "files").mkdir(parents=True, exist_ok=True)
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "done",
                "entrypoint": "experiment",
                "summary": "Old answer",
                "facts": [],
                "files": [],
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "demo"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]

    select_run = client.post(
        f"/api/session/{ctx}/run/select",
        json={"run_name": "run_old", "lane": "experiment"},
    )
    assert select_run.status_code == 200
    assert select_run.json()["selected_run"] == "run_old"

    response = client.post(
        f"/api/session/{ctx}/chat/create",
        json={"lane": "experiment"},
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_run"] == ""
    assert payload["result_text"] == ""
    assert payload["proposal"] == ""
    assert payload["todo_items"] == []
    assert payload["current_chat_session"]
    assert len(payload["chat_sessions"]) >= 2


def test_same_ctx_snapshot_is_scoped_by_project_space(tmp_path: Path) -> None:
    for name in ("alpha", "beta"):
        ws = tmp_path / name
        (ws / "files").mkdir(parents=True, exist_ok=True)
        (ws / "metadata" / "runs" / f"run_{name}").mkdir(parents=True, exist_ok=True)
        (ws / "metadata" / "runs" / f"run_{name}" / "run_state.json").write_text(
            json.dumps({"status": "done", "entrypoint": "experiment", "summary": name}),
            encoding="utf-8",
        )

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    boot = client.get("/api/bootstrap", params={"project_space": "alpha"})
    assert boot.status_code == 200
    ctx = boot.json()["ctx"]
    switch = client.get("/api/bootstrap", params={"ctx": ctx, "project_space": "beta"})
    assert switch.status_code == 200
    assert switch.json()["workspace_name"] == "beta"

    alpha = client.get(f"/api/session/{ctx}/snapshot", params={"project_space": "alpha", "lane": "experiment"})
    beta = client.get(f"/api/session/{ctx}/snapshot", params={"project_space": "beta", "lane": "experiment"})

    assert alpha.status_code == 200
    assert beta.status_code == 200
    assert alpha.json()["workspace_name"] == "alpha"
    assert beta.json()["workspace_name"] == "beta"
    assert alpha.json()["selected_run"] == "run_alpha"
    assert beta.json()["selected_run"] == "run_beta"


def test_bootstrap_recovers_active_run_for_lane(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    run_dir = ws / "metadata" / "runs" / "run_live"
    run_dir.mkdir(parents=True, exist_ok=True)
    (ws / "files").mkdir(parents=True, exist_ok=True)
    (ws / "metadata" / "active_runs.json").write_text(
        json.dumps({"experiment": "runs/run_live"}),
        encoding="utf-8",
    )
    (run_dir / "run_state.json").write_text(
        json.dumps(
            {
                "status": "running",
                "entrypoint": "experiment",
                "phase": "executing",
                "text_preview": "Still working.",
                "chat_session_id": "chat_demo",
            },
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )

    app = create_app(project_space_root=str(tmp_path), no_login=True)
    client = TestClient(app)
    response = client.get("/api/bootstrap", params={"project_space": "demo", "lane": "experiment"})

    assert response.status_code == 200
    payload = response.json()
    assert payload["selected_run"] == "run_live"
    assert payload["active_run"] == "run_live"
    assert payload["run_status"] == "running"
    assert payload["live_state"]["status"] == "running"
    assert payload["live_state"]["current_task_goal"] == "Still working."
