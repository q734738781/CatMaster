from __future__ import annotations

import json
import shutil
import sqlite3
from pathlib import Path

from ase import Atoms
from ase.io import write as ase_write
from fastapi.testclient import TestClient
from starlette.routing import Match

from catmaster.webui import server
from catmaster.webui.server import create_app
from catmaster.webui.session import WebSession


def _scope(path: str) -> dict:
    return {"type": "http", "path": path, "method": "GET", "root_path": ""}


def test_monitor_path_redirect_route_precedes_root_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor"


def test_monitor_path_with_slash_hits_monitor_mount(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    full_matches = [route for route in app.routes if route.matches(_scope("/monitor/"))[0] == Match.FULL]
    assert full_matches
    assert getattr(full_matches[0], "path", None) == "/monitor/"


def test_pages_load_react_static_bundle(tmp_path: Path) -> None:
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    home = client.get("/")
    assert home.status_code == 200
    assert 'rel="icon"' in home.text
    assert '/static/app.css' in home.text
    assert '/static/app.js' in home.text

    monitor = client.get("/monitor/")
    assert monitor.status_code == 200
    assert '/static/app.css' in monitor.text
    assert '/static/app.js' in monitor.text

    files_page = client.get("/files/")
    assert files_page.status_code == 200
    assert '/static/app.css' in files_page.text
    assert '/static/app.js' in files_page.text


def test_files_routes_list_preview_and_download(tmp_path: Path) -> None:
    ws = tmp_path / "demo"
    (ws / "files").mkdir(parents=True)
    (ws / "metadata").mkdir(parents=True)
    (ws / "files" / "notes.md").write_text("# Demo\n\nHello files view.\n", encoding="utf-8")
    shutil.copyfile("tests/assets/Fe.cif", ws / "files" / "Fe.cif")

    app = create_app(project_space_root=str(tmp_path))
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

    app = create_app(project_space_root=str(tmp_path))
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

    app = create_app(project_space_root=str(tmp_path))
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

    app = create_app(project_space_root=str(tmp_path))
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
    app = create_app(project_space_root=str(tmp_path))
    client = TestClient(app)

    response = client.get("/asset-KaTeX_Main-Regular.woff")
    assert response.status_code == 200
    assert response.content


def test_named_vasp_outputs_are_classified_as_structure(tmp_path: Path) -> None:
    assert server._entry_preview_kind(tmp_path / "OUTCAR") == "structure"
    assert server._entry_preview_kind(tmp_path / "XDATCAR") == "structure"


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

    app = create_app(project_space_root=str(tmp_path))
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

    app = create_app(project_space_root=str(tmp_path))
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
    app = create_app(project_space_root=str(tmp_path))
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

    app = create_app(project_space_root=str(tmp_path))
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


def test_runtime_snapshot_annotates_live_prompt_payload() -> None:
    class _DummyReporter:
        @staticmethod
        def get_snapshot():
            return {
                "run_name": "run_live",
                "seq": 7,
                "live_state": {},
                "llm": {},
                "graph": {},
                "prompt": {
                    "prompt_id": "prompt_live",
                    "kind": "proposal_review",
                    "payload": {"proposal_description": "# live proposal"},
                },
                "usage_totals": {},
                "recent_events": [],
            }

    class _DummyRunDir:
        name = "run_live"

    class _DummySession:
        reporter = _DummyReporter()

        @staticmethod
        def get_selected_run_dir():
            return _DummyRunDir()

        @staticmethod
        def _annotate_prompt_payload(run_dir, pending):
            assert run_dir.name == "run_live"
            return {
                **pending,
                "payload": {
                    **dict(pending.get("payload") or {}),
                    "guidance": 'Type "approve" to continue. Any other input requests a revised proposal.',
                },
            }

    snapshot = server._runtime_snapshot(_DummySession())
    prompt = snapshot["prompt"]
    assert isinstance(prompt, dict)
    assert prompt["payload"]["guidance"] == 'Type "approve" to continue. Any other input requests a revised proposal.'


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

    app = create_app(project_space_root=str(tmp_path))
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
