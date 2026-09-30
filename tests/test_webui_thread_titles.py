from __future__ import annotations

import asyncio
import threading
import time
from pathlib import Path
from typing import Any

import pytest
from fastapi.testclient import TestClient

from catmaster.tools.base import ensure_project_space_layout
from catmaster.webui import server
from catmaster.webui.agent_loop import ThreadAgentLoopService
from catmaster.webui.projections import project_event
from catmaster.webui.server import create_app
from catmaster.webui.thread_events import ThreadEventBroker
from catmaster.webui.thread_models import MessagePart, ThreadMessage, ThreadStatus
from catmaster.webui import thread_store as thread_store_module
from catmaster.webui.thread_store import ThreadStore, new_id
from catmaster.webui.thread_title_service import sanitize_generated_thread_title


def _workspace(tmp_path: Path) -> Path:
    workspace = tmp_path / "default"
    ensure_project_space_layout(workspace, create=True)
    return workspace


def _wait_for_title(store: ThreadStore, thread_id: str, expected: str) -> None:
    deadline = time.monotonic() + 3
    while time.monotonic() < deadline:
        if store.get_thread(thread_id).title == expected:
            return
        time.sleep(0.02)
    assert store.get_thread(thread_id).title == expected


async def _create_local_test_thread(
    self: ThreadAgentLoopService,
    *,
    title: str,
    entrypoint: str,
    parent_thread_id: str = "",
    thread_role: str = "primary",
    meta: dict[str, Any] | None = None,
) -> Any:
    """Keep title tests independent from an external Agent Server."""

    return self.store.create_thread(
        title=title,
        entrypoint=entrypoint,
        parent_thread_id=parent_thread_id,
        thread_role=thread_role,
        meta=meta,
    )


def test_thread_store_creates_local_titles_and_preserves_activity_on_cas(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    store.append_message(
        ThreadMessage(
            id="msg_first_title",
            thread_id=thread.thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id="part_first_title",
                    type="text",
                    text="  ##  CO2 加氢制 BTX 的高性能氧化物设计？  ",
                    status="completed",
                )
            ],
        )
    )

    provisional = store.get_thread(thread.thread_id)
    assert provisional.title == "CO2 加氢制 BTX 的高性能氧化物设计"
    updated = store.compare_and_set_title(
        thread.thread_id,
        expected_title=provisional.title,
        title="CO2 加氢氧化物设计",
    )
    assert updated is not None
    assert updated.updated_at == provisional.updated_at

    store.update_thread(thread.thread_id, title="实验组复现方案")
    assert store.compare_and_set_title(
        thread.thread_id,
        expected_title="CO2 加氢氧化物设计",
        title="晚到的模型标题",
    ) is None
    assert store.get_thread(thread.thread_id).title == "实验组复现方案"


def test_concurrent_manual_rename_finishes_after_background_cas(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = _workspace(tmp_path)
    store_a = ThreadStore(workspace=workspace, workspace_id="default")
    store_b = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store_a.create_thread(title="临时标题")
    cas_at_write = threading.Event()
    release_cas = threading.Event()
    original_write = thread_store_module._atomic_write_json

    def delayed_write(path: Path, payload: dict[str, Any]) -> None:
        if payload.get("title") == "后台标题":
            cas_at_write.set()
            assert release_cas.wait(timeout=2)
        original_write(path, payload)

    monkeypatch.setattr(thread_store_module, "_atomic_write_json", delayed_write)
    cas_thread = threading.Thread(
        target=lambda: store_a.compare_and_set_title(
            thread.thread_id,
            expected_title="临时标题",
            title="后台标题",
        )
    )
    rename_thread = threading.Thread(
        target=lambda: store_b.update_thread(thread.thread_id, title="人工标题")
    )
    cas_thread.start()
    assert cas_at_write.wait(timeout=1)
    rename_thread.start()
    time.sleep(0.05)
    assert rename_thread.is_alive()
    release_cas.set()
    cas_thread.join(timeout=2)
    rename_thread.join(timeout=2)

    assert not cas_thread.is_alive()
    assert not rename_thread.is_alive()
    assert store_a.get_thread(thread.thread_id).title == "人工标题"


def test_attachment_title_and_legacy_backfill_use_first_user_message(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    attachment_thread = store.create_thread(entrypoint="experiment")
    store.append_message(
        ThreadMessage(
            id="msg_attachment_title",
            thread_id=attachment_thread.thread_id,
            role="user",
            status="completed",
            parts=[MessagePart(id="part_empty", type="text", text="", status="completed")],
            structured_sidecar={"attachments": [{"filename": "catalyst_data.csv"}]},
        )
    )
    assert store.get_thread(attachment_thread.thread_id).title == "Experiment: catalyst_data.csv"

    legacy = store.create_thread(title="Legacy title")
    store.append_message(
        ThreadMessage(
            id="msg_legacy_first",
            thread_id=legacy.thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id="part_legacy_first",
                    type="text",
                    text="复现 CO2-FTS 高性能催化剂",
                    status="completed",
                )
            ],
        )
    )
    before = store.get_thread(legacy.thread_id)
    placeholder = store.compare_and_set_title(
        legacy.thread_id,
        expected_title=before.title,
        title="New thread",
    )
    assert placeholder is not None
    assert placeholder.updated_at == before.updated_at

    store.backfill_default_titles()
    backfilled = store.get_thread(legacy.thread_id)
    assert backfilled.title == "复现 CO2-FTS 高性能催化剂"
    assert backfilled.updated_at == before.updated_at


def test_steering_message_is_not_treated_as_the_first_title_source(
    tmp_path: Path,
) -> None:
    workspace = _workspace(tmp_path)
    store = ThreadStore(workspace=workspace, workspace_id="default")
    thread = store.create_thread()
    store.append_message(
        ThreadMessage(
            id="msg_title_steering",
            thread_id=thread.thread_id,
            role="user",
            status="completed",
            parts=[MessagePart(id="part_title_steering", type="text", text="补充条件", status="completed")],
            meta={"kind": "steering"},
        )
    )
    assert store.get_thread(thread.thread_id).title == ""

    store.append_message(
        ThreadMessage(
            id="msg_title_ordinary",
            thread_id=thread.thread_id,
            role="user",
            status="completed",
            parts=[MessagePart(id="part_title_ordinary", type="text", text="正式研究问题", status="completed")],
        )
    )
    assert store.get_thread(thread.thread_id).title == "正式研究问题"


def test_generated_title_cleanup_rejects_explanations_and_generic_labels() -> None:
    assert sanitize_generated_thread_title("标题：CO2 加氢氧化物设计。") == "CO2 加氢氧化物设计"
    assert sanitize_generated_thread_title("研究问题") == ""
    assert sanitize_generated_thread_title("标题一\n补充解释") == ""


@pytest.mark.parametrize("manual_rename", [False, True])
def test_submit_returns_provisional_title_and_background_update_respects_rename(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    manual_rename: bool,
) -> None:
    workspace = _workspace(tmp_path)
    started = threading.Event()
    release = threading.Event()

    async def fake_generate_semantic_thread_title(**_kwargs: Any) -> str:
        started.set()
        await asyncio.to_thread(release.wait, 3)
        return "CO2-FTS 氧化物复现"

    async def fake_submit(
        self: ThreadAgentLoopService,
        *,
        thread_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        message = ThreadMessage(
            id=new_id("msg"),
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=[
                MessagePart(
                    id=new_id("part_text"),
                    type="text",
                    text=str(payload.text or ""),
                    status="completed",
                )
            ],
        )
        self.store.append_message(message)
        thread = self.store.update_thread(thread_id, status=ThreadStatus.RUNNING)
        return {
            "accepted": True,
            "queued": False,
            "thread": thread,
            "message": message,
        }

    monkeypatch.setattr(server, "generate_semantic_thread_title", fake_generate_semantic_thread_title)
    monkeypatch.setattr(ThreadAgentLoopService, "create_thread", _create_local_test_thread)
    monkeypatch.setattr(ThreadAgentLoopService, "submit", fake_submit)

    with TestClient(create_app(project_space_root=str(tmp_path), no_login=True)) as client:
        created = client.post("/api/workspaces/default/threads", json={})
        assert created.status_code == 200
        thread_id = created.json()["thread"]["thread_id"]
        assert created.json()["thread"]["title"] == "New thread"
        assert ThreadStore(workspace=workspace, workspace_id="default").get_thread(thread_id).title == ""

        response = client.post(
            f"/api/threads/{thread_id}/submit",
            json={"text": "请调研 CO2-FTS 中高性能氧化物催化剂的复现路线"},
        )
        assert response.status_code == 200
        assert response.json()["thread"]["title"].startswith("请调研 CO2-FTS")
        assert started.wait(timeout=1)

        expected = "CO2-FTS 氧化物复现"
        if manual_rename:
            renamed = client.patch(
                f"/api/threads/{thread_id}",
                json={"title": "实验组复现计划"},
            )
            assert renamed.status_code == 200
            expected = "实验组复现计划"
        release.set()

        store = ThreadStore(workspace=workspace, workspace_id="default")
        _wait_for_title(store, thread_id, expected)
        if not manual_rename:
            title_events = [
                event
                for event in ThreadEventBroker(workspace=workspace).replay(thread_id)
                if event.event == "thread.updated"
                and isinstance(event.data.get("thread"), dict)
                and event.data["thread"].get("title") == expected
            ]
            assert title_events
            public_event = project_event(title_events[-1], workspace=workspace)
            assert public_event.data.thread is not None
            assert public_event.data.thread.title == expected


def test_explicit_thread_title_does_not_schedule_model(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    _workspace(tmp_path)
    called = threading.Event()

    async def fake_generate_semantic_thread_title(**_kwargs: Any) -> str:
        called.set()
        return "should not run"

    async def fake_submit(
        self: ThreadAgentLoopService,
        *,
        thread_id: str,
        payload: Any,
    ) -> dict[str, Any]:
        message = ThreadMessage(
            id=new_id("msg"),
            thread_id=thread_id,
            role="user",
            status="completed",
            parts=[MessagePart(id=new_id("part"), type="text", text=payload.text, status="completed")],
        )
        self.store.append_message(message)
        return {"queued": False, "thread": self.store.get_thread(thread_id), "message": message}

    monkeypatch.setattr(server, "generate_semantic_thread_title", fake_generate_semantic_thread_title)
    monkeypatch.setattr(ThreadAgentLoopService, "create_thread", _create_local_test_thread)
    monkeypatch.setattr(ThreadAgentLoopService, "submit", fake_submit)
    with TestClient(create_app(project_space_root=str(tmp_path), no_login=True)) as client:
        thread_id = client.post(
            "/api/workspaces/default/threads",
            json={"title": "Manual scientific title"},
        ).json()["thread"]["thread_id"]
        response = client.post(
            f"/api/threads/{thread_id}/submit",
            json={"text": "Start the review"},
        )
        assert response.status_code == 200
        time.sleep(0.05)
        assert not called.is_set()
        assert response.json()["thread"]["title"] == "Manual scientific title"
