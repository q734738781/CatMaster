from __future__ import annotations

import asyncio
import importlib.metadata
import json
import sqlite3
from contextlib import AsyncExitStack
from pathlib import Path
from types import SimpleNamespace
from typing import Any, ClassVar

import pytest
from deepagents import AsyncSubAgentMiddleware, create_deep_agent
from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import GlobResult
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage, ToolMessage
from langgraph.checkpoint.sqlite import SqliteSaver
from pydantic import Field

from catmaster.runtime.checkpoint_serde import FileSafeCheckpointSerializer
from catmaster.specialists import runtime as runtime_mod


class _BindableFakeModel(FakeMessagesListChatModel):
    bound_tools: ClassVar[list[dict[str, Any]]] = []

    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        _ = (tool_choice, kwargs)
        self.bound_tools.append({tool.name: tool for tool in tools})
        return self


class _CapturingBindableFakeModel(_BindableFakeModel):
    inline_image_counts: list[int] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        count = 0
        for message in messages:
            content = getattr(message, "content", None)
            if not isinstance(content, list):
                continue
            count += sum(
                1
                for block in content
                if isinstance(block, dict)
                and block.get("type") == "image"
                and bool(block.get("base64"))
            )
        self.inline_image_counts.append(count)
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


class _SystemPromptCapturingBindableFakeModel(_BindableFakeModel):
    system_prompts: list[str] = Field(default_factory=list)

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.system_prompts.append(
            "\n".join(
                str(message.content)
                for message in messages
                if isinstance(message, SystemMessage)
            )
        )
        return super()._generate(
            messages,
            stop=stop,
            run_manager=run_manager,
            **kwargs,
        )


def _runner(tmp_path: Path) -> runtime_mod.SpecialistRunner:
    workspace = tmp_path / "workspace"
    files_root = workspace / "files"
    run_dir = workspace / "metadata" / "runs" / "run_da074"
    files_root.mkdir(parents=True)
    run_dir.mkdir(parents=True)
    return runtime_mod.SpecialistRunner(
        llm_profile=SimpleNamespace(),
        run_context=SimpleNamespace(
            workspace=workspace,
            run_dir=run_dir,
            run_id="run_da074",
            project_id="project_da074",
        ),
    )


def _middleware_by_type(middleware: list[Any], type_name: str) -> Any:
    matches = [item for item in middleware if type(item).__name__ == type_name]
    assert len(matches) == 1
    return matches[0]


def test_sqlite_state_waits_for_concurrent_checkpoint_and_memory_writers(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)

    async def inspect_connections() -> tuple[int, int, str]:
        async with AsyncExitStack() as stack:
            saver, store = await runner._open_sqlite_state(stack)
            await saver.setup()
            checkpoint_timeout = int(
                (await (await saver.conn.execute("PRAGMA busy_timeout")).fetchone())[0]
            )
            memory_timeout = int(
                (await (await store.conn.execute("PRAGMA busy_timeout")).fetchone())[0]
            )
            checkpoint_mode = str(
                (await (await saver.conn.execute("PRAGMA journal_mode")).fetchone())[0]
            )
            return checkpoint_timeout, memory_timeout, checkpoint_mode

    checkpoint_timeout, memory_timeout, checkpoint_mode = asyncio.run(
        inspect_connections()
    )
    assert checkpoint_timeout == 30_000
    assert memory_timeout == 30_000
    assert checkpoint_mode.lower() == "wal"






def test_catmaster_deepagents_0711_middleware_is_explicit_and_graph_local(
    tmp_path: Path,
) -> None:
    assert importlib.metadata.version("deepagents") == "0.7.11"
    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    runtime = {"backend": backend}

    first = runner._catmaster_agent_middleware(runtime=runtime, skills=[])
    second = runner._catmaster_agent_middleware(runtime=runtime, skills=[])
    first_fs = _middleware_by_type(first, "FilesystemMiddleware")
    first_documents = _middleware_by_type(first, "BoundedDocumentReadMiddleware")
    first_todo = _middleware_by_type(first, "TodoListMiddleware")
    second_fs = _middleware_by_type(second, "FilesystemMiddleware")
    second_todo = _middleware_by_type(second, "TodoListMiddleware")

    assert first_fs is not second_fs
    assert first_todo is not second_todo
    assert first_fs.backend is backend
    assert first_documents.files_root == files_root.resolve()
    assert {tool.name for tool in first_fs.tools} == set(
        runtime_mod._CATMASTER_WRITABLE_FILESYSTEM_TOOLS
    )
    descriptions = {tool.name: tool.description for tool in first_fs.tools}
    assert "bounded text view" in descriptions["read_file"]
    assert "reported integer offset" in descriptions["read_file"]
    assert "replaced in its entirety" in descriptions["write_file"]
    assert "Read an existing file before replacing it" in descriptions["write_file"]
    assert "virtual workspace root `/`" in descriptions["delete"]
    assert descriptions["execute"] == runtime_mod.CATMASTER_EXECUTE_DESCRIPTION
    runtime_prompt = str(first_fs._custom_system_prompt)
    assert f"`{files_root.resolve()}`" in runtime_prompt
    assert "reference for interpreting shell output" in runtime_prompt
    assert "do not pass it to filesystem function tools" in runtime_prompt
    assert "Filesystem function tools use a virtual namespace whose `/`" in runtime_prompt
    assert "complete host-file inspection boundary" in runtime_prompt
    assert "`execute` already starts in the physical directory above" in runtime_prompt
    assert "prefer workspace-relative paths" in runtime_prompt
    assert "not the filesystem-tool virtual root" in runtime_prompt
    assert "host mappings do not authorize host-filesystem exploration" in runtime_prompt
    assert first_todo.system_prompt == ""
    assert first_todo.tools[0].name == "write_todos"
    assert "Every call replaces the whole current list" in first_todo.tools[0].description

    readonly = runner._build_filesystem_middleware(
        backend=backend,
        read_only=True,
    )
    assert readonly.backend is backend
    assert {tool.name for tool in readonly.tools} == set(
        runtime_mod._CATMASTER_READONLY_FILESYSTEM_TOOLS
    )
    readonly_prompt = str(readonly._custom_system_prompt)
    assert f"`{files_root.resolve()}`" in readonly_prompt
    assert "complete host-file inspection boundary" in readonly_prompt
    assert "`execute` already starts" not in readonly_prompt

    _BindableFakeModel.bound_tools = []
    model = _SystemPromptCapturingBindableFakeModel(
        responses=[AIMessage(content="ready")]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=first,
        backend=backend,
    )
    result = agent.invoke({"messages": [{"role": "user", "content": "Check readiness."}]})

    assert result["messages"][-1].content == "ready"
    assert model.system_prompts
    assert f"`{files_root.resolve()}`" in model.system_prompts[-1]
    assert "complete host-file inspection boundary" in model.system_prompts[-1]
    final_surface = next(
        surface
        for surface in _BindableFakeModel.bound_tools
        if "write_todos" in surface and "write_file" in surface
    )
    assert set(final_surface) == {
        "write_todos",
        "ls",
        "read_file",
        "write_file",
        "edit_file",
        "delete",
        "glob",
        "grep",
        "execute",
        "task",
    }
    assert final_surface["write_todos"].description == runtime_mod.CATMASTER_WRITE_TODOS_DESCRIPTION
    read_file_surface = json.dumps(
        {
            "description": final_surface["read_file"].description,
            "schema": final_surface["read_file"].args_schema.model_json_schema(),
        },
        ensure_ascii=False,
    ).lower()
    for forbidden in ("hash", "sha256", "checksum", "digest", "snapshot_ref", "cursor"):
        assert forbidden not in read_file_surface
    assert final_surface["write_file"].description == runtime_mod.CATMASTER_WRITE_FILE_DESCRIPTION
    assert final_surface["delete"].description == runtime_mod.CATMASTER_DELETE_DESCRIPTION
    assert final_surface["execute"].description == runtime_mod.CATMASTER_EXECUTE_DESCRIPTION


def test_native_async_subagent_surface_exposes_upstream_control_tools(
    tmp_path: Path,
) -> None:
    files_root = tmp_path / "workspace" / "files"
    files_root.mkdir(parents=True)
    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    middleware = AsyncSubAgentMiddleware(
        async_subagents=[
            {
                "name": "experiment_specialist",
                "description": "Run a bounded experiment in an independent thread.",
                "graph_id": "assistant-experiment",
            }
        ]
    )
    tool_names = {tool.name for tool in middleware.tools}
    assert tool_names == {
        "start_async_task",
        "check_async_task",
        "update_async_task",
        "cancel_async_task",
        "list_async_tasks",
    }

    _BindableFakeModel.bound_tools = []
    model = _BindableFakeModel(responses=[AIMessage(content="ready")])
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=[middleware],
        backend=backend,
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Plan independent work."}]}
    )

    assert result["messages"][-1].content == "ready"
    final_surface = next(
        surface
        for surface in _BindableFakeModel.bound_tools
        if "start_async_task" in surface
    )
    assert set(final_surface).issuperset(tool_names)
    assert "background" in final_surface["start_async_task"].description.lower()


def test_large_docx_is_bounded_before_the_next_model_call(tmp_path: Path) -> None:
    from docx import Document

    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    document = Document()
    document.add_paragraph("large document text " * 8_000)
    document.save(files_root / "large.docx")

    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    model = _BindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/large.docx"},
                        "id": "read-large-docx",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="bounded document inspected"),
        ]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=runner._catmaster_agent_middleware(
            runtime={"backend": backend},
            skills=[],
        ),
        backend=backend,
    )

    result = agent.invoke({"messages": [HumanMessage(content="Read the report.")]})

    tool_message = next(
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and message.tool_call_id == "read-large-docx"
    )
    assert tool_message.status == "success"
    assert tool_message.additional_kwargs["catmaster_bounded_document_read"] is True
    assert "Bounded DOCX text view" in str(tool_message.content)
    assert "base64" not in str(tool_message.content).lower()
    assert result["messages"][-1].content == "bounded document inspected"


def test_read_file_images_remain_in_active_history_without_custom_eviction(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    for index in range(5):
        (files_root / f"figure-{index}.png").write_bytes(
            b"\x89PNG\r\n\x1a\n" + bytes([index])
        )

    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    responses = [
        AIMessage(
            content="",
            tool_calls=[
                {
                    "name": "read_file",
                    "args": {"file_path": f"/figure-{index}.png"},
                    "id": f"read-image-{index}",
                    "type": "tool_call",
                }
            ],
        )
        for index in range(5)
    ]
    responses.append(AIMessage(content="all images inspected"))
    model = _CapturingBindableFakeModel(responses=responses)
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=runner._catmaster_agent_middleware(
            runtime={"backend": backend},
            skills=[],
        ),
        backend=backend,
    )

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [HumanMessage(content="Inspect the five figures in order.")]}
        )
    )
    image_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
        and str(message.name or "") == "read_file"
    ]

    assert model.inline_image_counts == [0, 1, 2, 3, 4, 5]
    assert [
        bool(message.content[0].get("base64"))
        if isinstance(message.content, list)
        and isinstance(message.content[0], dict)
        else False
        for message in image_results
    ] == [True, True, True, True, True]
    assert image_results[0].content[0]["base64"]
    assert image_results[0].tool_call_id == "read-image-0"
    assert result["messages"][-1].content == "all images inspected"


def test_native_file_todo_and_delete_contract_runs_through_final_graph(
    tmp_path: Path,
) -> None:
    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    runtime = {"backend": backend}
    middleware = runner._catmaster_agent_middleware(runtime=runtime, skills=[])
    todos_active = [{"content": "Exercise file contract", "status": "in_progress"}]
    todos_done = [{"content": "Exercise file contract", "status": "completed"}]
    model = _BindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": todos_active},
                        "id": "todo-active",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/notes/result.txt", "content": "first"},
                        "id": "write-first",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/notes/result.txt", "content": "second"},
                        "id": "write-second",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "edit_file",
                        "args": {
                            "file_path": "/notes/result.txt",
                            "old_string": "second",
                            "new_string": "third",
                        },
                        "id": "edit-third",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/delete_me/a.txt", "content": "a"},
                        "id": "write-delete-a",
                        "type": "tool_call",
                    },
                    {
                        "name": "write_file",
                        "args": {"file_path": "/delete_me/nested/b.txt", "content": "b"},
                        "id": "write-delete-b",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete",
                        "args": {"file_path": "/delete_me"},
                        "id": "delete-subtree",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/protected.txt", "content": "keep"},
                        "id": "write-protected",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "delete",
                        "args": {"file_path": "/"},
                        "id": "delete-root",
                        "type": "tool_call",
                    },
                    {
                        "name": "delete",
                        "args": {"file_path": "/memories"},
                        "id": "delete-memory-root",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_file",
                        "args": {"file_path": "/../escaped.txt", "content": "escape"},
                        "id": "write-escape",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "write_todos",
                        "args": {"todos": todos_done},
                        "id": "todo-done",
                        "type": "tool_call",
                    }
                ],
            ),
            AIMessage(content="contract complete"),
        ]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=middleware,
        backend=backend,
    )

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": "Exercise the file contract."}]}
        )
    )

    assert result["messages"][-1].content == "contract complete"
    assert result["todos"] == todos_done
    assert (files_root / "notes" / "result.txt").read_text(encoding="utf-8") == "third"
    assert not (files_root / "delete_me").exists()
    assert (files_root / "protected.txt").read_text(encoding="utf-8") == "keep"
    assert not (runner.run_context.workspace / "escaped.txt").exists()
    tool_messages = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    ]
    by_id = {message.tool_call_id: message for message in tool_messages}
    assert by_id["write-second"].status == "success"
    assert by_id["edit-third"].status == "success"
    assert by_id["delete-subtree"].status == "success"
    assert by_id["delete-root"].status == "error"
    assert by_id["delete-memory-root"].status == "error"
    assert "protected virtual" in str(by_id["delete-root"].content)
    assert "'/'" in str(by_id["delete-root"].content)
    assert by_id["write-escape"].status == "error"


def test_delete_guard_allows_specific_paths_but_blocks_root_aliases() -> None:
    guard = runtime_mod._CatMasterFilesystemGuardMiddleware()

    def handler(request: Any) -> ToolMessage:
        return ToolMessage(
            content="allowed",
            tool_call_id=request.tool_call["id"],
            name="delete",
            status="success",
        )

    for index, path in enumerate(("", "/", ".", "memories", "/memories/.", "/memories/..")):
        request = SimpleNamespace(
            tool_call={"name": "delete", "args": {"file_path": path}, "id": f"blocked-{index}"}
        )
        result = guard.wrap_tool_call(request, handler)
        assert result.status == "error"

    for index, path in enumerate(("/notes/a.txt", "/notes/subdir", "/memories/entry.md")):
        request = SimpleNamespace(
            tool_call={"name": "delete", "args": {"file_path": path}, "id": f"allowed-{index}"}
        )
        result = guard.wrap_tool_call(request, handler)
        assert result.status == "success"


def test_paginated_read_and_search_truncation_remain_model_visible(
    tmp_path: Path,
) -> None:
    class _TruncatedGlobBackend(LocalShellBackend):
        def glob(self, pattern: str, path: str | None = None) -> GlobResult:
            _ = (pattern, path)
            return GlobResult(matches=[{"path": "/partial.txt"}], truncated=True)

    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    (files_root / "many.txt").write_text(
        "first\nneedle second\nneedle third\nneedle fourth\nfifth\n",
        encoding="utf-8",
    )
    backend = _TruncatedGlobBackend(root_dir=files_root, virtual_mode=True)
    model = _BindableFakeModel(
        responses=[
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "read_file",
                        "args": {"file_path": "/many.txt", "offset": 1, "limit": 2},
                        "id": "read-window",
                        "type": "tool_call",
                    },
                    {
                        "name": "grep",
                        "args": {
                            "pattern": "needle",
                            "path": "/",
                            "output_mode": "content",
                            "max_count": 2,
                        },
                        "id": "grep-capped",
                        "type": "tool_call",
                    },
                    {
                        "name": "glob",
                        "args": {"pattern": "**/*.txt", "path": "/"},
                        "id": "glob-partial",
                        "type": "tool_call",
                    },
                ],
            ),
            AIMessage(content="navigation complete"),
        ]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=runner._catmaster_agent_middleware(
            runtime={"backend": backend},
            skills=[],
        ),
        backend=backend,
    )

    result = asyncio.run(
        agent.ainvoke(
            {"messages": [{"role": "user", "content": "Inspect the bounded fixture."}]}
        )
    )

    tool_messages = {
        message.tool_call_id: message
        for message in result["messages"]
        if isinstance(message, ToolMessage)
    }
    assert "remaining from offset 3" in str(tool_messages["read-window"].content)
    assert "maximum match count" in str(tool_messages["grep-capped"].content)
    assert "raise max_count" in str(tool_messages["grep-capped"].content)
    assert "/partial.txt" in str(tool_messages["glob-partial"].content)
    assert "paths above are valid but incomplete" in str(
        tool_messages["glob-partial"].content
    )


def test_compact_execution_policy_is_present_without_legacy_base_prompt() -> None:
    policy = runtime_mod.SpecialistRunner._deepagent_execution_policy()
    assert "diagnose the cause before retrying" in policy
    assert "close or remove pending task items" in policy
    assert "user's explicit priorities, scope, and verification request" in policy
    assert "unrelated hardware, build, hash, version, schema, manifest, or environment" in policy
    assert "BASE_AGENT_PROMPT" not in policy
    assert policy in runtime_mod.SpecialistRunner._tool_policy()
    assert policy in runtime_mod.SpecialistRunner._general_purpose_child_prompt()
    assert policy in runtime_mod.SpecialistRunner._hypothesis_proposer_prompt()
    assert policy in runtime_mod.SpecialistRunner._litreview_worker_prompt()


def test_deepagents_0612_checkpoint_state_resumes_in_074_without_replaying_task(
    tmp_path: Path,
) -> None:
    fixture_path = (
        Path(__file__).parent / "fixtures" / "deepagents_0612_checkpoint_state.json"
    )
    fixture = json.loads(fixture_path.read_text(encoding="utf-8"))
    assert fixture["generated_with"]["deepagents"] == "0.6.12"
    fixture_state = fixture["state"]
    message_types = {
        "human": HumanMessage,
        "ai": AIMessage,
        "tool": ToolMessage,
    }
    legacy_messages = [
        message_types[row["type"]](
            **{key: value for key, value in row.items() if key != "type"}
        )
        for row in fixture_state["messages"]
    ]
    connection = sqlite3.connect(":memory:", check_same_thread=False)
    checkpointer = SqliteSaver(
        connection,
        serde=FileSafeCheckpointSerializer(),
    )
    config = {
        "configurable": {
            "thread_id": fixture["thread"]["thread_id"],
            "checkpoint_ns": fixture["thread"]["checkpoint_ns"],
        }
    }
    runner = _runner(tmp_path)
    files_root = runner.run_context.workspace / "files"
    backend = LocalShellBackend(root_dir=files_root, virtual_mode=True)
    model = _BindableFakeModel(
        responses=[AIMessage(content="Resumed without replaying the completed task.")]
    )
    agent = create_deep_agent(
        model=model,
        tools=[],
        middleware=runner._catmaster_agent_middleware(
            runtime={"backend": backend},
            skills=[],
        ),
        backend=backend,
        checkpointer=checkpointer,
    )

    seeded_config = agent.update_state(
        config,
        {
            "messages": legacy_messages,
            "todos": fixture_state["todos"],
            "files": fixture_state["files"],
        },
        as_node="model",
    )
    loaded = agent.get_state(seeded_config)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": "Continue this saved thread."}]},
        config=config,
    )

    assert loaded.values["todos"] == fixture_state["todos"]
    assert result["todos"] == fixture_state["todos"]
    message_ids = [message.id for message in result["messages"]]
    assert message_ids[:3] == [
        "legacy-user-1",
        "legacy-assistant-task",
        "legacy-task-result",
    ]
    task_results = [
        message
        for message in result["messages"]
        if isinstance(message, ToolMessage) and message.tool_call_id == "legacy-task-call"
    ]
    assert len(task_results) == 1
    assert task_results[0].content == "Legacy delegated analysis completed once."
    assert result["messages"][-1].content == "Resumed without replaying the completed task."

    resumed_state = agent.get_state(config)
    assert resumed_state.config["configurable"]["thread_id"] == "fixture-deepagents-0612"
    assert resumed_state.values["todos"] == result["todos"]
    assert [message.id for message in resumed_state.values["messages"]] == message_ids
