from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel

import catmaster.specialists.runtime as runtime_mod
from catmaster.specialists.runtime import (
    RUN_STATE_FILE,
    _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST,
    _format_lightweight_internet_search_content,
    _LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES,
    _MATERIALS_WORKER_TOOL_ALLOWLIST,
    _METADATA_AGENT_TOOL_ALLOWLIST,
    _ML_WORKER_TOOL_ALLOWLIST,
    _ORCA_XTB_WORKER_TOOL_ALLOWLIST,
    _PROJECT_MEMORY_READ_TOOL_NAMES,
    _PROJECT_MEMORY_TOOL_NAMES,
    _LITREVIEW_AGENT_TOOL_ALLOWLIST,
    _LITREVIEW_COMPACT_KEEP_TOKENS,
    _LITREVIEW_COMPACT_TRIGGER_TOKENS,
    _RESEARCH_TOOL_ALLOWLIST,
    _WRITING_WORKER_TOOL_ALLOWLIST,
    _WRITING_TOOL_ALLOWLIST,
    build_specialist_runner,
)
from catmaster.runtime.usage_stats import load_usage_summary
from catmaster.runtime.artifact_callback import UIEventHandler
from catmaster.runtime.run_control import RunControl
from catmaster.tools.registry import get_tool_registry


class _FakeProfile:
    def config_for_role(self, role: str) -> SimpleNamespace:
        return SimpleNamespace(model=f"{role}-model", provider="langchain", base_url=None)


class _FakeToolStrategy:
    def __init__(self, schema, handle_errors: bool = False) -> None:
        self.schema = schema
        self.handle_errors = handle_errors


class _FakeSubAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCompiledSubAgent:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeSummarizationMiddleware:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeCompactConversationMiddleware:
    def __init__(self, summarizer) -> None:
        self.summarizer = summarizer


class _FakeMemoryMiddleware:
    def __init__(self, **kwargs) -> None:
        self.kwargs = kwargs


class _FakeSearchMemoryInput(BaseModel):
    query: str
    limit: int = 10
    offset: int = 0
    filter: dict | None = None


class _FakeManageMemoryInput(BaseModel):
    content: str | None = None
    action: str = "create"
    id: str | None = None


class _FakeDeepAgent:
    def __init__(self, *, kwargs: dict) -> None:
        self.kwargs = kwargs

    async def ainvoke(self, payload, config=None):
        _ = config
        assert payload["messages"][0]["role"] == "user"
        name = self.kwargs["name"]
        if name == "research_specialist":
            content = "## Summary\nresearch summary\n\n## Facts\n- grounded by literature agent when needed\n\n## Files\n- reports/research.md"
        elif name == "writing_specialist":
            content = "## Summary\nwriting summary\n\n## Facts\n- manuscript draft updated\n\n## Files\n- drafts/report.md"
        else:
            content = "## Summary\nexperiment summary\n\n## Facts\n- bounded execution completed\n\n## Files\n- experiments/out.json"
        return {"messages": [AIMessage(content=content)]}


def _fake_create_search_memory_tool(*, namespace, instructions="", response_format="content", name="search_memory", **kwargs):
    _ = (namespace, instructions, response_format, kwargs)

    def _search_memory(query: str, limit: int = 10, offset: int = 0, filter: dict | None = None):
        _ = (limit, offset, filter)
        return json.dumps([{"key": "mem-1", "value": {"content": f"memory about {query}"}}])

    return StructuredTool.from_function(
        func=_search_memory,
        name=name,
        description="fake search memory tool",
        args_schema=_FakeSearchMemoryInput,
        infer_schema=False,
        response_format="content",
    )


def _fake_create_manage_memory_tool(*, namespace, instructions="", actions_permitted=("create", "update", "delete"), name="manage_memory", **kwargs):
    _ = (namespace, instructions, actions_permitted, kwargs)

    def _manage_memory(content: str | None = None, action: str = "create", *, id: str | None = None):
        _ = content
        target = id or "generated-id"
        return f"{action}d memory {target}"

    return StructuredTool.from_function(
        func=_manage_memory,
        name=name,
        description="fake manage memory tool",
        args_schema=_FakeManageMemoryInput,
        infer_schema=False,
        response_format="content",
    )


class _FakeUsageCallback:
    def __init__(self) -> None:
        self.usage_metadata = {
            "task_runner-model": {
                "input_tokens": 123,
                "output_tokens": 17,
                "total_tokens": 140,
                "input_token_details": {"cache_read": 80},
                "output_token_details": {"reasoning": 5},
            }
        }
        self.call_counts_by_model = {"task_runner-model": 2}
        self.usage_metadata_by_role = {
            "experiment_specialist": {
                "task_runner-model": {
                    "input_tokens": 40,
                    "output_tokens": 7,
                    "total_tokens": 47,
                    "input_token_details": {"cache_read": 10},
                }
            }
        }
        self.call_counts_by_role = {"experiment_specialist": 1}


class _FailingToolInput(BaseModel):
    value: str


def test_real_registry_covers_specialist_allowlists() -> None:
    registry = get_tool_registry()
    registered = set(registry.tools)

    assert "write_note" not in registered
    assert _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST <= registered
    assert _RESEARCH_TOOL_ALLOWLIST <= registered
    assert _WRITING_TOOL_ALLOWLIST <= registered
    assert _MATERIALS_WORKER_TOOL_ALLOWLIST <= registered
    assert _METADATA_AGENT_TOOL_ALLOWLIST <= registered
    assert _LITREVIEW_AGENT_TOOL_ALLOWLIST <= registered
    assert _WRITING_WORKER_TOOL_ALLOWLIST <= registered
    assert "bash" not in _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
    assert "bash" not in _RESEARCH_TOOL_ALLOWLIST
    assert "bash" not in _WRITING_TOOL_ALLOWLIST
    assert "run_literature_research" in registered


def test_specialist_callbacks_include_ui_event_handler(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=SimpleNamespace(emit=lambda event: None),
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    callbacks = built.runner._langchain_callbacks(usage_handler=None, default_agent_name="experiment_specialist")
    ui_callbacks = [callback for callback in callbacks if isinstance(callback, UIEventHandler)]
    assert ui_callbacks
    assert ui_callbacks[0].default_agent_name == "experiment_specialist"


def test_specialist_tool_wrapper_returns_nonfatal_error_payload(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="writing",
    )

    def _boom(runtime=None, **kwargs):
        _ = (runtime, kwargs)
        raise RuntimeError("simulated failure")

    failing_tool = StructuredTool.from_function(
        func=_boom,
        name="polish_academic_prose",
        description="fail on purpose",
        args_schema=_FailingToolInput,
        infer_schema=False,
        response_format="content_and_artifact",
    )

    monkeypatch.setattr(
        built.runner.registry,
        "as_langchain_tools",
        lambda allowlist, run_dir, workspace: [failing_tool],
    )
    monkeypatch.setitem(built.runner.registry.tools, "polish_academic_prose", {"function": object()})

    wrapped = built.runner._named_tools({"polish_academic_prose"})
    content, artifact = wrapped[0].func(value="x")
    assert "simulated failure" in content
    assert artifact["tool_name"] == "polish_academic_prose"
    assert artifact["data"]["status"] == "error"
    assert artifact["data"]["tool_name"] == "polish_academic_prose"


def test_specialist_reporting_contract_requires_direct_answer_and_relative_paths() -> None:
    contract = runtime_mod.SpecialistRunner._soft_reporting_contract()
    assert "directly answer the user's actual question" in contract
    assert "workspace-relative output paths" in contract
    assert "optional `ReviewTarget` section" in contract
    assert "replace or delete stale incorrect reports/notes" in contract


def test_writing_reporting_contract_allows_summary_first_closeout() -> None:
    contract = runtime_mod.SpecialistRunner._writing_reporting_contract()
    assert "required section is `Summary`" in contract
    assert "Include a `Files` section only when" in contract
    assert "optional `ReviewTarget` section" in contract
    assert "Do not add a placeholder `Facts` section" in contract


def test_report_parser_supports_review_target() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    summary, facts, files, review_target = runner._parse_summary_and_files(
        "## Summary\nok\n\n## Facts\n- a\n\n## Files\n- `manuscript/paper.pdf`\n\n## ReviewTarget\n- `manuscript/paper.pdf`"
    )
    assert summary == "ok"
    assert facts == ["a"]
    assert files == ["manuscript/paper.pdf"]
    assert review_target == "manuscript/paper.pdf"


def test_materials_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._materials_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "Do not pass guessed input paths into tools" in prompt
    assert "never use leading-slash workspace paths like `/writing/...`" in prompt
    assert "Only persist key constraints, decisive results" in prompt
    assert "literature/" in prompt
    assert "structures/" in prompt
    assert "calculations/" in prompt
    assert "notes/" in prompt
    assert "writing/" in prompt


def test_orca_xtb_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._orca_xtb_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "molecular quantum-chemistry subtask" in prompt
    assert "`create_molecule_from_smiles`" in prompt
    assert "`xtb_run_batch`" in prompt
    assert "`orca_execute_batch`" in prompt
    assert "first create the structure under `<topic>/structures/`" in prompt
    assert "Do not guess that a path like `<topic>/structures/<name>.xyz` already exists" in prompt
    assert "default to `xtb_run_batch` or `crest_conformer_search`" in prompt
    assert "do not choose ORCA-XTB as the default fallback for routine preopt steps" in prompt


def test_common_worker_prompts_require_relevant_skill_check() -> None:
    expected = "Tool discipline: if a relevant skill is available to the current agent, read it before acting."

    assert expected in runtime_mod.SpecialistRunner._materials_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._ml_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._orca_xtb_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._writing_worker_prompt()
    assert expected in runtime_mod.SpecialistRunner._writing_polisher_prompt()
    assert expected in runtime_mod.SpecialistRunner._peer_review_worker_prompt()


def test_specialist_prompts_require_explicit_follow_on_delegate_judgment() -> None:
    research_prompt = runtime_mod.SpecialistRunner._base_system_prompt("research", thread_id="thread-1")
    experiment_prompt = runtime_mod.SpecialistRunner._base_system_prompt("experiment")
    writing_prompt = runtime_mod.SpecialistRunner._base_system_prompt("writing")
    peer_review_prompt = runtime_mod.SpecialistRunner._base_system_prompt("peer_review")

    assert "actively judge from the user's request, current evidence, and actual project state whether another bounded delegation round is needed" in research_prompt
    assert "do not default to closing in the research thread just because one delegate completed" in research_prompt
    assert "issue a bounded probe to `experiment_specialist` rather than deciding from absence in the research thread" in research_prompt
    assert "When one worker pass returns, actively decide whether another bounded delegate pass is needed" in experiment_prompt
    assert "delegate a bounded probe to the matching worker instead of concluding the capability is absent" in experiment_prompt
    assert "When one writing-worker pass returns, actively decide whether another bounded delegate pass is needed" in writing_prompt
    assert "When one worker review episode returns, actively decide whether another bounded delegate pass is needed" in peer_review_prompt


def test_execution_capability_contract_distinguishes_local_and_managed_runtime(tmp_path: Path) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="research",
    )

    research_contract = built.runner._execution_capability_contract(audience="research")
    materials_contract = built.runner._execution_capability_contract(audience="materials_worker")
    ml_contract = built.runner._execution_capability_contract(audience="ml_worker")
    orca_contract = built.runner._execution_capability_contract(audience="orca_xtb_worker")

    assert "Do not infer managed-execution availability from local shell probing alone." in research_contract
    assert "downgrading it to literature-only validation" in research_contract
    assert "`vasp_execute_batch`" in materials_contract
    assert "do not require a local periodic DFT engine to be directly runnable first" in materials_contract
    assert "`mace_train`" in ml_contract
    assert "registered managed-execution path" in ml_contract
    assert "prefer the registered managed tools when they fit the task" in ml_contract
    assert "continue by writing and running local workspace scripts instead of blocking on tool coverage" in ml_contract
    assert "`xtb_run_batch`" in orca_contract
    assert "`orca_execute_batch`" in orca_contract
    assert "serious molecular quantum-chemistry runs" in orca_contract


def test_writing_worker_and_proposal_prompts_include_workspace_layout_guidance() -> None:
    writing_prompt = runtime_mod.SpecialistRunner._writing_worker_prompt()
    proposal_prompt = runtime_mod.SpecialistRunner._proposal_system_prompt("experiment")
    assert "Project long-term memory tools" in writing_prompt
    assert "Only persist key constraints, decisive results" in writing_prompt
    assert "structures/" in writing_prompt
    assert "calculations/" in writing_prompt
    assert "notes/" in writing_prompt
    assert "writing/" in writing_prompt
    assert "Workspace path discipline" in proposal_prompt
    assert "literature/" in proposal_prompt
    assert "writing/" in proposal_prompt


def test_lightweight_internet_search_content_omits_raw_content() -> None:
    content = _format_lightweight_internet_search_content(
        {
            "query": "Pt(111) hydrogen adsorption benchmark DOI",
            "topic": "general",
            "results": [
                {
                    "title": "Hydrogen adsorption and diffusion on Pt {111}",
                    "url": "https://example.org/paper",
                    "content": "Representative DFT study discussing adsorption sites and diffusion barriers.",
                    "raw_content": "RAW " * 500,
                }
            ],
        },
        max_results=5,
    )

    assert "Query: Pt(111) hydrogen adsorption benchmark DOI" in content
    assert "Top results:" in content
    assert "Hydrogen adsorption and diffusion on Pt {111}" in content
    assert "Representative DFT study discussing adsorption sites" in content
    assert "raw page content was returned by Tavily but omitted" in content
    assert "RAW RAW RAW" not in content


def test_lightweight_internet_search_content_formats_error_compactly() -> None:
    content = _format_lightweight_internet_search_content(
        {
            "status": "error",
            "source": "tavily",
            "query": "Pt(111) H adsorption",
            "message": "temporary upstream failure",
        }
    )

    assert "internet_search failed" in content
    assert "Pt(111) H adsorption" in content
    assert "temporary upstream failure" in content
    assert "\n" not in content


def test_default_tool_error_middleware_returns_tool_message() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    handler_mw = middleware[-1]

    class _Request:
        tool_call = {
            "id": "call-1",
            "name": "create_molecule_from_smiles",
            "args": {"smiles": "C#O"},
        }

    async def _handler(_request):
        raise runtime_mod.CatMasterToolExecutionError(
            tool_name="create_molecule_from_smiles",
            public_message="Failed to build molecule from SMILES: Invalid SMILES: C#O",
            artifact={"tool_name": "create_molecule_from_smiles", "data": {"smiles": "C#O"}},
            error_code="molecule_build_failed",
        )

    async def _run():
        return await handler_mw.awrap_tool_call(_Request(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ToolMessage)
    assert result.status == "error"
    assert "Invalid SMILES" in str(result.content)
    assert result.tool_call_id == "call-1"


def test_tool_result_middleware_textualizes_multimodal_tool_messages() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    history_mw = middleware[1]

    class _Request:
        tool_call = {
            "id": "call-1",
            "name": "read_file",
            "args": {"file_path": "/paper/page.png"},
        }

    async def _handler(_request):
        return ToolMessage(
            content_blocks=[
                {
                    "type": "image",
                    "id": "img-1",
                    "base64": "not-for-history",
                    "mime_type": "image/png",
                }
            ],
            additional_kwargs={
                "read_file_path": "/paper/page.png",
                "read_file_media_type": "image/png",
            },
            tool_call_id="call-1",
            name="read_file",
            status="success",
        )

    async def _run():
        return await history_mw.awrap_tool_call(_Request(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ToolMessage)
    assert isinstance(result.content, str)
    assert "image tool content omitted from persistent history" in result.content
    assert "path=/paper/page.png" in result.content
    assert "not-for-history" not in result.content
    assert result.tool_call_id == "call-1"


def test_model_retry_middleware_retries_empty_ai_response(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    model_mw = middleware[0]
    sleeps: list[float] = []
    attempts = {"count": 0}

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _handler(_request):
        attempts["count"] += 1
        if attempts["count"] < 3:
            return ModelResponse(result=[AIMessage(content="")])
        return ModelResponse(result=[AIMessage(content="## Summary\nok")])

    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _fake_sleep)

    async def _run():
        return await model_mw.awrap_model_call(object(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ModelResponse)
    assert attempts["count"] == 3
    assert sleeps == [60.0, 180.0]


def test_model_retry_middleware_sanitizes_multimodal_tool_history() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    model_mw = middleware[0]
    seen_messages = []

    class _Request:
        def __init__(self, messages):
            self.messages = messages

        def override(self, **kwargs):
            return _Request(kwargs.get("messages", self.messages))

    request = _Request(
        [
            ToolMessage(
                content_blocks=[
                    {
                        "type": "image",
                        "id": "img-1",
                        "base64": "not-for-model-history",
                        "mime_type": "image/png",
                    }
                ],
                additional_kwargs={"read_file_path": "/paper/page.png"},
                tool_call_id="call-1",
                name="read_file",
            )
        ]
    )

    async def _handler(sanitized_request):
        seen_messages.extend(sanitized_request.messages)
        return ModelResponse(result=[AIMessage(content="## Summary\nok")])

    async def _run():
        return await model_mw.awrap_model_call(request, _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ModelResponse)
    assert isinstance(seen_messages[0], ToolMessage)
    assert isinstance(seen_messages[0].content, str)
    assert "not-for-model-history" not in seen_messages[0].content
    assert "path=/paper/page.png" in seen_messages[0].content


def test_model_retry_middleware_accepts_tool_calls_without_text() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    model_mw = middleware[0]

    async def _handler(_request):
        return ModelResponse(
            result=[
                AIMessage(
                    content="",
                    tool_calls=[{"name": "vasp_prepare", "args": {"path": "x"}, "id": "call-1", "type": "tool_call"}],
                    response_metadata={"finish_reason": "tool_calls"},
                )
            ]
        )

    async def _run():
        return await model_mw.awrap_model_call(object(), _handler)

    result = asyncio.run(_run())
    assert isinstance(result, ModelResponse)
    assert result.result[0].tool_calls


def test_model_retry_middleware_retries_openrouter_unmarshaller_error(monkeypatch: pytest.MonkeyPatch) -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware()
    model_mw = middleware[0]
    sleeps: list[float] = []
    attempts = {"count": 0}

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    async def _handler(_request):
        attempts["count"] += 1
        if attempts["count"] == 1:
            raise ValueError(
                "ValidationError: 40 validation errors for Unmarshaller "
                "body.174.tool.content.str Input should be a valid string"
            )
        return ModelResponse(result=[AIMessage(content="## Summary\nok")])

    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _fake_sleep)

    async def _run():
        return await model_mw.awrap_model_call(object(), _handler)

    result = asyncio.run(_run())

    assert isinstance(result, ModelResponse)
    assert attempts["count"] == 2
    assert sleeps == [60.0]


def test_extract_final_text_ignores_user_message_fallback() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    raw = {
        "messages": [
            {"role": "user", "content": "please do the calculation"},
            {"role": "assistant", "content": ""},
        ]
    }
    assert runner._extract_final_text(raw) == ""


def test_message_text_ignores_reasoning_blocks() -> None:
    message = AIMessage(
        content=[
            {"type": "reasoning", "text": "hidden chain"},
            {"type": "text", "text": "## Summary\nusable"},
        ]
    )
    assert runtime_mod.SpecialistRunner._message_text(message) == "## Summary\nusable"


def test_coerce_report_requires_summary_heading() -> None:
    runner = runtime_mod.SpecialistRunner(
        llm_profile=_FakeProfile(),
        run_context=SimpleNamespace(workspace=Path("/tmp"), run_dir=Path("/tmp"), run_id="r1", project_id="proj"),
        reporter=None,
        run_control=None,
    )
    with pytest.raises(runtime_mod.SpecialistInvalidFinalReportError, match="required `Summary` section"):
        runner._coerce_report(raw={"messages": [AIMessage(content="plain echo without headings")]})


def test_specialist_usage_callback_tracks_agent_scoped_usage() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(default_agent_name="writing_specialist")
    message = AIMessage(
        content="done",
        response_metadata={"model_name": "openai/gpt-5.4-20260305"},
        usage_metadata={
            "input_tokens": 25,
            "output_tokens": 4,
            "total_tokens": 29,
            "input_token_details": {"cache_read": 6},
            "output_token_details": {"reasoning": 2},
        },
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start({}, [[]], run_id="run-1", metadata={"agent_name": "writing_specialist"})
    handler.on_llm_end(result, run_id="run-1")

    assert handler.call_counts_by_model["openai/gpt-5.4-20260305"] == 1
    assert handler.call_counts_by_role["writing_specialist"] == 1
    assert handler.usage_metadata_by_role["writing_specialist"]["openai/gpt-5.4-20260305"]["input_tokens"] == 25


def test_specialist_usage_callback_falls_back_to_default_agent_name() -> None:
    handler = runtime_mod.SpecialistUsageCallbackHandler(default_agent_name="experiment_specialist")
    message = AIMessage(
        content="done",
        response_metadata={"model_name": "openai/gpt-5.4-20260305"},
        usage_metadata={"input_tokens": 10, "output_tokens": 3, "total_tokens": 13},
    )
    result = LLMResult(generations=[[ChatGeneration(message=message)]])

    handler.on_chat_model_start({}, [[]], run_id="run-2")
    handler.on_llm_end(result, run_id="run-2")

    assert handler.call_counts_by_role["experiment_specialist"] == 1
    assert handler.usage_metadata_by_role["experiment_specialist"]["openai/gpt-5.4-20260305"]["total_tokens"] == 13


def test_finalize_report_runs_compile_guard_for_tex_outputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    (workspace / "files" / "writeup").mkdir(parents=True)
    tex_path = workspace / "files" / "writeup" / "note.tex"
    tex_path.write_text("\\documentclass{article}\\begin{document}Hi\\end{document}\n", encoding="utf-8")

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="writing",
    )

    def _fake_compile(payload):
        assert payload == {"source_path": "writeup/note.tex"}
        return (
            "compiled",
            {
                "tool_name": "compile_text",
                "data": {
                    "compiled_ok": True,
                    "pdf_path": "writeup/note.pdf",
                    "bib_paths": ["writeup/references.bib"],
                    "inspected_files": ["writeup/note.tex", "writeup/references.bib"],
                    "remaining_diagnostics": [],
                },
            },
        )

    monkeypatch.setattr(built.runner.registry, "get_tool_function", lambda name: _fake_compile if name == "compile_text" else None)

    finalized = built.runner._finalize_report(
        {
            "text": "## Summary\nshort\n\n## Facts\n- one\n\n## Files\n- `writeup/note.tex`",
            "summary": "short",
            "facts": ["one"],
            "files": ["writeup/note.tex"],
        }
    )

    assert finalized["files"] == ["writeup/note.tex", "writeup/note.pdf", "writeup/references.bib"]
    assert any("Compile guard produced `writeup/note.pdf`" in fact for fact in finalized["facts"])
    assert "`writeup/note.pdf`" in finalized["text"]
    assert "`writeup/references.bib`" in finalized["text"]


def test_render_compact_report_omits_empty_sections() -> None:
    rendered = runtime_mod.SpecialistRunner._render_compact_report(
        summary="draft revised",
        facts=[],
        files=[],
    )

    assert rendered == "## Summary\ndraft revised"


def test_run_impl_retries_invalid_final_report_and_recovers(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)

    created_agents: list[dict] = []
    sleeps: list[float] = []

    class _RetryAgent:
        def __init__(self) -> None:
            self.calls = 0

        async def ainvoke(self, payload, config=None):
            _ = (payload, config)
            self.calls += 1
            if self.calls == 1:
                return {"messages": [{"role": "user", "content": "echoed prompt"}]}
            return {"messages": [AIMessage(content="## Summary\nrecovered\n\n## Facts\n- ok")]}

    retry_agent = _RetryAgent()

    def _fake_create_deep_agent(**kwargs):
        created_agents.append(kwargs)
        return retry_agent

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    async def _fake_sleep(delay: float) -> None:
        sleeps.append(delay)

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_entry_subagents", lambda self, entrypoint, runtime: [])
    monkeypatch.setattr(runtime_mod.asyncio, "sleep", _fake_sleep)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Design the stage-2/3 plan.",
            entrypoint="experiment",
            proposal_review=False,
        )
    )

    assert result["status"] == "done"
    assert result["summary"] == "recovered"
    assert retry_agent.calls == 2
    assert sleeps == [30.0]


@pytest.mark.parametrize(
    ("entrypoint", "expected_subagent_names"),
    [
        ("research", ["experiment_specialist", "writing_specialist", "peer_review_specialist", "litreview_agent"]),
        ("experiment", ["materials_worker", "ml_worker", "orca_xtb_worker", "literature_agent"]),
        ("writing", ["literature_agent", "writing_worker_agent", "writing_polisher_agent"]),
        ("peer_review", ["peer_review_worker_agent"]),
    ],
)
def test_three_specialist_lanes_start_with_staged_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected_subagent_names: list[str],
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    (workspace / "AGENTS.md").write_text("Project-level instructions.", encoding="utf-8")

    created_agents: list[dict] = []

    def _fake_create_deep_agent(**kwargs):
        created_agents.append(kwargs)
        return _FakeDeepAgent(kwargs=kwargs)

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_tool_strategy", staticmethod(lambda: _FakeToolStrategy))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_summarization_middleware",
        staticmethod(lambda: _FakeSummarizationMiddleware),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_summarization_tool_middleware",
        staticmethod(lambda: (lambda model, backend: _FakeCompactConversationMiddleware({"model": model, "backend": backend}))),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_memory_middleware",
        staticmethod(lambda: _FakeMemoryMiddleware),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_search_memory_tool",
        staticmethod(lambda: _fake_create_search_memory_tool),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_create_manage_memory_tool",
        staticmethod(lambda: _fake_create_manage_memory_tool),
    )
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj",
        preferred_entrypoint=entrypoint,
    )

    result = asyncio.run(
        built.runner.arun(
            "Run the lane smoke test.",
            entrypoint=entrypoint,
            proposal_review=False,
        )
    )

    assert result["status"] == "done"
    assert created_agents, "expected create_deep_agent to be called"
    agent_kwargs = created_agents[-1]
    assert agent_kwargs["name"] == f"{entrypoint}_specialist"
    assert "skills" not in agent_kwargs
    assert agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
    assert _PROJECT_MEMORY_TOOL_NAMES <= {tool.name for tool in agent_kwargs["tools"]}
    assert "Project long-term memory tools" in agent_kwargs["system_prompt"]
    assert "Never store transient task requests" in agent_kwargs["system_prompt"]
    assert all(getattr(tool, "name", None) != "bash" for tool in agent_kwargs["tools"])
    assert all(getattr(tool, "name", None) != "run_literature_research" for tool in agent_kwargs["tools"])
    top_subagents = list(agent_kwargs.get("subagents") or [])
    assert [subagent.kwargs["name"] for subagent in top_subagents] == expected_subagent_names
    for subagent in top_subagents:
        if "tools" in subagent.kwargs:
            assert all(getattr(tool, "name", None) != "bash" for tool in subagent.kwargs["tools"])
        if "middleware" in subagent.kwargs:
            middleware_names = {type(item).__name__ for item in (subagent.kwargs.get("middleware") or [])}
            assert any(name == "catmaster_nonfatal_tool_errors" for name in middleware_names)

    subagents_by_name = {subagent.kwargs["name"]: subagent.kwargs for subagent in top_subagents}

    def _created_agents_named(name: str) -> list[dict]:
        return [kwargs for kwargs in created_agents if kwargs["name"] == name]

    def _find_created_agent(
        name: str,
        *,
        tool_names: set[str] | None = None,
        prompt_contains: str | None = None,
    ) -> dict:
        matches = _created_agents_named(name)
        if tool_names is not None:
            matches = [kwargs for kwargs in matches if {tool.name for tool in kwargs["tools"]} == tool_names]
        if prompt_contains is not None:
            matches = [kwargs for kwargs in matches if prompt_contains in kwargs["system_prompt"]]
        assert matches, f"expected created agent {name!r}"
        return matches[0]

    if entrypoint == "research":
        assert "Maintain a lightweight Research Kernel" in agent_kwargs["system_prompt"]
        assert "/research_kernels/" in agent_kwargs["system_prompt"]
        assert "litreview_agent" in agent_kwargs["system_prompt"]
        assert "metadata_agent" in agent_kwargs["system_prompt"]
        assert "paper, manuscript, journal-style LaTeX draft" in agent_kwargs["system_prompt"]
        assert "experiment report, validation summary, QC note" in agent_kwargs["system_prompt"]
        assert "compact inline author packet" in agent_kwargs["system_prompt"]
        assert "compact inline report packet" in agent_kwargs["system_prompt"]
        assert "Default to not launching `peer_review_specialist`" in agent_kwargs["system_prompt"]
        assert "publication-level paper quality" in agent_kwargs["system_prompt"]
        assert "formal submission requirements" in agent_kwargs["system_prompt"]
        assert "explicitly hand it the canonical workspace-relative manuscript PDF path" in agent_kwargs["system_prompt"]
        assert "Do not rely on the Research Kernel to preserve full editor/reviewer comment text" in agent_kwargs["system_prompt"]
        assert "If `peer_review_specialist` gives you a saved review memo path, read that memo directly" in agent_kwargs["system_prompt"]
        assert "You remain the sole coordinator and final decision-maker" in agent_kwargs["system_prompt"]
        assert "you may relaunch `experiment_specialist` for bounded follow-up work" in agent_kwargs["system_prompt"]
        assert "do not default to closing in the research thread just because one delegate completed" in agent_kwargs["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in agent_kwargs["system_prompt"]
        assert "downgrading it to literature-only validation" in agent_kwargs["system_prompt"]
        assert "issue a bounded probe to `experiment_specialist`" in agent_kwargs["system_prompt"]
        assert "If a bounded local task needs a missing Python package, `execute` may install it with `python -m pip install ...`" in agent_kwargs["system_prompt"]
        assert "runnable" in subagents_by_name["experiment_specialist"]
        assert "runnable" in subagents_by_name["writing_specialist"]
        assert "runnable" in subagents_by_name["peer_review_specialist"]

        experiment_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "experiment_specialist"]
        assert experiment_agents, "expected nested experiment specialist to be created"
        experiment_agent_kwargs = experiment_agents[0]
        assert {tool.name for tool in experiment_agent_kwargs["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "mace_neb_batch" not in {tool.name for tool in experiment_agent_kwargs["tools"]}
        assert "skills" not in experiment_agent_kwargs
        assert experiment_agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
        assert [subagent.kwargs["name"] for subagent in experiment_agent_kwargs["subagents"]] == [
            "materials_worker",
            "ml_worker",
            "orca_xtb_worker",
            "literature_agent",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in experiment_agent_kwargs["middleware"])

        writing_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "writing_specialist"]
        assert writing_agents, "expected nested writing specialist to be created"
        writing_agent_kwargs = writing_agents[0]
        assert {tool.name for tool in writing_agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "skills" not in writing_agent_kwargs
        assert writing_agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
        assert [subagent.kwargs["name"] for subagent in writing_agent_kwargs["subagents"]] == [
            "literature_agent",
            "writing_worker_agent",
            "writing_polisher_agent",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in writing_agent_kwargs["middleware"])

        peer_review_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "peer_review_specialist"]
        assert peer_review_agents, "expected nested peer-review specialist to be created"
        peer_review_agent_kwargs = peer_review_agents[0]
        assert {tool.name for tool in peer_review_agent_kwargs["tools"]} == ({"peer_review_request"} | _PROJECT_MEMORY_TOOL_NAMES)
        assert "skills" not in peer_review_agent_kwargs
        assert "Act like a journal editor coordinating external peer review" in peer_review_agent_kwargs["system_prompt"]
        assert "explicit `ReviewTarget` or manuscript PDF path" in peer_review_agent_kwargs["system_prompt"]
        assert "delegate the bounded review episode to `peer_review_worker_agent`" in peer_review_agent_kwargs["system_prompt"]
        assert [subagent.kwargs["name"] for subagent in peer_review_agent_kwargs["subagents"]] == [
            "peer_review_worker_agent",
        ]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in peer_review_agent_kwargs["middleware"])
        litreview_compiled = subagents_by_name["litreview_agent"]
        assert "runnable" in litreview_compiled
        litreview_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "litreview_agent"]
        assert litreview_agents, "expected nested litreview agent to be created"
        litreview_agent_kwargs = litreview_agents[0]
        assert {tool.name for tool in litreview_agent_kwargs["tools"]} == _PROJECT_MEMORY_READ_TOOL_NAMES
        assert litreview_agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in litreview_agent_kwargs["middleware"])
        assert [subagent.kwargs["name"] for subagent in litreview_agent_kwargs["subagents"]] == ["literature_agent", "metadata_agent"]
        nested_subagents = {subagent.kwargs["name"]: subagent.kwargs for subagent in litreview_agent_kwargs["subagents"]}
        assert "runnable" in nested_subagents["literature_agent"]
        assert "runnable" in nested_subagents["metadata_agent"]
        litreview_literature_kwargs = _find_created_agent(
            "literature_agent",
            tool_names=_LITREVIEW_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES,
        )
        assert litreview_literature_kwargs["model"] == {"model": "literature_synthesizer-model"}
        metadata_agent_kwargs = _find_created_agent(
            "metadata_agent",
            tool_names=_METADATA_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES,
        )
        metadata_middleware = metadata_agent_kwargs["middleware"]
        compact_tool = next(item for item in metadata_middleware if isinstance(item, _FakeCompactConversationMiddleware))
        assert compact_tool.summarizer["model"] == {"model": "literature_deep_research-model"}
        assert compact_tool.summarizer["backend"] is not None
        assert not any(isinstance(item, _FakeSummarizationMiddleware) for item in metadata_middleware)
        literature_middleware = litreview_literature_kwargs["middleware"]
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in literature_middleware)
        assert not any(isinstance(item, _FakeMemoryMiddleware) for item in metadata_middleware)
        assert not any(isinstance(item, _FakeSummarizationMiddleware) for item in literature_middleware)
    elif entrypoint == "experiment":
        materials_worker_kwargs = _find_created_agent("materials_worker")
        ml_worker_kwargs = _find_created_agent("ml_worker")
        orca_worker_kwargs = _find_created_agent("orca_xtb_worker")
        literature_agent_kwargs = _find_created_agent(
            "literature_agent",
            tool_names=_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES,
        )
        assert "runnable" in subagents_by_name["materials_worker"]
        assert "runnable" in subagents_by_name["ml_worker"]
        assert "runnable" in subagents_by_name["orca_xtb_worker"]
        assert "runnable" in subagents_by_name["literature_agent"]
        assert {tool.name for tool in materials_worker_kwargs["tools"]} == (_MATERIALS_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert materials_worker_kwargs["skills"] == ["/.deepagents/skills/materials"]
        assert {tool.name for tool in ml_worker_kwargs["tools"]} == (_ML_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert ml_worker_kwargs["skills"] == ["/.deepagents/skills/machine_learning"]
        assert {tool.name for tool in orca_worker_kwargs["tools"]} == (_ORCA_XTB_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert orca_worker_kwargs["skills"] == ["/.deepagents/skills/quantum_chemistry"]
        assert {tool.name for tool in literature_agent_kwargs["tools"]} == (_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert literature_agent_kwargs["model"] == {"model": "literature_synthesizer-model"}
        assert {tool.name for tool in agent_kwargs["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "mace_neb_batch" not in {tool.name for tool in agent_kwargs["tools"]}
        assert "/notes/literature/" in literature_agent_kwargs["system_prompt"]
        assert "Only save a concise reusable markdown note" in literature_agent_kwargs["system_prompt"]
        assert "Web Evidence" in literature_agent_kwargs["system_prompt"]
        assert "You do not have permission to modify long-term project memory" in materials_worker_kwargs["system_prompt"]
        assert "dataset/model lifecycle tasks" in ml_worker_kwargs["system_prompt"]
        assert "default role is coordination, dispatch, and decision-making across the experiment lane" in agent_kwargs["system_prompt"]
        assert "Keep direct work in the specialist thread minimal and coordination-oriented" in agent_kwargs["system_prompt"]
        assert "Route by the current working artifact" in agent_kwargs["system_prompt"]
        assert "When a request clearly falls into one of those worker-owned domains, delegate first instead of doing the domain work yourself." in agent_kwargs["system_prompt"]
        assert "general materials or surface workflows belong to `materials_worker`" in agent_kwargs["system_prompt"]
        assert "model fine-tuning, training, evaluation, feature/data pipelines, and ML algorithm development belong to `ml_worker`" in agent_kwargs["system_prompt"]
        assert "use `orca_xtb_worker` for molecular or cluster quantum-chemistry work" in agent_kwargs["system_prompt"]
        assert "purely report writing from already completed evidence" in agent_kwargs["system_prompt"]
        assert "stays in `ExperimentSpecialist`" in agent_kwargs["system_prompt"]
        assert "Each worker should receive only one bounded execution episode around one primary artifact" in agent_kwargs["system_prompt"]
        assert "Do not hand an entire high-throughput campaign to one worker" in agent_kwargs["system_prompt"]
        assert "Do not personally absorb worker-owned tasks just because your own direct tool surface appears sufficient" in agent_kwargs["system_prompt"]
        assert "Only do the implementation directly in the specialist thread when no available worker matches the task" in agent_kwargs["system_prompt"]
        assert "Delegate domain-owned work to the proper specialized subagent first." in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in agent_kwargs["system_prompt"]
        assert "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context." in agent_kwargs["system_prompt"]
        assert "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread." in agent_kwargs["system_prompt"]
        assert "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents." in agent_kwargs["system_prompt"]
        assert "do not stop at that boundary alone" in agent_kwargs["system_prompt"]
        assert "prefer a quick built-in web check through the online model's native browsing capability" in agent_kwargs["system_prompt"]
        assert "prefer materializing it as a reusable workspace script under `scripts/`" in agent_kwargs["system_prompt"]
        assert "If a worker needs a handy Python package for a bounded local step and it is missing" in agent_kwargs["system_prompt"]
        assert "mace_neb_batch" in {tool.name for tool in materials_worker_kwargs["tools"]}
        assert "Typical MACE work here includes surrogate screening, relaxation, ranking, and post-analysis" in materials_worker_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in materials_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in materials_worker_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in materials_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context." in materials_worker_kwargs["system_prompt"]
        assert "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread." in materials_worker_kwargs["system_prompt"]
        assert "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents." in materials_worker_kwargs["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in materials_worker_kwargs["system_prompt"]
        assert "obtain POTCARs through the pymatgen interface" in materials_worker_kwargs["system_prompt"]
        assert "install it with `execute` via `python -m pip install ...`" in materials_worker_kwargs["system_prompt"]
        assert "If a handy Python package is missing for a bounded local step" in materials_worker_kwargs["system_prompt"]
        assert "use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check" in materials_worker_kwargs["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in materials_worker_kwargs["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in materials_worker_kwargs["system_prompt"]
        assert "do not require a local periodic DFT engine to be directly runnable first" in materials_worker_kwargs["system_prompt"]
        assert "Start here when the primary artifact is a curated dataset" in ml_worker_kwargs["system_prompt"]
        assert "When a registered managed ML tool fits the task, prefer that managed path first." in ml_worker_kwargs["system_prompt"]
        assert "prefer `build_dataset_from_runs`, `mace_train`, and `mace_evaluate` over ad hoc local wrapper scripts" in ml_worker_kwargs["system_prompt"]
        assert "Do not create or run a local `mace_run_train` wrapper when `mace_train` already fits the request" in ml_worker_kwargs["system_prompt"]
        assert "Prefer using libraries already available in the environment and reusable workspace code" in ml_worker_kwargs["system_prompt"]
        assert "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `joblib`, and `matminer`" in ml_worker_kwargs["system_prompt"]
        assert "If a handy Python package is still missing for a bounded local step" in ml_worker_kwargs["system_prompt"]
        assert "If the ML logic is longer than a short throwaway snippet and no managed tool covers it" in ml_worker_kwargs["system_prompt"]
        assert "Prefer organizing topic-specific ML scripts under `scripts/<topic>/`" in ml_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context." in ml_worker_kwargs["system_prompt"]
        assert "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread." in ml_worker_kwargs["system_prompt"]
        assert "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents." in ml_worker_kwargs["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in ml_worker_kwargs["system_prompt"]
        assert "install it with `execute` via `python -m pip install ...`" in ml_worker_kwargs["system_prompt"]
        assert "Prefer materializing training pipelines, feature generation, sweeps, evaluation harnesses, embedding workflows, and data-processing logic as reusable scripts" in ml_worker_kwargs["system_prompt"]
        assert "Use remote execution when the job is heavy, long-running, batch-oriented, or needs managed compute; MACE training/fine-tuning normally falls into this category." in ml_worker_kwargs["system_prompt"]
        assert "Treat the managed ML tools as preferred paths when they fit, not as an exclusive gate" in ml_worker_kwargs["system_prompt"]
        assert "keep going locally with reusable scripts under `scripts/` instead of stopping" in ml_worker_kwargs["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in ml_worker_kwargs["system_prompt"]
        assert "registered managed-execution path" in ml_worker_kwargs["system_prompt"]
        assert "use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check" in ml_worker_kwargs["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in ml_worker_kwargs["system_prompt"]
        assert "molecular quantum-chemistry subtask" in orca_worker_kwargs["system_prompt"]
        assert "`enumerate_molecular_conformers`" in orca_worker_kwargs["system_prompt"]
        assert "`orca_execute_batch`" in orca_worker_kwargs["system_prompt"]
        assert "Treat xTB/CREST as the fast exploration layer" in orca_worker_kwargs["system_prompt"]
        assert "default to `xtb_run_batch` or `crest_conformer_search`" in orca_worker_kwargs["system_prompt"]
        assert "install it with `execute` via `python -m pip install ...`" in orca_worker_kwargs["system_prompt"]
        assert "If a handy Python package is missing for a bounded local step" in orca_worker_kwargs["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in orca_worker_kwargs["system_prompt"]
        assert "When one worker pass returns, actively decide whether another bounded delegate pass is needed" in agent_kwargs["system_prompt"]
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in materials_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in orca_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in ml_worker_kwargs["middleware"]
        )
        assert not any(
            type(item).__name__ == "_FakeToolSelectorMiddleware"
            for item in literature_agent_kwargs["middleware"]
        )
    elif entrypoint == "writing":
        assert {tool.name for tool in agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "compile_text" not in {tool.name for tool in agent_kwargs["tools"]}
        writing_literature_kwargs = _find_created_agent(
            "literature_agent",
            tool_names=_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES,
        )
        writing_worker_kwargs = _find_created_agent("writing_worker_agent")
        writing_polisher_kwargs = _find_created_agent("writing_polisher_agent")
        assert "runnable" in subagents_by_name["literature_agent"]
        assert "runnable" in subagents_by_name["writing_worker_agent"]
        assert "runnable" in subagents_by_name["writing_polisher_agent"]
        assert {tool.name for tool in writing_literature_kwargs["tools"]} == (_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert writing_literature_kwargs["model"] == {"model": "literature_synthesizer-model"}
        assert "Use the lightweight `internet_search` tool only for tightly bounded writing-support lookups" in writing_literature_kwargs["system_prompt"]
        assert {tool.name for tool in writing_worker_kwargs["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert writing_worker_kwargs["skills"] == ["/.deepagents/skills/writing"]
        assert {tool.name for tool in writing_polisher_kwargs["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert writing_polisher_kwargs["skills"] == ["/.deepagents/skills/writing"]
        assert "This lane owns paper, manuscript, and author-facing scientific writing" in agent_kwargs["system_prompt"]
        assert "compact inline author packet" in agent_kwargs["system_prompt"]
        assert "Use `writing_polisher_agent` only for local prose cleanup" in agent_kwargs["system_prompt"]
        assert "You may use `literature_agent` only for narrow background supplementation" in agent_kwargs["system_prompt"]
        assert "Each writing-worker handoff should cover only one section or one bounded organization/integration task" in agent_kwargs["system_prompt"]
        assert "figures, tables, and concise explanatory schematics as part of the default deliverable" in agent_kwargs["system_prompt"]
        assert "Supporting Information / Supporting Data package" in agent_kwargs["system_prompt"]
        assert "keep Supporting Information in the same manuscript file" in agent_kwargs["system_prompt"]
        assert "place it after the references" in agent_kwargs["system_prompt"]
        assert "journal-style title centered on the chemical system and principal scientific finding" in agent_kwargs["system_prompt"]
        assert "figures to be inserted near their first substantive discussion rather than batched at the end" in agent_kwargs["system_prompt"]
        assert "run `review_pdf_manuscript` once on that PDF for comment-only publication-readiness review" in agent_kwargs["system_prompt"]
        assert "reconcile the manuscript against the accepted suggestions and run one more bounded polishing/revision pass" in agent_kwargs["system_prompt"]
        assert "clearly exposed as `ReviewTarget`" in agent_kwargs["system_prompt"]
        assert "publishable paper ready to enter peer review" in agent_kwargs["system_prompt"]
        assert "Do not mention the workspace, files, runs, prompts, tools, agents, interruptions" in agent_kwargs["system_prompt"]
        assert "Delegate domain-owned work to the proper specialized subagent first." in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in agent_kwargs["system_prompt"]
        assert "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context." in agent_kwargs["system_prompt"]
        assert "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread." in agent_kwargs["system_prompt"]
        assert "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents." in agent_kwargs["system_prompt"]
        assert "Handle only one section or one bounded organization/integration task at a time" in writing_worker_kwargs["system_prompt"]
        assert "compact author packet" in writing_worker_kwargs["system_prompt"]
        assert "organize what belongs in the main text versus Supporting Information / Supporting Data" in writing_worker_kwargs["system_prompt"]
        assert "keep Supporting Information in the same manuscript file" in writing_worker_kwargs["system_prompt"]
        assert "For short notes or compact summaries, do not manufacture extra visuals" in writing_worker_kwargs["system_prompt"]
        assert "Use `generate_nanobanana_figure` for conceptual, mechanistic, or workflow figures" in writing_worker_kwargs["system_prompt"]
        assert "produce a compact journal-style title" in writing_worker_kwargs["system_prompt"]
        assert "do not batch figures into a later block" in writing_worker_kwargs["system_prompt"]
        assert "Do not treat a successful TeX compile as sufficient" in writing_worker_kwargs["system_prompt"]
        assert "publishable paper ready to enter peer review" in writing_worker_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in writing_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in writing_worker_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in writing_worker_kwargs["system_prompt"]
        assert "Perform conservative section-level prose polish" in writing_polisher_kwargs["system_prompt"]
        assert "without changing claim strength, scientific scope, evidence selection" in writing_polisher_kwargs["system_prompt"]
        assert "For journal-facing citations and BibTeX, use publication-style metadata only" in writing_worker_kwargs["system_prompt"]
        assert "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context." in writing_worker_kwargs["system_prompt"]
        assert "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread." in writing_worker_kwargs["system_prompt"]
        assert "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents." in writing_worker_kwargs["system_prompt"]
    else:
        assert {tool.name for tool in agent_kwargs["tools"]} == ({"peer_review_request"} | _PROJECT_MEMORY_TOOL_NAMES)
        assert "Act like a journal editor coordinating external peer review" in agent_kwargs["system_prompt"]
        assert "Reviewer Comments" in agent_kwargs["system_prompt"]
        assert "peer_review_request" in agent_kwargs["system_prompt"]
        assert "save the full review as one durable workspace markdown memo" in agent_kwargs["system_prompt"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in agent_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in agent_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in agent_kwargs["system_prompt"]
        assert "do not compress away the editor comment or reviewer comment sections" in agent_kwargs["system_prompt"]
        assert "peer_review_worker_agent" in subagents_by_name
        peer_review_worker_kwargs = _find_created_agent("peer_review_worker_agent")
        assert "runnable" in subagents_by_name["peer_review_worker_agent"]
        assert {tool.name for tool in peer_review_worker_kwargs["tools"]} == ({"peer_review_request"} | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert peer_review_worker_kwargs["skills"] == ["/.deepagents/skills/writing"]
        assert "Tool discipline: if a relevant skill is available to the current agent, read it before acting." in peer_review_worker_kwargs["system_prompt"]
        assert "Prefer registered builtin tools when they fit the task." in peer_review_worker_kwargs["system_prompt"]
        assert "export_builtin_tool_source" in peer_review_worker_kwargs["system_prompt"]
        assert "call `peer_review_request` on that PDF exactly once for this episode" in peer_review_worker_kwargs["system_prompt"]

    staged_agents = workspace / "files" / ".deepagents" / "AGENTS.md"
    staged_materials = workspace / "files" / ".deepagents" / "skills" / "materials"
    staged_writing = workspace / "files" / ".deepagents" / "skills" / "writing"
    staged_quantum_chemistry = workspace / "files" / ".deepagents" / "skills" / "quantum_chemistry"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_materials.is_dir()
    assert staged_writing.is_dir()
    assert staged_quantum_chemistry.is_dir()
    staged_machine_learning = workspace / "files" / ".deepagents" / "skills" / "machine_learning"
    assert staged_machine_learning.is_dir()
    materials_skill_names = {path.name for path in staged_materials.iterdir() if path.is_dir()}
    writing_skill_names = {path.name for path in staged_writing.iterdir() if path.is_dir()}
    ml_worker_view_names = {path.name for path in staged_machine_learning.iterdir() if path.is_dir()}
    orca_xtb_worker_view_names = {path.name for path in staged_quantum_chemistry.iterdir() if path.is_dir()}
    assert "literature-grounding" in materials_skill_names
    assert "structure-visual-inspection" in materials_skill_names
    assert "materials-discovery-and-bulk-selection" in materials_skill_names
    assert "neb-prepare" in materials_skill_names
    assert "neb-calculation" in materials_skill_names
    assert "neb-analysis" in materials_skill_names
    assert {
        "mace-dataset-curation",
        "mace-finetuning-and-benchmark",
        "active-learning-relabel-loop",
    } <= ml_worker_view_names
    assert {
        "conformer-search-and-preopt",
        "xtb-screen-and-prune",
        "orca-optfreq-thermochemistry",
        "scan-to-ts",
        "nebts-and-irc",
        "nmr-ensemble-workup",
    } <= orca_xtb_worker_view_names
    assert "achemso-latex-manuscript" in writing_skill_names
    assert "results-and-discussion-writing" in writing_skill_names
    assert "scientific-writing" in writing_skill_names

    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["entrypoint"] == entrypoint
    assert run_state["status"] == "done"
    assert run_state["summary"]
    assert isinstance(run_state.get("facts"), list)
    if entrypoint == "research":
        assert run_state["research_kernel_path"].endswith("/kernel.json")
        assert run_state["research_kernel"]["question"] == "Run the lane smoke test."
        assert run_state["research_kernel"]["hypotheses"] == []
        kernel_file = workspace / "files" / run_state["research_kernel_path"]
        assert kernel_file.is_file()
    usage_summary = load_usage_summary(built.run_context.run_dir)
    assert usage_summary["source"] == "langchain_usage_metadata"
    assert usage_summary["input_tokens"] == 123
    assert usage_summary["input_cached_tokens"] == 80
    assert usage_summary["output_tokens"] == 17
    assert usage_summary["reasoning_tokens"] == 5
    assert usage_summary["calls"] == 2
    assert usage_summary["by_role"][0]["name"] == "experiment_specialist"
    assert usage_summary["by_role"][0]["calls"] == 1

def test_specialist_run_passes_project_id_to_langmem_namespace(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- stored\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_tool_strategy", staticmethod(lambda: _FakeToolStrategy))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_search_memory_tool", staticmethod(lambda: _fake_create_search_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_manage_memory_tool", staticmethod(lambda: _fake_create_manage_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_memory_ns",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Remember durable project facts when justified.",
            entrypoint="experiment",
            proposal_review=False,
            thread_id="thread-123",
        )
    )

    assert result["status"] == "done"
    config = captured["config"]
    assert isinstance(config, dict)
    assert config["configurable"]["thread_id"] == "thread-123"
    assert config["configurable"]["project_id"] == "proj_memory_ns"


def test_proposal_review_flag_is_ignored_and_run_executes_immediately(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- executed directly\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_search_memory_tool", staticmethod(lambda: _fake_create_search_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_manage_memory_tool", staticmethod(lambda: _fake_create_manage_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_proposal_gate",
        preferred_entrypoint="experiment",
    )

    result = asyncio.run(
        built.runner.arun(
            "Run the experiment lane directly.",
            entrypoint="experiment",
            proposal_review=True,
        )
    )

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Run the experiment lane directly."
    assert "Human review feedback" not in payload["messages"][0]["content"]
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
    assert run_state["proposal_review"] is False
    assert run_state["proposal_revision_count"] == 0


def test_legacy_proposal_wait_state_can_resume_into_normal_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {}

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- resumed legacy proposal run\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_memory_middleware", staticmethod(lambda: _FakeMemoryMiddleware))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_search_memory_tool", staticmethod(lambda: _fake_create_search_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_manage_memory_tool", staticmethod(lambda: _fake_create_manage_memory_tool))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_open_agent_runtime", _fake_open_agent_runtime)
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_new_usage_callback", staticmethod(lambda: _FakeUsageCallback()))

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_resume_legacy_proposal",
        preferred_entrypoint="experiment",
    )
    (built.run_context.run_dir / RUN_STATE_FILE).write_text(
        json.dumps(
            {
                "schema_version": 1,
                "entrypoint": "experiment",
                "status": "awaiting_human_feedback",
                "phase": "proposal_review",
                "active_specialist": "experiment",
                "thread_id": "thread-legacy",
                "proposal_review": True,
                "proposal_revision_count": 2,
                "pending_human_input": {
                    "kind": "proposal_review",
                    "approval_token": "approve",
                },
                "todo_items": [],
                "artifacts": [],
                "delegation_log": [],
                "user_prompt": "Resume this old stuck run.",
                "chat_session_id": "chat-legacy",
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )

    result = asyncio.run(built.runner.aresume("approve"))

    assert result["status"] == "done"
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Resume this old stuck run."
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
    assert run_state["proposal_review"] is False
    assert run_state["proposal_revision_count"] == 0


def test_specialist_runner_returns_interrupted_paused_when_interrupt_requested_before_start(
    tmp_path: Path,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    run_control = RunControl(run_id="run_interrupt")
    run_control.request_interrupt(source="ui", note="stop")

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=run_control,
        project_id="proj_interrupt_before_start",
        preferred_entrypoint="research",
    )

    result = asyncio.run(
        built.runner.arun(
            "Stop before any deepagent work starts.",
            entrypoint="research",
            proposal_review=False,
        )
    )

    assert result["status"] == "interrupted_paused"
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "interrupted_paused"
    assert run_state["summary"] == "Run interrupted by user."
