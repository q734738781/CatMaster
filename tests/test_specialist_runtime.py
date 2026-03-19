from __future__ import annotations

import asyncio
import json
from contextlib import asynccontextmanager
from pathlib import Path
from types import SimpleNamespace

import pytest
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
    _MATERIALS_WORKER_SELECTOR_ALWAYS_INCLUDE,
    _MATERIALS_WORKER_SELECTOR_MAX_TOOLS,
    _MATERIALS_WORKER_TOOL_ALLOWLIST,
    _METADATA_AGENT_TOOL_ALLOWLIST,
    _ML_WORKER_TOOL_ALLOWLIST,
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


class _FakeToolSelectorMiddleware:
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
    assert "replace or delete stale incorrect reports/notes" in contract


def test_materials_worker_prompt_includes_workspace_path_discipline() -> None:
    prompt = runtime_mod.SpecialistRunner._materials_worker_prompt()
    assert "Workspace path discipline" in prompt
    assert "Treat `/` only as the workspace virtual root" in prompt
    assert "Only persist key constraints, decisive results" in prompt
    assert "literature/" in prompt
    assert "structures/" in prompt
    assert "calculations/" in prompt
    assert "notes/" in prompt
    assert "writing/" in prompt


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


def test_default_middleware_uses_configurable_model_call_limit() -> None:
    middleware = runtime_mod.SpecialistRunner._build_default_middleware(model_call_run_limit=88)
    tracker = next(item for item in middleware if getattr(item, "run_limit", None) is not None)
    assert tracker.run_limit == 88


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


@pytest.mark.parametrize(
    ("entrypoint", "expected_skills", "expected_subagent_names"),
    [
        ("research", ["/.deepagents/skill_views/research_experiment", "/.deepagents/skill_views/research_writing"], ["experiment_specialist", "writing_specialist", "litreview_agent"]),
        ("experiment", ["/.deepagents/skill_views/experiment_specialist"], ["materials_worker", "ml_worker", "literature_agent"]),
        ("writing", ["/.deepagents/skill_views/writing_specialist"], ["writing_worker_agent"]),
    ],
)
def test_three_specialist_lanes_start_with_staged_skills(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    entrypoint: str,
    expected_skills: list[str],
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
        staticmethod(lambda: (lambda summarizer: _FakeCompactConversationMiddleware(summarizer))),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_memory_middleware",
        staticmethod(lambda: _FakeMemoryMiddleware),
    )
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_llm_tool_selector_middleware",
        staticmethod(lambda: _FakeToolSelectorMiddleware),
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
    assert agent_kwargs["skills"] == expected_skills
    assert agent_kwargs["memory"] == ["/.deepagents/AGENTS.md", "/memories/AGENTS.md"]
    assert _PROJECT_MEMORY_TOOL_NAMES <= {tool.name for tool in agent_kwargs["tools"]}
    assert "Project long-term memory tools" in agent_kwargs["system_prompt"]
    assert "Never store transient task requests" in agent_kwargs["system_prompt"]
    assert all(getattr(tool, "name", None) != "bash" for tool in agent_kwargs["tools"])
    assert all(getattr(tool, "name", None) != "run_literature_research" for tool in agent_kwargs["tools"])
    assert [subagent.kwargs["name"] for subagent in agent_kwargs["subagents"]] == expected_subagent_names
    for subagent in agent_kwargs["subagents"]:
        if "tools" in subagent.kwargs:
            assert all(getattr(tool, "name", None) != "bash" for tool in subagent.kwargs["tools"])
        if "middleware" in subagent.kwargs:
            middleware_names = {type(item).__name__ for item in (subagent.kwargs.get("middleware") or [])}
            assert any(name == "catmaster_nonfatal_tool_errors" for name in middleware_names)

    subagents_by_name = {subagent.kwargs["name"]: subagent.kwargs for subagent in agent_kwargs["subagents"]}
    if entrypoint == "research":
        assert "Maintain a lightweight Research Kernel" in agent_kwargs["system_prompt"]
        assert "/research_kernels/" in agent_kwargs["system_prompt"]
        assert "litreview_agent" in agent_kwargs["system_prompt"]
        assert "metadata_agent" in agent_kwargs["system_prompt"]
        assert {tool.name for tool in subagents_by_name["experiment_specialist"]["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert {tool.name for tool in subagents_by_name["writing_specialist"]["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["experiment_specialist"]["skills"] == ["/.deepagents/skill_views/experiment_specialist"]
        assert subagents_by_name["writing_specialist"]["skills"] == ["/.deepagents/skill_views/writing_specialist"]
        litreview_compiled = subagents_by_name["litreview_agent"]
        assert "runnable" in litreview_compiled
        litreview_agents = [kwargs for kwargs in created_agents if kwargs["name"] == "litreview_agent"]
        assert litreview_agents, "expected nested litreview agent to be created"
        litreview_agent_kwargs = litreview_agents[0]
        assert {tool.name for tool in litreview_agent_kwargs["tools"]} == _PROJECT_MEMORY_READ_TOOL_NAMES
        assert [subagent.kwargs["name"] for subagent in litreview_agent_kwargs["subagents"]] == ["literature_agent", "metadata_agent"]
        nested_subagents = {subagent.kwargs["name"]: subagent.kwargs for subagent in litreview_agent_kwargs["subagents"]}
        assert {tool.name for tool in nested_subagents["literature_agent"]["tools"]} == (_LITREVIEW_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert nested_subagents["literature_agent"]["model"] == {"model": "literature_synthesizer-model"}
        assert {tool.name for tool in nested_subagents["metadata_agent"]["tools"]} == (_METADATA_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        metadata_middleware = nested_subagents["metadata_agent"]["middleware"]
        metadata_summarizer = next(item for item in metadata_middleware if isinstance(item, _FakeSummarizationMiddleware))
        compact_tool = next(item for item in metadata_middleware if isinstance(item, _FakeCompactConversationMiddleware))
        assert metadata_summarizer.kwargs["trigger"] == ("tokens", _LITREVIEW_COMPACT_TRIGGER_TOKENS)
        assert metadata_summarizer.kwargs["keep"] == ("tokens", _LITREVIEW_COMPACT_KEEP_TOKENS)
        assert compact_tool.summarizer is metadata_summarizer
        literature_middleware = nested_subagents["literature_agent"]["middleware"]
        assert not any(isinstance(item, _FakeSummarizationMiddleware) for item in literature_middleware)
    elif entrypoint == "experiment":
        assert {tool.name for tool in subagents_by_name["materials_worker"]["tools"]} == (_MATERIALS_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["materials_worker"]["skills"] == ["/.deepagents/skill_views/materials_worker"]
        assert {tool.name for tool in subagents_by_name["ml_worker"]["tools"]} == (_ML_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["ml_worker"]["skills"] == ["/.deepagents/skill_views/ml_worker"]
        assert {tool.name for tool in subagents_by_name["literature_agent"]["tools"]} == (_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["literature_agent"]["model"] == {"model": "literature_synthesizer-model"}
        assert "/notes/literature/" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "Only save a concise reusable markdown note" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "Web Evidence" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "You do not have permission to modify long-term project memory" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "dataset/model lifecycle tasks" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Route by the current working artifact" in agent_kwargs["system_prompt"]
        assert "do not stop at that boundary alone" in agent_kwargs["system_prompt"]
        assert "Typical MACE work here includes surrogate screening, relaxation, ranking, and post-analysis" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "Start here when the primary artifact is a curated dataset" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in subagents_by_name["ml_worker"]["system_prompt"]
        materials_worker_selector = next(
            item
            for item in subagents_by_name["materials_worker"]["middleware"]
            if isinstance(item, _FakeToolSelectorMiddleware)
        )
        assert materials_worker_selector.kwargs["model"] == {"model": "tool_selector-model"}
        assert materials_worker_selector.kwargs["max_tools"] == _MATERIALS_WORKER_SELECTOR_MAX_TOOLS
        assert set(materials_worker_selector.kwargs["always_include"]) == set(_MATERIALS_WORKER_SELECTOR_ALWAYS_INCLUDE)
        assert not any(
            isinstance(item, _FakeToolSelectorMiddleware)
            for item in subagents_by_name["ml_worker"]["middleware"]
        )
        assert not any(
            isinstance(item, _FakeToolSelectorMiddleware)
            for item in subagents_by_name["literature_agent"]["middleware"]
        )
    else:
        assert {tool.name for tool in agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "compile_text" not in {tool.name for tool in agent_kwargs["tools"]}
        assert {tool.name for tool in subagents_by_name["writing_worker_agent"]["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["writing_worker_agent"]["skills"] == ["/.deepagents/skill_views/writing_worker_agent"]
        assert "literature_agent" not in subagents_by_name

    staged_agents = workspace / "files" / ".deepagents" / "AGENTS.md"
    staged_experiment = workspace / "files" / ".deepagents" / "skills" / "experiment"
    staged_writing = workspace / "files" / ".deepagents" / "skills" / "writing"
    staged_views = workspace / "files" / ".deepagents" / "skill_views"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_experiment.is_dir()
    assert staged_writing.is_dir()
    assert staged_views.is_dir()
    staged_machine_learning = workspace / "files" / ".deepagents" / "skills" / "machine_learning"
    assert staged_machine_learning.is_dir()
    experiment_view_names = {path.name for path in (staged_views / "experiment_specialist").iterdir() if path.is_dir()}
    writing_view_names = {path.name for path in (staged_views / "writing_specialist").iterdir() if path.is_dir()}
    materials_worker_view_names = {path.name for path in (staged_views / "materials_worker").iterdir() if path.is_dir()}
    ml_worker_view_names = {path.name for path in (staged_views / "ml_worker").iterdir() if path.is_dir()}
    assert "literature-grounding" not in experiment_view_names
    assert "structure-visual-inspection" in experiment_view_names
    assert "structure-visual-inspection" in materials_worker_view_names
    assert {
        "mace-dataset-curation",
        "mace-finetuning-and-benchmark",
        "active-learning-relabel-loop",
    } <= ml_worker_view_names
    assert "achemso-latex-manuscript" not in writing_view_names
    assert "scientific-writing" in writing_view_names

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
    monkeypatch.setattr(
        runtime_mod.SpecialistRunner,
        "_load_llm_tool_selector_middleware",
        staticmethod(lambda: _FakeToolSelectorMiddleware),
    )
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
