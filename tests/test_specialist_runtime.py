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
from catmaster.specialists.schemas import ProposalCheckpoint
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


def test_render_compact_report_omits_empty_sections() -> None:
    rendered = runtime_mod.SpecialistRunner._render_compact_report(
        summary="draft revised",
        facts=[],
        files=[],
    )

    assert rendered == "## Summary\ndraft revised"


@pytest.mark.parametrize(
    ("entrypoint", "expected_skills", "expected_subagent_names"),
    [
        ("research", ["/.deepagents/skill_views/research_experiment", "/.deepagents/skill_views/research_writing"], ["experiment_specialist", "writing_specialist", "peer_review_specialist", "litreview_agent"]),
        ("experiment", ["/.deepagents/skill_views/experiment_specialist"], ["materials_worker", "ml_worker", "orca_xtb_worker", "literature_agent", "report_worker_agent"]),
        ("writing", ["/.deepagents/skill_views/writing_specialist"], ["literature_agent", "writing_worker_agent", "writing_polisher_agent"]),
        ("peer_review", ["/.deepagents/skill_views/peer_review_specialist"], []),
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
        staticmethod(lambda: (lambda model, backend: _FakeCompactConversationMiddleware({"model": model, "backend": backend}))),
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
    top_subagents = list(agent_kwargs.get("subagents") or [])
    assert [subagent.kwargs["name"] for subagent in top_subagents] == expected_subagent_names
    for subagent in top_subagents:
        if "tools" in subagent.kwargs:
            assert all(getattr(tool, "name", None) != "bash" for tool in subagent.kwargs["tools"])
        if "middleware" in subagent.kwargs:
            middleware_names = {type(item).__name__ for item in (subagent.kwargs.get("middleware") or [])}
            assert any(name == "catmaster_nonfatal_tool_errors" for name in middleware_names)

    subagents_by_name = {subagent.kwargs["name"]: subagent.kwargs for subagent in top_subagents}
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
        assert "Do not infer managed-execution availability from local shell probing alone." in agent_kwargs["system_prompt"]
        assert "downgrading it to literature-only validation" in agent_kwargs["system_prompt"]
        assert "mace_neb_batch" in {tool.name for tool in subagents_by_name["experiment_specialist"]["tools"]}
        assert {tool.name for tool in subagents_by_name["experiment_specialist"]["tools"]} == (_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert {tool.name for tool in subagents_by_name["writing_specialist"]["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert {tool.name for tool in subagents_by_name["peer_review_specialist"]["tools"]} == ({"peer_review_request"} | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["experiment_specialist"]["skills"] == ["/.deepagents/skill_views/experiment_specialist"]
        assert subagents_by_name["writing_specialist"]["skills"] == ["/.deepagents/skill_views/writing_specialist"]
        assert subagents_by_name["peer_review_specialist"]["skills"] == ["/.deepagents/skill_views/peer_review_specialist"]
        assert "Act like a journal editor coordinating external peer review" in subagents_by_name["peer_review_specialist"]["system_prompt"]
        assert "explicit `ReviewTarget` or manuscript PDF path" in subagents_by_name["peer_review_specialist"]["system_prompt"]
        assert "call `peer_review_request` on that PDF exactly once per review episode" in subagents_by_name["peer_review_specialist"]["system_prompt"]
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
        assert {tool.name for tool in nested_subagents["literature_agent"]["tools"]} == (_LITREVIEW_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert nested_subagents["literature_agent"]["model"] == {"model": "literature_synthesizer-model"}
        assert {tool.name for tool in nested_subagents["metadata_agent"]["tools"]} == (_METADATA_AGENT_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        metadata_middleware = nested_subagents["metadata_agent"]["middleware"]
        compact_tool = next(item for item in metadata_middleware if isinstance(item, _FakeCompactConversationMiddleware))
        assert compact_tool.summarizer["model"] == {"model": "literature_deep_research-model"}
        assert compact_tool.summarizer["backend"] is not None
        assert not any(isinstance(item, _FakeSummarizationMiddleware) for item in metadata_middleware)
        literature_middleware = nested_subagents["literature_agent"]["middleware"]
        assert any(isinstance(item, _FakeMemoryMiddleware) for item in literature_middleware)
        assert any(isinstance(item, _FakeMemoryMiddleware) for item in metadata_middleware)
        assert not any(isinstance(item, _FakeSummarizationMiddleware) for item in literature_middleware)
    elif entrypoint == "experiment":
        assert {tool.name for tool in subagents_by_name["materials_worker"]["tools"]} == (_MATERIALS_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["materials_worker"]["skills"] == ["/.deepagents/skill_views/materials_worker"]
        assert {tool.name for tool in subagents_by_name["ml_worker"]["tools"]} == (_ML_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["ml_worker"]["skills"] == ["/.deepagents/skill_views/ml_worker"]
        assert {tool.name for tool in subagents_by_name["orca_xtb_worker"]["tools"]} == (_ORCA_XTB_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["orca_xtb_worker"]["skills"] == ["/.deepagents/skill_views/orca_xtb_worker"]
        assert {tool.name for tool in subagents_by_name["literature_agent"]["tools"]} == (_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["literature_agent"]["model"] == {"model": "literature_synthesizer-model"}
        assert {tool.name for tool in subagents_by_name["report_worker_agent"]["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["report_worker_agent"]["skills"] == ["/.deepagents/skill_views/report_worker_agent"]
        assert "/notes/literature/" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "Only save a concise reusable markdown note" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "Web Evidence" in subagents_by_name["literature_agent"]["system_prompt"]
        assert "You do not have permission to modify long-term project memory" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "dataset/model lifecycle tasks" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Route by the current working artifact" in agent_kwargs["system_prompt"]
        assert "use `report_worker_agent` for experiment reports, validation summaries, QC notes" in agent_kwargs["system_prompt"]
        assert "use `orca_xtb_worker` for molecular or cluster quantum-chemistry work" in agent_kwargs["system_prompt"]
        assert "purely report writing from already completed evidence" in agent_kwargs["system_prompt"]
        assert "Each worker should receive only one bounded execution episode around one primary artifact" in agent_kwargs["system_prompt"]
        assert "Do not hand an entire high-throughput campaign to one worker" in agent_kwargs["system_prompt"]
        assert "Do not rely on raw inline multimodal tool outputs remaining replay-safe" in agent_kwargs["system_prompt"]
        assert "do not stop at that boundary alone" in agent_kwargs["system_prompt"]
        assert "prefer a quick built-in web check through the online model's native browsing capability" in agent_kwargs["system_prompt"]
        assert "prefer materializing it as a reusable workspace script under `scripts/`" in agent_kwargs["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in agent_kwargs["system_prompt"]
        assert "For periodic DFT, the intended path is to prepare inputs locally, submit via `vasp_execute_batch`" in agent_kwargs["system_prompt"]
        assert "mace_neb_batch" in {tool.name for tool in subagents_by_name["materials_worker"]["tools"]}
        assert "Typical MACE work here includes surrogate screening, relaxation, ranking, and post-analysis" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "prefer keeping them as workspace artifacts and refer to them by path plus a short textual summary" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in subagents_by_name["materials_worker"]["system_prompt"]
        assert "do not require a local periodic DFT engine to be directly runnable first" in subagents_by_name["materials_worker"]["system_prompt"]
        assert "Start here when the primary artifact is a curated dataset" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "When a registered managed ML tool fits the task, prefer that managed path first." in subagents_by_name["ml_worker"]["system_prompt"]
        assert "prefer `build_dataset_from_runs`, `mace_train`, and `mace_evaluate` over ad hoc local wrapper scripts" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Do not create or run a local `mace_run_train` wrapper when `mace_train` already fits the request" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Prefer using libraries already available in the environment and reusable workspace code" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `joblib`, and `matminer`" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "If the ML logic is longer than a short throwaway snippet and no managed tool covers it" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Prefer organizing topic-specific ML scripts under `scripts/<topic>/`" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "prefer keeping them as workspace artifacts and refer to them by path plus a short textual summary" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "use `execute` to implement the missing step with Python and mature third-party libraries" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Prefer materializing training pipelines, feature generation, sweeps, evaluation harnesses, embedding workflows, and data-processing logic as reusable scripts" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Use remote execution when the job is heavy, long-running, batch-oriented, or needs managed compute; MACE training/fine-tuning normally falls into this category." in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Treat the managed ML tools as preferred paths when they fit, not as an exclusive gate" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "keep going locally with reusable scripts under `scripts/` instead of stopping" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in subagents_by_name["ml_worker"]["system_prompt"]
        assert "registered managed-execution path" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "write a reusable workspace script under `scripts/`" in subagents_by_name["ml_worker"]["system_prompt"]
        assert "molecular quantum-chemistry subtask" in subagents_by_name["orca_xtb_worker"]["system_prompt"]
        assert "`enumerate_molecular_conformers`" in subagents_by_name["orca_xtb_worker"]["system_prompt"]
        assert "`orca_execute_batch`" in subagents_by_name["orca_xtb_worker"]["system_prompt"]
        assert "Treat xTB/CREST as the fast exploration layer" in subagents_by_name["orca_xtb_worker"]["system_prompt"]
        assert "Do not infer managed-execution availability from local shell probing alone." in subagents_by_name["orca_xtb_worker"]["system_prompt"]
        assert "compact report packet" in subagents_by_name["report_worker_agent"]["system_prompt"]
        assert "it is not a paper/manuscript lane" in subagents_by_name["report_worker_agent"]["system_prompt"]
        assert "Do not restart calculations" in subagents_by_name["report_worker_agent"]["system_prompt"]
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
    elif entrypoint == "writing":
        assert {tool.name for tool in agent_kwargs["tools"]} == (_WRITING_TOOL_ALLOWLIST | _PROJECT_MEMORY_TOOL_NAMES)
        assert "compile_text" not in {tool.name for tool in agent_kwargs["tools"]}
        assert {tool.name for tool in subagents_by_name["literature_agent"]["tools"]} == (_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["literature_agent"]["model"] == {"model": "literature_synthesizer-model"}
        assert "Use the lightweight `internet_search` tool only for tightly bounded writing-support lookups" in subagents_by_name["literature_agent"]["system_prompt"]
        assert {tool.name for tool in subagents_by_name["writing_worker_agent"]["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["writing_worker_agent"]["skills"] == ["/.deepagents/skill_views/writing_worker_agent"]
        assert {tool.name for tool in subagents_by_name["writing_polisher_agent"]["tools"]} == (_WRITING_WORKER_TOOL_ALLOWLIST | _PROJECT_MEMORY_READ_TOOL_NAMES)
        assert subagents_by_name["writing_polisher_agent"]["skills"] == ["/.deepagents/skill_views/writing_polisher_agent"]
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
        assert "Do not rely on raw inline multimodal tool outputs remaining replay-safe" in agent_kwargs["system_prompt"]
        assert "Handle only one section or one bounded organization/integration task at a time" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "compact author packet" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "organize what belongs in the main text versus Supporting Information / Supporting Data" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "keep Supporting Information in the same manuscript file" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "For short notes or compact summaries, do not manufacture extra visuals" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "Use `generate_nanobanana_figure` for conceptual, mechanistic, or workflow figures" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "produce a compact journal-style title" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "do not batch figures into a later block" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "Do not treat a successful TeX compile as sufficient" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "publishable paper ready to enter peer review" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "Perform conservative section-level prose polish" in subagents_by_name["writing_polisher_agent"]["system_prompt"]
        assert "without changing claim strength, scientific scope, evidence selection" in subagents_by_name["writing_polisher_agent"]["system_prompt"]
        assert "For journal-facing citations and BibTeX, use publication-style metadata only" in subagents_by_name["writing_worker_agent"]["system_prompt"]
        assert "prefer keeping them as workspace artifacts and refer to them by path plus a short textual summary" in subagents_by_name["writing_worker_agent"]["system_prompt"]
    else:
        assert {tool.name for tool in agent_kwargs["tools"]} == ({"peer_review_request"} | _PROJECT_MEMORY_TOOL_NAMES)
        assert "Act like a journal editor coordinating external peer review" in agent_kwargs["system_prompt"]
        assert "Reviewer Comments" in agent_kwargs["system_prompt"]
        assert "peer_review_request" in agent_kwargs["system_prompt"]
        assert "save the full review as one durable workspace markdown memo" in agent_kwargs["system_prompt"]
        assert "do not compress away the editor comment or reviewer comment sections" in agent_kwargs["system_prompt"]

    staged_agents = workspace / "files" / ".deepagents" / "AGENTS.md"
    staged_experiment = workspace / "files" / ".deepagents" / "skills" / "experiment"
    staged_writing = workspace / "files" / ".deepagents" / "skills" / "writing"
    staged_quantum_chemistry = workspace / "files" / ".deepagents" / "skills" / "quantum_chemistry"
    staged_views = workspace / "files" / ".deepagents" / "skill_views"
    assert staged_agents.read_text(encoding="utf-8") == "Project-level instructions."
    assert staged_experiment.is_dir()
    assert staged_writing.is_dir()
    assert staged_quantum_chemistry.is_dir()
    assert staged_views.is_dir()
    staged_machine_learning = workspace / "files" / ".deepagents" / "skills" / "machine_learning"
    assert staged_machine_learning.is_dir()
    experiment_view_names = {path.name for path in (staged_views / "experiment_specialist").iterdir() if path.is_dir()}
    writing_view_names = {path.name for path in (staged_views / "writing_specialist").iterdir() if path.is_dir()}
    materials_worker_view_names = {path.name for path in (staged_views / "materials_worker").iterdir() if path.is_dir()}
    ml_worker_view_names = {path.name for path in (staged_views / "ml_worker").iterdir() if path.is_dir()}
    orca_xtb_worker_view_names = {path.name for path in (staged_views / "orca_xtb_worker").iterdir() if path.is_dir()}
    report_worker_view_names = {path.name for path in (staged_views / "report_worker_agent").iterdir() if path.is_dir()}
    writing_worker_view_names = {path.name for path in (staged_views / "writing_worker_agent").iterdir() if path.is_dir()}
    writing_polisher_view_names = {path.name for path in (staged_views / "writing_polisher_agent").iterdir() if path.is_dir()}
    peer_review_view_names = {path.name for path in (staged_views / "peer_review_specialist").iterdir() if path.is_dir()}
    assert "literature-grounding" in experiment_view_names
    assert "structure-visual-inspection" in experiment_view_names
    assert "structure-visual-inspection" in materials_worker_view_names
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
    assert "achemso-latex-manuscript" in writing_view_names
    assert "results-and-discussion-writing" in writing_view_names
    assert "scientific-writing" in writing_view_names
    assert "scientific-writing" in report_worker_view_names
    assert "achemso-latex-manuscript" in writing_worker_view_names
    assert "scientific-writing" in writing_polisher_view_names
    assert "scientific-writing" in peer_review_view_names

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

def test_sanitize_model_request_messages_collapses_inline_image_tool_messages() -> None:
    tool_message = ToolMessage(
        content=[{"type": "image", "id": "img_1", "base64": "abc", "mime_type": "image/png"}],
        tool_call_id="call_1",
        name="read_file",
        status="success",
        additional_kwargs={
            "read_file_path": "/writing/demo/figure.png",
            "read_file_media_type": "image/png",
        },
    )
    untouched = AIMessage(content="ok")

    sanitized = runtime_mod.SpecialistRunner._sanitize_model_request_messages([untouched, tool_message])

    assert sanitized[0] is untouched
    assert isinstance(sanitized[1], ToolMessage)
    assert isinstance(sanitized[1].content, str)
    assert "inline image tool output omitted from model history" in sanitized[1].content
    assert "source=/writing/demo/figure.png" in sanitized[1].content
    assert "mime=image/png" in sanitized[1].content


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


def test_proposal_review_requires_explicit_approve_before_execution(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    workspace = tmp_path / "project_space"
    workspace.mkdir(parents=True)
    captured: dict[str, object] = {"proposal_calls": []}
    proposal_versions = [
        ProposalCheckpoint(
            proposal_md="# Proposal v1\n\nInitial plan.",
            todo_items=["draft plan"],
            questions_for_human=["Approve or revise?"],
        ),
        ProposalCheckpoint(
            proposal_md="# Proposal v2\n\nRevised plan.",
            todo_items=["revised draft plan"],
            questions_for_human=["Approve now?"],
        ),
    ]

    class _CapturingAgent:
        async def ainvoke(self, payload, config=None):
            captured["payload"] = payload
            captured["config"] = config
            return {
                "messages": [
                    AIMessage(
                        content="## Summary\nok\n\n## Facts\n- proposal approved\n\n## Files\n- `(none reported)`"
                    )
                ]
            }

    def _fake_create_deep_agent(**kwargs):
        captured["agent_kwargs"] = kwargs
        return _CapturingAgent()

    async def _fake_build_proposal_checkpoint(
        self,
        *,
        entrypoint: str,
        prompt: str,
        usage_handler,
        current_proposal: str = "",
        review_feedback: str = "",
        revision_index: int = 0,
    ) -> ProposalCheckpoint:
        _ = usage_handler
        call = {
            "entrypoint": entrypoint,
            "prompt": prompt,
            "current_proposal": current_proposal,
            "review_feedback": review_feedback,
            "revision_index": revision_index,
        }
        proposal_calls = captured["proposal_calls"]
        assert isinstance(proposal_calls, list)
        proposal_calls.append(call)
        return proposal_versions[len(proposal_calls) - 1]

    @asynccontextmanager
    async def _fake_open_agent_runtime(self, *, files_root: Path):
        _ = files_root
        yield {"checkpointer": object(), "store": object(), "backend": object()}

    monkeypatch.setattr(runtime_mod, "build_chat_model", lambda cfg: {"model": cfg.model})
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_create_deep_agent", staticmethod(lambda: _fake_create_deep_agent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_compiled_subagent", staticmethod(lambda: _FakeCompiledSubAgent))
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_load_subagent", staticmethod(lambda: _FakeSubAgent))
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
    monkeypatch.setattr(runtime_mod.SpecialistRunner, "_build_proposal_checkpoint", _fake_build_proposal_checkpoint)

    built = build_specialist_runner(
        workspace=workspace,
        llm_profile=_FakeProfile(),
        reporter=None,
        run_control=None,
        project_id="proj_proposal_gate",
        preferred_entrypoint="experiment",
    )

    waiting = asyncio.run(
        built.runner.arun(
            "Run the experiment lane only after proposal approval.",
            entrypoint="experiment",
            proposal_review=True,
        )
    )
    assert waiting["status"] == "awaiting_human_feedback"
    assert (built.run_context.run_dir / "proposal.md").read_text(encoding="utf-8") == proposal_versions[0].proposal_md

    revised = asyncio.run(built.runner.aresume("needs a more concrete execution plan"))
    assert revised["status"] == "awaiting_human_feedback"
    assert (built.run_context.run_dir / "proposal.md").read_text(encoding="utf-8") == proposal_versions[1].proposal_md

    result = asyncio.run(built.runner.aresume("approve"))

    assert result["status"] == "done"
    proposal_calls = captured["proposal_calls"]
    assert isinstance(proposal_calls, list)
    assert len(proposal_calls) == 2
    assert proposal_calls[1]["current_proposal"] == proposal_versions[0].proposal_md
    assert proposal_calls[1]["review_feedback"] == "needs a more concrete execution plan"
    assert proposal_calls[1]["revision_index"] == 1
    payload = captured["payload"]
    assert isinstance(payload, dict)
    assert payload["messages"][0]["content"] == "Run the experiment lane only after proposal approval."
    assert "Human review feedback" not in payload["messages"][0]["content"]
    run_state = json.loads((built.run_context.run_dir / RUN_STATE_FILE).read_text(encoding="utf-8"))
    assert run_state["status"] == "done"
    assert run_state["proposal_revision_count"] == 1


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
