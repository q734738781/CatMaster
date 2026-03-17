from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Literal

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, LLMResult
from langchain_core.tools import StructuredTool
from pydantic import BaseModel, Field

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_callback import LangChainStepLogger, UIEventHandler
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.runtime.usage_stats import write_usage_summary_from_metadata
from catmaster.tools.base import system_root, workspace_root, workspace_scope
from catmaster.tools.registry import get_tool_registry
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .schemas import (
    ProposalCheckpoint,
    ResearchKernel,
    SpecialistEntrypoint,
)

logger = logging.getLogger(__name__)

RUN_STATE_FILE = "run_state.json"
PROPOSAL_FILE = "proposal.md"
MEMORY_STORE_FILE = "deepagent_memory.sqlite"
CHECKPOINT_STORE_FILE = "deepagent_threads.sqlite"
MEMORY_FILE_PATH = "/memories/AGENTS.md"
RESEARCH_KERNEL_DIR = "research_kernels"

_ENTRYPOINT_TO_MODEL_ROLE: dict[str, str] = {
    "research": "research_lead",
    "experiment": "task_runner",
    "writing": "write_director",
}

_EXPERIMENT_TOOL_ALLOWLIST = {
    "create_molecule_from_smiles",
    "mace_relax_batch",
    "mace_sp_batch",
    "vasp_relax_prepare",
    "vasp_sp_prepare",
    "build_slab",
    "fix_atoms_by_layers",
    "fix_atoms_by_height",
    "supercell",
    "enumerate_adsorption_sites",
    "place_adsorbate",
    "generate_batch_adsorption_structures",
    "make_neb_geometry",
    "make_neb_incar",
    "vasp_execute_batch",
    "mp_search_materials",
    "mp_download_structure",
    "render_structure_views",
    "analyze_images",
    "generate_schematic_figure",
}
_WRITING_TOOL_ALLOWLIST = {
    "analyze_images",
    "render_structure_views",
    "generate_schematic_figure",
}
_RESEARCH_TOOL_ALLOWLIST = {
    "analyze_images",
    "render_structure_views",
    "mp_search_materials",
    "mp_download_structure",
}
_TASK_WORKER_TOOL_ALLOWLIST = set(_EXPERIMENT_TOOL_ALLOWLIST)
_LITREVIEW_AGENT_TOOL_ALLOWLIST = {
    "search_openalex",
    "search_semantic_scholar",
    "get_openalex_record",
    "get_semantic_scholar_record",
    "recommend_semantic_scholar",
}
_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES = {"internet_search"}
_WRITING_WORKER_TOOL_ALLOWLIST = {
    "polish_academic_prose",
    "analyze_images",
    "render_structure_views",
    "generate_schematic_figure",
    "agentic_compile_tex",
}
_LITREVIEW_COMPACT_TRIGGER_TOKENS = 65_000
_LITREVIEW_COMPACT_KEEP_TOKENS = 6_500


class _InternetSearchInput(BaseModel):
    query: str = Field(..., description="Focused public-web query for background facts or literature orientation.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of search results to return.")
    topic: Literal["general", "news", "finance"] = Field(
        "general",
        description="Tavily search topic. Use `general` for scientific background lookup.",
    )


class SpecialistUsageCallbackHandler(UsageMetadataCallbackHandler):
    """Official LangChain usage tracker with per-model call counts for specialist runs."""

    def __init__(self, *, default_agent_name: str = "") -> None:
        super().__init__()
        self.default_agent_name = str(default_agent_name or "").strip()
        self.call_counts_by_model: dict[str, int] = {}
        self.call_counts_by_role: dict[str, int] = {}
        self.usage_metadata_by_role: dict[str, dict[str, Any]] = {}
        self._pending_agents_by_run: dict[str, str] = {}

    def on_llm_start(self, serialized: dict[str, Any], prompts: list[str], **kwargs: Any) -> None:
        _ = (serialized, prompts)
        self._remember_agent_for_run(**kwargs)

    def on_chat_model_start(
        self,
        serialized: dict[str, Any],
        messages: list[list[Any]],
        **kwargs: Any,
    ) -> None:
        _ = (serialized, messages)
        self._remember_agent_for_run(**kwargs)

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        model_name = self._extract_model_name(response)
        run_id = str(kwargs.get("run_id") or "").strip()
        agent_name = self._pending_agents_by_run.pop(run_id, "") if run_id else ""
        usage_metadata = self._extract_usage_metadata(response)
        super().on_llm_end(response, **kwargs)
        if model_name:
            self.call_counts_by_model[model_name] = int(self.call_counts_by_model.get(model_name, 0)) + 1
        if agent_name:
            self.call_counts_by_role[agent_name] = int(self.call_counts_by_role.get(agent_name, 0)) + 1
            if model_name and usage_metadata:
                current = self.usage_metadata_by_role.setdefault(agent_name, {})
                previous = current.get(model_name)
                if isinstance(previous, dict):
                    current[model_name] = self._merge_usage_dict(previous, usage_metadata)
                else:
                    current[model_name] = usage_metadata

    @staticmethod
    def _extract_model_name(response: LLMResult) -> str:
        try:
            generation = response.generations[0][0]
        except Exception:
            return ""
        if not isinstance(generation, ChatGeneration):
            return ""
        message = getattr(generation, "message", None)
        if not isinstance(message, AIMessage):
            return ""
        response_metadata = getattr(message, "response_metadata", None)
        if not isinstance(response_metadata, dict):
            return ""
        return str(response_metadata.get("model_name") or "").strip()

    @staticmethod
    def _agent_name_from_kwargs(default_agent_name: str = "", **kwargs: Any) -> str:
        for source in (kwargs.get("metadata"), kwargs.get("inheritable_metadata")):
            if not isinstance(source, dict):
                continue
            for key in ("lc_agent_name", "agent_name", "agent", "subagent"):
                value = str(source.get(key) or "").strip()
                if value:
                    return value
        return str(default_agent_name or "").strip()

    def _remember_agent_for_run(self, **kwargs: Any) -> None:
        run_id = str(kwargs.get("run_id") or "").strip()
        if not run_id:
            return
        agent_name = self._agent_name_from_kwargs(self.default_agent_name, **kwargs)
        if agent_name:
            self._pending_agents_by_run[run_id] = agent_name

    @staticmethod
    def _extract_usage_metadata(response: LLMResult) -> dict[str, Any]:
        try:
            generation = response.generations[0][0]
        except Exception:
            return {}
        if not isinstance(generation, ChatGeneration):
            return {}
        message = getattr(generation, "message", None)
        if not isinstance(message, AIMessage):
            return {}
        usage = getattr(message, "usage_metadata", None)
        return dict(usage) if isinstance(usage, dict) else {}

    @classmethod
    def _merge_usage_dict(cls, base: dict[str, Any], update: dict[str, Any]) -> dict[str, Any]:
        merged = dict(base)
        for key, value in update.items():
            if isinstance(value, dict):
                current = merged.get(key)
                if isinstance(current, dict):
                    merged[key] = cls._merge_usage_dict(current, value)
                else:
                    merged[key] = dict(value)
                continue
            if isinstance(value, bool):
                merged[key] = int(bool(merged.get(key, 0))) + int(value)
                continue
            if isinstance(value, int):
                merged[key] = int(merged.get(key, 0) or 0) + value
                continue
            if isinstance(value, float):
                merged[key] = float(merged.get(key, 0.0) or 0.0) + value
                continue
            if value is not None:
                merged[key] = value
        return merged


@dataclass(frozen=True)
class BuiltSpecialistRunner:
    runner: "SpecialistRunner"
    run_context: RunContext


def build_specialist_runner(
    *,
    workspace: Path,
    llm_profile: LLMProfile,
    reporter: Reporter | None,
    run_control: RunControl | None,
    project_id: str,
    run_dir: Path | None = None,
    preferred_entrypoint: SpecialistEntrypoint = "research",
) -> BuiltSpecialistRunner:
    if run_dir is not None and Path(run_dir).exists():
        run_ctx = RunContext.load(Path(run_dir))
    else:
        entry_role = _ENTRYPOINT_TO_MODEL_ROLE[preferred_entrypoint]
        entry_cfg = llm_profile.config_for_role(entry_role)
        run_ctx = RunContext.create(
            workspace=workspace,
            run_dir=run_dir,
            project_id=project_id,
            model_name=entry_cfg.model,
            provider=entry_cfg.provider,
            base_url=entry_cfg.base_url,
        )
    if run_control is not None:
        run_control.run_id = run_ctx.run_id
    runner = SpecialistRunner(
        llm_profile=llm_profile,
        run_context=run_ctx,
        reporter=reporter or NullReporter(),
        run_control=run_control,
    )
    return BuiltSpecialistRunner(runner=runner, run_context=run_ctx)


class SpecialistRunner:
    def __init__(
        self,
        *,
        llm_profile: LLMProfile,
        run_context: RunContext,
        reporter: Reporter | None = None,
        run_control: RunControl | None = None,
    ) -> None:
        self.llm_profile = llm_profile
        self.run_context = run_context
        self.reporter = reporter or NullReporter()
        self.run_control = run_control
        self.registry = get_tool_registry()

    def run(
        self,
        prompt: str,
        *,
        entrypoint: SpecialistEntrypoint,
        proposal_review: bool,
        chat_session_id: str = "",
        thread_id: str = "",
    ) -> dict[str, Any]:
        return asyncio.run(
            self.arun(
                prompt,
                entrypoint=entrypoint,
                proposal_review=proposal_review,
                chat_session_id=chat_session_id,
                thread_id=thread_id,
            )
        )

    def resume(self, human_feedback: str = "") -> dict[str, Any]:
        return asyncio.run(self.aresume(human_feedback=human_feedback))

    async def arun(
        self,
        prompt: str,
        *,
        entrypoint: SpecialistEntrypoint,
        proposal_review: bool,
        chat_session_id: str = "",
        thread_id: str = "",
    ) -> dict[str, Any]:
        payload = {
            "entrypoint": entrypoint,
            "user_prompt": str(prompt or "").strip(),
            "proposal_review": bool(proposal_review),
            "chat_session_id": str(chat_session_id or "").strip(),
            "thread_id": str(thread_id or "").strip(),
        }
        return await self._run_impl(payload=payload, resume_feedback=None)

    async def aresume(self, human_feedback: str = "") -> dict[str, Any]:
        run_state = self._read_run_state()
        if not run_state:
            raise ValueError("Cannot resume run without run_state.json")
        if str(run_state.get("status") or "").strip() != "awaiting_human_feedback":
            raise ValueError("Selected run is not waiting for human feedback.")
        return await self._run_impl(payload=run_state, resume_feedback=str(human_feedback or "").strip())

    async def _run_impl(self, *, payload: dict[str, Any], resume_feedback: str | None) -> dict[str, Any]:
        entrypoint = str(payload.get("entrypoint") or "research").strip() or "research"
        if entrypoint not in {"research", "experiment", "writing"}:
            raise ValueError(f"Unsupported specialist entrypoint: {entrypoint}")

        prompt = str(payload.get("user_prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        thread_id = self._resolve_thread_id(payload)

        files_root = workspace_root(self.run_context.workspace)
        files_root.mkdir(parents=True, exist_ok=True)
        self._stage_deepagent_assets(files_root)
        research_kernel_relpath = ""
        if entrypoint == "research":
            research_kernel_relpath = self._ensure_research_kernel_seed(files_root=files_root, thread_id=thread_id, prompt=prompt)
        self._emit("RUN_START", payload={"entrypoint": entrypoint, "status": "running"})
        usage_handler = self._new_usage_callback()
        usage_flushed = False

        def _flush_usage() -> None:
            nonlocal usage_flushed
            if usage_flushed:
                return
            usage_flushed = True
            self._write_usage_summary(usage_handler)
        try:
            if resume_feedback is None and bool(payload.get("proposal_review", False)):
                checkpoint = await self._build_proposal_checkpoint(
                    entrypoint=entrypoint,
                    prompt=prompt,
                    usage_handler=usage_handler,
                )
                proposal_path = self.run_context.run_dir / PROPOSAL_FILE
                proposal_path.write_text(checkpoint.proposal_md, encoding="utf-8")
                self._write_run_state(
                    {
                        "schema_version": 1,
                        "entrypoint": entrypoint,
                        "status": "awaiting_human_feedback",
                        "phase": "proposal_review",
                        "active_specialist": entrypoint,
                        "thread_id": thread_id,
                        "proposal_review": True,
                        "pending_human_input": {
                            "kind": "proposal_review",
                            "questions_for_human": list(checkpoint.questions_for_human),
                            "todo_items": list(checkpoint.todo_items),
                        },
                        "todo_items": list(checkpoint.todo_items),
                        "artifacts": [],
                        "delegation_log": [],
                        "text_preview": checkpoint.proposal_md[:280],
                        "user_prompt": prompt,
                        "chat_session_id": str(payload.get("chat_session_id") or ""),
                        **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
                    }
                )
                self._emit(
                    "RUN_WAITING_INPUT",
                    payload={
                        "interrupt_type": "proposal_review",
                        "message": "Proposal review is required before execution continues.",
                    },
                )
                _flush_usage()
                feedback = self.reporter.prompt_proposal_feedback(
                    todo=list(checkpoint.todo_items),
                    proposal_description=checkpoint.proposal_md,
                )
                self._emit(
                    "RUN_INPUT_RECEIVED",
                    payload={"interrupt_type": "proposal_review", "feedback_len": len(str(feedback or ""))},
                )
                return await self._run_impl(
                    payload=self._read_run_state() or payload,
                    resume_feedback=str(feedback or "").strip(),
                )

            if resume_feedback is not None:
                self._write_run_state(
                    {
                        **payload,
                        "status": "running",
                        "phase": "executing",
                        "pending_human_input": None,
                        "text_preview": str(resume_feedback or "")[:280],
                    }
                )

            async with self._open_agent_runtime(files_root=files_root) as runtime:
                agent = await self._build_entry_agent(
                    entrypoint=entrypoint,
                    runtime=runtime,
                    thread_id=thread_id,
                )
                message_text = prompt if resume_feedback in (None, "") else (
                    f"{prompt}\n\nHuman review feedback:\n{resume_feedback}"
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": message_text}]},
                    config={
                        "configurable": {"thread_id": thread_id},
                        "callbacks": self._langchain_callbacks(
                            usage_handler=usage_handler,
                            default_agent_name=f"{entrypoint}_specialist",
                        ),
                        "metadata": {"lc_agent_name": f"{entrypoint}_specialist"},
                    },
                )
            parsed = self._finalize_report(self._coerce_report(raw=result))
            final_answer = parsed["text"]
            artifacts = self._artifact_rows(parsed["files"])
            status = "done"
            self._write_run_state(
                {
                    "schema_version": 1,
                    "entrypoint": entrypoint,
                    "status": status,
                    "phase": "finalized",
                    "active_specialist": entrypoint,
                    "thread_id": thread_id,
                    "proposal_review": bool(payload.get("proposal_review", False)),
                    "pending_human_input": None,
                    "todo_items": [],
                    "artifacts": artifacts,
                    "delegation_log": [],
                    "text_preview": final_answer[:280],
                    "user_prompt": prompt,
                    "chat_session_id": str(payload.get("chat_session_id") or ""),
                    "final_answer": final_answer,
                    "summary": parsed["summary"],
                    "facts": list(parsed["facts"]),
                    **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
                }
            )
            self._emit("RUN_END", payload={"entrypoint": entrypoint, "status": status})
            _flush_usage()
            return {
                "run_id": self.run_context.run_id,
                "run_dir": str(self.run_context.run_dir),
                "status": status,
                "summary": parsed["summary"],
                "facts": list(parsed["facts"]),
                "final_answer": final_answer,
                "artifacts": artifacts,
                "delegation_log": [],
            }
        finally:
            _flush_usage()

    async def _build_proposal_checkpoint(
        self,
        *,
        entrypoint: SpecialistEntrypoint,
        prompt: str,
        usage_handler: SpecialistUsageCallbackHandler,
    ) -> ProposalCheckpoint:
        create_agent = self._load_create_agent()
        ToolStrategy = self._load_tool_strategy()
        system_prompt = self._proposal_system_prompt(entrypoint=entrypoint)
        model = build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint]))
        agent = create_agent(
            model=model,
            tools=[],
            system_prompt=system_prompt,
            response_format=ToolStrategy(ProposalCheckpoint, handle_errors=False),
        )
        result = await agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": prompt,
                    }
                ]
            },
            config={
                "callbacks": self._langchain_callbacks(
                    usage_handler=usage_handler,
                    default_agent_name=f"{entrypoint}_specialist",
                ),
                "metadata": {"lc_agent_name": f"{entrypoint}_specialist"},
            },
        )
        raw = result.get("structured_response") if isinstance(result, dict) else None
        if isinstance(raw, ProposalCheckpoint):
            return raw
        if isinstance(raw, dict):
            return ProposalCheckpoint.model_validate(raw)
        raise RuntimeError("Proposal checkpoint generation failed.")

    async def _build_entry_agent(
        self,
        *,
        entrypoint: SpecialistEntrypoint,
        runtime: dict[str, Any],
        thread_id: str,
    ) -> Any:
        create_deep_agent = self._load_create_deep_agent()
        tools = self._specialist_tools(entrypoint)
        skills = self._virtual_skill_paths(entrypoint)
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint])),
            "tools": tools,
            "system_prompt": self._system_prompt(entrypoint, thread_id=thread_id),
            "middleware": self._build_default_middleware(),
            "checkpointer": runtime["checkpointer"],
            "store": runtime["store"],
            "backend": runtime["backend"],
            "name": f"{entrypoint}_specialist",
            "skills": skills,
            "memory": self._memory_sources(),
        }
        subagents = self._entry_subagents(entrypoint, runtime=runtime)
        if subagents:
            kwargs["subagents"] = subagents
        return create_deep_agent(**kwargs)

    def _entry_subagents(self, entrypoint: SpecialistEntrypoint, *, runtime: dict[str, Any]) -> list[Any]:
        if entrypoint == "research":
            return self._research_subagents(runtime=runtime)
        if entrypoint == "experiment":
            return self._experiment_subagents(runtime=runtime)
        if entrypoint == "writing":
            return self._writing_subagents(runtime=runtime)
        return []

    def _research_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        subagent_middleware = self._subagent_middleware(runtime=runtime)
        return [
            SubAgent(
                name="experiment_specialist",
                description="Run bounded computational experiment work and return compact evidence summaries.",
                system_prompt=self._system_prompt("experiment"),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._specialist_tools("experiment"),
                skills=self._virtual_skill_paths("experiment"),
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="writing_specialist",
                description="Turn existing evidence into reports, outlines, sections, or manuscript-ready outputs.",
                system_prompt=self._system_prompt("writing"),
                model=build_chat_model(self.llm_profile.config_for_role("write_director")),
                tools=self._specialist_tools("writing"),
                skills=self._virtual_skill_paths("writing"),
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="litreview_agent",
                description="Retrieve scholarly literature grounding, benchmark conventions, and representative citations when deeper review is needed.",
                system_prompt=self._litreview_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
                tools=self._named_tools(_LITREVIEW_AGENT_TOOL_ALLOWLIST),
                middleware=self._litreview_middleware(runtime=runtime),
            ),
        ]

    def _experiment_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        subagent_middleware = self._subagent_middleware(runtime=runtime)
        return [
            SubAgent(
                name="task_worker_agent",
                description="Handle bounded, context-heavy execution subtasks in isolation and return concise results with artifact paths.",
                system_prompt=self._task_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._named_tools(_TASK_WORKER_TOOL_ALLOWLIST),
                skills=self._virtual_skill_paths("experiment"),
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="literature_agent",
                description="Run lightweight Tavily-backed public-web search for quick background grounding and literature orientation.",
                system_prompt=self._lightweight_literature_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("summary")),
                tools=self._lightweight_literature_tools(),
                middleware=subagent_middleware,
            ),
        ]

    def _writing_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        subagent_middleware = self._subagent_middleware(runtime=runtime)
        return [
            SubAgent(
                name="writing_worker_agent",
                description="Draft or revise context-heavy sections in isolation and return compact manuscript-ready outputs.",
                system_prompt=self._writing_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("section_writer")),
                tools=self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                skills=self._virtual_skill_paths("writing"),
                middleware=subagent_middleware,
            ),
        ]

    def _subagent_middleware(self, *, runtime: dict[str, Any]) -> list[Any]:
        return [
            *self._build_default_middleware(),
            self._new_memory_middleware(backend=runtime["backend"]),
        ]

    def _litreview_middleware(self, *, runtime: dict[str, Any]) -> list[Any]:
        middleware = self._subagent_middleware(runtime=runtime)
        try:
            SummarizationMiddleware = self._load_summarization_middleware()
            create_summarization_tool_middleware = self._load_create_summarization_tool_middleware()
            summarizer = SummarizationMiddleware(
                model=build_chat_model(self.llm_profile.config_for_role("summary")),
                backend=runtime["backend"],
                trigger=("tokens", _LITREVIEW_COMPACT_TRIGGER_TOKENS),
                keep=("tokens", _LITREVIEW_COMPACT_KEEP_TOKENS),
            )
        except Exception as exc:
            logger.warning("litreview compaction middleware unavailable; continuing without extra compaction: %s", exc)
            return middleware
        middleware.extend([summarizer, create_summarization_tool_middleware(summarizer)])
        return middleware

    @asynccontextmanager
    async def _open_agent_runtime(self, *, files_root: Path):
        stack = AsyncExitStack()
        try:
            checkpointer, store = await self._open_sqlite_state(stack)
            backend_factory = self._make_backend_factory(files_root=files_root, store=store)
            yield {
                "checkpointer": checkpointer,
                "store": store,
                "backend": backend_factory,
            }
        finally:
            await stack.aclose()

    async def _open_sqlite_state(self, stack: AsyncExitStack) -> tuple[Any, Any]:
        checkpoint_path = system_root(self.run_context.workspace) / CHECKPOINT_STORE_FILE
        store_path = system_root(self.run_context.workspace) / MEMORY_STORE_FILE
        try:
            from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
        except Exception as exc:
            raise RuntimeError(
                "DeepAgent runtime requires sqlite checkpoint support. "
                "Install langgraph-checkpoint-sqlite."
            ) from exc
        try:
            from langgraph.store.sqlite.aio import AsyncSqliteStore
        except Exception:
            try:
                from langgraph.store.sqlite import AsyncSqliteStore
            except Exception as exc:
                raise RuntimeError(
                    "DeepAgent runtime requires sqlite store support."
                ) from exc
        saver_cm = AsyncSqliteSaver.from_conn_string(str(checkpoint_path))
        store_cm = AsyncSqliteStore.from_conn_string(str(store_path))
        saver = await stack.enter_async_context(saver_cm)
        store = await stack.enter_async_context(store_cm)
        setup = getattr(store, "setup", None)
        if callable(setup):
            maybe = setup()
            if asyncio.iscoroutine(maybe):
                await maybe
        await self._ensure_memory_seed(store)
        return saver, store

    def _make_backend_factory(self, *, files_root: Path, store: Any):
        def _factory(runtime: Any) -> Any:
            from deepagents.backends import CompositeBackend, LocalShellBackend, StoreBackend

            return CompositeBackend(
                default=LocalShellBackend(
                    root_dir=files_root,
                    virtual_mode=True,
                    timeout=120,
                    inherit_env=True,
                ),
                routes={
                    "/memories/": StoreBackend(runtime, namespace=lambda _ctx: self._memory_namespace()),
                },
            )

        _ = store
        return _factory

    def _specialist_tools(self, entrypoint: SpecialistEntrypoint) -> list[Any]:
        if entrypoint == "writing":
            requested = _WRITING_TOOL_ALLOWLIST
        elif entrypoint == "research":
            requested = _RESEARCH_TOOL_ALLOWLIST
        else:
            requested = _EXPERIMENT_TOOL_ALLOWLIST
        return self._named_tools(requested)

    def _named_tools(self, requested: set[str] | list[str] | tuple[str, ...]) -> list[Any]:
        requested_names = {str(name).strip() for name in requested if str(name).strip()}
        all_names = set(self.registry.tools.keys())
        missing = sorted(name for name in requested_names if name not in all_names)
        if missing:
            raise RuntimeError(
                f"Missing registered tools: {', '.join(missing)}"
            )
        allowlist = sorted(requested_names)
        tools = self.registry.as_langchain_tools(
            allowlist=allowlist,
            run_dir=str(self.run_context.run_dir),
            workspace=str(self.run_context.workspace),
        )
        return [self._wrap_nonfatal_tool(tool) for tool in tools]

    def _lightweight_literature_tools(self) -> list[Any]:
        def internet_search(
            query: str,
            max_results: int = 5,
            topic: Literal["general", "news", "finance"] = "general",
        ) -> tuple[str, dict[str, Any]]:
            data: dict[str, Any]
            try:
                from catmaster.runtime.literature.online_search_adapter import OnlineSearchAdapter

                adapter = OnlineSearchAdapter(topic=topic)
                result = adapter.search_public_web(query, max_results=max_results)
                data = {
                    "status": "ok",
                    "source": "tavily",
                    "query": query,
                    "topic": topic,
                    "count": len(result.results),
                    "results": [hit.model_dump() for hit in result.results],
                }
            except Exception as exc:
                data = {
                    "status": "error",
                    "source": "tavily",
                    "query": query,
                    "topic": topic,
                    "message": str(exc),
                }
            return json.dumps(data, ensure_ascii=False), {
                "tool_name": "internet_search",
                "data": data,
            }

        return [
            StructuredTool.from_function(
                func=internet_search,
                name="internet_search",
                description="Search the public web for targeted scientific background facts and literature orientation.",
                args_schema=_InternetSearchInput,
                infer_schema=False,
                response_format="content_and_artifact",
            )
        ]

    @staticmethod
    def _nonfatal_tool_error_result(tool_name: str, exc: Exception, tool_args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
        if isinstance(exc, CatMasterToolExecutionError):
            message = str(exc.public_message or f"{tool_name} failed.").strip()
            data = dict(exc.artifact.get("data") or {}) if isinstance(exc.artifact, dict) else {}
            data.update(
                {
                    "status": "error",
                    "tool_name": tool_name,
                    "message": message,
                    "retryable": bool(exc.retryable),
                    "error_code": str(exc.error_code or ""),
                    "tool_args": dict(tool_args or {}),
                }
            )
            artifact = {"tool_name": tool_name, "data": data}
            return content_to_text(message), artifact
        message = f"{type(exc).__name__}: {exc}".strip()
        artifact = {
            "tool_name": tool_name,
            "data": {
                "status": "error",
                "tool_name": tool_name,
                "message": message,
                "error_type": type(exc).__name__,
                "tool_args": dict(tool_args or {}),
            },
        }
        return content_to_text(message), artifact

    def _wrap_nonfatal_tool(self, tool: Any) -> Any:
        if not isinstance(tool, StructuredTool):
            return tool
        args_schema = getattr(tool, "args_schema", None)
        if not isinstance(args_schema, type) or not issubclass(args_schema, BaseModel):
            return tool
        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        if func is None and coroutine is None:
            return tool

        def _wrapped(runtime=None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            if func is None:
                raise NotImplementedError(f"Tool {tool.name} does not support sync invocation.")
            try:
                return func(runtime=runtime, **kwargs)
            except Exception as exc:
                return self._nonfatal_tool_error_result(tool.name, exc, kwargs)

        async def _awrapped(runtime=None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            if coroutine is not None:
                try:
                    return await coroutine(runtime=runtime, **kwargs)
                except Exception as exc:
                    return self._nonfatal_tool_error_result(tool.name, exc, kwargs)
            if func is None:
                raise NotImplementedError(f"Tool {tool.name} does not support async invocation.")
            try:
                return func(runtime=runtime, **kwargs)
            except Exception as exc:
                return self._nonfatal_tool_error_result(tool.name, exc, kwargs)

        _wrapped.__name__ = tool.name
        _awrapped.__name__ = f"{tool.name}_async"
        return StructuredTool.from_function(
            func=_wrapped if func is not None else None,
            coroutine=_awrapped,
            name=tool.name,
            description=str(getattr(tool, "description", "") or "").strip(),
            args_schema=args_schema,
            infer_schema=False,
            response_format="content_and_artifact",
        )

    def _stage_deepagent_assets(self, files_root: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        deepagents_root = files_root / ".deepagents"
        base = deepagents_root / "skills"
        layouts = {
            base / "experiment": repo_root / "skills" / "experiment",
            base / "writing": repo_root / "skills" / "writing",
        }
        for target, source in layouts.items():
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
        staged_agents = deepagents_root / "AGENTS.md"
        if not staged_agents.exists():
            workspace_agents = Path(self.run_context.workspace) / "AGENTS.md"
            if workspace_agents.exists():
                staged_agents.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace_agents, staged_agents)

    @staticmethod
    def _virtual_skill_paths(entrypoint: SpecialistEntrypoint) -> list[str]:
        if entrypoint == "experiment":
            return ["/.deepagents/skills/experiment"]
        if entrypoint == "writing":
            return ["/.deepagents/skills/writing"]
        return [
            "/.deepagents/skills/experiment",
            "/.deepagents/skills/writing",
        ]

    def _resolve_thread_id(self, payload: dict[str, Any]) -> str:
        thread_id = str(payload.get("thread_id") or "").strip()
        if thread_id:
            return thread_id
        chat_session_id = str(payload.get("chat_session_id") or "").strip()
        if chat_session_id:
            return chat_session_id
        return self.run_context.run_id

    def _system_prompt(self, entrypoint: SpecialistEntrypoint, *, thread_id: str = "") -> str:
        return self._base_system_prompt(entrypoint, thread_id=thread_id)

    def _memory_sources(self) -> list[str]:
        return ["/.deepagents/AGENTS.md", MEMORY_FILE_PATH]

    def _memory_namespace(self) -> tuple[str, ...]:
        project_id = str(self.run_context.project_id or "default").strip() or "default"
        return ("catmaster", project_id, "filesystem")

    async def _ensure_memory_seed(self, store: Any) -> None:
        namespace = self._memory_namespace()
        existing = await store.aget(namespace, "/AGENTS.md")
        if existing is not None:
            return
        from deepagents.backends.utils import create_file_data

        content = (
            "# Persistent Project Memory\n\n"
            "Use this file to store durable project conventions, user preferences, "
            "validated scientific background, and stable workflow guidance that should persist across runs.\n\n"
            "- Do not store transient task requests.\n"
            "- Do not store step-by-step execution logs, temporary status notes, or intermediate tool outputs.\n"
            "- Do not store one-off artifact paths or run-specific scratch details unless they encode a stable convention.\n"
            "- Do not store speculative or unverified findings from an unfinished task.\n"
            "- Do not store secrets, credentials, or API keys.\n"
        )
        await store.aput(namespace, "/AGENTS.md", create_file_data(content))

    def _new_memory_middleware(self, *, backend: Any) -> Any:
        MemoryMiddleware = self._load_memory_middleware()
        return MemoryMiddleware(backend=backend, sources=self._memory_sources())

    @classmethod
    def _base_system_prompt(cls, entrypoint: SpecialistEntrypoint, *, thread_id: str = "") -> str:
        if entrypoint == "research":
            kernel_path = cls._research_kernel_virtual_path(thread_id)
            return (
                "You are ResearchSpecialist, the only orchestration-capable specialist.\n"
                "You coordinate scientific campaigns, decide when bounded experiment work is justified, "
                "and decide when writing/report generation should start.\n"
                "You may delegate only to `experiment_specialist`, `writing_specialist`, and `litreview_agent`.\n"
                "Use `litreview_agent` whenever external scholarly grounding, benchmark conventions, or representative citations are needed.\n"
                "If the user requests a report, manuscript, note, LaTeX document, or other substantial written artifact, delegate that work to `writing_specialist` rather than drafting it directly in the research thread.\n"
                f"{cls._writing_handoff_policy()}\n"
                "Do not perform large direct execution yourself when delegation is more appropriate.\n"
                f"{cls._research_kernel_contract(kernel_path)}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._soft_reporting_contract()}"
            )
        if entrypoint == "writing":
            return (
                "You are WritingSpecialist.\n"
                "Write from existing workspace evidence only. Do not initiate new computational experiments.\n"
                "Do not reopen literature review from the writing thread; if external grounding is still missing, report that research lane should gather it first.\n"
                "Your default role is coordination, not long-form drafting in the main thread.\n"
                "For any substantive note writing, section writing, manuscript drafting, or major revision, immediately delegate to `writing_worker_agent` with a bounded brief.\n"
                "Keep the main writing thread focused on planning, dispatch, evidence selection, and final reconciliation.\n"
                "Do not handle TeX compile/fix passes in the main thread.\n"
                "If you create or substantially revise a TeX manuscript bundle, require `writing_worker_agent` to run the compile tool itself and repair issues from the returned diagnostics before concluding.\n"
                "Do not leave final cited TeX deliverables with an inline `thebibliography` block. Prefer a separate bibliography file and a `\\bibliography{references}` entry so the bundle includes `.tex`, `.bib`, and `.pdf` outputs when compilation succeeds.\n"
                f"{cls._writing_handoff_policy()}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._soft_reporting_contract()}"
            )
        return (
            "You are ExperimentSpecialist.\n"
            "Perform bounded computational execution in the current workspace using available tools and skills.\n"
            "Use `task_worker_agent` for context-heavy isolated execution subtasks, and `literature_agent` for fast Tavily-backed public-web grounding when a quick external check is needed.\n"
            f"Do not orchestrate other specialists. {cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @staticmethod
    def _memory_write_policy() -> str:
        return (
            "If you update persistent memory, store only durable user preferences, project conventions, "
            "validated reusable scientific knowledge, or stable workflow guidance. "
            "Never store transient task requests, step-by-step execution history, intermediate tool output, "
            "one-off file paths from a single run, temporary status notes, or speculative findings."
        )

    @staticmethod
    def _writing_handoff_policy() -> str:
        return (
            "When handing off writing work, pass only a bounded brief containing the writing goal, target audience, exact evidence file paths or compact run-card facts to rely on, "
            "the key facts that must be preserved, the requested output artifact path(s), the desired section structure, and any citation or style constraints. "
            "For TeX deliverables, require a separate `.tex` file, a separate `.bib` file when citations are used, and at least one direct compile pass that should produce a `.pdf` when the environment supports compilation. "
            "Do not paste long transcripts or ask the writing agent to rediscover evidence already available in the workspace."
        )

    @staticmethod
    def _soft_reporting_contract() -> str:
        return (
            "For multi-step work, use `write_todos` early to maintain a concise checklist and update it when the plan changes. "
            "When you finish, reply with a concise markdown report containing only three sections in this order: "
            "`Summary`, `Facts`, and `Files`. "
            "`Summary` should be a short user-facing wrap-up. "
            "`Facts` should be a flat bullet list of the few most important archival facts. "
            "`Files` should be a flat bullet list of relevant output paths, or `(none reported)` if there are none."
        )

    @classmethod
    def _research_kernel_contract(cls, kernel_path: str) -> str:
        return (
            f"Maintain a lightweight Research Kernel in `{kernel_path}` as valid JSON. "
            "It must contain exactly these top-level fields: `question`, `hypotheses`, `run_cards`, `frontier`, `conclusion_draft`. "
            "Keep `hypotheses` to only the currently active 3-5 lines. "
            "Every time a subagent returns, immediately update `run_cards` with one compact card containing only `source`, `summary`, `facts`, and `artifacts`. "
            "Keep only the minimum decision-relevant facts needed for the next choice. "
            "Use `frontier` for the next unresolved questions or actions to validate. "
            "When delegating, write a clear bounded brief and explicitly require the subagent to answer with the compact `Summary` / `Facts` / `Files` contract so its result can be distilled into a run card."
        )

    @classmethod
    def _task_worker_prompt(cls) -> str:
        return (
            "You are task_worker_agent for ExperimentSpecialist.\n"
            "Handle a bounded execution subtask autonomously inside the workspace.\n"
            "Use available execution and analysis tools, keep the run focused, and return a compact result with the key finding, relevant artifact paths, and any blocking issue.\n"
            "Do not perform literature search; that belongs to literature_agent.\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _litreview_agent_prompt(cls) -> str:
        return (
            "You are litreview_agent.\n"
            "Use the scholarly literature tools directly to gather external literature grounding, benchmark conventions, citations, or background evidence.\n"
            "Prefer OpenAlex and Semantic Scholar for exact metadata, DOI/year/venue/authors/citation details, and seed-based recommendation expansion.\n"
            "Stay focused on representative, decision-relevant papers instead of broad browsing.\n"
            "You may write concise reusable literature artifacts into the workspace when helpful, such as notes, citation lists, or evidence summaries.\n"
            "Return concise findings with clear separation between retrieved facts and inference.\n"
            "Do not perform computational execution.\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _lightweight_literature_agent_prompt(cls) -> str:
        return (
            "You are literature_agent.\n"
            "Use the lightweight `internet_search` tool for quick public-web orientation when ExperimentSpecialist needs external background facts or literature hints.\n"
            "Keep search narrow, prefer concise result sets, cite source URLs, and separate retrieved facts from your inference.\n"
            "Do not attempt full citation curation or long-form literature review here.\n"
            "Do not perform computational execution.\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @staticmethod
    def _sanitize_kernel_component(text: str) -> str:
        normalized = re.sub(r"[^A-Za-z0-9._-]+", "_", str(text or "").strip())
        normalized = normalized.strip("._") or "default"
        return normalized[:80]

    @classmethod
    def _research_kernel_virtual_path(cls, thread_id: str) -> str:
        safe_thread = cls._sanitize_kernel_component(thread_id)
        return f"/{RESEARCH_KERNEL_DIR}/{safe_thread}/kernel.json"

    @classmethod
    def _research_kernel_fs_path(cls, files_root: Path, thread_id: str) -> Path:
        safe_thread = cls._sanitize_kernel_component(thread_id)
        return files_root / RESEARCH_KERNEL_DIR / safe_thread / "kernel.json"

    def _ensure_research_kernel_seed(self, *, files_root: Path, thread_id: str, prompt: str) -> str:
        path = self._research_kernel_fs_path(files_root, thread_id)
        if path.exists():
            return str(path.relative_to(files_root)).replace("\\", "/")
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = ResearchKernel(question=str(prompt or "").strip())
        path.write_text(payload.model_dump_json(indent=2), encoding="utf-8")
        return str(path.relative_to(files_root)).replace("\\", "/")

    def _load_research_kernel(self, *, files_root: Path, thread_id: str) -> dict[str, Any]:
        path = self._research_kernel_fs_path(files_root, thread_id)
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        if not isinstance(payload, dict):
            return {}
        try:
            kernel = ResearchKernel.model_validate(payload)
        except Exception:
            return {}
        return kernel.model_dump()

    def _research_kernel_state_fields(self, *, files_root: Path, thread_id: str, relpath: str = "") -> dict[str, Any]:
        if not thread_id:
            return {}
        kernel = self._load_research_kernel(files_root=files_root, thread_id=thread_id)
        if not kernel and not relpath:
            return {}
        result: dict[str, Any] = {}
        if relpath:
            result["research_kernel_path"] = relpath
        if kernel:
            result["research_kernel"] = kernel
        return result

    @classmethod
    def _writing_worker_prompt(cls) -> str:
        return (
            "You are writing_worker_agent for WritingSpecialist.\n"
            "Draft, revise, or polish bounded writing subtasks from existing workspace evidence only.\n"
            "Do not reopen broad research loops or re-read large unrelated workspace trees on your own.\n"
            "Return concise manuscript-ready output summaries and any output artifact paths.\n"
            "If the output is a TeX bundle, you must run `agentic_compile_tex` yourself before returning and use its diagnostics/log summary to fix compile-facing issues.\n"
            "If you draft TeX with citations, structure it to use a separate bibliography file rather than leaving inline `thebibliography` in the final bundle.\n"
            f"{cls._writing_handoff_policy()}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @staticmethod
    def _proposal_system_prompt(entrypoint: SpecialistEntrypoint) -> str:
        return (
            f"You are {entrypoint.capitalize()}Specialist in proposal review mode.\n"
            "Produce a compact executable proposal only. Do not perform the work yet.\n"
            "Return a ProposalCheckpoint with a markdown proposal, short todo list, and only blocking human questions."
        )

    def _workspace_agent_instructions(self) -> str:
        candidates = [
            workspace_root(self.run_context.workspace) / ".deepagents" / "AGENTS.md",
            Path(self.run_context.workspace) / "AGENTS.md",
        ]
        chunks: list[str] = []
        seen: set[Path] = set()
        for path in candidates:
            resolved = path.resolve()
            if resolved in seen or not path.exists():
                continue
            seen.add(resolved)
            try:
                text = path.read_text(encoding="utf-8").strip()
            except Exception:
                continue
            if text:
                chunks.append(text)
        return "\n\n".join(chunks)

    @staticmethod
    def _build_default_middleware() -> list[Any]:
        middleware: list[Any] = []
        try:
            from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
        except Exception:
            pass
        else:
            middleware.append(ModelCallLimitMiddleware(run_limit=40))
        try:
            from langchain.agents.middleware import wrap_tool_call
        except Exception:
            return middleware

        @wrap_tool_call(name="catmaster_nonfatal_tool_errors")
        async def _handle_tool_errors(request: Any, handler: Any) -> Any:
            try:
                return await handler(request)
            except Exception as exc:
                tool_call = getattr(request, "tool_call", None)
                if not isinstance(tool_call, dict):
                    tool_call = {}
                tool_name = str(tool_call.get("name") or getattr(request, "name", "") or "tool").strip() or "tool"
                tool_call_id = str(tool_call.get("id") or "").strip() or f"{tool_name}_error"
                content, artifact = SpecialistRunner._nonfatal_tool_error_result(
                    tool_name,
                    exc,
                    tool_call.get("args") if isinstance(tool_call.get("args"), dict) else {},
                )
                return ToolMessage(
                    content=content,
                    artifact=artifact,
                    tool_call_id=tool_call_id,
                    name=tool_name,
                    status="error",
                )

        middleware.append(_handle_tool_errors)
        return middleware

    def _langchain_callbacks(
        self,
        *,
        usage_handler: SpecialistUsageCallbackHandler | None,
        default_agent_name: str = "",
    ) -> list[Any]:
        callbacks: list[Any] = []
        if usage_handler is not None:
            usage_handler.default_agent_name = str(default_agent_name or "").strip()
            callbacks.append(usage_handler)
        if not isinstance(self.reporter, NullReporter):
            callbacks.append(
                UIEventHandler(
                    self.reporter,
                    run_id=self.run_context.run_id,
                    default_agent_name=default_agent_name,
                )
            )
        agent_runtime = getattr(self.llm_profile, "agent_runtime", None)
        if bool(getattr(agent_runtime, "print_state_messages", False)):
            callbacks.append(LangChainStepLogger(run_id=self.run_context.run_id))
        return callbacks

    @staticmethod
    def _new_usage_callback() -> SpecialistUsageCallbackHandler:
        return SpecialistUsageCallbackHandler()

    def _write_usage_summary(self, usage_handler: SpecialistUsageCallbackHandler) -> None:
        usage_metadata = getattr(usage_handler, "usage_metadata", None)
        if not isinstance(usage_metadata, dict) or not usage_metadata:
            return
        call_counts_by_model = getattr(usage_handler, "call_counts_by_model", None)
        usage_metadata_by_role = getattr(usage_handler, "usage_metadata_by_role", None)
        call_counts_by_role = getattr(usage_handler, "call_counts_by_role", None)
        write_usage_summary_from_metadata(
            self.run_context.run_dir,
            usage_metadata=usage_metadata,
            call_counts_by_model=call_counts_by_model if isinstance(call_counts_by_model, dict) else {},
            usage_metadata_by_role=usage_metadata_by_role if isinstance(usage_metadata_by_role, dict) else {},
            call_counts_by_role=call_counts_by_role if isinstance(call_counts_by_role, dict) else {},
            append=True,
        )

    def _coerce_report(self, *, raw: dict[str, Any] | Any) -> dict[str, Any]:
        text = self._extract_final_text(raw)
        if not text:
            raise RuntimeError("specialist failed to return a final text report.")
        summary, facts, files = self._parse_summary_and_files(text)
        return {
            "text": text,
            "summary": summary,
            "facts": facts,
            "files": files,
        }

    def _finalize_report(self, parsed: dict[str, Any]) -> dict[str, Any]:
        summary = str(parsed.get("summary") or "").strip()
        facts = [str(item).strip() for item in list(parsed.get("facts") or []) if str(item).strip()]
        files = [self._normalize_artifact_path(str(item).strip()) for item in list(parsed.get("files") or []) if str(item).strip()]
        files, facts = self._ensure_tex_bundle_outputs(files=files, facts=facts)
        return {
            "text": self._render_compact_report(summary=summary, facts=facts, files=files),
            "summary": summary,
            "facts": facts,
            "files": files,
        }

    def _extract_final_text(self, raw: dict[str, Any] | Any) -> str:
        if isinstance(raw, AIMessage):
            return self._message_text(raw)
        if isinstance(raw, str):
            return str(raw).strip()
        if isinstance(raw, dict):
            messages = raw.get("messages")
            if isinstance(messages, list):
                for message in reversed(messages):
                    text = self._message_text(message)
                    if text:
                        return text
        return ""

    @staticmethod
    def _message_text(message: Any) -> str:
        content = getattr(message, "content", message)
        if isinstance(content, str):
            return content.strip()
        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, str):
                    if item.strip():
                        chunks.append(item.strip())
                    continue
                if isinstance(item, dict):
                    text = str(item.get("text") or "").strip()
                    if text:
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return str(content or "").strip()

    def _parse_summary_and_files(self, text: str) -> tuple[str, list[str], list[str]]:
        summary_lines: list[str] = []
        facts: list[str] = []
        files: list[str] = []
        current_section: str | None = None
        for raw_line in text.splitlines():
            line = raw_line.rstrip()
            heading = self._match_report_heading(line)
            if heading is not None:
                current_section = heading
                continue
            if current_section == "summary":
                if line.strip():
                    summary_lines.append(line.strip())
                continue
            if current_section == "facts":
                fact = self._extract_reported_fact(line)
                if fact:
                    facts.append(fact)
                continue
            if current_section == "files":
                path = self._extract_reported_file(line)
                if path:
                    files.append(path)
        summary = "\n".join(summary_lines).strip()
        if not summary:
            summary = self._fallback_summary(text)
        deduped_facts: list[str] = []
        seen_facts: set[str] = set()
        for item in facts:
            normalized = str(item).strip()
            if not normalized or normalized in seen_facts:
                continue
            seen_facts.add(normalized)
            deduped_facts.append(normalized)
        deduped_files: list[str] = []
        seen: set[str] = set()
        for item in files:
            normalized = self._normalize_artifact_path(item)
            if normalized in seen:
                continue
            seen.add(normalized)
            deduped_files.append(normalized)
        return summary, deduped_facts, deduped_files

    @staticmethod
    def _match_report_heading(line: str) -> str | None:
        normalized = re.sub(r"^[#\-\s]+", "", str(line or "").strip()).lower().rstrip(":")
        if normalized == "summary":
            return "summary"
        if normalized == "facts":
            return "facts"
        if normalized == "files":
            return "files"
        return None

    @staticmethod
    def _extract_reported_fact(line: str) -> str:
        candidate = re.sub(r"^[-*]\s*", "", str(line or "").strip()).strip()
        return candidate

    @staticmethod
    def _extract_reported_file(line: str) -> str:
        stripped = str(line or "").strip()
        if not stripped:
            return ""
        code_match = re.search(r"`([^`]+)`", stripped)
        if code_match:
            return code_match.group(1).strip()
        candidate = re.sub(r"^[-*]\s*", "", stripped).strip()
        if ":" in candidate:
            candidate = candidate.split(":", 1)[0].strip()
        return candidate

    @staticmethod
    def _fallback_summary(text: str) -> str:
        chunks = [chunk.strip() for chunk in re.split(r"\n\s*\n", text) if chunk.strip()]
        return chunks[0] if chunks else text.strip()

    @staticmethod
    def _render_compact_report(*, summary: str, facts: list[str], files: list[str]) -> str:
        fact_lines = [f"- {item}" for item in facts] or ["- (none reported)"]
        file_lines = [f"- `{item}`" for item in files] or ["- `(none reported)`"]
        return "\n".join(
            [
                "## Summary",
                summary.strip() or "(no summary reported)",
                "",
                "## Facts",
                *fact_lines,
                "",
                "## Files",
                *file_lines,
            ]
        ).strip()

    def _ensure_tex_bundle_outputs(self, *, files: list[str], facts: list[str]) -> tuple[list[str], list[str]]:
        normalized_files: list[str] = []
        seen_files: set[str] = set()
        for item in files:
            normalized = self._normalize_artifact_path(item)
            if not normalized or normalized in seen_files:
                continue
            seen_files.add(normalized)
            normalized_files.append(normalized)

        tex_paths = [item for item in normalized_files if item.lower().endswith(".tex")]
        if not tex_paths:
            return normalized_files, facts

        updated_facts = list(facts)
        compile_tool = self.registry.get_tool_function("agentic_compile_tex")
        for tex_path in tex_paths:
            has_pdf = any(self._tex_bundle_matches(tex_path, item, suffix=".pdf") for item in normalized_files)
            has_bib = any(self._tex_bundle_matches(tex_path, item, suffix=".bib") for item in normalized_files)
            if has_pdf and has_bib:
                continue
            with workspace_scope(self.run_context.workspace):
                try:
                    _content, artifact = compile_tool({"source_path": tex_path})
                except Exception as exc:
                    _content, artifact = self._nonfatal_tool_error_result(
                        "agentic_compile_tex",
                        exc,
                        {"source_path": tex_path},
                    )
            data = dict(artifact.get("data") or {}) if isinstance(artifact, dict) else {}
            compiled_ok = bool(data.get("compiled_ok"))
            pdf_path = self._normalize_artifact_path(str(data.get("pdf_path") or "").strip()) if data.get("pdf_path") else ""
            touched = [
                self._normalize_artifact_path(str(item).strip())
                for item in list(data.get("rewritten_files") or [])
                if str(item).strip()
            ]
            bib_paths = [
                self._normalize_artifact_path(str(item).strip())
                for item in list(data.get("bib_paths") or [])
                if str(item).strip()
            ]
            inspected = [
                self._normalize_artifact_path(str(item).strip())
                for item in list(data.get("inspected_files") or [])
                if str(item).strip()
            ]
            for candidate in [pdf_path, *bib_paths, *touched, *inspected]:
                if not candidate:
                    continue
                if candidate.lower().endswith(".pdf") or candidate.lower().endswith(".bib"):
                    if candidate not in seen_files:
                        seen_files.add(candidate)
                        normalized_files.append(candidate)
            if compiled_ok and pdf_path:
                updated_facts.append(f"Compile guard produced `{pdf_path}` from `{tex_path}`.")
            else:
                diagnostics = [str(item).strip() for item in list(data.get("remaining_diagnostics") or []) if str(item).strip()]
                if diagnostics:
                    updated_facts.append(f"Compile guard for `{tex_path}` reported: {diagnostics[0]}")
                else:
                    updated_facts.append(f"Compile guard ran for `{tex_path}` but no PDF was produced.")

        deduped_facts: list[str] = []
        seen_facts: set[str] = set()
        for item in updated_facts:
            normalized = str(item or "").strip()
            if not normalized or normalized in seen_facts:
                continue
            seen_facts.add(normalized)
            deduped_facts.append(normalized)
        return normalized_files, deduped_facts

    @staticmethod
    def _tex_bundle_matches(tex_path: str, candidate: str, *, suffix: str) -> bool:
        try:
            tex = Path(str(tex_path))
            other = Path(str(candidate))
        except Exception:
            return False
        if other.suffix.lower() != suffix.lower():
            return False
        if other.parent != tex.parent:
            return False
        if suffix.lower() == ".bib":
            return True
        return other.stem == tex.stem

    def _artifact_rows(self, files: list[str]) -> list[dict[str, str]]:
        rows: list[dict[str, str]] = []
        for raw_path in files:
            path = str(raw_path or "").strip()
            if not path:
                continue
            rows.append(
                {
                    "path": self._normalize_artifact_path(path),
                    "description": "reported file",
                    "kind": "file",
                }
            )
        return rows

    def _normalize_artifact_path(self, path: str) -> str:
        candidate = Path(path)
        if not candidate.is_absolute():
            return path.replace("\\", "/")
        try:
            return str(candidate.resolve().relative_to(workspace_root(self.run_context.workspace))).replace("\\", "/")
        except Exception:
            try:
                return str(candidate.resolve().relative_to(system_root(self.run_context.workspace))).replace("\\", "/")
            except Exception:
                return str(candidate)

    def _read_run_state(self) -> dict[str, Any]:
        path = self.run_context.run_dir / RUN_STATE_FILE
        if not path.exists():
            return {}
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
        return payload if isinstance(payload, dict) else {}

    def _write_run_state(self, payload: dict[str, Any]) -> None:
        path = self.run_context.run_dir / RUN_STATE_FILE
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _emit(self, name: str, *, payload: dict[str, Any] | None = None) -> None:
        try:
            self.reporter.emit(
                make_event(
                    name,
                    category="run",
                    run_id=self.run_context.run_id,
                    payload=payload or {},
                )
            )
        except Exception:
            logger.debug("failed to emit event %s", name, exc_info=True)

    @staticmethod
    def _load_create_deep_agent():
        try:
            from deepagents import create_deep_agent
        except Exception as exc:
            raise RuntimeError("deepagents is required for the new specialist runtime.") from exc
        return create_deep_agent

    @staticmethod
    def _load_create_agent():
        try:
            from langchain.agents import create_agent
        except Exception as exc:
            raise RuntimeError("langchain>=1.0 is required for proposal checkpoint generation.") from exc
        return create_agent

    @staticmethod
    def _load_tool_strategy():
        try:
            from langchain.agents.structured_output import ToolStrategy
        except Exception as exc:
            raise RuntimeError("LangChain ToolStrategy is required.") from exc
        return ToolStrategy

    @staticmethod
    def _load_subagent():
        try:
            from deepagents.middleware.subagents import SubAgent
        except Exception as exc:
            raise RuntimeError("deepagents subagent support is required.") from exc
        return SubAgent

    @staticmethod
    def _load_memory_middleware():
        try:
            from deepagents.middleware.memory import MemoryMiddleware
        except Exception as exc:
            raise RuntimeError("deepagents memory middleware is required.") from exc
        return MemoryMiddleware

    @staticmethod
    def _load_summarization_middleware():
        try:
            from deepagents.middleware.summarization import SummarizationMiddleware
        except Exception as exc:
            raise RuntimeError("deepagents summarization middleware is required.") from exc
        return SummarizationMiddleware

    @staticmethod
    def _load_create_summarization_tool_middleware():
        try:
            from deepagents.middleware.summarization import create_summarization_tool_middleware
        except Exception as exc:
            raise RuntimeError("deepagents compact_conversation middleware is required.") from exc
        return create_summarization_tool_middleware


__all__ = ["BuiltSpecialistRunner", "RUN_STATE_FILE", "SpecialistRunner", "build_specialist_runner"]
