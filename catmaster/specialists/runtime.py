from __future__ import annotations

import asyncio
import json
import logging
import re
import shutil
from contextlib import AsyncExitStack, asynccontextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from langchain_core.callbacks import UsageMetadataCallbackHandler
from langchain_core.messages import AIMessage
from langchain_core.outputs import ChatGeneration, LLMResult

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_callback import LangChainStepLogger, UIEventHandler
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.usage_stats import write_usage_summary_from_metadata
from catmaster.tools.base import system_root, workspace_root
from catmaster.tools.registry import get_tool_registry
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .schemas import (
    ProposalCheckpoint,
    SpecialistEntrypoint,
)

logger = logging.getLogger(__name__)

RUN_STATE_FILE = "run_state.json"
PROPOSAL_FILE = "proposal.md"
MEMORY_STORE_FILE = "deepagent_memory.sqlite"
MEMORY_FILE_PATH = "/memories/AGENTS.md"

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
    "run_literature_research",
}
_WRITING_TOOL_ALLOWLIST = {
    "agentic_compile_tex",
    "polish_academic_prose",
    "analyze_images",
    "render_structure_views",
    "generate_schematic_figure",
    "run_literature_research",
}
_RESEARCH_TOOL_ALLOWLIST = {
    "run_literature_research",
    "analyze_images",
    "render_structure_views",
    "mp_search_materials",
    "mp_download_structure",
}
_TASK_WORKER_TOOL_ALLOWLIST = _EXPERIMENT_TOOL_ALLOWLIST - {"run_literature_research"}
_LITERATURE_AGENT_TOOL_ALLOWLIST = {"run_literature_research"}
_WRITING_WORKER_TOOL_ALLOWLIST = _WRITING_TOOL_ALLOWLIST - {"agentic_compile_tex", "run_literature_research"}
_COMPILE_AGENT_TOOL_ALLOWLIST = {"agentic_compile_tex"}


class SpecialistUsageCallbackHandler(UsageMetadataCallbackHandler):
    """Official LangChain usage tracker with per-model call counts for specialist runs."""

    def __init__(self) -> None:
        super().__init__()
        self.call_counts_by_model: dict[str, int] = {}

    def on_llm_end(self, response: LLMResult, **kwargs: Any) -> None:
        model_name = self._extract_model_name(response)
        super().on_llm_end(response, **kwargs)
        if model_name:
            self.call_counts_by_model[model_name] = int(self.call_counts_by_model.get(model_name, 0)) + 1

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
        session_context_text: str = "",
        chat_session_id: str = "",
        entry_context_tokens_estimate: int = 0,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.arun(
                prompt,
                entrypoint=entrypoint,
                proposal_review=proposal_review,
                session_context_text=session_context_text,
                chat_session_id=chat_session_id,
                entry_context_tokens_estimate=entry_context_tokens_estimate,
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
        session_context_text: str = "",
        chat_session_id: str = "",
        entry_context_tokens_estimate: int = 0,
    ) -> dict[str, Any]:
        payload = {
            "entrypoint": entrypoint,
            "user_prompt": str(prompt or "").strip(),
            "proposal_review": bool(proposal_review),
            "session_context_text": str(session_context_text or "").strip(),
            "chat_session_id": str(chat_session_id or "").strip(),
            "entry_context_tokens_estimate": int(entry_context_tokens_estimate or 0),
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

        files_root = workspace_root(self.run_context.workspace)
        files_root.mkdir(parents=True, exist_ok=True)
        self._stage_deepagent_assets(files_root)
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
                    session_context_text=str(payload.get("session_context_text") or ""),
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
                        "thread_id": self.run_context.run_id,
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
                        "session_context_text": str(payload.get("session_context_text") or ""),
                        "chat_session_id": str(payload.get("chat_session_id") or ""),
                        "entry_context_tokens_estimate": int(payload.get("entry_context_tokens_estimate") or 0),
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
                agent = await self._build_entry_agent(entrypoint=entrypoint, runtime=runtime)
                message_text = prompt if resume_feedback in (None, "") else (
                    f"{prompt}\n\nHuman review feedback:\n{resume_feedback}"
                )
                result = await agent.ainvoke(
                    {"messages": [{"role": "user", "content": message_text}]},
                    config={
                        "configurable": {"thread_id": self.run_context.run_id},
                        "callbacks": self._langchain_callbacks(usage_handler=usage_handler),
                    },
                )
            parsed = self._coerce_report(raw=result)
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
                    "thread_id": self.run_context.run_id,
                    "proposal_review": bool(payload.get("proposal_review", False)),
                    "pending_human_input": None,
                    "todo_items": [],
                    "artifacts": artifacts,
                    "delegation_log": [],
                    "text_preview": final_answer[:280],
                    "user_prompt": prompt,
                    "session_context_text": str(payload.get("session_context_text") or ""),
                    "chat_session_id": str(payload.get("chat_session_id") or ""),
                    "entry_context_tokens_estimate": int(payload.get("entry_context_tokens_estimate") or 0),
                    "final_answer": final_answer,
                    "summary": parsed["summary"],
                    "facts": list(parsed["facts"]),
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
        session_context_text: str,
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
                        "content": self._compose_request_text(
                            prompt=prompt,
                            session_context_text=session_context_text,
                        ),
                    }
                ]
            },
            config={"callbacks": self._langchain_callbacks(usage_handler=usage_handler)},
        )
        raw = result.get("structured_response") if isinstance(result, dict) else None
        if isinstance(raw, ProposalCheckpoint):
            return raw
        if isinstance(raw, dict):
            return ProposalCheckpoint.model_validate(raw)
        raise RuntimeError("Proposal checkpoint generation failed.")

    async def _build_entry_agent(self, *, entrypoint: SpecialistEntrypoint, runtime: dict[str, Any]) -> Any:
        create_deep_agent = self._load_create_deep_agent()
        tools = self._specialist_tools(entrypoint)
        skills = self._virtual_skill_paths(entrypoint)
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint])),
            "tools": tools,
            "system_prompt": self._system_prompt(entrypoint),
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
        memory_middleware = [self._new_memory_middleware(backend=runtime["backend"])]
        return [
            SubAgent(
                name="experiment_specialist",
                description="Run bounded computational experiment work and return compact evidence summaries.",
                system_prompt=self._system_prompt("experiment"),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._specialist_tools("experiment"),
                skills=self._virtual_skill_paths("experiment"),
                middleware=memory_middleware,
            ),
            SubAgent(
                name="writing_specialist",
                description="Turn existing evidence into reports, outlines, sections, or manuscript-ready outputs.",
                system_prompt=self._system_prompt("writing"),
                model=build_chat_model(self.llm_profile.config_for_role("write_director")),
                tools=self._specialist_tools("writing"),
                skills=self._virtual_skill_paths("writing"),
                middleware=memory_middleware,
            ),
            SubAgent(
                name="literature_agent",
                description="Retrieve compact literature grounding, citations, benchmark conventions, and background evidence when external scholarly context is needed.",
                system_prompt=self._literature_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
                tools=self._named_tools(_LITERATURE_AGENT_TOOL_ALLOWLIST),
                middleware=memory_middleware,
            ),
        ]

    def _experiment_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        memory_middleware = [self._new_memory_middleware(backend=runtime["backend"])]
        return [
            SubAgent(
                name="task_worker_agent",
                description="Handle bounded, context-heavy execution subtasks in isolation and return concise results with artifact paths.",
                system_prompt=self._task_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._named_tools(_TASK_WORKER_TOOL_ALLOWLIST),
                skills=self._virtual_skill_paths("experiment"),
                middleware=memory_middleware,
            ),
            SubAgent(
                name="literature_agent",
                description="Retrieve compact literature grounding, citations, benchmark conventions, and background evidence for experiment planning or interpretation.",
                system_prompt=self._literature_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
                tools=self._named_tools(_LITERATURE_AGENT_TOOL_ALLOWLIST),
                middleware=memory_middleware,
            ),
        ]

    def _writing_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        memory_middleware = [self._new_memory_middleware(backend=runtime["backend"])]
        return [
            SubAgent(
                name="writing_worker_agent",
                description="Draft or revise context-heavy sections in isolation and return compact manuscript-ready outputs.",
                system_prompt=self._writing_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("section_writer")),
                tools=self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                skills=self._virtual_skill_paths("writing"),
                middleware=memory_middleware,
            ),
            SubAgent(
                name="compile_agent",
                description="Run compile or static compile-fix passes on TeX bundles and report only compile-facing issues and repaired artifact paths.",
                system_prompt=self._compile_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("tex_compile_fixer")),
                tools=self._named_tools(_COMPILE_AGENT_TOOL_ALLOWLIST),
                middleware=memory_middleware,
            ),
        ]

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
        checkpoint_path = self.run_context.run_dir / "deepagent_state.sqlite"
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
        return self.registry.as_langchain_tools(
            allowlist=allowlist,
            run_dir=str(self.run_context.run_dir),
            workspace=str(self.run_context.workspace),
        )

    def _stage_deepagent_assets(self, files_root: Path) -> None:
        repo_root = Path(__file__).resolve().parents[2]
        deepagents_root = files_root / ".deepagents"
        base = deepagents_root / "skills"
        layouts = {
            base / "experiment": repo_root / "skills",
            base / "writing": repo_root / "writing_skills",
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

    def _compose_request_text(self, *, prompt: str, session_context_text: str) -> str:
        context = str(session_context_text or "").strip()
        if not context:
            return prompt
        return f"{prompt}\n\nRelevant prior session context:\n{context}"

    def _system_prompt(self, entrypoint: SpecialistEntrypoint) -> str:
        return self._base_system_prompt(entrypoint)

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
    def _base_system_prompt(cls, entrypoint: SpecialistEntrypoint) -> str:
        if entrypoint == "research":
            return (
                "You are ResearchSpecialist, the only orchestration-capable specialist.\n"
                "You coordinate scientific campaigns, decide when bounded experiment work is justified, "
                "and decide when writing/report generation should start.\n"
                "You may delegate only to `experiment_specialist`, `writing_specialist`, and `literature_agent`.\n"
                "Use `literature_agent` whenever external literature grounding, benchmark conventions, or representative citations are needed.\n"
                "Do not perform large direct execution yourself when delegation is more appropriate.\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._soft_reporting_contract()}"
            )
        if entrypoint == "writing":
            return (
                "You are WritingSpecialist.\n"
                "Write from existing workspace evidence only. Do not initiate new computational experiments.\n"
                "Use `writing_worker_agent` for context-heavy drafting or revision subtasks.\n"
                "Use `compile_agent` for TeX compile/fix passes instead of handling compile details in the main thread.\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._soft_reporting_contract()}"
            )
        return (
            "You are ExperimentSpecialist.\n"
            "Perform bounded computational execution in the current workspace using available tools and skills.\n"
            "Use `task_worker_agent` for context-heavy isolated execution subtasks, and `literature_agent` for external literature/background grounding.\n"
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
    def _literature_agent_prompt(cls) -> str:
        return (
            "You are literature_agent.\n"
            "Use `run_literature_research` to gather compact external literature grounding, benchmark conventions, citations, or background evidence.\n"
            "Return concise findings with clear separation between retrieved facts and inference.\n"
            "Do not perform local file manipulation or computational execution.\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _writing_worker_prompt(cls) -> str:
        return (
            "You are writing_worker_agent for WritingSpecialist.\n"
            "Draft, revise, or polish bounded writing subtasks from existing workspace evidence only.\n"
            "Return concise manuscript-ready output summaries and any output artifact paths.\n"
            "Do not handle TeX compile/fix passes; that belongs to compile_agent.\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _compile_agent_prompt(cls) -> str:
        return (
            "You are compile_agent for WritingSpecialist.\n"
            "Use `agentic_compile_tex` to run compile or static compile-fix passes on TeX manuscript bundles.\n"
            "Focus strictly on compile-facing issues, path/reference fixes, and repaired output paths.\n"
            "Do not rewrite scientific content beyond what is necessary for compile correctness.\n"
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
        try:
            from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
        except Exception:
            return []
        return [ModelCallLimitMiddleware(run_limit=40)]

    def _langchain_callbacks(self, *, usage_handler: SpecialistUsageCallbackHandler | None) -> list[Any]:
        callbacks: list[Any] = []
        if usage_handler is not None:
            callbacks.append(usage_handler)
        if not isinstance(self.reporter, NullReporter):
            callbacks.append(UIEventHandler(self.reporter, run_id=self.run_context.run_id))
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
        write_usage_summary_from_metadata(
            self.run_context.run_dir,
            usage_metadata=usage_metadata,
            call_counts_by_model=call_counts_by_model if isinstance(call_counts_by_model, dict) else {},
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


__all__ = ["BuiltSpecialistRunner", "RUN_STATE_FILE", "SpecialistRunner", "build_specialist_runner"]
