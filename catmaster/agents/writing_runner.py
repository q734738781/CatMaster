from __future__ import annotations

import asyncio
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catmaster.agents.graph import _build_role_middleware, _load_create_agent, _load_tool_strategy
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import ResearchStore
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.skills import CatMasterSkillsRuntime
from catmaster.runtime.tool_surface import _dedupe_tools, _make_role_scoped_literature_tool
from catmaster.runtime.writing import (
    WritingBoard,
    WritingStore,
    WritingToolDeps,
    make_read_research_pack_tool,
    make_review_research_context_tool,
)
from catmaster.tools.base import system_root
from catmaster.tools.registry import get_tool_registry
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .writing_graph import build_writing_graph
from .writing_prompts import (
    get_section_writer_system_prompt,
    get_write_director_system_prompt,
    get_write_finalizer_system_prompt,
)
from .writing_schemas import SectionDraftOutput, WritingFinalizeOutput, WritingPlanOutput, WritingRequest


def _build_agent(*, model, tools, system_prompt: str, schema: type, role: str, skills_runtime, mounted_skill_tokens) -> Any:
    create_agent = _load_create_agent()
    ToolStrategy = _load_tool_strategy()
    middleware = _build_role_middleware(
        role=role,
        lane="writing",
        max_tool_calls=40 if role == "section_writer" else 12,
        skills_runtime=skills_runtime,
        mounted_skill_tokens=mounted_skill_tokens,
        selector_model=None,
        enable_selector=False,
    )
    return create_agent(
        model=model,
        tools=list(tools),
        system_prompt=system_prompt,
        response_format=ToolStrategy(schema, handle_errors=False),
        middleware=middleware,
    )


class WritingRunner:
    def __init__(
        self,
        *,
        llm_profile: LLMProfile,
        run_context: RunContext,
        reporter: Reporter | None = None,
        skills_runtime: CatMasterSkillsRuntime | None = None,
    ) -> None:
        self.llm_profile = llm_profile
        self.run_context = run_context
        self.reporter = reporter or NullReporter()
        self.skills_runtime = skills_runtime
        self.store = WritingStore(workspace=run_context.workspace, run_id=run_context.run_id)

    def run(self, request: WritingRequest | dict[str, Any]) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("WritingRunner.run() cannot be called inside a running event loop; use arun().")
        return asyncio.run(self.arun(request))

    def resume(self) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("WritingRunner.resume() cannot be called inside a running event loop; use aresume().")
        return asyncio.run(self.aresume())

    async def arun(self, request: WritingRequest | dict[str, Any]) -> dict[str, Any]:
        request_model = request if isinstance(request, WritingRequest) else WritingRequest.model_validate(request)
        self.store.prepare_new_run()
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "writing",
                "status": "planning",
                "current_phase": "planning",
                "current_work_label": "Initialize writing run",
                "request": request_model.request,
                "chat_session_id": request_model.chat_session_id,
                "entry_context_tokens_estimate": request_model.entry_context_tokens_estimate,
                "source_campaign_id": request_model.source_campaign_id,
                "summary": "",
                "final_answer": "",
            }
        )
        return await self._run_graph(
            {
                "request": request_model.model_dump(),
                "status": "planning",
                "resume_mode": False,
            }
        )

    async def aresume(self) -> dict[str, Any]:
        request = self.store.load_request()
        board = self.store.load_board()
        if request is None or board is None:
            raise ValueError("Writing resume failed: missing persisted request or board.")
        resume_goto = (
            "summarize_writing"
            if board.status == "done"
            else (
                ("finalize_markdown" if getattr(board, "output_format", "tex") == "md" else "finalize_writing")
                if board.status == "finalizing"
                else (("assemble_markdown" if getattr(board, "output_format", "tex") == "md" else "assemble_manuscript") if board.status == "reviewing" else "write_section" if board.status == "drafting" else "plan_writing")
            )
        )
        state = {
            "request": request,
            "board": board,
            "status": board.status,
            "resume_mode": True,
            "resume_goto": resume_goto,
        }
        return await self._run_graph(state)

    async def _run_graph(self, initial_state: dict[str, Any]) -> dict[str, Any]:
        request_model = WritingRequest.model_validate(initial_state["request"])
        requested_output_format = str(request_model.output_format or "tex").strip().lower() or "tex"
        source_campaign_id = str(request_model.source_campaign_id or "").strip() or None
        source_store = (
            ResearchStore(workspace=self.run_context.workspace, campaign_id=source_campaign_id)
            if source_campaign_id is not None
            else None
        )
        registry = get_tool_registry()
        local_tools = registry.as_langchain_tools(
            allowlist=[
                "apply_aider_edits",
                "render_structure_views",
                "analyze_images",
                "generate_schematic_figure",
                "agentic_compile_tex",
                "polish_academic_prose",
                "run_literature_research",
            ],
            run_dir=str(self.run_context.run_dir),
            workspace=str(self.run_context.workspace),
        )
        deps = WritingToolDeps(
            workspace=self.run_context.workspace,
            memory_store=MemoryStore.create_default(workspace=self.run_context.workspace),
            project_id=self.run_context.project_id,
        )
        deps.memory_store.ensure_exists()
        mounted_skill_tokens: tuple[str, ...] = ()
        local_by_name = {str(getattr(tool, "name", "") or ""): tool for tool in local_tools}
        compile_tool = local_by_name.get("agentic_compile_tex") if requested_output_format == "tex" else None
        polish_tool = local_by_name.get("polish_academic_prose")
        section_tools = [
            _make_role_scoped_literature_tool(tool, role="section_writer")
            if getattr(tool, "name", "") == "run_literature_research"
            else tool
            for tool in local_tools
            if getattr(tool, "name", "") != "polish_academic_prose"
        ]
        section_tools.extend(
            [
                make_read_research_pack_tool(deps),
                make_review_research_context_tool(deps),
            ]
        )
        director_tools = [
            *([compile_tool] if compile_tool is not None else []),
            *([polish_tool] if polish_tool is not None else []),
            make_read_research_pack_tool(deps),
            make_review_research_context_tool(deps),
        ]
        write_director_agent = _build_agent(
            model=build_chat_model(self.llm_profile.config_for_role("write_director")),
            tools=_dedupe_tools(director_tools),
            system_prompt=get_write_director_system_prompt(requested_output_format),
            schema=WritingPlanOutput,
            role="write_director",
            skills_runtime=self.skills_runtime,
            mounted_skill_tokens=mounted_skill_tokens,
        )
        write_finalizer_agent = _build_agent(
            model=build_chat_model(self.llm_profile.config_for_role("write_director")),
            tools=_dedupe_tools(director_tools),
            system_prompt=get_write_finalizer_system_prompt(requested_output_format),
            schema=WritingFinalizeOutput,
            role="write_director",
            skills_runtime=self.skills_runtime,
            mounted_skill_tokens=mounted_skill_tokens,
        )
        section_writer_agent = _build_agent(
            model=build_chat_model(self.llm_profile.config_for_role("section_writer")),
            tools=_dedupe_tools(section_tools),
            system_prompt=get_section_writer_system_prompt(requested_output_format),
            schema=SectionDraftOutput,
            role="section_writer",
            skills_runtime=self.skills_runtime,
            mounted_skill_tokens=mounted_skill_tokens,
        )
        graph = build_writing_graph(
            writing_store=self.store,
            write_director_agent=write_director_agent,
            write_finalizer_agent=write_finalizer_agent,
            section_writer_agent=section_writer_agent,
            write_reviewer_model=build_chat_model(self.llm_profile.config_for_role("write_reviewer")),
            source_store=source_store,
            skills_runtime=self.skills_runtime,
            writing_config=self.llm_profile.writing,
            progress_callback=self._update_task_state_progress,
        )
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": self.run_context.run_id}})
        summary = str(result.get("summary") or "").strip()
        status = str(result.get("status") or "done")
        board_loader = getattr(self.store, "load_board", None)
        board = board_loader() if callable(board_loader) else None
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "writing",
                "status": status,
                "current_phase": "done" if status == "done" else status,
                "current_work_label": "Writing completed" if status == "done" else (summary[:180] if summary else ""),
                "request": request_model.request,
                "chat_session_id": request_model.chat_session_id,
                "entry_context_tokens_estimate": request_model.entry_context_tokens_estimate,
                "source_campaign_id": request_model.source_campaign_id,
                "writing_mode": str(getattr(board, "writing_mode", "") or ""),
                "output_format": str(getattr(board, "output_format", request_model.output_format) or request_model.output_format or ""),
                "summary": summary,
                "final_answer": summary,
            }
        )
        return {
            "status": status,
            "summary": summary,
            "final_answer": summary,
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
        }

    def _write_task_state(self, payload: dict[str, Any]) -> None:
        path = self.run_context.run_dir / "task_state.json"
        path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")

    def _update_task_state_progress(self, *, current_phase: str, current_work_label: str) -> None:
        path = self.run_context.run_dir / "task_state.json"
        existing: dict[str, Any] = {}
        if path.exists():
            try:
                loaded = json.loads(path.read_text(encoding="utf-8"))
                if isinstance(loaded, dict):
                    existing = loaded
            except Exception:
                existing = {}
        existing["current_phase"] = str(current_phase or "").strip()
        existing["current_work_label"] = str(current_work_label or "").strip()
        path.write_text(json.dumps(existing, ensure_ascii=False, indent=2), encoding="utf-8")

__all__ = ["WritingRunner"]
