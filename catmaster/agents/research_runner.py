from __future__ import annotations

import asyncio
import json
import logging
from contextlib import asynccontextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from langchain_core.tools import StructuredTool

from catmaster.agents.graph import _build_role_middleware, _load_create_agent, _load_tool_strategy
from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.mcp_filesystem import MCPFilesystemRuntime
from catmaster.runtime.manager_tools import memory_read_index
from catmaster.runtime.research import ResearchStore
from catmaster.runtime.research.experiment_runner import ExperimentLaneRunner
from catmaster.runtime.research.literature_runner import ResearchLiteratureRunner
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.skills import CatMasterSkillsRuntime
from catmaster.runtime.tool_surface import _dedupe_tools
from catmaster.tools.registry import get_tool_registry
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .research_graph import build_research_graph
from .research_prompts import RESEARCH_LEAD_SYSTEM_PROMPT, RESEARCH_STATE_UPDATER_SYSTEM_PROMPT
from .research_schemas import ResearchLeadOutput, ResearchRequest, ResearchStateSyncOutput, RunWriterPayload
from .writing_runner import WritingRunner
from .writing_schemas import WritingRequest

logger = logging.getLogger(__name__)


RESEARCH_TO_WRITER_HOUSE_PROMPT = """Write from the current workspace evidence and available research artifacts only.
Do not perform new expensive computations.
Choose the writing workflow strictly from the requested writing mode and output format.
If the requested output format is markdown, do not switch into LaTeX/manuscript compilation workflow.
If the requested output format is TeX, write TeX-ready scientific content and keep compilation concerns inside the TeX branch only.
Do not drift into experiment logs or execution traces unless the requested writing mode explicitly implies a report-style deliverable."""


class ResearchRunner:
    def __init__(
        self,
        *,
        llm_profile: LLMProfile,
        run_context: RunContext,
        memory_store: MemoryStore,
        reporter: Reporter | None = None,
        skills_runtime: CatMasterSkillsRuntime | None = None,
    ) -> None:
        self.llm_profile = llm_profile
        self.run_context = run_context
        self.memory_store = memory_store
        self.reporter = reporter or NullReporter()
        self.skills_runtime = skills_runtime
        self.store = ResearchStore(workspace=run_context.workspace, campaign_id=run_context.run_id)
        self.store.ensure_exists()
        self.literature_runner = ResearchLiteratureRunner(allow_deep_report=False)
        self.experiment_runner = ExperimentLaneRunner(
            workspace=run_context.workspace,
            llm_profile=llm_profile,
            project_id=run_context.project_id,
            reporter=self.reporter,
        )

    @staticmethod
    def _build_structured_agent(*, role: str, model, tools, skills_runtime, mounted_skill_tokens, system_prompt: str, schema: Any) -> Any:
        create_agent = _load_create_agent()
        ToolStrategy = _load_tool_strategy()
        middleware = _build_role_middleware(
            role=role,
            lane="research",
            max_tool_calls=12,
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

    def _build_memory_read_tool(self) -> StructuredTool:
        return StructuredTool.from_function(
            func=lambda max_lines=None, max_chars=1800: memory_read_index(
                max_lines=max_lines,
                max_chars=max_chars,
                workspace=self.run_context.workspace,
            ),
            name="memory_read_index",
            description="Read the project MEMORY.md index to ground high-level planning decisions. Read-only.",
        )

    @asynccontextmanager
    async def _open_mcp_filesystem_runtime(self):
        cfg = getattr(getattr(self.llm_profile, "mcp", None), "filesystem", None)
        if cfg is None or not getattr(cfg, "enabled", False):
            yield None
            return
        runtime = MCPFilesystemRuntime(config=cfg, run_context=self.run_context, reporter=self.reporter)
        async with runtime as opened:
            yield opened

    def _emit(self, name: str, *, payload: dict[str, Any] | None = None) -> None:
        try:
            self.reporter.emit(
                make_event(name, category="run", run_id=self.run_context.run_id, payload=payload or {})
            )
        except Exception:
            pass

    def run(self, request: ResearchRequest | dict[str, Any]) -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("ResearchRunner.run() cannot be called inside a running event loop; use arun().")
        return asyncio.run(self.arun(request))

    def resume(self, *, resume_feedback: str = "") -> dict[str, Any]:
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            pass
        else:
            raise RuntimeError("ResearchRunner.resume() cannot be called inside a running event loop; use aresume().")
        return asyncio.run(self.aresume(resume_feedback=resume_feedback))

    async def arun(self, request: ResearchRequest | dict[str, Any]) -> dict[str, Any]:
        request_model = request if isinstance(request, ResearchRequest) else ResearchRequest.model_validate(request)
        initial_state = {
            "request": request_model.model_dump(),
            "literature_packs": [],
            "experiment_packs": [],
            "status": "running",
            "summary": "",
            "final_answer": "",
            "resume_mode": False,
        }
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "research",
                "question": request_model.question,
                "chat_session_id": request_model.chat_session_id,
                "entry_context_tokens_estimate": request_model.entry_context_tokens_estimate,
                "campaign_id": self.run_context.run_id,
                "status": "running",
                "current_phase": "planning",
                "current_work_label": "Initialize research campaign",
                "cycle_index": 0,
                "summary": "",
            }
        )
        self._emit("RUN_START", payload={"lane": "research", "question": request_model.question})
        return await self._run_graph(initial_state, request_model=request_model)

    async def aresume(self, *, resume_feedback: str = "") -> dict[str, Any]:
        state = self._build_resume_state(resume_feedback=resume_feedback)
        request_raw = state.get("request")
        if not isinstance(request_raw, dict):
            raise ValueError("Research resume failed: missing persisted request.json")
        request_model = ResearchRequest.model_validate(request_raw)
        self._emit("RUN_START", payload={"lane": "research", "question": request_model.question, "mode": "resume"})
        return await self._run_graph(state, request_model=request_model)

    async def _execute_writer_request(
        self,
        *,
        request_model: ResearchRequest,
        writer_payload: RunWriterPayload,
    ) -> dict[str, Any]:
        run_ctx = RunContext.create(
            workspace=self.run_context.workspace,
            project_id=self.run_context.project_id,
            model_name=self.llm_profile.config_for_role("write_director").model,
            provider=self.llm_profile.config_for_role("write_director").provider,
            base_url=self.llm_profile.config_for_role("write_director").base_url,
        )
        runner = WritingRunner(
            llm_profile=self.llm_profile,
            run_context=run_ctx,
            reporter=self.reporter,
            skills_runtime=self.skills_runtime,
        )
        result = await runner.arun(
            self._build_writer_request(
                request_model=request_model,
                source_campaign_id=self.run_context.run_id,
                writer_payload=writer_payload,
            )
        )
        final_output_path = str((result or {}).get("final_output_path") or "").strip()
        if not final_output_path:
            final_output_path = str((result or {}).get("final_report_path") or "").strip()
        enriched = dict(result or {})
        enriched["final_output_path"] = final_output_path
        return enriched

    async def _run_graph(
        self,
        initial_state: dict[str, Any],
        *,
        request_model: ResearchRequest,
    ) -> dict[str, Any]:
        registry = get_tool_registry()
        local_tools = registry.as_langchain_tools(
            allowlist=["memory_read_index"],
            run_dir=str(self.run_context.run_dir),
            workspace=str(self.run_context.workspace),
        )
        if not local_tools:
            local_tools = [self._build_memory_read_tool()]
        async with self._open_mcp_filesystem_runtime() as mcp_fs_runtime:
            planner_tools = list(local_tools)
            mounted_skill_tokens: tuple[str, ...] = ()
            if mcp_fs_runtime is not None:
                planner_tools.extend(mcp_fs_runtime.role_filtered_tools(role="director"))
                mounted_skill_tokens = tuple(mcp_fs_runtime.skill_mounts.keys())
            planner_agent = self._build_structured_agent(
                role="research_lead",
                model=build_chat_model(self.llm_profile.config_for_role("research_lead")),
                tools=_dedupe_tools(planner_tools),
                skills_runtime=self.skills_runtime,
                mounted_skill_tokens=mounted_skill_tokens,
                system_prompt=RESEARCH_LEAD_SYSTEM_PROMPT,
                schema=ResearchLeadOutput,
            )
            state_updater_agent = self._build_structured_agent(
                role="research_state_updater",
                model=build_chat_model(self.llm_profile.config_for_role("research_state_updater")),
                tools=_dedupe_tools(planner_tools),
                skills_runtime=self.skills_runtime,
                mounted_skill_tokens=mounted_skill_tokens,
                system_prompt=RESEARCH_STATE_UPDATER_SYSTEM_PROMPT,
                schema=ResearchStateSyncOutput,
            )
            graph = build_research_graph(
                store=self.store,
                planner_agent=planner_agent,
                state_updater_agent=state_updater_agent,
                memory_store=self.memory_store,
                literature_runner=ResearchLiteratureRunner(
                    allow_deep_report=request_model.allow_deep_report
                ),
                experiment_runner=self.experiment_runner,
                writer_runner=lambda payload: self._execute_writer_request(
                    request_model=request_model,
                    writer_payload=payload,
                ),
                skills_runtime=self.skills_runtime,
                progress_callback=self._update_task_state_progress,
            )
            result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": self.run_context.run_id}})
        summary = str(result.get("summary") or "").strip()
        status = str(result.get("status") or "done")
        writing_result = result.get("writing_result") if isinstance(result.get("writing_result"), dict) else {}
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "research",
                "question": request_model.question,
                "chat_session_id": request_model.chat_session_id,
                "entry_context_tokens_estimate": request_model.entry_context_tokens_estimate,
                "campaign_id": self.run_context.run_id,
                "status": status,
                "current_phase": "done" if status == "done" else status,
                "current_work_label": "Research campaign completed" if status == "done" else (summary[:180] if summary else ""),
                "cycle_index": self._read_board_cycle_index(),
                "board_path": f"research_campaigns/{self.run_context.run_id}/board.json",
                "latest_action": self._lead_action_state(result.get("lead_action")),
                "latest_writing_run_id": str((result.get("writing_run_id") or writing_result.get("run_id") or "")),
                "summary": summary,
            }
        )
        report_paths = self._publish_report(question=request_model.question, final_answer=summary)
        self._emit("RUN_END", payload={"lane": "research", "status": status})
        return {
            "summary": summary,
            "final_answer": summary,
            "status": status,
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "final_report_path": report_paths.get("final_report", ""),
            "run_export_path": "",
            "writing_run_id": str((result.get("writing_run_id") or writing_result.get("run_id") or "")),
        }

    @staticmethod
    def _build_writer_request(
        *,
        request_model: ResearchRequest,
        source_campaign_id: str,
        writer_payload: RunWriterPayload | None = None,
    ) -> WritingRequest:
        payload = writer_payload
        request_text = (
            str(payload.request or "").strip()
            if payload is not None and str(payload.request or "").strip()
            else "\n".join(
                line
                for line in [
                    RESEARCH_TO_WRITER_HOUSE_PROMPT,
                    f"Research question: {request_model.question}",
                    f"Preferred writing mode: {request_model.writing_mode}." if request_model.writing_mode != "none" else "",
                    f"Preferred output format: {request_model.output_format}." if request_model.writing_mode != "none" else "",
                    f"Prefer focusing on section: {request_model.target_section}." if str(request_model.target_section or "").strip() else "",
                    f"Preferred title direction: {request_model.campaign_title}." if str(request_model.campaign_title or "").strip() else "",
                ]
                if line
            ).strip()
        )
        writing_mode = (
            str(payload.writing_mode or "").strip()
            if payload is not None
            else str(request_model.writing_mode or "").strip()
        ) or "internal_report"
        output_format = (
            str(payload.output_format or "").strip()
            if payload is not None
            else str(request_model.output_format or "").strip()
        ) or "tex"
        target_section = (
            str(payload.target_section or "").strip()
            if payload is not None
            else str(request_model.target_section or "").strip()
        )
        return WritingRequest(
            request=request_text,
            session_context_text=request_model.session_context_text,
            chat_session_id=request_model.chat_session_id,
            entry_context_tokens_estimate=request_model.entry_context_tokens_estimate,
            source_campaign_id=source_campaign_id,
            writing_mode=writing_mode,
            output_format=output_format,
            target_section=(target_section or None),
        )

    def _build_resume_state(self, *, resume_feedback: str = "") -> dict[str, Any]:
        request = self.store.load_request()
        if request is None:
            return {}
        board = self.store.load_board()
        feedback_text = self._normalize_feedback(resume_feedback)
        if board.status == "needs_human" and feedback_text:
            board = self._ingest_human_feedback(board=board, feedback_text=feedback_text)
        literature_packs = self.store.load_literature_packs()
        experiment_packs = self.store.load_experiment_packs()
        conclusion = self.store.load_conclusion()
        dossier = self.store.load_dossier()
        request_model = ResearchRequest.model_validate(request)
        resume_goto = "plan_research"
        status = "running"
        summary = ""
        final_answer = ""
        if board.status == "running":
            resume_goto = "plan_research"
            status = "running"
        elif board.status == "needs_human":
            if feedback_text:
                resume_goto = "sync_research_state"
                status = "running"
            else:
                resume_goto = "summarize_research"
                status = "needs_human"
                summary = self._build_needs_human_summary(board)
                final_answer = summary
        elif board.status == "done":
            status = "done"
            if dossier is None and request_model.writing_mode == "none":
                resume_goto = "build_dossier"
            else:
                resume_goto = "summarize_research"
                summary = self._build_done_summary(board=board, dossier=dossier, conclusion=conclusion)
                final_answer = summary
        else:
            resume_goto = "summarize_research"
            status = board.status
            summary = board.current_best_answer_md or "Research campaign is not resumable from the current status."
            final_answer = summary
        return {
            "request": request,
            "board": board,
            "latest_literature": literature_packs[-1] if literature_packs else None,
            "latest_experiment": experiment_packs[-1] if experiment_packs else None,
            "literature_packs": literature_packs,
            "experiment_packs": experiment_packs,
            "conclusion": conclusion,
            "dossier": dossier,
            "resume_mode": True,
            "resume_goto": resume_goto,
            "status": status,
            "summary": summary,
            "final_answer": final_answer,
        }

    def _write_task_state(self, body: dict[str, Any]) -> None:
        path = self.run_context.run_dir / "task_state.json"
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _publish_report(self, *, question: str, final_answer: str) -> dict[str, str]:
        reports_dir = self.run_context.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        final_report = reports_dir / "FINAL_REPORT.md"
        dossier_path = self.run_context.workspace / "files" / "research" / self.run_context.run_id / "dossier" / "RESEARCH_DOSSIER.md"
        lines = [
            "# Research Final Report",
            "",
            "## Research Question",
            question,
            "",
            "## Current Best Conclusion",
            final_answer,
            "",
            "## Campaign Artifacts",
            f"- Board: metadata/research_campaigns/{self.run_context.run_id}/board.json",
        ]
        if dossier_path.exists():
            lines.append(f"- Dossier: files/research/{self.run_context.run_id}/dossier/RESEARCH_DOSSIER.md")
            lines.append(f"- Dossier file exists: {dossier_path}")
        final_report.write_text("\n".join(lines).rstrip() + "\n", encoding="utf-8")
        return {"final_report": str(final_report)}

    def _read_board_cycle_index(self) -> int:
        board_path = self.store.metadata_root / "board.json"
        if not board_path.exists():
            return 0
        try:
            data = json.loads(board_path.read_text(encoding="utf-8"))
        except Exception:
            return 0
        try:
            return int(data.get("cycle_index") or 0)
        except Exception:
            return 0

    @staticmethod
    def _lead_action_state(lead_action: Any) -> str:
        if lead_action is None:
            return ""
        state = getattr(lead_action, "state", None)
        if state is not None:
            text = str(state).strip()
            if text:
                return text
        if isinstance(lead_action, dict):
            return str(lead_action.get("state") or "").strip()
        return ""

    def _ingest_human_feedback(self, *, board, feedback_text: str):
        action_id = f"human_feedback_{len(board.action_refs) + 1:03d}"
        summary = self._normalize_feedback(feedback_text, max_chars=1600)
        record = {
            "ts": datetime.now(timezone.utc).isoformat(),
            "action_id": action_id,
            "kind": "human_feedback",
            "status": "received",
            "summary": summary,
            "feedback": feedback_text,
            "questions": list(board.latest_human_questions),
            "run_id": None,
        }
        ref_path = self.store.append_action_log(record)
        updated = type(board).model_validate(
            {
                **board.model_dump(),
                "status": "running",
                "human_feedback_summary": summary,
                "action_refs": list(board.action_refs)
                + [
                    {
                        "action_id": action_id,
                        "kind": "human_feedback",
                        "status": "received",
                        "summary": summary,
                        "ref_path": ref_path,
                        "run_id": None,
                    }
                ],
            }
        )
        self.store.save_board(updated)
        return updated

    @staticmethod
    def _normalize_feedback(text: str, *, max_chars: int = 2400) -> str:
        compact = " ".join(str(text or "").split()).strip()
        if len(compact) <= max_chars:
            return compact
        return compact[: max(0, max_chars - 3)] + "..."

    @staticmethod
    def _build_needs_human_summary(board) -> str:
        questions = list(board.latest_human_questions) or list(board.open_questions)
        question_lines = [f"- {item}" for item in questions[:10]] or ["- (none)"]
        lines = [
            board.current_best_answer_md or "Research campaign is waiting for human input.",
            "",
            "Questions for human:",
            *question_lines,
        ]
        if board.human_feedback_summary:
            lines.extend(["", "Latest human feedback:", board.human_feedback_summary])
        return "\n".join(lines).strip()

    @staticmethod
    def _build_done_summary(*, board, dossier, conclusion) -> str:
        lines = [
            (conclusion.final_answer_md if conclusion is not None else board.current_best_answer_md) or "Research campaign completed.",
        ]
        if dossier is not None:
            lines.append("")
            lines.append(f"Dossier: research/{board.campaign_id}/dossier/RESEARCH_DOSSIER.md")
        return "\n".join(lines).strip()


__all__ = ["ResearchRunner"]
