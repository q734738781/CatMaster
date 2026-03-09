from __future__ import annotations

import asyncio
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.memory_store import MemoryStore
from catmaster.runtime.research import ResearchStore
from catmaster.runtime.research.experiment_runner import ExperimentLaneRunner
from catmaster.runtime.research.literature_runner import ResearchLiteratureRunner
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_ledger.blob_builder import build_run_search_blob
from catmaster.runtime.run_ledger.history_reader import HistoryReader
from catmaster.runtime.run_ledger.models import RunLedgerEntry
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.runtime.skills import CatMasterSkillsRuntime
from catmaster.tools.base import system_root
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .research_graph import build_research_graph
from .research_schemas import ResearchRequest
from .writing_runner import WritingRunner
from .writing_schemas import WritingRequest

logger = logging.getLogger(__name__)


RESEARCH_TO_WRITER_HOUSE_PROMPT = """Write a manuscript for current results presented workspace in TeX, using only the existing workspace evidence, and available research artifacts.
Do not perform new expensive computations.
Do not write this as an experiment log, execution trace, or lab report.
Write it as a compact journal-style scientific sections with Abstract, Introduction, Results and Discussion and Methods.
Prefer ACS-like manuscript tone when appropriate.
You can generate lightweight schematic figures and using existing data to create new result figures to enhance the manuscript."""


class ResearchRunner:
    def __init__(
        self,
        *,
        llm_profile: LLMProfile,
        run_context: RunContext,
        memory_store: MemoryStore,
        reporter: Reporter | None = None,
        run_ledger_store: RunLedgerStore | None = None,
        history_reader: HistoryReader | None = None,
        skills_runtime: CatMasterSkillsRuntime | None = None,
    ) -> None:
        self.llm_profile = llm_profile
        self.run_context = run_context
        self.memory_store = memory_store
        self.reporter = reporter or NullReporter()
        self.run_ledger_store = run_ledger_store
        self.history_reader = history_reader
        self.skills_runtime = skills_runtime
        self.store = ResearchStore(workspace=run_context.workspace, campaign_id=run_context.run_id)
        self.store.ensure_exists()
        self.planner_model = build_chat_model(llm_profile.config_for_role("research_lead"))
        self.literature_runner = ResearchLiteratureRunner(allow_deep_report=False)
        self.experiment_runner = ExperimentLaneRunner(
            workspace=run_context.workspace,
            llm_profile=llm_profile,
            project_id=run_context.project_id,
            reporter=self.reporter,
        )

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
            "history_context_summary": await self._load_history_context(request_model.question),
            "resume_mode": False,
        }
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "research",
                "question": request_model.question,
                "campaign_id": self.run_context.run_id,
                "status": "running",
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

    async def _run_graph(
        self,
        initial_state: dict[str, Any],
        *,
        request_model: ResearchRequest,
    ) -> dict[str, Any]:
        graph = build_research_graph(
            store=self.store,
            planner_model=self.planner_model,
            memory_store=self.memory_store,
            literature_runner=ResearchLiteratureRunner(
                allow_deep_report=request_model.allow_deep_report
            ),
            experiment_runner=self.experiment_runner,
            history_reader=self.history_reader,
            project_id=self.run_context.project_id,
            skills_runtime=self.skills_runtime,
        )
        result = await graph.ainvoke(initial_state, config={"configurable": {"thread_id": self.run_context.run_id}})
        summary = str(result.get("summary") or "").strip()
        status = str(result.get("status") or "done")
        writing_result: dict[str, Any] | None = None
        if status == "done" and request_model.writing_mode != "none":
            writing_result = await self._launch_writing_handoff(request_model=request_model)
            writing_summary = str((writing_result or {}).get("summary") or "").strip()
            if writing_summary:
                summary = "\n\n".join([summary, writing_summary]).strip()
        self._write_task_state(
            {
                "schema_version": 1,
                "lane": "research",
                "question": request_model.question,
                "campaign_id": self.run_context.run_id,
                "status": status,
                "cycle_index": self._read_board_cycle_index(),
                "board_path": f"research_campaigns/{self.run_context.run_id}/board.json",
                "latest_action": self._lead_action_state(result.get("lead_action")),
                "latest_writing_run_id": str((writing_result or {}).get("run_id") or ""),
                "summary": summary,
            }
        )
        report_paths = self._publish_report(question=request_model.question, final_answer=summary)
        export_paths = self._publish_run_export(
            request=request_model,
            status=status,
            final_answer=summary,
            report_paths=report_paths,
        )
        await self._upsert_run_ledger(
            request=request_model,
            status=status,
            final_answer=summary,
            report_paths=report_paths,
            export_paths=export_paths,
        )
        self._emit("RUN_END", payload={"lane": "research", "status": status})
        return {
            "summary": summary,
            "final_answer": summary,
            "status": status,
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "final_report_path": report_paths.get("final_report", ""),
            "run_export_path": export_paths.get("run_export", ""),
            "writing_run_id": str((writing_result or {}).get("run_id") or ""),
        }

    async def _launch_writing_handoff(self, *, request_model: ResearchRequest) -> dict[str, Any]:
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
            run_ledger_store=self.run_ledger_store,
            history_reader=self.history_reader,
            skills_runtime=self.skills_runtime,
        )
        writing_prompt_lines = [
            RESEARCH_TO_WRITER_HOUSE_PROMPT,
            f"Research question: {request_model.question}",
        ]
        if request_model.writing_mode != "none":
            writing_prompt_lines.append(f"Preferred writing mode: {request_model.writing_mode}.")
        if str(request_model.target_section or "").strip():
            writing_prompt_lines.append(f"Prefer focusing on section: {request_model.target_section}.")
        if str(request_model.campaign_title or "").strip():
            writing_prompt_lines.append(f"Preferred title direction: {request_model.campaign_title}.")
        return await runner.arun(
            WritingRequest(
                request="\n".join(writing_prompt_lines).strip(),
                source_campaign_id=self.run_context.run_id,
            )
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
                resume_goto = "plan_research"
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
            "history_context_summary": board.history_context_summary or "",
            "resume_mode": True,
            "resume_goto": resume_goto,
            "status": status,
            "summary": summary,
            "final_answer": final_answer,
        }

    async def _load_history_context(self, question: str) -> str:
        if self.history_reader is None:
            return ""
        try:
            pack = await self.history_reader.aload_context(
                query=question,
                project_id=self.run_context.project_id,
                lane=None,
            )
        except Exception as exc:
            logger.warning("research history prefetch failed: %s", exc)
            return ""
        return str(pack.context_text or "").strip()

    def _write_task_state(self, body: dict[str, Any]) -> None:
        path = self.run_context.run_dir / "task_state.json"
        path.write_text(json.dumps(body, ensure_ascii=False, indent=2), encoding="utf-8")

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

    def _publish_run_export(
        self,
        *,
        request: ResearchRequest,
        status: str,
        final_answer: str,
        report_paths: dict[str, str],
    ) -> dict[str, str]:
        reports_dir = self.run_context.run_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        export_path = reports_dir / "RUN_EXPORT.json"
        blob = build_run_search_blob(self.run_context.run_dir)
        payload = {
            "request": request.question,
            "answer_summary": final_answer,
            "lane": "research",
            "status": status,
            "task_goals": [request.question],
            "top_observations": [],
            "tool_names": blob.tool_names,
            "artifact_paths": blob.artifact_paths,
            "final_report_path": self._system_relpath(report_paths.get("final_report", "")),
            "run_dir": self._system_relpath(self.run_context.run_dir),
            "run_id": self.run_context.run_id,
            "project_id": self.run_context.project_id,
        }
        export_path.write_text(json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
        return {
            "run_export": str(export_path),
            "run_export_relpath": self._system_relpath(export_path),
        }

    async def _upsert_run_ledger(
        self,
        *,
        request: ResearchRequest,
        status: str,
        final_answer: str,
        report_paths: dict[str, str],
        export_paths: dict[str, str],
    ) -> None:
        if self.run_ledger_store is None:
            return
        blob = build_run_search_blob(self.run_context.run_dir)
        entry = RunLedgerEntry(
            project_id=self.run_context.project_id,
            run_id=self.run_context.run_id,
            lane="research",
            status=status,
            request=request.question,
            answer_summary=final_answer,
            search_blob_text=blob.search_blob_text,
            final_report_relpath=self._system_relpath(report_paths.get("final_report", "")),
            run_export_relpath=str(export_paths.get("run_export_relpath") or ""),
            ts_start=self.run_context.start_time,
            ts_end=datetime.now(timezone.utc).isoformat(),
            model_name=self.run_context.model_name,
            provider=str(self.run_context.provider or ""),
        )
        self.run_ledger_store.upsert_entry(entry)
        if self.history_reader is not None:
            try:
                await self.history_reader.aindex_entry(entry)
            except Exception as exc:
                logger.warning("research run ledger index update failed: %s", exc)

    def _system_relpath(self, raw: str | Path) -> str:
        path = Path(str(raw)).expanduser().resolve()
        root = system_root(self.run_context.workspace).resolve()
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            return str(path)

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
