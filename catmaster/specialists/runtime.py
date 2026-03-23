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
import yaml

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_callback import LangChainStepLogger, UIEventHandler
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.runtime.usage_stats import write_usage_summary_from_metadata
from catmaster.tools.base import system_root, workspace_root, workspace_scope
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
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
    "peer_review": "write_reviewer",
}

_MATERIALS_WORKER_TOOL_ALLOWLIST = {
    "create_molecule_from_smiles",
    "mace_neb_batch",
    "mace_relax_batch",
    "mace_sp_batch",
    "vasp_prepare",
    "vasp_band_prepare",
    "build_slab",
    "fix_atoms_by_layers",
    "fix_atoms_by_height",
    "fix_atoms_by_indices",
    "supercell",
    "enumerate_unique_sites",
    "create_vacancy",
    "substitute_species",
    "insert_interstitial_at_coords",
    "enumerate_adsorption_sites",
    "place_adsorbate",
    "generate_batch_adsorption_structures",
    "make_neb_geometry",
    "make_dimer_mode_from_neb",
    "make_dimer_mode_from_mace",
    "generate_strained_structures",
    "generate_kpath",
    "generate_phonon_displacements",
    "vasp_neb_prepare",
    "vasp_dimer_prepare",
    "vasp_execute_batch",
    "mp_search_materials",
    "mp_download_structure",
    "render_structure_views",
    "analyze_images",
    "identify_structure_fragments",
    "analyze_vasp_results",
    "analyze_neb_results",
    "analyze_trajectory",
    "generate_nanobanana_figure",
    "vaspkit_adsorbate_thermo_correction",
    "vaspkit_gas_thermo_correction",
}
_ML_WORKER_TOOL_ALLOWLIST: set[str] = {
    "build_dataset_from_runs",
    "mace_train",
    "mace_evaluate",
    "calculate_al_candidates",
}
_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST = set(_MATERIALS_WORKER_TOOL_ALLOWLIST) | set(_ML_WORKER_TOOL_ALLOWLIST)
_WRITING_TOOL_ALLOWLIST = {
    "analyze_images",
    "render_structure_views",
    "generate_nanobanana_figure",
    "review_pdf_manuscript",
}
_RESEARCH_TOOL_ALLOWLIST = {
    "analyze_images",
    "render_structure_views",
    "mp_search_materials",
    "mp_download_structure",
}
_PEER_REVIEW_TOOL_ALLOWLIST = {"peer_review_request"}
_METADATA_AGENT_TOOL_ALLOWLIST = {
    "search_openalex",
    "search_semantic_scholar",
    "get_openalex_record",
    "get_semantic_scholar_record",
    "recommend_semantic_scholar",
}
_LITREVIEW_AGENT_TOOL_ALLOWLIST = {
    "search_public_web",
    "open_public_page",
    "find_in_page",
}
_LIGHTWEIGHT_LITERATURE_AGENT_TOOL_NAMES = {"internet_search"}
_PROJECT_MEMORY_READ_TOOL_NAMES = {"search_memory"}
_PROJECT_MEMORY_WRITE_TOOL_NAMES = {"manage_memory"}
_PROJECT_MEMORY_TOOL_NAMES = {"manage_memory", "search_memory"}
_DEEPAGENT_BUILTIN_TOOL_NAMES = {
    "write_todos",
    "ls",
    "read_file",
    "write_file",
    "edit_file",
    "glob",
    "grep",
    "execute",
}
_MATERIALS_WORKER_SELECTOR_MAX_TOOLS = 8
_MATERIALS_WORKER_SELECTOR_ALWAYS_INCLUDE = tuple(
    sorted(_DEEPAGENT_BUILTIN_TOOL_NAMES | _PROJECT_MEMORY_READ_TOOL_NAMES)
)
_WRITING_WORKER_TOOL_ALLOWLIST = {
    "polish_academic_prose",
    "analyze_images",
    "render_structure_views",
    "generate_nanobanana_figure",
    "compile_text",
}
_LITREVIEW_COMPACT_TRIGGER_TOKENS = 65_000
_LITREVIEW_COMPACT_KEEP_TOKENS = 6_500
_PROJECT_LONG_TERM_MEMORY_NAMESPACE = ("catmaster", "{project_id}", "long_term_memory")
_PROJECT_MEMORY_TOOL_INSTRUCTIONS = (
    "Project long-term memory tools:\n"
    "- `search_memory` and `manage_memory` target the project-level LangMem store.\n"
    "- Use them for durable project facts, validated reusable conclusions, stable project state, "
    "and correction or removal of stale incorrect memories.\n"
    "- Do not use them for transient requests, step logs, one-off scratch paths, or unfinished speculation.\n"
    "- Before updating or deleting an existing long-term memory, call `search_memory` first to find the correct MEMORY IDs.\n"
    "- Treat `/.deepagents/AGENTS.md` and `/memories/AGENTS.md` as instruction memory, not the project fact store."
)
_PROJECT_MEMORY_READONLY_INSTRUCTIONS = (
    "Project long-term memory tools:\n"
    "- You have `search_memory` access to the project-level LangMem store.\n"
    "- Use it to retrieve durable project facts, validated reusable conclusions, and stable project conventions before starting or when you suspect prior relevant work exists.\n"
    "- You do not have permission to modify long-term project memory from this subagent. If durable memory should be added, corrected, or deleted, report that explicitly so the parent specialist can decide whether to call `manage_memory`.\n"
    "- Treat `/.deepagents/AGENTS.md` and `/memories/AGENTS.md` as instruction memory, not the project fact store."
)
_PROJECT_MANAGE_MEMORY_INSTRUCTIONS = (
    "Proactively call this tool when current work yields durable project memory. "
    "Store concise validated facts, reusable conclusions, stable project conventions, or durable project status. "
    "If a stored project memory is wrong, outdated, or superseded, update or delete it instead of creating duplicates. "
    "Do not store transient prompts, scratch calculations, one-off file paths, or speculative unfinished findings."
)
_PROJECT_SEARCH_MEMORY_INSTRUCTIONS = (
    "Search project long-term memory before creating a similar memory, and always search before updating or deleting memories."
)
_SKILL_VIEW_ROOT = "/.deepagents/skill_views"
_SPECIALIST_SKILL_VIEW_SPECS: dict[str, tuple[str, set[str]]] = {
    "research_experiment": ("experiment", _DEEPAGENT_BUILTIN_TOOL_NAMES | _RESEARCH_TOOL_ALLOWLIST),
    "research_writing": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _RESEARCH_TOOL_ALLOWLIST),
    "experiment_specialist": ("experiment", _DEEPAGENT_BUILTIN_TOOL_NAMES | _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST),
    "materials_worker": ("experiment", _DEEPAGENT_BUILTIN_TOOL_NAMES | _MATERIALS_WORKER_TOOL_ALLOWLIST),
    "ml_worker": ("machine_learning", _DEEPAGENT_BUILTIN_TOOL_NAMES | _ML_WORKER_TOOL_ALLOWLIST),
    "report_worker_agent": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _WRITING_WORKER_TOOL_ALLOWLIST),
    "writing_specialist": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _WRITING_TOOL_ALLOWLIST),
    "writing_worker_agent": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _WRITING_WORKER_TOOL_ALLOWLIST),
    "writing_polisher_agent": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _WRITING_WORKER_TOOL_ALLOWLIST),
    "peer_review_specialist": ("writing", _DEEPAGENT_BUILTIN_TOOL_NAMES | _PEER_REVIEW_TOOL_ALLOWLIST),
}


class _InternetSearchInput(BaseModel):
    query: str = Field(..., description="Focused public-web query for background facts or literature orientation.")
    max_results: int = Field(5, ge=1, le=10, description="Maximum number of search results to return.")
    topic: Literal["general", "news", "finance"] = Field(
        "general",
        description="Tavily search topic. Use `general` for scientific background lookup.",
    )
    include_raw_content: bool = Field(
        False,
        description="Whether Tavily should include raw page content excerpts in the response.",
    )


def _compact_search_text(value: Any, *, max_chars: int) -> str:
    text = re.sub(r"\s+", " ", str(value or "")).strip()
    if len(text) <= max_chars:
        return text
    return text[: max(0, max_chars - 1)].rstrip() + "…"


def _format_lightweight_internet_search_content(
    data: dict[str, Any],
    *,
    max_results: int = 5,
) -> str:
    status = str(data.get("status") or "").strip().lower()
    if status == "error":
        query = _compact_search_text(data.get("query") or "", max_chars=160)
        message = _compact_search_text(data.get("message") or "unknown error", max_chars=280)
        source = _compact_search_text(data.get("source") or "search backend", max_chars=40)
        return f"internet_search failed for query={query!r} via {source}: {message}"

    query = _compact_search_text(data.get("query") or "", max_chars=200)
    topic = _compact_search_text(data.get("topic") or "general", max_chars=20)
    answer = _compact_search_text(data.get("answer") or "", max_chars=360)
    follow_up_raw = data.get("follow_up_questions") or []
    follow_up = [
        _compact_search_text(item, max_chars=140)
        for item in follow_up_raw
        if str(item or "").strip()
    ][:3]
    results_raw = data.get("results") or []
    result_lines: list[str] = []
    raw_content_present = False
    for idx, item in enumerate(results_raw[: max(1, int(max_results or 1))], start=1):
        if not isinstance(item, dict):
            continue
        title = _compact_search_text(item.get("title") or "Untitled result", max_chars=120)
        url = _compact_search_text(item.get("url") or "", max_chars=220)
        snippet = _compact_search_text(item.get("content") or "", max_chars=220)
        if not snippet:
            snippet = "(no summary provided)"
        if str(item.get("raw_content") or "").strip():
            raw_content_present = True
        result_lines.append(f"- [{idx}] {title} | {url} | {snippet}")

    lines = [
        f"Query: {query or '(none)'}",
        f"Topic: {topic}",
        f"Results returned: {len(results_raw)}",
    ]
    if answer:
        lines.append(f"Answer summary: {answer}")
    if follow_up:
        lines.append("Follow-up questions:")
        lines.extend(f"- {item}" for item in follow_up)
    if result_lines:
        lines.append("Top results:")
        lines.extend(result_lines)
    else:
        lines.append("Top results: (none)")
    if raw_content_present:
        lines.append("Note: raw page content was returned by Tavily but omitted from tool content.")
    return "\n".join(lines)


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

    def _raise_if_interrupt_requested(self, *, phase: str, details: dict[str, Any] | None = None) -> None:
        control = self.run_control
        if control is None or not control.is_interrupt_requested():
            return
        control.ack_interrupt(phase=phase, details=dict(details or {}))
        raise asyncio.CancelledError(f"Interrupted during {phase}")

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
        if entrypoint not in {"research", "experiment", "writing", "peer_review"}:
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
            self._raise_if_interrupt_requested(phase="run_start", details={"entrypoint": entrypoint})
            proposal_review_enabled = bool(payload.get("proposal_review", False))
            proposal_revision_count = max(0, int(payload.get("proposal_revision_count") or 0))

            if proposal_review_enabled:
                checkpoint: ProposalCheckpoint | None = None
                if resume_feedback is None:
                    checkpoint = await self._build_proposal_checkpoint(
                        entrypoint=entrypoint,
                        prompt=prompt,
                        usage_handler=usage_handler,
                    )
                else:
                    feedback_text = str(resume_feedback or "").strip()
                    if not self._is_proposal_approval(feedback_text):
                        proposal_revision_count += 1
                        checkpoint = await self._build_proposal_checkpoint(
                            entrypoint=entrypoint,
                            prompt=prompt,
                            usage_handler=usage_handler,
                            current_proposal=self._read_current_proposal_text(),
                            review_feedback=feedback_text,
                            revision_index=proposal_revision_count,
                        )
                    else:
                        resume_feedback = None

                while checkpoint is not None:
                    self._persist_proposal_review_state(
                        entrypoint=entrypoint,
                        prompt=prompt,
                        checkpoint=checkpoint,
                        thread_id=thread_id,
                        chat_session_id=str(payload.get("chat_session_id") or ""),
                        files_root=files_root,
                        research_kernel_relpath=research_kernel_relpath,
                        revision_count=proposal_revision_count,
                    )
                    self._emit(
                        "RUN_WAITING_INPUT",
                        payload={
                            "interrupt_type": "proposal_review",
                            "message": (
                                "Proposal review is required before execution continues. "
                                "Type `approve` to continue; any other input requests a revised proposal."
                            ),
                            "approval_token": self._proposal_approval_token(),
                            "revision_count": proposal_revision_count,
                        },
                    )
                    if not self.reporter.is_live():
                        self._write_usage_summary(usage_handler)
                        return {
                            "run_id": self.run_context.run_id,
                            "run_dir": str(self.run_context.run_dir),
                            "status": "awaiting_human_feedback",
                            "summary": (
                                "Proposal review is waiting for explicit `approve`. "
                                "Any other feedback will request a revised proposal."
                            ),
                            "facts": [],
                            "final_answer": "",
                            "artifacts": [],
                            "delegation_log": [],
                        }

                    feedback = self.reporter.prompt_proposal_feedback(
                        todo=list(checkpoint.todo_items),
                        proposal_description=checkpoint.proposal_md,
                    )
                    self._emit(
                        "RUN_INPUT_RECEIVED",
                        payload={"interrupt_type": "proposal_review", "feedback_len": len(str(feedback or ""))},
                    )
                    feedback_text = str(feedback or "").strip()
                    if self._is_proposal_approval(feedback_text):
                        checkpoint = None
                        resume_feedback = None
                        break
                    proposal_revision_count += 1
                    checkpoint = await self._build_proposal_checkpoint(
                        entrypoint=entrypoint,
                        prompt=prompt,
                        usage_handler=usage_handler,
                        current_proposal=self._read_current_proposal_text(),
                        review_feedback=feedback_text,
                        revision_index=proposal_revision_count,
                    )

                self._write_run_state(
                    {
                        **(self._read_run_state() or payload),
                        "schema_version": 1,
                        "entrypoint": entrypoint,
                        "status": "running",
                        "phase": "executing",
                        "active_specialist": entrypoint,
                        "thread_id": thread_id,
                        "proposal_review": True,
                        "proposal_revision_count": proposal_revision_count,
                        "pending_human_input": None,
                        "todo_items": [],
                        "artifacts": [],
                        "delegation_log": [],
                        "text_preview": prompt[:280],
                        "user_prompt": prompt,
                        "chat_session_id": str(payload.get("chat_session_id") or ""),
                        **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
                    }
                )

            elif resume_feedback is not None:
                self._write_run_state(
                    {
                        **payload,
                        "status": "running",
                        "phase": "executing",
                        "pending_human_input": None,
                        "text_preview": str(resume_feedback or "")[:280],
                    }
                )

            while True:
                self._raise_if_interrupt_requested(phase="before_agent_invoke", details={"entrypoint": entrypoint})
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
                            "configurable": {
                                "thread_id": thread_id,
                                "project_id": str(self.run_context.project_id or "").strip(),
                            },
                            "callbacks": self._langchain_callbacks(
                                usage_handler=usage_handler,
                                default_agent_name=f"{entrypoint}_specialist",
                            ),
                            "metadata": {"lc_agent_name": f"{entrypoint}_specialist"},
                        },
                    )
                parsed = self._finalize_report(self._coerce_report(raw=result))
                artifacts = self._artifact_rows(parsed["files"])

                final_answer = parsed["text"]
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
                        "proposal_revision_count": proposal_revision_count,
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
                        "review_target": str(parsed.get("review_target") or "").strip(),
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
        except asyncio.CancelledError:
            if self.run_control is None or not self.run_control.is_interrupt_requested():
                raise
            self.run_control.ack_interrupt(phase="specialist_runtime", details={"entrypoint": entrypoint})
            interrupted_state = {
                "schema_version": 1,
                "entrypoint": entrypoint,
                "status": "interrupted_paused",
                "phase": "interrupted",
                "active_specialist": entrypoint,
                "thread_id": thread_id,
                "proposal_review": bool(payload.get("proposal_review", False)),
                "proposal_revision_count": max(0, int(payload.get("proposal_revision_count") or 0)),
                "pending_human_input": None,
                "todo_items": [],
                "artifacts": list(payload.get("artifacts") or []),
                "delegation_log": list(payload.get("delegation_log") or []),
                "text_preview": "Run interrupted by user.",
                "user_prompt": prompt,
                "chat_session_id": str(payload.get("chat_session_id") or ""),
                "final_answer": "",
                "summary": "Run interrupted by user.",
                "facts": [],
                **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
            }
            self._write_run_state(interrupted_state)
            self._emit(
                "RUN_PAUSED",
                payload={"entrypoint": entrypoint, "status": "interrupted_paused", "phase": "specialist_runtime"},
            )
            _flush_usage()
            return self._final_response_from_state(interrupted_state)
        finally:
            _flush_usage()

    async def _build_proposal_checkpoint(
        self,
        *,
        entrypoint: SpecialistEntrypoint,
        prompt: str,
        usage_handler: SpecialistUsageCallbackHandler,
        current_proposal: str = "",
        review_feedback: str = "",
        revision_index: int = 0,
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
                        "content": self._proposal_request_text(
                            prompt=prompt,
                            current_proposal=current_proposal,
                            review_feedback=review_feedback,
                            revision_index=revision_index,
                        ),
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

    @classmethod
    def _proposal_approval_token(cls) -> str:
        return "approve"

    @classmethod
    def _is_proposal_approval(cls, text: str) -> bool:
        return str(text or "").strip().lower() == cls._proposal_approval_token()

    @staticmethod
    def _proposal_request_text(
        *,
        prompt: str,
        current_proposal: str = "",
        review_feedback: str = "",
        revision_index: int = 0,
    ) -> str:
        base_prompt = str(prompt or "").strip()
        proposal_text = str(current_proposal or "").strip()
        feedback_text = str(review_feedback or "").strip()
        if not proposal_text or not feedback_text:
            return base_prompt
        return (
            f"Original user request:\n{base_prompt}\n\n"
            f"Current proposal:\n{proposal_text}\n\n"
            f"Human review feedback:\n{feedback_text}\n\n"
            f"Revise the proposal from scratch to address the feedback. "
            f"This is revision {max(1, int(revision_index or 1))}. "
            "Do not start execution. Return the full revised ProposalCheckpoint only."
        )

    def _read_current_proposal_text(self) -> str:
        proposal_path = self.run_context.run_dir / PROPOSAL_FILE
        if not proposal_path.exists():
            return ""
        try:
            return proposal_path.read_text(encoding="utf-8").strip()
        except Exception:
            return ""

    def _persist_proposal_review_state(
        self,
        *,
        entrypoint: str,
        prompt: str,
        checkpoint: ProposalCheckpoint,
        thread_id: str,
        chat_session_id: str,
        files_root: Path,
        research_kernel_relpath: str,
        revision_count: int,
    ) -> None:
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
                "proposal_revision_count": max(0, int(revision_count or 0)),
                "pending_human_input": {
                    "kind": "proposal_review",
                    "questions_for_human": list(checkpoint.questions_for_human),
                    "todo_items": list(checkpoint.todo_items),
                    "revision_count": max(0, int(revision_count or 0)),
                    "approval_token": self._proposal_approval_token(),
                },
                "todo_items": list(checkpoint.todo_items),
                "artifacts": [],
                "delegation_log": [],
                "text_preview": checkpoint.proposal_md[:280],
                "user_prompt": prompt,
                "chat_session_id": chat_session_id,
                **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
            }
        )

    def _final_response_from_state(self, payload: dict[str, Any]) -> dict[str, Any]:
        artifacts = list(payload.get("artifacts") or [])
        return {
            "run_id": self.run_context.run_id,
            "run_dir": str(self.run_context.run_dir),
            "status": str(payload.get("status") or "done"),
            "summary": str(payload.get("summary") or "").strip(),
            "facts": [str(item).strip() for item in list(payload.get("facts") or []) if str(item).strip()],
            "final_answer": str(payload.get("final_answer") or "").strip(),
            "artifacts": artifacts,
            "delegation_log": list(payload.get("delegation_log") or []),
        }

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
        agent_runtime = getattr(self.llm_profile, "agent_runtime", None)
        model_call_run_limit = max(1, int(getattr(agent_runtime, "max_model_calls", 120)))
        # TODO: Revisit explicit summarization tuning for OpenRouter-backed specialists
        # via an official config path instead of patching model.profile at runtime.
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint])),
            "tools": tools,
            "system_prompt": self._system_prompt(entrypoint, thread_id=thread_id),
            "middleware": self._build_default_middleware(model_call_run_limit=model_call_run_limit),
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
                system_prompt=self._system_prompt("experiment", allow_manage_memory=False),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._specialist_subagent_tools("experiment"),
                skills=[self._skill_view_virtual_path("experiment_specialist")],
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="writing_specialist",
                description="Turn existing evidence into reports, outlines, sections, or manuscript-ready outputs.",
                system_prompt=self._system_prompt("writing", allow_manage_memory=False),
                model=build_chat_model(self.llm_profile.config_for_role("write_director")),
                tools=self._specialist_subagent_tools("writing"),
                skills=[self._skill_view_virtual_path("writing_specialist")],
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="peer_review_specialist",
                description="Act like a journal editor: inspect the manuscript PDF, request reviewer-style reports, and return an editor decision with raw reviewer comments.",
                system_prompt=self._system_prompt("peer_review", allow_manage_memory=False),
                model=build_chat_model(self.llm_profile.config_for_role("write_reviewer")),
                tools=self._specialist_subagent_tools("peer_review"),
                skills=[self._skill_view_virtual_path("peer_review_specialist")],
                middleware=subagent_middleware,
            ),
            self._compiled_litreview_subagent(runtime=runtime),
        ]

    def _experiment_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        subagent_middleware = self._subagent_middleware(runtime=runtime)
        materials_worker_middleware = self._subagent_middleware(
            runtime=runtime,
            enable_tool_selector=True,
            selector_max_tools=_MATERIALS_WORKER_SELECTOR_MAX_TOOLS,
            selector_always_include=_MATERIALS_WORKER_SELECTOR_ALWAYS_INCLUDE,
        )
        return [
            SubAgent(
                name="materials_worker",
                description="Handle bounded, context-heavy materials execution subtasks in isolation and return concise results with artifact paths.",
                system_prompt=self._materials_worker_prompt(
                    execution_contract=self._execution_capability_contract(audience="materials_worker")
                ),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_MATERIALS_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skill_view_virtual_path("materials_worker")],
                middleware=materials_worker_middleware,
            ),
            SubAgent(
                name="ml_worker",
                description="Handle bounded machine-learning subtasks in isolation using the default DeepAgent tool surface until ML-specific tools are added.",
                system_prompt=self._ml_worker_prompt(
                    execution_contract=self._execution_capability_contract(audience="ml_worker")
                ),
                model=build_chat_model(self.llm_profile.config_for_role("task_runner")),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_ML_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skill_view_virtual_path("ml_worker")],
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="literature_agent",
                description="Run lightweight Tavily-backed public-web search for quick background grounding and literature orientation.",
                system_prompt=self._lightweight_literature_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("literature_synthesizer")),
                tools=self._lightweight_literature_tools(),
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="report_worker_agent",
                description="Write bounded experiment-facing reports, validation summaries, or QC notes from existing workspace evidence.",
                system_prompt=self._report_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("section_writer")),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skill_view_virtual_path("report_worker_agent")],
                middleware=subagent_middleware,
            ),
        ]

    def _writing_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        SubAgent = self._load_subagent()
        subagent_middleware = self._subagent_middleware(runtime=runtime)
        return [
            SubAgent(
                name="literature_agent",
                description="Provide tightly bounded background/context lookups for writing tasks when introduction or discussion needs focused external grounding.",
                system_prompt=self._writing_literature_agent_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("literature_synthesizer")),
                tools=self._lightweight_literature_tools(),
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="writing_worker_agent",
                description="Draft or revise context-heavy sections in isolation and return compact manuscript-ready outputs.",
                system_prompt=self._writing_worker_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("section_writer")),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skill_view_virtual_path("writing_worker_agent")],
                middleware=subagent_middleware,
            ),
            SubAgent(
                name="writing_polisher_agent",
                description="Apply conservative section-level prose polish without changing the manuscript's scientific stance or structure.",
                system_prompt=self._writing_polisher_prompt(),
                model=build_chat_model(self.llm_profile.config_for_role("academic_polisher")),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skill_view_virtual_path("writing_polisher_agent")],
                middleware=subagent_middleware,
            ),
        ]

    def _subagent_middleware(
        self,
        *,
        runtime: dict[str, Any],
        include_memory_middleware: bool = True,
        enable_tool_selector: bool = False,
        selector_max_tools: int | None = None,
        selector_always_include: tuple[str, ...] = (),
    ) -> list[Any]:
        agent_runtime = getattr(self.llm_profile, "agent_runtime", None)
        model_call_run_limit = max(1, int(getattr(agent_runtime, "max_model_calls", 120)))
        middleware = [
            *self._build_default_middleware(model_call_run_limit=model_call_run_limit),
        ]
        if include_memory_middleware:
            middleware.append(self._new_memory_middleware(backend=runtime["backend"]))
        if enable_tool_selector:
            middleware.append(
                self._new_tool_selector_middleware(
                    max_tools=selector_max_tools,
                    always_include=selector_always_include,
                )
            )
        return middleware

    def _metadata_middleware(self, *, runtime: dict[str, Any]) -> list[Any]:
        middleware = self._subagent_middleware(runtime=runtime)
        try:
            create_summarization_tool_middleware = self._load_create_summarization_tool_middleware()
            compact_tool_middleware = create_summarization_tool_middleware(
                build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
                runtime["backend"],
            )
        except Exception as exc:
            logger.warning("litreview compaction middleware unavailable; continuing without extra compaction: %s", exc)
            return middleware
        middleware.append(compact_tool_middleware)
        return middleware

    def _compiled_litreview_subagent(self, *, runtime: dict[str, Any]) -> Any:
        CompiledSubAgent = self._load_compiled_subagent()
        return CompiledSubAgent(
            name="litreview_agent",
            description="Orchestrate literature review by delegating broad public-web review to `literature_agent` and exact DOI/venue/author resolution to `metadata_agent`.",
            runnable=self._build_litreview_agent(runtime=runtime),
        )

    def _build_litreview_agent(self, *, runtime: dict[str, Any]) -> Any:
        create_deep_agent = self._load_create_deep_agent()
        SubAgent = self._load_subagent()
        return create_deep_agent(
            model=build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
            tools=self._augment_with_project_memory_tools([], include_manage_memory=False),
            system_prompt=self._litreview_wrapper_prompt(),
            middleware=self._subagent_middleware(runtime=runtime, include_memory_middleware=False),
            checkpointer=runtime["checkpointer"],
            store=runtime["store"],
            backend=runtime["backend"],
            name="litreview_agent",
            memory=self._memory_sources(),
            subagents=[
                SubAgent(
                    name="literature_agent",
                    description="Use Tavily-backed public web search and page reading for broader literature review, background grounding, and public-source synthesis.",
                    system_prompt=self._litreview_agent_prompt(),
                    model=build_chat_model(self.llm_profile.config_for_role("literature_synthesizer")),
                    tools=self._augment_with_project_memory_tools(
                        self._named_tools(_LITREVIEW_AGENT_TOOL_ALLOWLIST),
                        include_manage_memory=False,
                    ),
                    middleware=self._subagent_middleware(runtime=runtime),
                ),
                SubAgent(
                    name="metadata_agent",
                    description="Resolve exact paper metadata, DOI/year/venue/authors, and citation details from scholarly databases.",
                    system_prompt=self._metadata_agent_prompt(),
                    model=build_chat_model(self.llm_profile.config_for_role("literature_deep_research")),
                    tools=self._augment_with_project_memory_tools(
                        self._named_tools(_METADATA_AGENT_TOOL_ALLOWLIST),
                        include_manage_memory=False,
                    ),
                    middleware=self._metadata_middleware(runtime=runtime),
                ),
            ],
        )

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
        elif entrypoint == "peer_review":
            requested = _PEER_REVIEW_TOOL_ALLOWLIST
        elif entrypoint == "research":
            requested = _RESEARCH_TOOL_ALLOWLIST
        else:
            requested = _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
        return self._augment_with_project_memory_tools(self._named_tools(requested))

    def _specialist_subagent_tools(self, entrypoint: SpecialistEntrypoint) -> list[Any]:
        if entrypoint == "writing":
            requested = _WRITING_TOOL_ALLOWLIST
        elif entrypoint == "peer_review":
            requested = _PEER_REVIEW_TOOL_ALLOWLIST
        elif entrypoint == "research":
            requested = _RESEARCH_TOOL_ALLOWLIST
        else:
            requested = _EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST
        return self._augment_with_project_memory_tools(
            self._named_tools(requested),
            include_manage_memory=False,
        )

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

    def _augment_with_project_memory_tools(
        self,
        tools: list[Any],
        *,
        include_manage_memory: bool = True,
    ) -> list[Any]:
        existing = {str(getattr(tool, "name", "") or "").strip() for tool in tools}
        augmented = list(tools)
        for tool in self._project_memory_tools(include_manage_memory=include_manage_memory):
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in existing:
                augmented.append(tool)
                existing.add(name)
        return augmented

    def _project_memory_tools(self, *, include_manage_memory: bool = True) -> list[Any]:
        create_search_memory_tool = self._load_create_search_memory_tool()
        search_tool = create_search_memory_tool(
            namespace=_PROJECT_LONG_TERM_MEMORY_NAMESPACE,
            instructions=_PROJECT_SEARCH_MEMORY_INSTRUCTIONS,
            response_format="content",
            name="search_memory",
        )
        tools = [self._wrap_project_memory_tool(search_tool)]
        if include_manage_memory:
            create_manage_memory_tool = self._load_create_manage_memory_tool()
            manage_tool = create_manage_memory_tool(
                namespace=_PROJECT_LONG_TERM_MEMORY_NAMESPACE,
                instructions=_PROJECT_MANAGE_MEMORY_INSTRUCTIONS,
                actions_permitted=("create", "update", "delete"),
                name="manage_memory",
            )
            tools.append(self._wrap_project_memory_tool(manage_tool))
        return tools

    def _lightweight_literature_tools(self) -> list[Any]:
        def internet_search(
            query: str,
            max_results: int = 5,
            topic: Literal["general", "news", "finance"] = "general",
            include_raw_content: bool = False,
        ) -> tuple[str, dict[str, Any]]:
            data: dict[str, Any]
            try:
                import os
                from tavily import TavilyClient

                api_key = str(os.environ.get("TAVILY_API_KEY", "")).strip()
                if not api_key:
                    raise RuntimeError("TAVILY_API_KEY is required for public web search.")
                tavily_client = TavilyClient(api_key=api_key)
                response = tavily_client.search(
                    query,
                    max_results=max_results,
                    include_raw_content=include_raw_content,
                    topic=topic,
                )
                payload = response if isinstance(response, dict) else {"result": response}
                if isinstance(payload, dict):
                    payload.setdefault("query", query)
                    payload.setdefault("topic", topic)
                data = payload
            except Exception as exc:
                data = {
                    "status": "error",
                    "source": "tavily",
                    "query": query,
                    "topic": topic,
                    "include_raw_content": bool(include_raw_content),
                    "message": str(exc),
                }
            return _format_lightweight_internet_search_content(
                data,
                max_results=max_results,
            ), {
                "tool_name": "internet_search",
                "data": data,
            }

        tools = [
            StructuredTool.from_function(
                func=internet_search,
                name="internet_search",
                description="Search the public web for targeted scientific background facts and literature orientation.",
                args_schema=_InternetSearchInput,
                infer_schema=False,
                response_format="content_and_artifact",
            )
        ]
        return self._augment_with_project_memory_tools(tools, include_manage_memory=False)

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
            self._raise_if_interrupt_requested(phase="before_tool_call", details={"tool": tool.name})
            try:
                result = func(runtime=runtime, **kwargs)
                self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                return result
            except Exception as exc:
                return self._nonfatal_tool_error_result(tool.name, exc, kwargs)

        async def _awrapped(runtime=None, **kwargs: Any) -> tuple[Any, dict[str, Any]]:
            self._raise_if_interrupt_requested(phase="before_tool_call", details={"tool": tool.name})
            if coroutine is not None:
                try:
                    result = await coroutine(runtime=runtime, **kwargs)
                    self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                    return result
                except Exception as exc:
                    return self._nonfatal_tool_error_result(tool.name, exc, kwargs)
            if func is None:
                raise NotImplementedError(f"Tool {tool.name} does not support async invocation.")
            try:
                result = func(runtime=runtime, **kwargs)
                self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                return result
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

    def _wrap_project_memory_tool(self, tool: Any) -> Any:
        if not isinstance(tool, StructuredTool):
            return tool
        args_schema = getattr(tool, "args_schema", None)
        if not isinstance(args_schema, type) or not issubclass(args_schema, BaseModel):
            return tool
        func = getattr(tool, "func", None)
        coroutine = getattr(tool, "coroutine", None)
        if func is None and coroutine is None:
            return tool

        def _artifact(payload: Any, tool_args: dict[str, Any]) -> tuple[str, dict[str, Any]]:
            text = content_to_text(payload)
            return text, {
                "tool_name": tool.name,
                "data": {
                    "status": "ok",
                    "tool_name": tool.name,
                    "message": text,
                    "tool_args": dict(tool_args or {}),
                },
            }

        def _wrapped(runtime=None, **kwargs: Any) -> tuple[str, dict[str, Any]]:
            _ = runtime
            if func is None:
                raise NotImplementedError(f"Tool {tool.name} does not support sync invocation.")
            self._raise_if_interrupt_requested(phase="before_tool_call", details={"tool": tool.name})
            try:
                result = _artifact(func(**kwargs), kwargs)
                self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                return result
            except Exception as exc:
                return self._nonfatal_tool_error_result(tool.name, exc, kwargs)

        async def _awrapped(runtime=None, **kwargs: Any) -> tuple[str, dict[str, Any]]:
            _ = runtime
            self._raise_if_interrupt_requested(phase="before_tool_call", details={"tool": tool.name})
            if coroutine is not None:
                try:
                    result = _artifact(await coroutine(**kwargs), kwargs)
                    self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                    return result
                except Exception as exc:
                    return self._nonfatal_tool_error_result(tool.name, exc, kwargs)
            if func is None:
                raise NotImplementedError(f"Tool {tool.name} does not support async invocation.")
            try:
                result = _artifact(func(**kwargs), kwargs)
                self._raise_if_interrupt_requested(phase="after_tool_call", details={"tool": tool.name})
                return result
            except Exception as exc:
                return self._nonfatal_tool_error_result(tool.name, exc, kwargs)

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
            base / "machine_learning": repo_root / "skills" / "machine_learning",
            base / "writing": repo_root / "skills" / "writing",
        }
        for target, source in layouts.items():
            if not source.exists():
                continue
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copytree(source, target, dirs_exist_ok=True)
        self._stage_filtered_skill_views(
            deepagents_root=deepagents_root,
            repo_root=repo_root,
        )
        staged_agents = deepagents_root / "AGENTS.md"
        if not staged_agents.exists():
            workspace_agents = Path(self.run_context.workspace) / "AGENTS.md"
            if workspace_agents.exists():
                staged_agents.parent.mkdir(parents=True, exist_ok=True)
                shutil.copyfile(workspace_agents, staged_agents)

    @staticmethod
    def _skill_view_virtual_path(view_name: str) -> str:
        return f"{_SKILL_VIEW_ROOT}/{str(view_name or '').strip()}"

    @staticmethod
    def _read_skill_allowed_tools(skill_md: Path) -> list[str]:
        aliases = {
            "bash_exec": "bash",
            "apply_aider_edits": "edit_file",
            "read_research_pack": "read_file",
            "review_research_context": "read_file",
            "run_literature_research": "execute",
        }
        try:
            text = skill_md.read_text(encoding="utf-8")
        except Exception:
            return []
        if not text.startswith("---\n"):
            return []
        end_idx = text.find("\n---\n", 4)
        if end_idx < 0:
            return []
        try:
            frontmatter = yaml.safe_load(text[4:end_idx]) or {}
        except Exception:
            return []
        if not isinstance(frontmatter, dict):
            return []
        raw = frontmatter.get("allowed-tools")
        if isinstance(raw, str):
            tokens = [item.strip().strip(",") for item in raw.split()]
        elif isinstance(raw, (list, tuple)):
            tokens = [str(item).strip().strip(",") for item in raw]
        else:
            tokens = []
        out: list[str] = []
        seen: set[str] = set()
        for token in tokens:
            if not token:
                continue
            normalized = aliases.get(token, token)
            if normalized in seen:
                continue
            seen.add(normalized)
            out.append(normalized)
        return out

    def _stage_filtered_skill_views(self, *, deepagents_root: Path, repo_root: Path) -> None:
        view_root = deepagents_root / "skill_views"
        for view_name, (source_group, available_tools) in _SPECIALIST_SKILL_VIEW_SPECS.items():
            source_root = repo_root / "skills" / source_group
            if not source_root.is_dir():
                continue
            target_root = view_root / view_name
            if target_root.exists():
                shutil.rmtree(target_root)
            target_root.mkdir(parents=True, exist_ok=True)
            for skill_dir in sorted(source_root.iterdir(), key=lambda p: p.name):
                if not skill_dir.is_dir():
                    continue
                skill_md = skill_dir / "SKILL.md"
                if not skill_md.is_file():
                    continue
                declared_tools = self._read_skill_allowed_tools(skill_md)
                if declared_tools and any(tool_name not in available_tools for tool_name in declared_tools):
                    continue
                shutil.copytree(skill_dir, target_root / skill_dir.name, dirs_exist_ok=True)

    @staticmethod
    def _virtual_skill_paths(entrypoint: SpecialistEntrypoint) -> list[str]:
        if entrypoint == "experiment":
            return [SpecialistRunner._skill_view_virtual_path("experiment_specialist")]
        if entrypoint == "writing":
            return [SpecialistRunner._skill_view_virtual_path("writing_specialist")]
        if entrypoint == "peer_review":
            return [SpecialistRunner._skill_view_virtual_path("peer_review_specialist")]
        return [
            SpecialistRunner._skill_view_virtual_path("research_experiment"),
            SpecialistRunner._skill_view_virtual_path("research_writing"),
        ]

    def _resolve_thread_id(self, payload: dict[str, Any]) -> str:
        thread_id = str(payload.get("thread_id") or "").strip()
        if thread_id:
            return thread_id
        chat_session_id = str(payload.get("chat_session_id") or "").strip()
        if chat_session_id:
            return chat_session_id
        return self.run_context.run_id

    def _system_prompt(
        self,
        entrypoint: SpecialistEntrypoint,
        *,
        thread_id: str = "",
        allow_manage_memory: bool = True,
    ) -> str:
        execution_contract = ""
        if entrypoint in {"research", "experiment"}:
            execution_contract = self._execution_capability_contract(audience=entrypoint)
        return self._base_system_prompt(
            entrypoint,
            thread_id=thread_id,
            allow_manage_memory=allow_manage_memory,
            execution_contract=execution_contract,
        )

    def _execution_capability_contract(
        self,
        *,
        audience: Literal["research", "experiment", "materials_worker", "ml_worker"],
    ) -> str:
        available = set(self.registry.tools.keys())
        managed_tools = [
            name
            for name in ("vasp_execute_batch", "mace_neb_batch", "mace_relax_batch", "mace_sp_batch", "mace_train", "mace_evaluate")
            if name in available
        ]
        prepare_tools = [
            name
            for name in ("vasp_prepare", "vasp_band_prepare", "vasp_neb_prepare", "build_dataset_from_runs")
            if name in available
        ]
        machine_names: list[str] = []
        resource_names: list[str] = []
        task_names: list[str] = []
        try:
            machine_register = MachineRegister()
            machine_names = sorted(machine_register.list_machines().keys())
            resource_names = sorted(machine_register.list_resources().keys())
        except Exception:
            machine_names = []
            resource_names = []
        try:
            task_registry = TaskRegistry()
            task_names = sorted(task_registry.list_tasks().keys())
        except Exception:
            task_names = []

        lines = [
            "Execution capability contract: distinguish local interactive availability from managed execution availability.",
            "Do not infer managed-execution availability from local shell probing alone. If a registered submission tool exists, treat that capability as available through the platform unless the tool itself fails at runtime.",
        ]
        if audience == "research":
            lines.append(
                "When a bounded experiment stage needs periodic DFT or other managed compute, route it to ExperimentSpecialist instead of downgrading it to literature-only validation just because no local executable is visible in the shell."
            )
        else:
            lines.append(
                "Treat local shell checks as relevant only for local interactive steps; use the registered remote-execution tools for managed runs instead of concluding the capability is unavailable."
            )
        if prepare_tools:
            lines.append(f"Local preparation/analysis tools currently available here: {', '.join(f'`{name}`' for name in prepare_tools)}.")
        if managed_tools:
            lines.append(f"Managed execution tools currently available here: {', '.join(f'`{name}`' for name in managed_tools)}.")
        if machine_names or resource_names or task_names:
            facts: list[str] = []
            if machine_names:
                facts.append(f"machines={', '.join(f'`{name}`' for name in machine_names)}")
            if resource_names:
                facts.append(f"resources={', '.join(f'`{name}`' for name in resource_names)}")
            if task_names:
                facts.append(f"tasks={', '.join(f'`{name}`' for name in task_names)}")
            lines.append("DPDispatcher runtime config visible in this workspace: " + "; ".join(facts) + ".")
        if audience in {"experiment", "materials_worker"} and "vasp_execute_batch" in available:
            lines.append(
                "For periodic DFT, the intended path is to prepare inputs locally, submit via `vasp_execute_batch`, then analyze the returned results; do not require a local periodic DFT engine to be directly runnable first."
            )
        if audience in {"experiment", "materials_worker"} and any(name in available for name in ("mace_neb_batch", "mace_relax_batch", "mace_sp_batch")):
            lines.append(
                "For surrogate screening or MACE-based materials workflows, prefer the registered batch execution tools over ad hoc shell probing of remote resources."
            )
        if audience in {"experiment", "ml_worker"} and any(name in available for name in ("mace_train", "mace_evaluate")):
            lines.append(
                "For heavy ML training or evaluation, use the registered managed-execution path when the run is long, batch-oriented, or compute-intensive."
            )
        if audience in {"experiment", "ml_worker"}:
            lines.append(
                "ML work is especially heterogeneous: prefer the registered managed tools when they fit the task, but if no managed tool covers the needed code path, continue by writing and running local workspace scripts instead of blocking on tool coverage."
            )
        return "\n".join(lines)

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
            "# Persistent Instruction Memory\n\n"
            "Use this file only for durable user preferences, project conventions, and stable workflow guidance "
            "that should be loaded into future prompts.\n\n"
            "- Do not store project-state facts, experiment conclusions, or evolving status here; use long-term memory tools for those.\n"
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

    def _new_tool_selector_middleware(
        self,
        *,
        max_tools: int | None,
        always_include: tuple[str, ...] = (),
    ) -> Any:
        LLMToolSelectorMiddleware = self._load_llm_tool_selector_middleware()
        selector_model = build_chat_model(self.llm_profile.config_for_role("tool_selector"))
        # TODO: Revisit skill/tool coupling after the tool surface expansion stabilizes.
        # Official LLMToolSelectorMiddleware does not enforce skill `allowed-tools`;
        # it only sees the current candidate tool list plus the last human message.
        # If we later want skill-aware hard routing, insert a matched-skill-based
        # tool filter before this selector middleware runs.
        return LLMToolSelectorMiddleware(
            model=selector_model,
            max_tools=max_tools,
            always_include=list(always_include),
        )

    @classmethod
    def _base_system_prompt(
        cls,
        entrypoint: SpecialistEntrypoint,
        *,
        thread_id: str = "",
        allow_manage_memory: bool = True,
        execution_contract: str = "",
    ) -> str:
        memory_policy = cls._long_term_memory_policy(allow_manage_memory=allow_manage_memory)
        if entrypoint == "research":
            kernel_path = cls._research_kernel_virtual_path(thread_id)
            return (
                "You are ResearchSpecialist, the only orchestration-capable specialist.\n"
                "You coordinate scientific campaigns, decide when bounded experiment work is justified, "
                "and decide when writing/report generation should start.\n"
                "You may delegate only to `experiment_specialist`, `writing_specialist`, `peer_review_specialist`, and `litreview_agent`.\n"
                "Use `litreview_agent` for all literature-review work. It can internally delegate to `literature_agent` for Tavily-backed public-web review and to `metadata_agent` for exact DOI/year/venue/authors/citation metadata.\n"
                "If the user requests a paper, manuscript, journal-style LaTeX draft, cover letter, rebuttal-style response, or other author-facing publication artifact, delegate that work to `writing_specialist` rather than drafting it directly in the research thread.\n"
                "If the user requests an experiment report, validation summary, QC note, execution-facing memo, or other report-style artifact grounded in completed workspace evidence, delegate that work to `experiment_specialist` and have it use `report_worker_agent` when the writing task is substantial.\n"
                "Default to not launching `peer_review_specialist`.\n"
                "Launch `peer_review_specialist` only when the user explicitly asks for publication-level paper quality, submission-ready or peer-review-ready manuscript quality, formal submission requirements, journal submission standards, or another equivalent formal publication bar.\n"
                "When you do launch it, treat it as an editor-style review process over the manuscript PDF, not as the primary scientific decision-maker.\n"
                "When delegating to `peer_review_specialist`, explicitly hand it the canonical workspace-relative manuscript PDF path; if one PDF is the review target, state that path clearly in the handoff instead of making the reviewer infer it.\n"
                "When `peer_review_specialist` returns, treat its returned markdown or saved review memo as the authoritative revision brief. Do not rely on the Research Kernel to preserve full editor/reviewer comment text.\n"
                "If `peer_review_specialist` gives you a saved review memo path, read that memo directly before deciding the next revision or experiment step.\n"
                "You remain the sole coordinator and final decision-maker for the run.\n"
                "If peer-review or revision comments show that additional experiments are needed, you may relaunch `experiment_specialist` for bounded follow-up work as long as that work still respects the user's stated scope, budget, evidence limits, and time constraints.\n"
                "If peer review indicates the work cannot reach the requested publication bar within the user's stated scope, budget, evidence limits, or time constraints, stop and tell the user that directly instead of looping.\n"
                f"{cls._author_packet_policy()}\n"
                f"{cls._report_packet_policy()}\n"
                f"{cls._multimodal_tool_history_policy()}\n"
                f"{execution_contract}\n"
                "Do not perform large direct execution yourself when delegation is more appropriate.\n"
                f"{cls._research_kernel_contract(kernel_path)}\n"
                f"{memory_policy}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._workspace_path_discipline()}\n"
                f"{cls._soft_reporting_contract()}"
            )
        if entrypoint == "peer_review":
            return (
                "You are PeerReviewSpecialist.\n"
                "Act like a journal editor coordinating external peer review for one manuscript PDF.\n"
                "If the parent or user gives you an explicit `ReviewTarget` or manuscript PDF path, treat that as the canonical review target.\n"
                "Use DeepAgent file tools to locate the manuscript PDF only when that path is missing, ambiguous, or invalid.\n"
                "Once you have identified the canonical manuscript PDF, call `peer_review_request` on that PDF exactly once per review episode.\n"
                "The tool will collect raw reviewer-style reports from the configured peer-review models.\n"
                "Do not run experiments, do not rewrite the manuscript, and do not take over research planning.\n"
                "Your job is to synthesize an editor decision and editor comment from the reviewer reports, then include the raw reviewer comments for ResearchSpecialist or the user.\n"
                "Use decision language such as reject, major revision, minor revision, or conditionally acceptable only when supported by the reviewer comments and the manuscript evidence.\n"
                "Keep the review grounded in ACS-style expectations: scientific soundness, evidence-claim fit, controls, validation quality, novelty positioning, comparison quality, figure logic, and publication readiness.\n"
                "Return the full review markdown directly to the parent; do not compress away the editor comment or reviewer comment sections.\n"
                "Also save the full review as one durable workspace markdown memo under `notes/peer_review/` or another stable path, and include that memo path in `Files`, so the parent can reuse the exact text without depending on kernel summaries.\n"
                f"{memory_policy}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._workspace_path_discipline()}\n"
                "When you finish, return a concise markdown report with sections `Summary`, `Facts`, `Files`, `Editor Decision`, `Editor Comment`, and `Reviewer Comments`.\n"
                "In `Files`, include the reviewed manuscript PDF path.\n"
                "In `Reviewer Comments`, preserve each reviewer's raw comments with clear reviewer labels."
            )
        if entrypoint == "writing":
            return (
                "You are WritingSpecialist.\n"
                "Write from existing workspace evidence only. Do not initiate new computational experiments.\n"
                "Do not reopen broad literature review from the writing thread.\n"
                "This lane owns paper, manuscript, and author-facing scientific writing. It is not the default lane for experiment reports, QC summaries, or execution-facing internal reports.\n"
                "You may use `literature_agent` only for narrow background supplementation when the user explicitly asks to expand background/context, or when a paper/manuscript draft lacks the minimal external background needed for a credible introduction or discussion.\n"
                "Keep such literature work tightly bounded to the current writing need; do not let it expand into a new autonomous research campaign.\n"
                "Your default role is coordination, not long-form drafting in the main thread.\n"
                "For any substantive note writing, section writing, manuscript drafting, or major revision, immediately delegate to `writing_worker_agent` with a bounded brief.\n"
                "Before a substantial paper/manuscript rewrite, first condense the task into one compact inline author packet, then dispatch the section or integration brief from that packet instead of forwarding raw run logs.\n"
                "Each writing-worker handoff should cover only one section or one bounded organization/integration task. "
                "Give it one primary goal and one completion criterion. "
                "If the next step still requires deciding what to write next, how to restructure the manuscript, or whether to change direction, bring that decision back to WritingSpecialist instead of letting the worker continue to expand.\n"
                "For paper/manuscript titles, require a journal-style title centered on the chemical system and principal scientific finding. Avoid workflow-led or meta titles such as 'same-template comparison', 'unified screen', 'evidence hierarchy', or sentence-like conclusion titles unless that framing is scientifically essential.\n"
                "When the requested deliverable is a paper, manuscript, or journal-style draft, treat figures, tables, and concise explanatory schematics as part of the default deliverable when the workspace evidence supports them; do not return text-only manuscript output if key visual evidence is still missing.\n"
                "When the requested deliverable is a paper, manuscript, or journal-style draft, also plan the Supporting Information / Supporting Data package from the existing workspace evidence. Keep claim-critical figures, tables, and arguments in the main text; move extended methods, robustness checks, exhaustive tables, extra figures, structure lists, and machine-readable data exports into supporting content when that organization improves publication readiness.\n"
                "For the current implementation, keep Supporting Information in the same manuscript file rather than a separate SI manuscript: place it after the references as a clear supporting-information section or appendix so compilation and downstream PDF review operate on one manuscript PDF. Supporting data files may still live in separate workspace folders.\n"
                "For LaTeX manuscripts, require figures to be inserted near their first substantive discussion rather than batched at the end. If float drift appears after compilation, require the worker to repair placement by moving the figure block closer to first mention and using conservative float controls such as `[htbp]` or `\\FloatBarrier` when the template permits.\n"
                "Use `writing_polisher_agent` only for local prose cleanup on already drafted sections. It must not change claim strength, scientific scope, section logic, figure order, or evidence selection.\n"
                "When a full paper/manuscript draft has been assembled and a compiled PDF is available, run `review_pdf_manuscript` once on that PDF for comment-only publication-readiness review before final reconciliation.\n"
                "After that review returns, reconcile the manuscript against the accepted suggestions and run one more bounded polishing/revision pass before treating the manuscript as final.\n"
                "If an external-model peer review is requested from the parent, make sure the canonical manuscript PDF is clearly exposed as `ReviewTarget` in your closeout so downstream review uses the right artifact.\n"
                "When the requested deliverable is a short note, compact summary, or quick status writeup, prioritize clarity and sufficiency over making it figure-heavy unless the user explicitly asks for visuals.\n"
                "Keep the main writing thread focused on planning, dispatch, evidence selection, and final reconciliation.\n"
                "Do not handle TeX compile/fix passes in the main thread.\n"
                "If you create or substantially revise a TeX manuscript bundle, require `writing_worker_agent` to run the compile tool itself and repair issues from the returned diagnostics before concluding.\n"
                "Do not leave final cited TeX deliverables with an inline `thebibliography` block. Prefer a separate bibliography file and a `\\bibliography{references}` entry so the bundle includes `.tex`, `.bib`, and `.pdf` outputs when compilation succeeds.\n"
                f"{cls._peer_review_ready_paper_policy()}\n"
                f"{cls._journal_manuscript_policy()}\n"
                f"{cls._author_packet_policy()}\n"
                f"{cls._multimodal_tool_history_policy()}\n"
                f"{memory_policy}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._workspace_path_discipline()}\n"
                f"{cls._writing_reporting_contract()}"
            )
        return (
            "You are ExperimentSpecialist.\n"
            "Perform bounded computational execution in the current workspace using available tools and skills.\n"
            "Route by the current working artifact: use `materials_worker` for structure/calc/result work, including MACE-based surrogate screening inside a materials workflow; use `ml_worker` for dataset/model lifecycle work such as dataset curation, training, benchmark evaluation, and active-learning selection; use `literature_agent` for fast Tavily-backed public-web grounding when a quick external check is needed; use `report_worker_agent` for experiment reports, validation summaries, QC notes, and other execution-facing written artifacts drawn from existing workspace evidence.\n"
            "Each worker should receive only one bounded execution episode around one primary artifact, such as one screening round, one training/evaluation pass, or one post-analysis step. "
            "Each brief should contain one primary goal and one completion criterion. "
            "If direction still needs to be chosen after the step finishes, bring that choice back to ExperimentSpecialist instead of letting the worker continue to expand. "
            "Do not hand an entire high-throughput campaign to one worker; split it into episodes and decide the next episode yourself after each return.\n"
            "If the task is purely report writing from already completed evidence, do not restart calculations just to make the report look more complete. Summarize the executed scope honestly and keep unresolved points explicit.\n"
            "If a bounded workspace task is not covered by a dedicated registered tool, do not stop at that boundary alone; route it to the relevant worker so it can use `execute` plus Python and mature third-party libraries for a focused custom implementation when the environment supports it.\n"
            "When method settings, software behavior, or scientific best practice are uncertain, prefer a quick built-in web check through the online model's native browsing capability to align with current official or primary-source guidance before improvising a custom implementation. Keep that check narrow and implementation-oriented; do not turn it into a broad literature review.\n"
            "When that custom implementation becomes heavy, batch-oriented, high-throughput, or clearly worth rerunning, prefer materializing it as a reusable workspace script under `scripts/` instead of burying the logic inside one long ephemeral shell command.\n"
            f"{execution_contract}\n"
            f"Do not orchestrate other specialists. {memory_policy}\n"
            f"{cls._report_packet_policy()}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @staticmethod
    def _memory_write_policy() -> str:
        return (
            "Instruction memory files (`/.deepagents/AGENTS.md` and `/memories/AGENTS.md`) are for durable user preferences, "
            "project conventions, and stable workflow guidance only. "
            "Do not store project-state facts or run conclusions there. "
            "Never store transient task requests, step-by-step execution history, "
            "intermediate tool output, one-off file paths, temporary status notes, or speculative findings there."
        )

    @staticmethod
    def _author_packet_policy() -> str:
        return (
            "For paper/manuscript handoffs, pass one compact inline author packet rather than raw run history. "
            "Keep it minimal and decision-relevant with exactly these fields: `thesis`, `novelty`, `core_claims` (2-4 bullets), `evidence_refs`, `main_text_keep`, `supporting_only`, and `target_outputs`. "
            "Then issue a bounded writing brief containing the specific section goal, target audience, requested output path(s), desired local section structure, and any citation or style constraints. "
            "For TeX deliverables, require a separate `.tex` file, a separate `.bib` file when citations are used, and at least one direct compile pass that should produce a `.pdf` when the environment supports compilation. "
            "Do not paste long transcripts or ask the writing agent to rediscover evidence already available in the workspace."
        )

    @staticmethod
    def _report_packet_policy() -> str:
        return (
            "For experiment-report handoffs, pass one compact inline report packet with exactly these fields: "
            "`objective`, `executed_scope`, `key_methods`, `key_results`, `failures_or_qc`, and `target_outputs`. "
            "Keep it terse, execution-facing, and grounded in completed workspace evidence. "
            "Do not pad it with paper-style novelty framing or raw transcript excerpts."
        )

    @staticmethod
    def _peer_review_ready_paper_policy() -> str:
        return (
            "Peer-review-ready paper standard: the final manuscript should read like a publishable paper ready to enter peer review from the existing evidence. "
            "State the strongest evidence-supported scientific claim in a direct scientific voice. "
            "Do not reflexively underclaim, and do not weaken the main result with self-cancelling hedge sentences. "
            "Do not ask for new experiments inside the manuscript body; mention limitations only when they materially affect interpretation, scope, or reviewer confidence, and keep those notes brief."
        )

    @staticmethod
    def _multimodal_tool_history_policy() -> str:
        return (
            "Multimodal tool-content discipline: if a tool produces images, PDFs, or other non-text payloads, prefer keeping them as workspace artifacts and refer to them by path plus a short textual summary. "
            "Do not rely on raw inline multimodal tool outputs remaining replay-safe in long-lived thread history across provider bridges. "
            "When later reasoning needs that visual or file content again, re-open the artifact in that turn or use a dedicated analysis tool to turn it into text."
        )

    @staticmethod
    def _journal_manuscript_policy() -> str:
        return (
            "Journal-manuscript discipline: when the deliverable is a paper, manuscript, or journal-style draft, write as an author of the scientific work, not as an agent narrating the workflow. "
            "Do not mention the workspace, files, runs, prompts, tools, agents, interruptions, or that the manuscript was assembled from existing workspace evidence. "
            "Keep process provenance such as 'no new calculations were run in this writing pass', 'accessible snippets', or 'workspace notes' out of the title, abstract, main text, acknowledgements, and supporting-information prose unless the user explicitly asked for an internal note instead of a paper. "
            "State scientific scope limits and evidence limits in field-appropriate prose, not as internal workflow disclaimers. "
            "For journal-facing citations and BibTeX, use publication-style metadata only; do not leave internal snippet notes or workspace-provenance explanations in final reference entries. "
            "If citation metadata is unresolved or only weak snippet evidence exists, prefer a cleaner uncited statement, a visible citation gap to be resolved before submission, or a request for literature cleanup rather than fabricating a journal-facing reference."
        )

    @staticmethod
    def _workspace_path_discipline() -> str:
        return (
            "Workspace path discipline: treat the project files root as your working directory. "
            "Prefer workspace-relative paths. Treat `/` only as the workspace virtual root, not as a host filesystem root. "
            "If you see a host absolute path like `/home/...`, convert it back to a workspace-relative path before using it, "
            "and never recreate host absolute path segments inside the workspace. "
            "For shell or `execute` commands, never use leading-slash workspace paths like `/writing/...`; use workspace-relative paths such as `writing/...` instead. "
            "Do not proactively materialize every intermediate observation into files; default to keeping transient reasoning in the conversation/tool stream. "
            "Only persist key constraints, decisive results, reusable handoff material, or user-requested deliverables. "
            "Prefer a topic-centric layout for user-facing outputs: create one folder per user topic when the work naturally clusters that way. "
            "Within a topic folder, place literature-grounding material under `literature/`, experiment geometry/setup artifacts under `structures/`, "
            "and execution outputs under `calculations/`. "
            "Place reusable generated helper code and heavier deterministic execution logic under `scripts/`, especially for high-throughput screening, batch analysis, or multi-step reproducible pipelines. "
            "If a small saved note or compact report is actually needed, place it under `notes/` (either topic-local or shared when appropriate). "
            "Keep manuscript drafting and writing-focused outputs under a dedicated `writing/` workspace area instead of scattering them across experiment folders. "
            "If the workspace already has a clear established layout, extend that layout consistently instead of creating parallel folder schemes."
        )

    @staticmethod
    def _long_term_memory_policy(*, allow_manage_memory: bool = True) -> str:
        if allow_manage_memory:
            return _PROJECT_MEMORY_TOOL_INSTRUCTIONS
        return _PROJECT_MEMORY_READONLY_INSTRUCTIONS

    @staticmethod
    def _soft_reporting_contract() -> str:
        return (
            "For multi-step work, use `write_todos` early to maintain a concise checklist and update it when the plan changes. "
            "When you finish, reply with a concise markdown report containing only three sections in this order: "
            "`Summary`, `Facts`, and `Files`. "
            "`Summary` must directly answer the user's actual question with the key result, key numbers/conditions when available, and the main conclusion; do not say only that a report was written. "
            "`Facts` should be a flat bullet list of the few most important archival facts. "
            "`Files` should be a flat bullet list of relevant workspace-relative output paths with enough directory context to be unambiguous; do not return bare filenames, and use `(none reported)` if there are none. "
            "If one manuscript PDF is the canonical downstream review target, you may add an optional `ReviewTarget` section with exactly one workspace-relative PDF path. "
            "If you are correcting a previously wrong result after the user pointed out an error, replace or delete stale incorrect reports/notes when feasible and do not leave superseded wrong paths in `Files`."
        )

    @staticmethod
    def _writing_reporting_contract() -> str:
        return (
            "For multi-step work, use `write_todos` early to maintain a concise checklist and update it when the plan changes. "
            "When you finish, reply with a concise markdown report whose required section is `Summary`. "
            "`Summary` must directly answer the user's current writing request by stating what was drafted, revised, or recommended and the current manuscript status. "
            "Include a `Files` section only when you created or materially updated durable workspace artifacts that the parent should inspect. "
            "If one manuscript PDF is the canonical downstream review target, add an optional `ReviewTarget` section with exactly one workspace-relative PDF path. "
            "Do not add a placeholder `Facts` section for writing-only closeout."
        )

    @classmethod
    def _research_kernel_contract(cls, kernel_path: str) -> str:
        return (
            f"Maintain a lightweight Research Kernel in `{kernel_path}` as valid JSON. "
            "It must contain exactly these top-level fields: `question`, `hypotheses`, `run_cards`, `frontier`, `conclusion_draft`. "
            "Keep `hypotheses` to only the currently active 3-5 lines. "
            "Every time a subagent returns, immediately update `run_cards` with one compact card containing only `source`, `summary`, `facts`, and `artifacts`. "
            "Do not inline long editor comments, reviewer comments, or other bulky source text into the kernel; keep only a short summary plus artifact paths pointing to any saved full memo. "
            "Keep only the minimum decision-relevant facts needed for the next choice. "
            "Use `frontier` for the next unresolved questions or actions to validate. "
            "When delegating, write a clear bounded brief. For normal execution lanes, prefer the compact `Summary` / `Facts` / `Files` contract so the result can be distilled into a run card. For `peer_review_specialist`, keep the full editor/reviewer markdown in the returned text or saved memo instead of forcing it into the compact kernel shape."
        )

    @classmethod
    def _materials_worker_prompt(cls, *, execution_contract: str = "") -> str:
        return (
            "You are materials_worker for ExperimentSpecialist.\n"
            "Handle a bounded materials execution subtask autonomously inside the workspace.\n"
            "This worker owns structure/calc/result workflows: modeling, VASP execution, surrogate-forcefield screening, and materials-side analysis.\n"
            "Typical MACE work here includes surrogate screening, relaxation, ranking, and post-analysis when those steps serve one materials workflow.\n"
            "When no dedicated tool covers a bounded materials task, use `execute` to implement the missing step with Python and mature third-party libraries inside the workspace instead of stopping at the missing-tool boundary.\n"
            "When configuration details, package behavior, or methodological best practice are uncertain, use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check before finalizing the workflow; do not wait for a dedicated search tool.\n"
            "For heavier custom logic such as high-throughput screening helpers, large batch post-processing, or multi-step deterministic pipelines, write a reusable workspace script under `scripts/` and run that script instead of leaving the whole implementation embedded in one `execute` call.\n"
            "When your result naturally becomes a dataset, a training/evaluation job, or an active-learning update loop, return the artifacts needed for a clean handoff to `ml_worker`.\n"
            "Use available execution and analysis tools, keep the run focused, and return a compact result with the key finding, relevant artifact paths, and any blocking issue.\n"
            "Do not perform broad literature review; that belongs to literature_agent.\n"
            f"{execution_contract}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _ml_worker_prompt(cls, *, execution_contract: str = "") -> str:
        return (
            "You are ml_worker for ExperimentSpecialist.\n"
            "Handle a bounded machine-learning subtask autonomously inside the workspace.\n"
            "This worker owns dataset/model lifecycle tasks: dataset building, model training, benchmark evaluation, and active-learning candidate selection.\n"
            "Start here when the primary artifact is a curated dataset, a training/evaluation run, a model checkpoint, or an active-learning selection ledger.\n"
            "Assume most ML work here will be done by writing reusable Python scripts under `scripts/` and running them, not by waiting for dedicated narrow ML tools to exist.\n"
            "Prefer using libraries already available in the environment and reusable workspace code before introducing new dependencies or parallel implementations.\n"
            "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, and `joblib`; prefer them first unless the task clearly needs something else.\n"
            "If the ML logic is longer than a short throwaway snippet, materialize it as a script instead of keeping it inline in the conversation or a one-off command.\n"
            "Prefer organizing topic-specific ML scripts under `scripts/<topic>/`, and use shared `scripts/` only for genuinely cross-topic utilities.\n"
            "When no dedicated tool covers a bounded ML task, use `execute` to implement the missing step with Python and mature third-party libraries inside the workspace instead of stopping at the missing-tool boundary.\n"
            "Prefer materializing training pipelines, feature generation, sweeps, evaluation harnesses, embedding workflows, and data-processing logic as reusable scripts rather than burying them in one-off shell snippets.\n"
            "Use remote execution when the job is heavy, long-running, batch-oriented, or needs managed compute; otherwise keep the script local and lightweight.\n"
            "Treat the managed ML tools as preferred paths when they fit, not as an exclusive gate. If the current ML task is not covered by those managed tools, keep going locally with reusable scripts under `scripts/` instead of stopping.\n"
            "When framework behavior, hyperparameter conventions, or implementation best practice are uncertain, use the online model's built-in web-browsing capability for a narrow official-docs or primary-source check before locking the workflow; do not wait for a dedicated search tool.\n"
            "For heavier custom logic such as dataset sweeps, benchmark harnesses, or other multi-run deterministic pipelines, write a reusable workspace script under `scripts/` and run that script instead of leaving the whole implementation embedded in one `execute` call.\n"
            "When the loop needs new structures, new reference calculations, or materials-side post-analysis, return the artifacts needed for a clean handoff to `materials_worker`.\n"
            "Do not perform broad literature review; that belongs to literature_agent.\n"
            f"{execution_contract}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _litreview_agent_prompt(cls) -> str:
        return (
            "You are literature_agent.\n"
            "Use Tavily-backed public web search and public-page reading to gather external literature grounding, benchmark conventions, broader background evidence, and public-source synthesis.\n"
            "You are the broad-review and orientation layer, not the exact scholarly metadata resolver. If exact DOI/year/venue/authors/citation details are missing or uncertain, ResearchSpecialist should delegate that part to `metadata_agent`.\n"
            "Stay focused on representative, decision-relevant sources instead of broad browsing.\n"
            "You may write concise reusable literature artifacts into the workspace when helpful, such as notes, evidence summaries, source lists, or background briefs.\n"
            "Return concise findings with clear separation between retrieved facts and inference.\n"
            "Do not perform computational execution.\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Only save a reusable markdown note under `/notes/literature/` or another stable workspace path when the user asked for a saved artifact or when a durable handoff/writing reference is clearly worth the extra file.\n"
            "Return a polished markdown answer with exactly these sections in order: `Answer`, `Public Evidence`, `Interpretation`, and `Files`.\n"
            "`Answer` should synthesize the best available public evidence in a few compact paragraphs.\n"
            "`Public Evidence` should be a flat bullet list with source titles, concrete factual takeaways, and source URLs.\n"
            "`Interpretation` should separate direct evidence from your inference, identify uncertainty, and say when metadata verification is still needed from `metadata_agent`.\n"
            "`Files` should list any saved reusable note paths, or `(none reported)` if nothing was persisted."
        )

    @classmethod
    def _litreview_wrapper_prompt(cls) -> str:
        return (
            "You are litreview_agent.\n"
            "You are the top-level literature-review orchestrator used by ResearchSpecialist.\n"
            "Delegate broad public-web orientation, review synthesis, landing-page inspection, and public-source evidence gathering to `literature_agent`.\n"
            "Delegate exact DOI/year/venue/authors/citation verification and scholarly record disambiguation to `metadata_agent`.\n"
            "Use whichever subagent is necessary, and use both when a review needs both broad evidence and citation-grade metadata.\n"
            "Keep the final answer compact and decision-relevant. Save a reusable note under `/notes/literature/` or another stable workspace path only when the user asked for it or when a durable handoff artifact is clearly justified.\n"
            "Do not perform computational execution.\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _metadata_agent_prompt(cls) -> str:
        return (
            "You are metadata_agent.\n"
            "Use only the scholarly metadata tools to resolve exact paper matches, DOI/year/venue/authors/citation details, recommendation expansion, and citation-grade metadata.\n"
            "You are not the broad-review layer. Do not do public-web orientation here.\n"
            "Prefer precision over breadth. When the query is ambiguous, narrow the candidate set and explicitly state uncertainty instead of guessing.\n"
            "You may write concise reusable citation notes or metadata tables into the workspace when helpful.\n"
            "Do not perform computational execution.\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Return a polished markdown answer with exactly these sections in order: `Metadata Answer`, `Candidate Records`, `Gaps`, and `Files`.\n"
            "`Metadata Answer` should directly state the best exact matches or the best disambiguation you could establish.\n"
            "`Candidate Records` should be a flat bullet list with title, year, venue, DOI/identifier, and why each record is relevant.\n"
            "`Gaps` should explain any unresolved ambiguity or missing metadata.\n"
            "`Files` should list any saved reusable metadata-note paths, or `(none reported)` if nothing was persisted."
        )

    @classmethod
    def _lightweight_literature_agent_prompt(cls) -> str:
        return (
            "You are literature_agent for ExperimentSpecialist.\n"
            "Use the lightweight `internet_search` tool for focused external web research when ExperimentSpecialist needs quick literature hints, benchmark conventions, or general public-web answers.\n"
            "Treat Tavily results as preprocessed web evidence. Prefer a few narrow searches over one vague broad search.\n"
            "This agent is not limited to academic literature: it may answer with high-quality web evidence when the user asks for broader background, methods, safety notes, public documentation, or benchmark references.\n"
            "Use the standard DeepAgent workspace capabilities when helpful to save reusable notes or evidence summaries into the workspace.\n"
            "Only save a concise reusable markdown note under `/notes/literature/` or another stable workspace path when the user asked for a saved artifact or when a durable handoff/writing reference is clearly justified, and include that path in the final `Files` section.\n"
            "If a durable cross-run fact should be stored, report it explicitly so ExperimentSpecialist can decide whether to update project memory.\n"
            "Do not perform computational execution.\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Return a polished markdown answer with exactly these sections in order: `Answer`, `Web Evidence`, `Interpretation`, and `Files`.\n"
            "`Answer` should directly answer the request in a few compact paragraphs.\n"
            "`Web Evidence` should be a flat bullet list with source titles, concrete factual takeaways, and source URLs.\n"
            "`Interpretation` should separate direct evidence from your inference, note uncertainty, and explain why the evidence matters for the experiment context.\n"
            "`Files` should list any saved reusable note paths, or `(none reported)` if nothing was persisted."
        )

    @classmethod
    def _writing_literature_agent_prompt(cls) -> str:
        return (
            "You are literature_agent for WritingSpecialist.\n"
            "Use the lightweight `internet_search` tool only for tightly bounded writing-support lookups: missing introduction background, a specific benchmark/context check, or a small citation-support query needed by the current manuscript draft.\n"
            "Do not expand the task into a broad literature review, a new scientific campaign, or open-ended research planning.\n"
            "Prioritize a few targeted, high-signal sources that directly support the current writing need.\n"
            "Return concise findings with clear separation between retrieved facts and inference.\n"
            "If the request really needs broad review or citation-grade metadata disambiguation beyond this narrow scope, say so explicitly so the user can switch to the research lane.\n"
            "Do not perform computational execution.\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Only save a concise reusable markdown note under `/notes/literature/` or another stable workspace path when a durable writing handoff artifact is clearly justified.\n"
            "Return a polished markdown answer with exactly these sections in order: `Answer`, `Web Evidence`, `Interpretation`, and `Files`.\n"
            "`Answer` should directly address the bounded writing-context question in a few compact paragraphs.\n"
            "`Web Evidence` should be a flat bullet list with source titles, concrete factual takeaways, and source URLs.\n"
            "`Interpretation` should explain how the evidence should influence the manuscript wording and note uncertainty.\n"
            "`Files` should list any saved reusable note paths, or `(none reported)` if nothing was persisted."
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
            "Handle only one section or one bounded organization/integration task at a time.\n"
            "Assume the parent has already reduced the task into a compact author packet. Follow that packet instead of rediscovering the paper story from raw run logs.\n"
            "Do not reopen broad research loops or re-read large unrelated workspace trees on your own.\n"
            "For paper/manuscript/journal-style writing, complete the evidence presentation: add or update figures, tables, structure renders, and concise schematics when they materially improve the manuscript and the workspace contains enough evidence to support them.\n"
            "For paper/manuscript/journal-style writing, explicitly organize what belongs in the main text versus Supporting Information / Supporting Data. Keep claim-critical evidence in the main manuscript; move extended methods, exhaustive tables, auxiliary figures, coordinate inventories, and machine-readable exports into supporting content when that makes the package cleaner and more submission-ready.\n"
            "For the current implementation, keep Supporting Information in the same manuscript file rather than a separate SI manuscript: place it after the references as a clear supporting-information section or appendix. Supporting data files may still live in separate workspace folders.\n"
            "Use `generate_nanobanana_figure` for conceptual, mechanistic, or workflow figures. Prefer it over hand-built matplotlib diagrams for those figure types, and reserve plotting libraries for quantitative or data-native visualizations.\n"
            "For short notes or compact summaries, do not manufacture extra visuals unless they are explicitly requested or clearly necessary for comprehension.\n"
            "When writing a paper/manuscript title, produce a compact journal-style title that foregrounds the material system and the main scientific result. Avoid titles that read like project summaries, workflow descriptions, or sentence-length claims.\n"
            "For LaTeX manuscripts, do not batch figures into a later block. Insert each figure environment close to the paragraph that first discusses it, prefer conservative placement controls such as `[htbp]`, and if compilation still pushes a figure too far away, repair it by moving the float closer to first mention or inserting `\\FloatBarrier` when the template already supports it.\n"
            "Return concise manuscript-ready output summaries and any output artifact paths.\n"
            "If the output is a TeX bundle, you must run `compile_text` yourself before returning and use its diagnostics/log summary to fix compile-facing issues.\n"
            "Do not treat a successful TeX compile as sufficient if the PDF still has obviously misplaced figures or a weak title.\n"
            "If you draft TeX with citations, structure it to use a separate bibliography file rather than leaving inline `thebibliography` in the final bundle.\n"
            f"{cls._peer_review_ready_paper_policy()}\n"
            f"{cls._journal_manuscript_policy()}\n"
            f"{cls._author_packet_policy()}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._writing_reporting_contract()}"
        )

    @classmethod
    def _writing_polisher_prompt(cls) -> str:
        return (
            "You are writing_polisher_agent for WritingSpecialist.\n"
            "Perform conservative section-level prose polish on already drafted manuscript text.\n"
            "Improve sentence flow, readability, local transitions, and journal-style phrasing without changing claim strength, scientific scope, evidence selection, paragraph order, figure order, numbers, units, citations, labels, or notation.\n"
            "Do not delete or add substantive scientific content on your own. If a needed change is structural, evidentiary, or argumentative rather than local prose polish, report that in the summary instead of rewriting around it.\n"
            "Do not introduce new experiments, new references, or new limitations.\n"
            "When polishing TeX, preserve commands, labels, citation keys, math, and float structure unless the parent explicitly asks for a local TeX fix.\n"
            f"{cls._peer_review_ready_paper_policy()}\n"
            f"{cls._journal_manuscript_policy()}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._writing_reporting_contract()}"
        )

    @classmethod
    def _report_worker_prompt(cls) -> str:
        return (
            "You are report_worker_agent for ExperimentSpecialist.\n"
            "Write bounded experiment-facing reports from existing workspace evidence only.\n"
            "This lane is for experiment reports, validation summaries, QC notes, execution memos, and other report-style artifacts; it is not a paper/manuscript lane.\n"
            "Assume the parent has already reduced the task into a compact report packet. Follow that packet instead of replaying raw run history.\n"
            "Do not restart calculations, broaden the scientific scope, or convert the report into a manuscript-style narrative.\n"
            "Prioritize reproducibility, executed scope, method specificity, key results, and unresolved execution/QC items. Preserve failed attempts or QC caveats when they are material to interpreting the result.\n"
            "If visuals are explicitly requested and the evidence supports them, include only the visuals that help an experiment-facing reader understand the result. Prefer `generate_nanobanana_figure` for concise workflow or mechanism sketches, and keep matplotlib-style plotting for quantitative figures.\n"
            "If the output is a TeX bundle, run `compile_text` yourself before returning and fix compile-facing issues from the diagnostics.\n"
            f"{cls._report_packet_policy()}\n"
            f"{cls._multimodal_tool_history_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._writing_reporting_contract()}"
        )

    @staticmethod
    def _proposal_system_prompt(entrypoint: SpecialistEntrypoint) -> str:
        return (
            f"You are {entrypoint.capitalize()}Specialist in proposal review mode.\n"
            "Produce a compact executable proposal only. Do not perform the work yet.\n"
            "Return a ProposalCheckpoint with a markdown proposal, short todo list, and only blocking human questions.\n"
            f"{SpecialistRunner._workspace_path_discipline()}"
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
    def _build_default_middleware(*, model_call_run_limit: int = 120) -> list[Any]:
        middleware: list[Any] = []
        try:
            from langchain.agents.middleware.model_call_limit import ModelCallLimitMiddleware
        except Exception:
            pass
        else:
            if int(model_call_run_limit) > 0:
                middleware.append(ModelCallLimitMiddleware(run_limit=int(model_call_run_limit)))
        try:
            from langchain.agents.middleware import AgentMiddleware
        except Exception:
            AgentMiddleware = None
        if AgentMiddleware is not None:
            class _ToolMessageHistorySanitizerMiddleware(AgentMiddleware):
                def wrap_model_call(self, request: Any, handler: Any) -> Any:
                    sanitized = SpecialistRunner._sanitize_model_request_messages(getattr(request, "messages", []))
                    if sanitized is getattr(request, "messages", None):
                        return handler(request)
                    return handler(request.override(messages=sanitized))

                async def awrap_model_call(self, request: Any, handler: Any) -> Any:
                    sanitized = SpecialistRunner._sanitize_model_request_messages(getattr(request, "messages", []))
                    if sanitized is getattr(request, "messages", None):
                        return await handler(request)
                    return await handler(request.override(messages=sanitized))

            middleware.append(_ToolMessageHistorySanitizerMiddleware())
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

    @staticmethod
    def _sanitize_model_request_messages(messages: Any) -> Any:
        if not isinstance(messages, list):
            return messages
        changed = False
        sanitized: list[Any] = []
        for message in messages:
            normalized = SpecialistRunner._sanitize_model_request_message(message)
            if normalized is not message:
                changed = True
            sanitized.append(normalized)
        return sanitized if changed else messages

    @staticmethod
    def _sanitize_model_request_message(message: Any) -> Any:
        if not isinstance(message, ToolMessage):
            return message
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return message
        replacement = SpecialistRunner._tool_message_content_placeholder(message)
        if replacement == content:
            return message
        try:
            return message.model_copy(update={"content": replacement})
        except Exception:
            return ToolMessage(
                content=replacement,
                artifact=getattr(message, "artifact", None),
                tool_call_id=str(getattr(message, "tool_call_id", "") or ""),
                name=str(getattr(message, "name", "") or None) or None,
                status=str(getattr(message, "status", "success") or "success"),
                additional_kwargs=dict(getattr(message, "additional_kwargs", {}) or {}),
                response_metadata=dict(getattr(message, "response_metadata", {}) or {}),
                id=getattr(message, "id", None),
            )

    @staticmethod
    def _tool_message_content_placeholder(message: ToolMessage) -> str:
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content
        additional = dict(getattr(message, "additional_kwargs", {}) or {})
        path = str(additional.get("read_file_path") or "").strip()
        media_type = str(additional.get("read_file_media_type") or "").strip()
        if isinstance(content, list):
            text_bits: list[str] = []
            image_blocks = 0
            for item in content:
                if isinstance(item, str) and item.strip():
                    text_bits.append(item.strip())
                    continue
                if not isinstance(item, dict):
                    continue
                item_type = str(item.get("type") or "").strip().lower()
                if item_type == "text":
                    text = str(item.get("text") or "").strip()
                    if text:
                        text_bits.append(text)
                    continue
                if item_type in {"image", "image_url"}:
                    image_blocks += 1
                    if not media_type:
                        media_type = str(item.get("mime_type") or "").strip()
            if image_blocks:
                details: list[str] = []
                if path:
                    details.append(f"source={path}")
                if media_type:
                    details.append(f"mime={media_type}")
                note = (
                    f"[inline image tool output omitted from model history; "
                    f"{' '.join(details) if details else f'images={image_blocks}'}]"
                )
                if text_bits:
                    return "\n".join(text_bits + [note]).strip()
                return note
        text = content_to_text(content).strip()
        return text or "[non-text tool output omitted from model history]"

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
        summary, facts, files, review_target = self._parse_summary_and_files(text)
        return {
            "text": text,
            "summary": summary,
            "facts": facts,
            "files": files,
            "review_target": review_target,
        }

    def _finalize_report(self, parsed: dict[str, Any]) -> dict[str, Any]:
        summary = str(parsed.get("summary") or "").strip()
        facts = [str(item).strip() for item in list(parsed.get("facts") or []) if str(item).strip()]
        files = [self._normalize_artifact_path(str(item).strip()) for item in list(parsed.get("files") or []) if str(item).strip()]
        review_target = self._normalize_artifact_path(str(parsed.get("review_target") or "").strip()) if parsed.get("review_target") else ""
        files, facts = self._ensure_tex_bundle_outputs(files=files, facts=facts)
        return {
            "text": self._render_compact_report(summary=summary, facts=facts, files=files, review_target=review_target),
            "summary": summary,
            "facts": facts,
            "files": files,
            "review_target": review_target,
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

    def _parse_summary_and_files(self, text: str) -> tuple[str, list[str], list[str], str]:
        summary_lines: list[str] = []
        facts: list[str] = []
        files: list[str] = []
        review_target = ""
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
                continue
            if current_section == "review_target":
                path = self._extract_reported_file(line)
                if path and not review_target:
                    review_target = path
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
        return summary, deduped_facts, deduped_files, self._normalize_artifact_path(review_target) if review_target else ""

    @staticmethod
    def _match_report_heading(line: str) -> str | None:
        normalized = re.sub(r"^[#\-\s]+", "", str(line or "").strip()).lower().rstrip(":")
        if normalized == "summary":
            return "summary"
        if normalized == "facts":
            return "facts"
        if normalized == "files":
            return "files"
        if normalized in {"reviewtarget", "review target"}:
            return "review_target"
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
    def _render_compact_report(*, summary: str, facts: list[str], files: list[str], review_target: str = "") -> str:
        lines = [
            "## Summary",
            summary.strip() or "(no summary reported)",
        ]
        if facts:
            lines.extend(
                [
                    "",
                    "## Facts",
                    *[f"- {item}" for item in facts],
                ]
            )
        if files:
            lines.extend(
                [
                    "",
                    "## Files",
                    *[f"- `{item}`" for item in files],
                ]
            )
        if review_target:
            lines.extend(
                [
                    "",
                    "## ReviewTarget",
                    f"- `{review_target}`",
                ]
            )
        return "\n".join(lines).strip()

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
        compile_tool = self.registry.get_tool_function("compile_text")
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
                        "compile_text",
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
    def _load_compiled_subagent():
        try:
            from deepagents.middleware.subagents import CompiledSubAgent
        except Exception as exc:
            raise RuntimeError("deepagents compiled subagent support is required.") from exc
        return CompiledSubAgent

    @staticmethod
    def _load_memory_middleware():
        try:
            from deepagents.middleware.memory import MemoryMiddleware
        except Exception as exc:
            raise RuntimeError("deepagents memory middleware is required.") from exc
        return MemoryMiddleware

    @staticmethod
    def _load_llm_tool_selector_middleware():
        try:
            from langchain.agents.middleware.tool_selection import LLMToolSelectorMiddleware
        except Exception as exc:
            raise RuntimeError(
                "LangChain LLMToolSelectorMiddleware is unavailable. Install 'langchain>=1.0'."
            ) from exc
        return LLMToolSelectorMiddleware

    @staticmethod
    def _load_create_manage_memory_tool():
        try:
            from langmem import create_manage_memory_tool
        except Exception as exc:
            raise RuntimeError("langmem is required for project long-term memory tools.") from exc
        return create_manage_memory_tool

    @staticmethod
    def _load_create_search_memory_tool():
        try:
            from langmem import create_search_memory_tool
        except Exception as exc:
            raise RuntimeError("langmem is required for project long-term memory tools.") from exc
        return create_search_memory_tool

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
