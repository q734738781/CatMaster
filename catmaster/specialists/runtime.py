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
from langgraph.types import Command
from pydantic import BaseModel

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.artifact_callback import LangChainStepLogger, UIEventHandler
from catmaster.runtime.run_context import RunContext
from catmaster.runtime.run_control import RunControl
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError, content_to_text
from catmaster.runtime.usage_stats import write_usage_summary_from_metadata
from catmaster.runtime.workspace_python_env import workspace_python_env_overrides
from catmaster.tools.base import system_root, workspace_root, workspace_scope
from catmaster.tools.execution.machine_registry import MachineRegister
from catmaster.tools.execution.task_registry import TaskRegistry
from catmaster.tools.registry import get_tool_registry
from catmaster.ui import make_event
from catmaster.ui.reporters import NullReporter, Reporter

from .schemas import ProposalCheckpoint, ResearchKernel, SpecialistEntrypoint

logger = logging.getLogger(__name__)


class SpecialistRetryableModelResponseError(RuntimeError):
    """Transient model/provider response was syntactically successful but unusable."""


class SpecialistInvalidFinalReportError(RuntimeError):
    """Final assistant output did not satisfy the specialist reporting contract."""

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
    "literature_review": "literature_deep_research",
}
_SUPPORTED_ENTRYPOINTS = set(_ENTRYPOINT_TO_MODEL_ROLE)
_ENTRYPOINT_ALIASES = {
    "litreview": "literature_review",
    "literature": "literature_review",
}

_MATERIALS_WORKER_TOOL_ALLOWLIST = {
    "create_molecule_from_smiles",
    "mace_neb_batch",
    "mace_relax_batch",
    "mace_md_batch",
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
    "estimate_neb_image_count",
    "remap_neb_endpoint_atoms",
    "make_neb_geometry",
    "make_dimer_mode_from_neb",
    "make_dimer_mode_from_mace",
    "mace_analyze_frequencies",
    "generate_strained_structures",
    "generate_kpath",
    "generate_phonon_displacements",
    "vasp_neb_prepare",
    "vasp_dimer_prepare",
    "vasp_execute_batch",
    "mp_search_materials",
    "mp_download_structure",
    "render_structure_views",
    "identify_structure_fragments",
    "analyze_vasp_neb_results",
    "analyze_trajectory",
    "generate_nanobanana_figure",
    "vaspkit_adsorbate_thermo_correction",
    "vaspkit_gas_thermo_correction",
    "export_builtin_tool_source",
}
_ML_WORKER_TOOL_ALLOWLIST: set[str] = {
    "build_dataset_from_runs",
    "mace_train",
    "mace_evaluate",
    "calculate_al_candidates",
    "export_builtin_tool_source",
}
_ORCA_XTB_WORKER_TOOL_ALLOWLIST: set[str] = {
    "create_molecule_from_smiles",
    "enumerate_molecular_conformers",
    "filter_conformer_ensemble",
    "extract_optimized_molecules",
    "identify_structure_fragments",
    "crest_conformer_search",
    "xtb_run_batch",
    "analyze_xtb_results",
    "orca_prepare",
    "orca_scan_prepare",
    "orca_optts_prepare",
    "orca_nebts_prepare",
    "orca_irc_prepare",
    "orca_execute_batch",
    "analyze_orca_results",
    "export_builtin_tool_source",
}
_WRITING_TOOL_ALLOWLIST = {
    "generate_nanobanana_figure",
    "review_pdf_manuscript",
}
_RESEARCH_TOOL_ALLOWLIST: set[str] = set()
_EXPERIMENT_SPECIALIST_TOOL_ALLOWLIST = set()
_PEER_REVIEW_TOOL_ALLOWLIST = {"peer_review_request"}
_PEER_REVIEW_WORKER_TOOL_ALLOWLIST = set(_PEER_REVIEW_TOOL_ALLOWLIST)
_METADATA_AGENT_TOOL_ALLOWLIST = {
    "search_openalex",
    "search_semantic_scholar",
    "get_openalex_record",
    "get_semantic_scholar_record",
    "recommend_semantic_scholar",
}
_LITREVIEW_AGENT_TOOL_ALLOWLIST = {
    "web_search",
    "open_public_page",
    "find_in_page",
}
_DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES = {"web_search"}
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
_WRITING_WORKER_TOOL_ALLOWLIST = {
    "polish_academic_prose",
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
_SKILLS_ROOT = "/.deepagents/skills"


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
    _MODEL_RESPONSE_RETRY_DELAYS_S: tuple[float, ...] = (60.0, 180.0, 300.0)
    _FINAL_REPORT_RETRY_DELAYS_S: tuple[float, ...] = (30.0, 120.0)

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
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        return asyncio.run(
            self.arun(
                prompt,
                entrypoint=entrypoint,
                proposal_review=proposal_review,
                chat_session_id=chat_session_id,
                thread_id=thread_id,
                conversation_messages=conversation_messages,
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
        conversation_messages: list[dict[str, Any]] | None = None,
    ) -> dict[str, Any]:
        payload = {
            "entrypoint": entrypoint,
            "user_prompt": str(prompt or "").strip(),
            "proposal_review": bool(proposal_review),
            "chat_session_id": str(chat_session_id or "").strip(),
            "thread_id": str(thread_id or "").strip(),
            "conversation_messages": list(conversation_messages or []),
        }
        return await self._run_impl(payload=payload, resume_feedback=None)

    async def aresume(self, human_feedback: str = "") -> dict[str, Any]:
        run_state = self._read_run_state()
        if not run_state:
            raise ValueError("Cannot resume run without run_state.json")
        status = str(run_state.get("status") or "").strip().lower() or "unknown"
        if status in {"done", "failure"}:
            raise ValueError("Selected run is already finished.")
        feedback = str(human_feedback or "").strip() or "Continue the previous interrupted request."
        run_state["status"] = "running"
        run_state["phase"] = "executing"
        run_state["pending_human_input"] = None
        run_state["proposal_review"] = False
        run_state["proposal_revision_count"] = 0
        run_state["resume_guidance"] = feedback
        run_state["resume_source_status"] = status
        return await self._run_impl(payload=run_state, resume_feedback=feedback)

    async def _run_impl(self, *, payload: dict[str, Any], resume_feedback: str | None) -> dict[str, Any]:
        raw_entrypoint = str(payload.get("entrypoint") or "research").strip() or "research"
        entrypoint = _ENTRYPOINT_ALIASES.get(raw_entrypoint, raw_entrypoint)
        if entrypoint not in _SUPPORTED_ENTRYPOINTS:
            raise ValueError(f"Unsupported specialist entrypoint: {entrypoint}")

        prompt = str(payload.get("user_prompt") or "").strip()
        if not prompt:
            raise ValueError("Prompt is required.")
        thread_id = self._resolve_thread_id(payload)
        conversation_messages = self._coerce_conversation_messages(payload.get("conversation_messages"))

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
            proposal_revision_count = 0

            if resume_feedback is not None:
                self._write_run_state(
                    {
                        **payload,
                        "status": "running",
                        "phase": "executing",
                        "pending_human_input": None,
                        "proposal_review": False,
                        "proposal_revision_count": 0,
                        "text_preview": str(resume_feedback or "")[:280],
                    }
                )

            retryable_exceptions = (
                SpecialistRetryableModelResponseError,
                SpecialistInvalidFinalReportError,
            )
            max_attempts = len(self._FINAL_REPORT_RETRY_DELAYS_S) + 1
            for attempt_index in range(max_attempts):
                self._raise_if_interrupt_requested(phase="before_agent_invoke", details={"entrypoint": entrypoint})
                try:
                    async with self._open_agent_runtime(files_root=files_root) as runtime:
                        agent = await self._build_entry_agent(
                            entrypoint=entrypoint,
                            runtime=runtime,
                            thread_id=thread_id,
                        )
                        if resume_feedback is None:
                            messages = [
                                *conversation_messages,
                                {"role": "user", "content": prompt},
                            ]
                        else:
                            messages = [{"role": "user", "content": str(resume_feedback or "").strip() or prompt}]
                        result = await agent.ainvoke(
                            {"messages": messages},
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
                            "proposal_review": False,
                            "proposal_revision_count": 0,
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
                except retryable_exceptions as exc:
                    if attempt_index >= max_attempts - 1:
                        raise RuntimeError(
                            f"{entrypoint}_specialist failed after {max_attempts} attempts due to transient model/output instability."
                        ) from exc
                    delay_s = self._FINAL_REPORT_RETRY_DELAYS_S[attempt_index]
                    logger.warning(
                        "%s retrying after retryable model/output failure on attempt %d/%d in %.1fs: %s",
                        entrypoint,
                        attempt_index + 1,
                        max_attempts,
                        delay_s,
                        exc,
                    )
                    self._emit(
                        "RUN_RETRY",
                        payload={
                            "entrypoint": entrypoint,
                            "attempt": attempt_index + 1,
                            "max_attempts": max_attempts,
                            "delay_s": delay_s,
                            "reason": str(exc),
                        },
                    )
                    await asyncio.sleep(delay_s)
                except Exception as exc:
                    if not self._is_retryable_model_exception(exc):
                        raise
                    if attempt_index >= max_attempts - 1:
                        raise RuntimeError(
                            f"{entrypoint}_specialist failed after {max_attempts} attempts due to transient provider/schema instability."
                        ) from exc
                    delay_s = self._FINAL_REPORT_RETRY_DELAYS_S[attempt_index]
                    logger.warning(
                        "%s retrying after transient provider/schema failure on attempt %d/%d in %.1fs: %s",
                        entrypoint,
                        attempt_index + 1,
                        max_attempts,
                        delay_s,
                        exc,
                    )
                    self._emit(
                        "RUN_RETRY",
                        payload={
                            "entrypoint": entrypoint,
                            "attempt": attempt_index + 1,
                            "max_attempts": max_attempts,
                            "delay_s": delay_s,
                            "reason": str(exc),
                        },
                    )
                    await asyncio.sleep(delay_s)
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
                "proposal_review": False,
                "proposal_revision_count": 0,
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
        except Exception as exc:
            failed_state = {
                "schema_version": 1,
                "entrypoint": entrypoint,
                "status": "error",
                "phase": "failed",
                "active_specialist": entrypoint,
                "thread_id": thread_id,
                "proposal_review": False,
                "proposal_revision_count": 0,
                "pending_human_input": None,
                "todo_items": [],
                "artifacts": list(payload.get("artifacts") or []),
                "delegation_log": list(payload.get("delegation_log") or []),
                "text_preview": str(exc)[:280],
                "user_prompt": prompt,
                "chat_session_id": str(payload.get("chat_session_id") or ""),
                "final_answer": "",
                "summary": str(exc).strip() or "Run failed.",
                "facts": [],
                **self._research_kernel_state_fields(files_root=files_root, thread_id=thread_id, relpath=research_kernel_relpath),
            }
            self._write_run_state(failed_state)
            self._emit("RUN_END", payload={"entrypoint": entrypoint, "status": "error", "error": str(exc)})
            raise
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
        if entrypoint == "literature_review":
            _ = thread_id
            return self._build_litreview_agent(runtime=runtime)
        create_deep_agent = self._load_create_deep_agent()
        tools = self._specialist_tools(entrypoint)
        # TODO: Revisit explicit summarization tuning for OpenRouter-backed specialists
        # via an official config path instead of patching model.profile at runtime.
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint])),
            "tools": tools,
            "system_prompt": self._system_prompt(entrypoint, thread_id=thread_id),
            "middleware": self._build_default_middleware(),
            "checkpointer": runtime["checkpointer"],
            "store": runtime["store"],
            "backend": runtime["backend"],
            "name": f"{entrypoint}_specialist",
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
        if entrypoint == "peer_review":
            return self._peer_review_subagents(runtime=runtime)
        return []

    def _research_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        return [
            self._compiled_specialist_subagent(
                name="experiment_specialist",
                description="Run bounded computational experiment work and return compact evidence summaries.",
                entrypoint="experiment",
                runtime=runtime,
            ),
            self._compiled_specialist_subagent(
                name="writing_specialist",
                description="Turn existing evidence into reports, outlines, sections, or manuscript-ready outputs.",
                entrypoint="writing",
                runtime=runtime,
            ),
            self._compiled_specialist_subagent(
                name="peer_review_specialist",
                description="Act like a journal editor: inspect the manuscript PDF, request reviewer-style reports, and return an editor decision with raw reviewer comments.",
                entrypoint="peer_review",
                runtime=runtime,
            ),
            self._compiled_litreview_subagent(runtime=runtime),
        ]

    def _experiment_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        return [
            self._compiled_worker_subagent(
                name="materials_worker",
                description="Handle bounded, context-heavy materials execution subtasks in isolation and return concise results with artifact paths.",
                model_role="task_runner",
                system_prompt=self._materials_worker_prompt(
                    execution_contract=self._execution_capability_contract(audience="materials_worker")
                ),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_MATERIALS_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("materials")],
                runtime=runtime,
            ),
            self._compiled_worker_subagent(
                name="ml_worker",
                description="Handle bounded machine-learning subtasks in isolation using the default DeepAgent tool surface until ML-specific tools are added.",
                model_role="task_runner",
                system_prompt=self._ml_worker_prompt(
                    execution_contract=self._execution_capability_contract(audience="ml_worker")
                ),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_ML_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("machine_learning")],
                runtime=runtime,
            ),
            self._compiled_worker_subagent(
                name="orca_xtb_worker",
                description="Handle bounded molecular quantum-chemistry subtasks, including conformer search, xTB screening, and ORCA preparation/execution/analysis.",
                model_role="task_runner",
                system_prompt=self._orca_xtb_worker_prompt(
                    execution_contract=self._execution_capability_contract(audience="orca_xtb_worker")
                ),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_ORCA_XTB_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("quantum_chemistry")],
                runtime=runtime,
            ),
        ]

    def _writing_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        return [
            self._compiled_worker_subagent(
                name="writing_worker_agent",
                description="Draft or revise context-heavy sections in isolation and return compact manuscript-ready outputs.",
                model_role="section_writer",
                system_prompt=self._writing_worker_prompt(),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("writing")],
                runtime=runtime,
            ),
            self._compiled_worker_subagent(
                name="writing_polisher_agent",
                description="Apply conservative section-level prose polish without changing the manuscript's scientific stance or structure.",
                model_role="academic_polisher",
                system_prompt=self._writing_polisher_prompt(),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_WRITING_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("writing")],
                runtime=runtime,
            ),
        ]

    def _peer_review_subagents(self, *, runtime: dict[str, Any]) -> list[Any]:
        return [
            self._compiled_worker_subagent(
                name="peer_review_worker_agent",
                description="Run one bounded peer-review episode over one canonical manuscript PDF and return the full review plus memo path.",
                model_role="write_reviewer",
                system_prompt=self._peer_review_worker_prompt(),
                tools=self._augment_with_project_memory_tools(
                    self._named_tools(_PEER_REVIEW_WORKER_TOOL_ALLOWLIST),
                    include_manage_memory=False,
                ),
                skills=[self._skills_group_virtual_path("writing")],
                runtime=runtime,
            ),
        ]

    def _subagent_middleware(
        self,
        *,
        runtime: dict[str, Any],
        include_memory_middleware: bool = True,
    ) -> list[Any]:
        middleware = [
            *self._build_default_middleware(),
        ]
        if include_memory_middleware:
            middleware.append(self._new_memory_middleware(backend=runtime["backend"]))
        return middleware

    def _metadata_middleware(self, *, runtime: dict[str, Any]) -> list[Any]:
        middleware: list[Any] = []
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
            description="Orchestrate literature review by combining `web_search`, public-page inspection, and exact DOI/venue/author resolution via `metadata_agent`.",
            runnable=self._build_litreview_agent(runtime=runtime),
        )

    def _compiled_specialist_subagent(
        self,
        *,
        name: str,
        description: str,
        entrypoint: SpecialistEntrypoint,
        runtime: dict[str, Any],
    ) -> Any:
        CompiledSubAgent = self._load_compiled_subagent()
        return CompiledSubAgent(
            name=name,
            description=description,
            runnable=self._build_nested_specialist_agent(entrypoint=entrypoint, runtime=runtime),
        )

    def _compiled_worker_subagent(
        self,
        *,
        name: str,
        description: str,
        model_role: str,
        system_prompt: str,
        tools: list[Any],
        runtime: dict[str, Any],
        skills: list[str] | None = None,
        middleware: list[Any] | None = None,
    ) -> Any:
        CompiledSubAgent = self._load_compiled_subagent()
        return CompiledSubAgent(
            name=name,
            description=description,
            runnable=self._build_nested_worker_agent(
                name=name,
                model_role=model_role,
                system_prompt=system_prompt,
                tools=tools,
                runtime=runtime,
                skills=skills,
                middleware=middleware,
            ),
        )

    def _build_nested_specialist_agent(
        self,
        *,
        entrypoint: SpecialistEntrypoint,
        runtime: dict[str, Any],
    ) -> Any:
        create_deep_agent = self._load_create_deep_agent()
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(_ENTRYPOINT_TO_MODEL_ROLE[entrypoint])),
            "tools": self._specialist_tools(entrypoint),
            "system_prompt": self._system_prompt(entrypoint),
            # `create_deep_agent` already injects its standard stack, including
            # MemoryMiddleware when `memory=` is provided. Only pass additional
            # CatMaster middleware here to avoid duplicate middleware instances.
            "middleware": self._build_default_middleware(),
            "checkpointer": runtime["checkpointer"],
            "store": runtime["store"],
            "backend": runtime["backend"],
            "name": f"{entrypoint}_specialist",
            "memory": self._memory_sources(),
        }
        subagents = self._entry_subagents(entrypoint, runtime=runtime)
        if subagents:
            kwargs["subagents"] = subagents
        return create_deep_agent(**kwargs)

    def _build_nested_worker_agent(
        self,
        *,
        name: str,
        model_role: str,
        system_prompt: str,
        tools: list[Any],
        runtime: dict[str, Any],
        skills: list[str] | None = None,
        middleware: list[Any] | None = None,
    ) -> Any:
        create_deep_agent = self._load_create_deep_agent()
        kwargs: dict[str, Any] = {
            "model": build_chat_model(self.llm_profile.config_for_role(model_role)),
            "tools": tools,
            "system_prompt": system_prompt,
            "middleware": [
                *self._build_default_middleware(),
                *(middleware or []),
            ],
            "checkpointer": runtime["checkpointer"],
            "store": runtime["store"],
            "backend": runtime["backend"],
            "name": name,
            "memory": self._memory_sources(),
        }
        if skills:
            kwargs["skills"] = skills
        return create_deep_agent(**kwargs)

    def _build_litreview_agent(self, *, runtime: dict[str, Any]) -> Any:
        create_deep_agent = self._load_create_deep_agent()
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
                self._compiled_worker_subagent(
                    name="literature_agent",
                    description="Use `web_search`, `open_public_page`, and `find_in_page` for broader literature review, background grounding, and public-source synthesis.",
                    model_role="literature_synthesizer",
                    system_prompt=self._litreview_agent_prompt(),
                    tools=self._augment_with_project_memory_tools(
                        self._named_tools(_LITREVIEW_AGENT_TOOL_ALLOWLIST),
                        include_manage_memory=False,
                    ),
                    runtime=runtime,
                ),
                self._compiled_worker_subagent(
                    name="metadata_agent",
                    description="Resolve exact paper metadata, DOI/year/venue/authors, and citation details from scholarly databases.",
                    model_role="literature_deep_research",
                    system_prompt=self._metadata_agent_prompt(),
                    tools=self._augment_with_project_memory_tools(
                        self._named_tools(_METADATA_AGENT_TOOL_ALLOWLIST),
                        include_manage_memory=False,
                    ),
                    middleware=self._metadata_middleware(runtime=runtime),
                    runtime=runtime,
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

            workspace_env = workspace_python_env_overrides(self.run_context.workspace)
            return CompositeBackend(
                default=LocalShellBackend(
                    root_dir=files_root,
                    virtual_mode=True,
                    timeout=14_400,
                    env=workspace_env,
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
        for tool in self._named_tools(_DEFAULT_AUTONOMOUS_AGENT_TOOL_NAMES):
            name = str(getattr(tool, "name", "") or "").strip()
            if name and name not in existing:
                augmented.append(tool)
                existing.add(name)
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
            base / "materials": repo_root / "skills" / "materials",
            base / "machine_learning": repo_root / "skills" / "machine_learning",
            base / "quantum_chemistry": repo_root / "skills" / "quantum_chemistry",
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
    def _skills_group_virtual_path(group_name: str) -> str:
        return f"{_SKILLS_ROOT}/{str(group_name or '').strip()}"

    def _resolve_thread_id(self, payload: dict[str, Any]) -> str:
        thread_id = str(payload.get("thread_id") or "").strip()
        if thread_id:
            return thread_id
        chat_session_id = str(payload.get("chat_session_id") or "").strip()
        if chat_session_id:
            return chat_session_id
        return self.run_context.run_id

    @staticmethod
    def _coerce_conversation_messages(raw: Any) -> list[dict[str, str]]:
        if not isinstance(raw, list):
            return []
        out: list[dict[str, str]] = []
        for item in raw:
            if not isinstance(item, dict):
                continue
            role = str(item.get("role") or "").strip().lower()
            if role not in {"user", "assistant"}:
                continue
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            out.append({"role": role, "content": content})
        return out

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
        audience: Literal["research", "experiment", "materials_worker", "ml_worker", "orca_xtb_worker"],
    ) -> str:
        available = set(self.registry.tools.keys())
        managed_tools = [
            name
            for name in (
                "vasp_execute_batch",
                "mace_neb_batch",
                "mace_relax_batch",
                "mace_md_batch",
                "mace_sp_batch",
                "mace_train",
                "mace_evaluate",
                "xtb_run_batch",
                "crest_conformer_search",
                "orca_execute_batch",
            )
            if name in available
        ]
        prepare_tools = [
            name
            for name in (
                "vasp_prepare",
                "vasp_band_prepare",
                "vasp_neb_prepare",
                "build_dataset_from_runs",
                "orca_prepare",
                "orca_scan_prepare",
                "orca_optts_prepare",
                "orca_nebts_prepare",
                "orca_irc_prepare",
            )
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
            "Your current tool surface is not the whole platform surface; some managed execution, resource visibility, or environment checks exist only behind delegated specialists or workers.",
            "If a bounded local task needs a missing Python package, `execute` may install it with `python -m pip install ...` when that is the most direct way to complete the step.",
        ]
        if audience == "research":
            lines.append(
                "When a bounded experiment stage needs periodic DFT or other managed compute, route it to ExperimentSpecialist instead of downgrading it to literature-only validation because no local executable is visible in the shell."
            )
            lines.append(
                "If execution-path, managed-resource, or remote-environment availability matters and the relevant capability is not visible in your own tool surface, delegate a bounded probe to ExperimentSpecialist instead of concluding the capability is unavailable."
            )
        else:
            lines.append(
                "Treat local shell checks as relevant only for local interactive steps; use the registered remote-execution tools for managed runs."
            )
            lines.append(
                "If execution-path, managed-resource, or remote-environment availability matters and the relevant capability is not visible in your current tool surface, delegate a bounded probe to the most relevant downstream worker instead of trying to prove absence from the current thread."
            )
        if prepare_tools:
            lines.append(f"Local preparation/analysis tools here: {', '.join(f'`{name}`' for name in prepare_tools)}.")
        if managed_tools:
            lines.append(f"Managed execution tools here: {', '.join(f'`{name}`' for name in managed_tools)}.")
        if machine_names or resource_names or task_names:
            facts: list[str] = []
            if machine_names:
                facts.append(f"machines={', '.join(f'`{name}`' for name in machine_names)}")
            if resource_names:
                facts.append(f"resources={', '.join(f'`{name}`' for name in resource_names)}")
            if task_names:
                facts.append(f"tasks={', '.join(f'`{name}`' for name in task_names)}")
            lines.append("DPDispatcher config visible here: " + "; ".join(facts) + ".")
        if audience in {"experiment", "materials_worker"} and "vasp_execute_batch" in available:
            lines.append(
                "For periodic DFT, the intended path is to prepare inputs locally, submit via `vasp_execute_batch`, then analyze the returned results; do not require a local periodic DFT engine to be directly runnable first."
            )
        if audience in {"experiment", "orca_xtb_worker"} and "xtb_run_batch" in available:
            lines.append(
                "For molecular preoptimization, quick screening, or conformer refinement, use `xtb_run_batch`; do not block on local xTB visibility when the registered submission tool is available."
            )
        if audience in {"experiment", "orca_xtb_worker"} and "crest_conformer_search" in available:
            lines.append(
                "For broad conformer/rotamer exploration, prefer `crest_conformer_search` over ad hoc shell wrappers when the task is naturally a CREST batch."
            )
        if audience in {"experiment", "orca_xtb_worker"} and "orca_execute_batch" in available:
            lines.append(
                "For serious molecular quantum-chemistry runs, prepare jobs locally with the ORCA preparation tools, submit them with `orca_execute_batch`, then close the loop with `analyze_orca_results`."
            )
        if audience in {"experiment", "materials_worker"} and any(name in available for name in ("mace_neb_batch", "mace_relax_batch", "mace_md_batch", "mace_sp_batch")):
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
                "Use `litreview_agent` for literature-review work that needs synthesis, source inspection, or metadata verification. It internally uses `web_search`, `open_public_page`, and `find_in_page`, and can delegate exact DOI/year/venue/authors/citation metadata resolution to `metadata_agent`.\n"
                "If the user requests a paper, manuscript, journal-style LaTeX draft, cover letter, rebuttal-style response, or other author-facing publication artifact, delegate that work to `writing_specialist` rather than drafting it directly in the research thread.\n"
                "If the user requests an experiment report, validation summary, QC note, execution-facing memo, or other report-style artifact grounded in completed workspace evidence, delegate that work to `experiment_specialist` as a bounded report-writing episode.\n"
                "Default to not launching `peer_review_specialist`.\n"
                "Launch `peer_review_specialist` only when the user explicitly asks for publication-level paper quality, submission-ready or peer-review-ready manuscript quality, formal submission requirements, journal submission standards, or another equivalent formal publication bar.\n"
                "When you do launch it, treat it as an editor-style review process over the manuscript PDF, not as the primary scientific decision-maker.\n"
                "When delegating to `peer_review_specialist`, explicitly hand it the canonical workspace-relative manuscript PDF path; if one PDF is the review target, state that path clearly in the handoff instead of making the reviewer infer it.\n"
                "When `peer_review_specialist` returns, treat its returned markdown or saved review memo as the authoritative revision brief. Do not rely on the Research Kernel to preserve full editor/reviewer comment text.\n"
                "If `peer_review_specialist` gives you a saved review memo path, read that memo directly before deciding the next revision or experiment step.\n"
                "You remain the sole coordinator and final decision-maker for the run.\n"
                "After any delegated specialist returns, actively judge from the user's request, current evidence, and actual project state whether another bounded delegation round is needed; do not default to closing in the research thread just because one delegate completed.\n"
                "If peer-review or revision comments show that additional experiments are needed, you may relaunch `experiment_specialist` for bounded follow-up work as long as that work still respects the user's stated scope, budget, evidence limits, and time constraints.\n"
                "If peer review indicates the work cannot reach the requested publication bar within the user's stated scope, budget, evidence limits, or time constraints, stop and tell the user that directly instead of looping.\n"
                "Do not treat your own local shell view or direct tool view as authoritative for managed experiment capability. If submission-path, remote-environment, or resource visibility matters, issue a bounded probe to `experiment_specialist` rather than deciding from absence in the research thread.\n"
                f"{cls._author_packet_policy()}\n"
                f"{cls._report_packet_policy()}\n"
                f"{cls._tool_policy()}\n"
                f"{cls._general_purpose_specialist_policy()}\n"
                f"{cls._multimodal_policy()}\n"
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
                "Your default role is coordination, target validation, and final editorial synthesis rather than direct tool execution.\n"
                "If the parent or user gives you an explicit `ReviewTarget` or manuscript PDF path, treat that as the canonical review target.\n"
                "Use DeepAgent file tools to locate the manuscript PDF only when that path is missing, ambiguous, or invalid.\n"
                "Once you have identified the canonical manuscript PDF, delegate the bounded review episode to `peer_review_worker_agent` and pass that canonical PDF path explicitly.\n"
                "When one worker review episode returns, actively decide whether another bounded delegate pass is needed or whether the result is ready to send upstream; do not default to closing in the specialist thread just because one worker finished.\n"
                "Have `peer_review_worker_agent` call `peer_review_request` on that PDF exactly once per review episode and return the full review plus any saved review memo path.\n"
                "Do not run experiments, do not rewrite the manuscript, and do not take over research planning.\n"
                "Your job is to synthesize an editor decision and editor comment from the reviewer reports, then include the raw reviewer comments for ResearchSpecialist or the user.\n"
                "Use decision language such as reject, major revision, minor revision, or conditionally acceptable only when supported by the reviewer comments and the manuscript evidence.\n"
                "Keep the review grounded in ACS-style expectations: scientific soundness, evidence-claim fit, controls, validation quality, novelty positioning, comparison quality, figure logic, and publication readiness.\n"
                "Return the full review markdown directly to the parent; do not compress away the editor comment or reviewer comment sections.\n"
                "Also save the full review as one durable workspace markdown memo under `notes/peer_review/` or another stable path, and include that memo path in `Files`, so the parent can reuse the exact text without depending on kernel summaries.\n"
                f"{cls._tool_policy()}\n"
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
                "Use `web_search` directly only for narrow background supplementation when the user explicitly asks to expand background/context, or when a paper/manuscript draft lacks the minimal external background needed for a credible introduction or discussion.\n"
                "Keep such `web_search` use tightly bounded to the current writing need; do not let it expand into a new autonomous research campaign.\n"
                "Your default role is coordination, not long-form drafting in the main thread.\n"
                "For any substantive note writing, section writing, manuscript drafting, or major revision, immediately delegate to `writing_worker_agent` with a bounded brief.\n"
                "Before a substantial paper/manuscript rewrite, first condense the task into one compact inline author packet, then dispatch the section or integration brief from that packet instead of forwarding raw run logs.\n"
                "Each writing-worker handoff should cover only one section or one bounded organization/integration task. "
                "Give it one primary goal and one completion criterion. "
                "If the next step still requires deciding what to write next, how to restructure the manuscript, or whether to change direction, bring that decision back to WritingSpecialist instead of letting the worker continue to expand.\n"
                "When one writing-worker pass returns, actively decide whether another bounded delegate pass is needed or whether the result is ready for final reconciliation; do not default to closing in the specialist thread just because one worker finished.\n"
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
                f"{cls._tool_policy()}\n"
                f"{cls._general_purpose_specialist_policy()}\n"
                f"{cls._multimodal_policy()}\n"
                f"{memory_policy}\n"
                f"{cls._memory_write_policy()}\n"
                f"{cls._workspace_path_discipline()}\n"
                f"{cls._writing_reporting_contract()}"
            )
        return (
            "You are ExperimentSpecialist.\n"
            "Your default role is coordination, dispatch, and decision-making across the experiment lane, not personally executing the substantive domain work.\n"
            "Keep direct work in the specialist thread minimal and coordination-oriented: quick workspace inspection, artifact triage, memory updates, deciding the next bounded handoff, and bounded experiment-facing summaries grounded in completed workspace evidence.\n"
            "Route by the current working artifact and domain: use `materials_worker` for periodic materials and surface work, including structure preparation, VASP execution, MACE screening/NEB/relaxation, and materials-side post-analysis; use `ml_worker` for dataset construction, model fine-tuning or training, benchmark evaluation, ML workflow development, and active-learning algorithm work; use `orca_xtb_worker` for molecular or cluster quantum-chemistry work such as conformer generation, xTB screening, ORCA preparation/execution, and molecular post-analysis; use `web_search` directly when a quick external check is needed.\n"
            "When a request clearly falls into one of those worker-owned domains, delegate first instead of doing the domain work yourself.\n"
            "In particular, general materials or surface workflows belong to `materials_worker`; model fine-tuning, training, evaluation, feature/data pipelines, and ML algorithm development belong to `ml_worker`; molecular or cluster quantum-chemistry workflows belong to `orca_xtb_worker`; purely report writing from already completed evidence stays in `ExperimentSpecialist` rather than being delegated further.\n"
                "Each worker should receive only one bounded execution episode around one primary artifact, such as one screening round, one training/evaluation pass, or one post-analysis step. "
                "Each brief should contain one primary goal and one completion criterion. "
                "If direction still needs to be chosen after the step finishes, bring that choice back to ExperimentSpecialist instead of letting the worker continue to expand. "
                "When one worker pass returns, actively decide whether another bounded delegate pass is needed; do not default to closing in the specialist thread just because one worker finished.\n"
                "Do not hand an entire high-throughput campaign to one worker; split it into episodes and decide the next episode yourself after each return.\n"
            "Do not personally absorb worker-owned tasks just because your own direct tool surface appears sufficient for a small piece of them; the worker boundary is part of the design contract.\n"
            "Do not assume your own specialist thread can directly verify every execution path or remote environment. Some submission or resource checks are only visible through worker-owned managed tools.\n"
            "If execution-path, remote-environment, or resource availability is relevant and the relevant managed tool is not directly visible here, delegate a bounded probe to the matching worker instead of concluding the capability is absent.\n"
            "Only do the implementation directly in the specialist thread when no available worker matches the task, or when the action is a tiny coordination-only step that would not justify a delegation round.\n"
            "If the task is purely report writing from already completed evidence, do not restart calculations just to make the report look more complete. Summarize the executed scope honestly and keep unresolved points explicit.\n"
            "If a bounded workspace task is not covered by a dedicated registered tool, do not stop at that boundary alone; route it to the relevant worker so it can use `execute` plus Python and mature third-party libraries for a focused custom implementation when the environment supports it.\n"
            "If a worker needs a handy Python package for a bounded local step and it is missing, let it install that package with `execute` via `python -m pip install ...`.\n"
            "When method settings, software behavior, or scientific best practice are uncertain, use `web_search` for a narrow official-docs or primary-source check before improvising a custom implementation. Keep that check narrow and implementation-oriented; do not turn it into a broad literature review.\n"
            "When that custom implementation becomes heavy, batch-oriented, high-throughput, or clearly worth rerunning, prefer materializing it as a reusable workspace script under `scripts/` instead of burying the logic inside one long ephemeral shell command.\n"
            f"Do not orchestrate other specialists. {memory_policy}\n"
            f"{cls._report_packet_policy()}\n"
            f"{cls._tool_policy()}\n"
            f"{cls._general_purpose_specialist_policy()}\n"
            f"{cls._multimodal_policy()}\n"
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
            "Use exactly these fields: `thesis`, `novelty`, `core_claims` (2-4 bullets), `evidence_refs`, `main_text_keep`, `supporting_only`, and `target_outputs`. "
            "Then issue one bounded writing brief with the section goal, target audience, requested output path(s), local section structure, and any citation/style constraints. "
            "For TeX deliverables, require separate `.tex` and `.bib` files plus at least one direct compile pass when the environment supports compilation. "
            "Do not paste long transcripts or ask the writing agent to rediscover workspace evidence."
        )

    @staticmethod
    def _report_packet_policy() -> str:
        return (
            "For experiment-report handoffs, pass one compact inline report packet with exactly these fields: "
            "`objective`, `executed_scope`, `key_methods`, `key_results`, `failures_or_qc`, and `target_outputs`. "
            "Keep it terse, execution-facing, and grounded in completed workspace evidence. Do not pad it with paper-style novelty framing or raw transcript excerpts."
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
    def _general_purpose_specialist_policy() -> str:
        return (
            "Delegate domain-owned work to the proper specialized subagent first. "
            "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context. "
            "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents. It is for context isolation, not responsibility transfer."
        )

    @staticmethod
    def _general_purpose_worker_policy() -> str:
        return (
            "Use `general-purpose` only for bounded work that still belongs to your current lane when the main risk is context bloat from heavy local context. "
            "`general-purpose` uses only the current layer's tools and cannot delegate to other subagents. It is for context isolation, not responsibility transfer."
        )

    @staticmethod
    def _multimodal_policy() -> str:
        return (
            "Multimodal discipline: use `general-purpose` for multimodal analysis so that multimodal context stays isolated from the parent thread. "
            "For PDFs, first extract with fitz into workspace text artifacts and analyze those artifacts."
        )

    @staticmethod
    def _tool_policy() -> str:
        return (
            "Tool discipline: if a relevant skill is available to the current agent, read it before acting. "
            "Prefer registered builtin tools when they fit the task. "
            "Before writing custom code, first try to satisfy the task by adjusting the parameters and supported variants of a relevant builtin tool. "
            "Builtin tools often already encode validated parameter choices, implementation optimizations, and known pitfall handling, so avoid reimplementing that logic unless the builtin boundary truly does not fit. "
            "For any code that overlaps internal tool functionality, first export the builtin tool source into the workspace with `export_builtin_tool_source` and code against that reference instead of starting from scratch; if custom code is still necessary, inspect that exported source carefully and preserve its validated behavior rather than writing an approximate look-alike from memory."
        )

    @staticmethod
    def _journal_manuscript_policy() -> str:
        return (
            "Journal-manuscript discipline: when the deliverable is a paper, manuscript, or journal-style draft, write as an author of the scientific work, not as an agent narrating workflow. "
            "Do not mention the workspace, files, runs, prompts, tools, agents, interruptions, or that the draft was assembled from workspace evidence. "
            "Keep process provenance such as 'no new calculations were run in this writing pass' out of title, abstract, main text, acknowledgements, and supporting-information prose unless the user explicitly asked for an internal note. "
            "State scientific scope limits and evidence limits in field-appropriate prose, not as internal workflow disclaimers. "
            "For journal-facing citations and BibTeX, use publication-style metadata only; if citation metadata is unresolved, prefer a visible citation gap or a request for literature cleanup rather than fabricating a reference."
        )

    @staticmethod
    def _workspace_path_discipline() -> str:
        return (
            "Workspace path discipline: treat the project files root as your working directory and prefer workspace-relative paths. "
            "Treat `/` only as the workspace virtual root, not as a host filesystem root. "
            "If you see a host absolute path like `/home/...`, convert it back to a workspace-relative path and never recreate host absolute path segments inside the workspace. "
            "Do not pass guessed input paths into tools: if a structure or dataset does not already exist under the workspace files root, create or fetch it first, then reuse that exact returned path. "
            "For shell or `execute` commands, never use leading-slash workspace paths like `/writing/...`; use workspace-relative paths such as `writing/...` instead. "
            "Keep transient observations in the conversation/tool stream unless persistence is needed. Only persist key constraints, decisive results, reusable handoff material, or user-requested deliverables. "
            "Prefer a topic-centric layout: `literature/` for grounding material, `structures/` for geometry/setup artifacts, `calculations/` for execution outputs, `scripts/` for reusable code, `notes/` for compact saved notes, and `writing/` for manuscript outputs. "
            "If the workspace already has a clear established layout, extend it instead of creating a parallel scheme."
        )

    @staticmethod
    def _long_term_memory_policy(*, allow_manage_memory: bool = True) -> str:
        if allow_manage_memory:
            return _PROJECT_MEMORY_TOOL_INSTRUCTIONS
        return _PROJECT_MEMORY_READONLY_INSTRUCTIONS

    @staticmethod
    def _soft_reporting_contract() -> str:
        return (
            "For multi-step work, use `write_todos` early and keep it updated when the plan changes. "
            "When you finish, reply with only three markdown sections in this order: `Summary`, `Facts`, and `Files`. "
            "`Summary` must directly answer the user's actual question with the key result and conclusion; do not say only that a report was written. "
            "`Facts` should be a short flat list of the most important archival facts. "
            "`Files` should be a flat list of relevant workspace-relative output paths; do not return bare filenames, and use `(none reported)` if there are none. "
            "If one manuscript PDF is the canonical downstream review target, you may add an optional `ReviewTarget` section with exactly one workspace-relative PDF path. "
            "If you are correcting a previously wrong result after the user pointed out an error, replace or delete stale incorrect reports/notes when feasible and do not leave superseded wrong paths in `Files`."
        )

    @staticmethod
    def _writing_reporting_contract() -> str:
        return (
            "For multi-step work, use `write_todos` early and keep it updated when the plan changes. "
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
            "Typical MACE work here includes surrogate screening, relaxation, MD sampling, ranking, and post-analysis when those steps serve one materials workflow.\n"
            "When no dedicated tool covers a bounded materials task, use `execute` to implement the missing step with Python and mature third-party libraries inside the workspace instead of stopping at the missing-tool boundary.\n"
            "When preparing VASP inputs or scripts that need POTCAR access, obtain POTCARs through the pymatgen interface rather than ad hoc shell copying or manual symbol-to-file mapping.\n"
            "If a handy Python package is missing for a bounded local step, install it with `execute` via `python -m pip install ...`.\n"
            "When configuration details, package behavior, or methodological best practice are uncertain, use `web_search` for a narrow official-docs or primary-source check before finalizing the workflow.\n"
            "For heavier custom logic such as high-throughput screening helpers, large batch post-processing, or multi-step deterministic pipelines, write a reusable workspace script under `scripts/` and run that script instead of leaving the whole implementation embedded in one `execute` call.\n"
            "When your result naturally becomes a dataset, a training/evaluation job, or an active-learning update loop, return the artifacts needed for a clean handoff to `ml_worker`.\n"
            "Use available execution and analysis tools, keep the run focused, and return a compact result with the key finding, relevant artifact paths, and any blocking issue.\n"
            "Do not perform broad literature review; that belongs to `litreview_agent` in the research lane.\n"
            f"{cls._tool_policy()}\n"
            f"{execution_contract}\n"
            f"{cls._general_purpose_worker_policy()}\n"
            f"{cls._multimodal_policy()}\n"
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
            "When a registered managed ML tool fits the task, prefer that managed path first.\n"
            "For MACE dataset curation, training, and benchmark evaluation, prefer `build_dataset_from_runs`, `mace_train`, and `mace_evaluate` over ad hoc local wrapper scripts.\n"
            "Do not create or run a local `mace_run_train` wrapper when `mace_train` already fits the request; use the managed remote training path instead.\n"
            "Prefer using libraries already available in the environment and reusable workspace code before introducing new dependencies or parallel implementations.\n"
            "Common libraries already available here include `numpy`, `pandas`, `scipy`, `matplotlib`, `torch`, `joblib`, and `matminer`; prefer them first unless the task clearly needs something else.\n"
            "If a handy Python package is still missing for a bounded local step, install it with `execute` via `python -m pip install ...`.\n"
            "If the ML logic is longer than a short throwaway snippet and no managed tool covers it, materialize it as a script instead of keeping it inline in the conversation or a one-off command.\n"
            "Prefer organizing topic-specific ML scripts under `scripts/<topic>/`, and use shared `scripts/` only for genuinely cross-topic utilities.\n"
            "When no dedicated tool covers a bounded ML task, use `execute` to implement the missing step with Python and mature third-party libraries inside the workspace instead of stopping at the missing-tool boundary.\n"
            "Prefer materializing training pipelines, feature generation, sweeps, evaluation harnesses, embedding workflows, and data-processing logic as reusable scripts rather than burying them in one-off shell snippets.\n"
            "Use remote execution when the job is heavy, long-running, batch-oriented, or needs managed compute; MACE training/fine-tuning normally falls into this category.\n"
            "Treat the managed ML tools as preferred paths when they fit, not as an exclusive gate. If the current ML task is not covered by those managed tools, keep going locally with reusable scripts under `scripts/` instead of stopping.\n"
            "When framework behavior, hyperparameter conventions, or implementation best practice are uncertain, use `web_search` for a narrow official-docs or primary-source check before locking the workflow.\n"
            "For heavier custom logic such as dataset sweeps, benchmark harnesses, or other multi-run deterministic pipelines, write a reusable workspace script under `scripts/` and run that script instead of leaving the whole implementation embedded in one `execute` call.\n"
            "When the loop needs new structures, new reference calculations, or materials-side post-analysis, return the artifacts needed for a clean handoff to `materials_worker`.\n"
            "Do not perform broad literature review; that belongs to `litreview_agent` in the research lane.\n"
            f"{cls._tool_policy()}\n"
            f"{execution_contract}\n"
            f"{cls._general_purpose_worker_policy()}\n"
            f"{cls._multimodal_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _orca_xtb_worker_prompt(cls, *, execution_contract: str = "") -> str:
        return (
            "You are orca_xtb_worker for ExperimentSpecialist.\n"
            "Handle a bounded molecular quantum-chemistry subtask autonomously inside the workspace.\n"
            "This worker owns molecule/cluster workflows: SMILES-to-3D conversion, conformer generation and pruning, xTB or CREST preoptimization/screening, ORCA preparation/execution, and molecular post-analysis for optimization, frequencies, scans, TS/IRC, TDDFT, or NMR-style jobs.\n"
            "Prefer the dedicated managed tools when they fit: `create_molecule_from_smiles`, `enumerate_molecular_conformers`, `filter_conformer_ensemble`, `crest_conformer_search`, `xtb_run_batch`, `orca_prepare`, `orca_scan_prepare`, `orca_optts_prepare`, `orca_nebts_prepare`, `orca_irc_prepare`, `orca_execute_batch`, `analyze_xtb_results`, and `analyze_orca_results`.\n"
            "If the user names a small molecule or cluster but does not provide a structure file, first create the structure under `<topic>/structures/` and only then launch xTB/CREST/ORCA tools against that exact workspace-relative path.\n"
            "Do not guess that a path like `<topic>/structures/<name>.xyz` already exists; verify it exists or create it before calling `xtb_run_batch`, `crest_conformer_search`, or any ORCA preparation tool.\n"
            "Treat xTB/CREST as the fast exploration layer and ORCA as the higher-fidelity molecular quantum layer unless the task explicitly calls for a different partition.\n"
            "For cheap preoptimization, conformer cleanup, low-cost screening, or geometry relaxation before higher-level ORCA work, default to `xtb_run_batch` or `crest_conformer_search` instead of `orca_prepare(method=XTB1/XTB2)`.\n"
            "Use ORCA with XTB-family methods only when the request explicitly needs an ORCA-native XTB workflow or another ORCA-side feature that `xtb_run_batch` does not cover; do not choose ORCA-XTB as the default fallback for routine preopt steps.\n"
            "When the request is about one mechanistic step or one catalyst-side molecular episode, keep the run on the molecular lane instead of trying to translate it into a periodic workflow.\n"
            "When no dedicated tool covers a bounded molecular task, use `execute` to implement the missing step with Python and mature third-party libraries inside the workspace instead of stopping at the missing-tool boundary.\n"
            "If a handy Python package is missing for a bounded local step, install it with `execute` via `python -m pip install ...`.\n"
            "For heavier custom logic such as ensemble post-processing, Boltzmann aggregation, or multi-step deterministic screening helpers, write a reusable workspace script under `scripts/` and run that script instead of leaving the whole implementation embedded in one `execute` call.\n"
            "When configuration details, software behavior, or methodological best practice are uncertain, use `web_search` for a narrow official-docs or primary-source check before finalizing the workflow.\n"
            "Return a compact result with the key finding, relevant artifact paths, and any blocking issue.\n"
            "Do not perform broad literature review; that belongs to `litreview_agent` in the research lane.\n"
            f"{cls._tool_policy()}\n"
            f"{execution_contract}\n"
            f"{cls._general_purpose_worker_policy()}\n"
            f"{cls._multimodal_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._soft_reporting_contract()}"
        )

    @classmethod
    def _litreview_agent_prompt(cls) -> str:
        return (
            "You are literature_agent.\n"
            "Use `web_search`, `open_public_page`, and `find_in_page` to gather external literature grounding, benchmark conventions, broader background evidence, and public-source synthesis.\n"
            "You are the broad-review and orientation layer, not the exact scholarly metadata resolver. If exact DOI/year/venue/authors/citation details are missing or uncertain, the parent LitReview Agent should delegate that part to `metadata_agent`.\n"
            "Stay focused on representative, decision-relevant sources instead of broad browsing.\n"
            "You may write concise reusable literature artifacts into the workspace when helpful, such as notes, evidence summaries, source lists, or background briefs.\n"
            "Return concise findings with clear separation between retrieved facts and inference.\n"
            "Do not perform computational execution.\n"
            f"{cls._tool_policy()}\n"
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
            "You are the top-level literature-review orchestrator used by ResearchSpecialist and the direct Literature Review lane.\n"
            "Delegate broad public-web orientation, review synthesis, landing-page inspection, and public-source evidence gathering to `literature_agent`, which uses `web_search`, `open_public_page`, and `find_in_page`.\n"
            "Delegate exact DOI/year/venue/authors/citation verification and scholarly record disambiguation to `metadata_agent`.\n"
            "Use whichever subagent is necessary, and use both when a review needs both broad evidence and citation-grade metadata.\n"
            "Keep the final answer compact and decision-relevant. Save a reusable note under `/notes/literature/` or another stable workspace path only when the user asked for it or when a durable handoff artifact is clearly justified.\n"
            "Do not perform computational execution.\n"
            f"{cls._tool_policy()}\n"
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
            f"{cls._tool_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Return a polished markdown answer with exactly these sections in order: `Metadata Answer`, `Candidate Records`, `Gaps`, and `Files`.\n"
            "`Metadata Answer` should directly state the best exact matches or the best disambiguation you could establish.\n"
            "`Candidate Records` should be a flat bullet list with title, year, venue, DOI/identifier, and why each record is relevant.\n"
            "`Gaps` should explain any unresolved ambiguity or missing metadata.\n"
            "`Files` should list any saved reusable metadata-note paths, or `(none reported)` if nothing was persisted."
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
            f"{cls._tool_policy()}\n"
            f"{cls._general_purpose_worker_policy()}\n"
            f"{cls._multimodal_policy()}\n"
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
            f"{cls._tool_policy()}\n"
            f"{cls._general_purpose_worker_policy()}\n"
            f"{cls._multimodal_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            f"{cls._writing_reporting_contract()}"
        )

    @classmethod
    def _peer_review_worker_prompt(cls) -> str:
        return (
            "You are peer_review_worker_agent for PeerReviewSpecialist.\n"
            "Handle one bounded peer-review execution episode over one canonical manuscript PDF.\n"
            "If the parent gives you an explicit `ReviewTarget` or manuscript PDF path, treat that as the canonical review target.\n"
            "Use DeepAgent file tools to locate the manuscript PDF only when that path is missing, ambiguous, or invalid.\n"
            "Once the canonical manuscript PDF is identified, call `peer_review_request` on that PDF exactly once for this episode.\n"
            "Do not run experiments, do not rewrite the manuscript, and do not broaden the task into research planning.\n"
            "Your job is to collect the reviewer-style reports, synthesize an editor decision and editor comment grounded in those reports, and preserve the raw reviewer comments for the parent specialist.\n"
            "Use decision language such as reject, major revision, minor revision, or conditionally acceptable only when supported by the reviewer comments and manuscript evidence.\n"
            "Keep the review grounded in ACS-style expectations: scientific soundness, evidence-claim fit, controls, validation quality, novelty positioning, comparison quality, figure logic, and publication readiness.\n"
            "Also save the full review as one durable workspace markdown memo under `notes/peer_review/` or another stable path, and include that memo path in `Files`.\n"
            f"{cls._tool_policy()}\n"
            f"{cls._long_term_memory_policy(allow_manage_memory=False)}\n"
            f"{cls._memory_write_policy()}\n"
            f"{cls._workspace_path_discipline()}\n"
            "Return a concise markdown report with sections `Summary`, `Facts`, `Files`, `Editor Decision`, `Editor Comment`, and `Reviewer Comments`.\n"
            "In `Files`, include the reviewed manuscript PDF path.\n"
            "In `Reviewer Comments`, preserve each reviewer's raw comments with clear reviewer labels."
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

    @classmethod
    def _build_default_middleware(cls) -> list[Any]:
        middleware: list[Any] = []
        try:
            from langchain.agents.middleware import wrap_model_call, wrap_tool_call
        except Exception:
            return middleware

        @wrap_model_call(name="catmaster_retry_semantic_model_failures")
        async def _retry_invalid_model_responses(request: Any, handler: Any) -> Any:
            request = cls._sanitize_model_request_for_history(request)
            max_attempts = len(cls._MODEL_RESPONSE_RETRY_DELAYS_S) + 1
            for attempt_index in range(max_attempts):
                try:
                    response = await handler(request)
                    cls._validate_model_response_for_retry(response)
                    return response
                except Exception as exc:
                    if not isinstance(exc, SpecialistRetryableModelResponseError) and not cls._is_retryable_model_exception(exc):
                        raise
                    if attempt_index >= max_attempts - 1:
                        raise SpecialistRetryableModelResponseError(str(exc)) from exc
                    delay_s = cls._MODEL_RESPONSE_RETRY_DELAYS_S[attempt_index]
                    await asyncio.sleep(delay_s)
            raise SpecialistRetryableModelResponseError("Unexpected model retry loop exit.")

        middleware.append(_retry_invalid_model_responses)

        @wrap_tool_call(name="catmaster_textualize_multimodal_tool_results")
        async def _textualize_multimodal_tool_results(request: Any, handler: Any) -> Any:
            result = await handler(request)
            return cls._sanitize_tool_result_for_history(result)

        middleware.append(_textualize_multimodal_tool_results)

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
            raise SpecialistInvalidFinalReportError("specialist failed to return a final assistant text report.")
        if not self._has_required_summary_heading(text):
            raise SpecialistInvalidFinalReportError("specialist final report is missing the required `Summary` section.")
        summary, facts, files, review_target = self._parse_summary_and_files(text)
        if not str(summary or "").strip():
            raise SpecialistInvalidFinalReportError("specialist final report did not contain a usable summary.")
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
                    if not self._is_assistant_message(message):
                        continue
                    text = self._message_text(message)
                    if text:
                        return text
        return ""

    @classmethod
    def _message_text(cls, message: Any) -> str:
        if isinstance(message, dict):
            content = message.get("content")
        else:
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
                    item_type = str(item.get("type") or "").strip().lower()
                    if item_type in {"reasoning", "thinking", "reasoning_text", "redacted_reasoning"}:
                        continue
                    text = str(item.get("text") or "").strip()
                    if text:
                        chunks.append(text)
            return "\n".join(chunks).strip()
        return str(content or "").strip()

    @staticmethod
    def _is_assistant_message(message: Any) -> bool:
        if isinstance(message, AIMessage):
            return True
        role = ""
        if isinstance(message, dict):
            role = str(message.get("role") or message.get("type") or "").strip().lower()
        else:
            role = str(getattr(message, "role", "") or getattr(message, "type", "") or "").strip().lower()
        return role in {"assistant", "ai"}

    @staticmethod
    def _has_required_summary_heading(text: str) -> bool:
        for raw_line in str(text or "").splitlines():
            if SpecialistRunner._match_report_heading(raw_line) == "summary":
                return True
        return False

    @classmethod
    def _is_retryable_model_exception(cls, exc: Exception) -> bool:
        text = str(exc or "").lower()
        if not text:
            return False
        retryable_fragments = (
            "response validation failed",
            "eof while parsing",
            "validation errors for unmarshaller",
            "validation error for unmarshaller",
            "union_tag_invalid",
            "body.",
            ".tool.content",
        )
        if "validationerror" in text and "unmarshaller" in text:
            return True
        if "openrouter" in text and "validation" in text:
            return True
        if "body." in text and ".tool.content" in text:
            return True
        return any(fragment in text for fragment in retryable_fragments[:4])

    @classmethod
    def _sanitize_tool_result_for_history(cls, result: Any) -> Any:
        if isinstance(result, ToolMessage):
            return cls._sanitize_tool_message_for_history(result)
        if isinstance(result, Command):
            update = getattr(result, "update", None)
            if isinstance(update, dict) and isinstance(update.get("messages"), list):
                updated_messages = [
                    cls._sanitize_tool_message_for_history(item) if isinstance(item, ToolMessage) else item
                    for item in update["messages"]
                ]
                return Command(
                    graph=getattr(result, "graph", None),
                    update={**update, "messages": updated_messages},
                    resume=getattr(result, "resume", None),
                    goto=getattr(result, "goto", ()),
                )
        return result

    @classmethod
    def _sanitize_model_request_for_history(cls, request: Any) -> Any:
        messages = getattr(request, "messages", None)
        override = getattr(request, "override", None)
        if not isinstance(messages, list) or not callable(override):
            return request
        sanitized = [cls._sanitize_tool_message_for_history(item) if isinstance(item, ToolMessage) else item for item in messages]
        if all(left is right for left, right in zip(messages, sanitized, strict=False)):
            return request
        return override(messages=sanitized)

    @classmethod
    def _sanitize_tool_message_for_history(cls, message: ToolMessage) -> ToolMessage:
        content = getattr(message, "content", None)
        if not cls._tool_content_needs_textualization(content):
            return message
        text = cls._textualized_tool_content(
            content,
            tool_name=str(getattr(message, "name", "") or ""),
            additional_kwargs=getattr(message, "additional_kwargs", None),
        )
        return ToolMessage(
            content=text,
            additional_kwargs=dict(getattr(message, "additional_kwargs", None) or {}),
            response_metadata=dict(getattr(message, "response_metadata", None) or {}),
            name=getattr(message, "name", None),
            id=getattr(message, "id", None),
            tool_call_id=str(getattr(message, "tool_call_id", "") or "tool_result"),
            artifact=getattr(message, "artifact", None),
            status=getattr(message, "status", "success"),
        )

    @staticmethod
    def _tool_content_needs_textualization(content: Any) -> bool:
        if not isinstance(content, list):
            return False
        for item in content:
            if isinstance(item, str):
                continue
            if not isinstance(item, dict):
                return True
            item_type = str(item.get("type") or "").strip().lower()
            if item_type != "text":
                return True
        return False

    @classmethod
    def _textualized_tool_content(
        cls,
        content: Any,
        *,
        tool_name: str = "",
        additional_kwargs: Any = None,
    ) -> str:
        if not isinstance(content, list):
            return str(content or "").strip()
        lines: list[str] = []
        for item in content:
            if isinstance(item, str):
                if item.strip():
                    lines.append(item.strip())
                continue
            if isinstance(item, dict):
                item_type = str(item.get("type") or "content").strip() or "content"
                if item_type == "text" and isinstance(item.get("text"), str):
                    text = str(item.get("text") or "").strip()
                    if text:
                        lines.append(text)
                    continue
                lines.append(cls._multimodal_tool_block_reference(item, tool_name=tool_name, additional_kwargs=additional_kwargs))
                continue
            lines.append(f"[{type(item).__name__} tool content omitted from persistent history]")
        return "\n".join(line for line in lines if line).strip() or "[tool result omitted from persistent history]"

    @staticmethod
    def _multimodal_tool_block_reference(
        block: dict[str, Any],
        *,
        tool_name: str = "",
        additional_kwargs: Any = None,
    ) -> str:
        block_type = str(block.get("type") or "content").strip() or "content"
        mime_type = str(block.get("mime_type") or "").strip()
        path = ""
        if isinstance(additional_kwargs, dict):
            path = str(additional_kwargs.get("read_file_path") or "").strip()
            mime_type = mime_type or str(additional_kwargs.get("read_file_media_type") or "").strip()
        block_id = str(block.get("id") or "").strip()
        details = []
        if tool_name:
            details.append(f"tool={tool_name}")
        if path:
            details.append(f"path={path}")
        if mime_type:
            details.append(f"mime_type={mime_type}")
        if block_id:
            details.append(f"id={block_id}")
        suffix = f" ({', '.join(details)})" if details else ""
        return f"[{block_type} tool content omitted from persistent history{suffix}]"

    @classmethod
    def _validate_model_response_for_retry(cls, response: Any) -> None:
        if isinstance(response, AIMessage):
            cls._validate_ai_message_for_retry(response)
            return

        result_messages = list(getattr(response, "result", []) or [])
        if not result_messages:
            raise SpecialistRetryableModelResponseError("model returned no messages.")

        ai_messages = [message for message in result_messages if isinstance(message, AIMessage)]
        if not ai_messages:
            raise SpecialistRetryableModelResponseError("model response contained no assistant message.")
        cls._validate_ai_message_for_retry(ai_messages[-1])

    @classmethod
    def _validate_ai_message_for_retry(cls, message: AIMessage) -> None:
        finish_reason = str((getattr(message, "response_metadata", None) or {}).get("finish_reason") or "").strip().lower()
        if list(getattr(message, "invalid_tool_calls", None) or []):
            raise SpecialistRetryableModelResponseError("assistant returned invalid tool calls.")
        has_tool_calls = bool(list(getattr(message, "tool_calls", None) or []))
        has_visible_text = bool(cls._message_text(message))
        if finish_reason == "tool_calls" and not has_tool_calls:
            raise SpecialistRetryableModelResponseError("assistant reported tool_calls finish_reason without usable tool calls.")
        if has_tool_calls:
            return
        if has_visible_text:
            return
        raise SpecialistRetryableModelResponseError("assistant returned neither visible text nor tool calls.")

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
