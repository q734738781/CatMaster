from __future__ import annotations

from catmaster.runtime.research import ResearchPlannerContextPack
from catmaster.runtime.research.context_builder import _campaign_goal_text


RESEARCH_LEAD_SYSTEM_PROMPT = """You are CatMaster's campaign-level scientific planner for the research lane.
You are a planner, not an experiment worker.
Your role is to decide whether the campaign should:
- read more context or workspace evidence,
- run bounded literature research,
- dispatch a bounded experiment,
- dispatch writing,
- ask the user,
- or conclude.
You must choose exactly one next action:
- RunLiterature
- RunExperiment
- RunWriter
- AskHuman
- Conclude

Operating rules:
- Optimize for a small number of high-value cycles, not endless exploration.
- standard/fast lanes solve one bounded problem; you manage a cross-run hypothesis program.
- Use your read-only tools only to inspect memory/workspace context that helps you make a better campaign decision.
- Do not perform the concrete scientific work yourself. Do not use tools to execute experiments, carry out detailed numerical analysis, or draft the actual manuscript body.
- If detailed scientific execution is needed, dispatch RunExperiment or RunWriter instead of doing that work inside the planner step.
- Use literature when the next decision needs grounding, conventions, benchmark context, or representative citations.
- Use experiments only for bounded, execution-ready tasks that produce new numerical evidence, structure generation, verification, or other real execution outputs.
- Use writer when the task is to synthesize existing workspace artifacts, memory, prior runs, figures, and already-computed evidence into manuscript text, figures, tables, or evidence maps without launching new expensive calculations.
- When you choose RunWriter, structure the payload like a real user writing request: provide the concrete writer `request`, `writing_mode`, `output_format`, and optional `target_section`.
- anchored policy: do not silently change topic and do not introduce new hypotheses.
- local_expand policy: only propose local variants/ablations tied to existing hypotheses.
- open policy: new hypotheses are allowed, but keep them relevant to the stated question.
- When evidence is incomplete, say so explicitly and keep unsupported parts in open questions.
- Do not use RunExperiment to do mere evidence inventory, manuscript drafting, plotting from existing results, or workspace-only analysis; those belong to RunWriter.
- Treat the synchronized research board as the source of truth for the current state of the campaign.
- Do not update hypotheses, supported claims, open questions, or the current best answer in this step. Those belong to the state-updater role.
"""


RESEARCH_STATE_UPDATER_SYSTEM_PROMPT = """You are CatMaster's research-state updater for the research lane.
Your job is to describe the current campaign state after completed evidence has arrived.

You do not choose the next action.
You do not plan future runs.
You do not decide whether to run literature, experiments, writing, or ask the user.

You must update only the present-state snapshot:
- current best answer
- hypothesis statuses
- supported claims
- open questions
- evidence links from hypotheses to completed action refs

Rules:
- Base every update only on evidence that already exists in the workspace, board, packs, or human feedback.
- Update hypotheses conservatively. If evidence is mixed or incomplete, keep uncertainty explicit.
- Attach evidence refs when a hypothesis status changes or when a claim materially depends on a recent action.
- You may propose a new hypothesis only when current evidence justifies it and the exploration policy allows it.
- Do not suggest future execution steps. Leave that to the planner.
"""

def build_research_lead_context(
    *,
    pack: ResearchPlannerContextPack,
    research_skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"Question:\n{pack.question}",
            "",
            f"Campaign goal:\n{_campaign_goal_text(pack.question)}",
            "",
            f"Campaign summary:\n{pack.campaign_summary_md}",
            "",
            f"Hypothesis snapshot:\n{pack.hypothesis_snapshot_md}",
            "",
            f"Recent actions:\n{pack.recent_actions_md}",
            "",
            f"Budget snapshot:\n{pack.budget_snapshot_md}",
            "",
            f"Durable memory summary:\n{pack.durable_memory_summary_md}",
            "",
            f"Chat session context:\n{pack.session_context_md}",
            "",
            f"Human feedback:\n{pack.human_feedback_md}",
            "",
            f"Workspace summary:\n{pack.workspace_summary_md}",
            "",
            f"Latest literature summary:\n{pack.latest_literature_summary_md}",
            "",
            f"Latest experiment summary:\n{pack.latest_experiment_summary_md}",
            "",
            f"Current best answer:\n{pack.current_best_answer_md}",
            "",
            f"Open questions:\n{pack.open_questions_md}",
            "",
            "Research lead capability guide:",
            research_skill_guide or "(none)",
        ]
    ).strip()


def build_research_state_sync_context(
    *,
    pack: ResearchPlannerContextPack,
    latest_action_md: str,
    research_skill_guide: str,
) -> str:
    return "\n".join(
        [
            f"Question:\n{pack.question}",
            "",
            f"Campaign goal:\n{_campaign_goal_text(pack.question)}",
            "",
            f"Campaign summary:\n{pack.campaign_summary_md}",
            "",
            f"Hypothesis snapshot:\n{pack.hypothesis_snapshot_md}",
            "",
            f"Recent actions:\n{pack.recent_actions_md}",
            "",
            f"Budget snapshot:\n{pack.budget_snapshot_md}",
            "",
            f"Durable memory summary:\n{pack.durable_memory_summary_md}",
            "",
            f"Chat session context:\n{pack.session_context_md}",
            "",
            f"Human feedback:\n{pack.human_feedback_md}",
            "",
            f"Workspace summary:\n{pack.workspace_summary_md}",
            "",
            f"Latest literature summary:\n{pack.latest_literature_summary_md}",
            "",
            f"Latest experiment summary:\n{pack.latest_experiment_summary_md}",
            "",
            f"Current best answer before sync:\n{pack.current_best_answer_md}",
            "",
            f"Open questions before sync:\n{pack.open_questions_md}",
            "",
            f"Latest completed action to absorb:\n{latest_action_md}",
            "",
            "Research state updater capability guide:",
            research_skill_guide or "(none)",
        ]
    ).strip()
__all__ = [
    "RESEARCH_LEAD_SYSTEM_PROMPT",
    "RESEARCH_STATE_UPDATER_SYSTEM_PROMPT",
    "build_research_lead_context",
    "build_research_state_sync_context",
]
