from __future__ import annotations

from catmaster.runtime.research import ResearchPlannerContextPack
from catmaster.runtime.research.context_builder import _campaign_goal_text


RESEARCH_LEAD_SYSTEM_PROMPT = """You are CatMaster's campaign-level scientific planner for the research lane.
You do not execute primitive tools yourself.
You must choose exactly one next action:
- RunLiterature
- RunExperiment
- RunWriter
- AskHuman
- Conclude

Operating rules:
- Optimize for a small number of high-value cycles, not endless exploration.
- standard/fast lanes solve one bounded problem; you manage a cross-run hypothesis program.
- Use literature when the next decision needs grounding, conventions, benchmark context, or representative citations.
- Use experiments only for bounded, execution-ready tasks that produce new numerical evidence, structure generation, verification, or other real execution outputs.
- Use writer when the task is to synthesize existing workspace artifacts, memory, prior runs, figures, and already-computed evidence into manuscript text, figures, tables, or evidence maps without launching new expensive calculations.
- anchored policy: do not silently change topic and do not introduce new hypotheses.
- local_expand policy: only propose local variants/ablations tied to existing hypotheses.
- open policy: new hypotheses are allowed, but keep them relevant to the stated question.
- When evidence is incomplete, say so explicitly and keep unsupported parts in open questions.
- Do not use RunExperiment to do mere evidence inventory, manuscript drafting, plotting from existing results, or workspace-only analysis; those belong to RunWriter.
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
            f"Historical context:\n{pack.history_summary_md}",
            "",
            f"Human feedback:\n{pack.human_feedback_md}",
            "",
            f"Reviewed supplemental context:\n{pack.context_review_md}",
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
__all__ = [
    "RESEARCH_LEAD_SYSTEM_PROMPT",
    "build_research_lead_context",
]
