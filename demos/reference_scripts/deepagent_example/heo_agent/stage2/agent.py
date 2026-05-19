from __future__ import annotations

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy

from ..core.agent_utils import build_checkpointer, build_default_middleware, build_store, make_backend
from ..core.config import CampaignPaths
from ..core.llm import build_llm, build_search_subagent
from ..core.schemas import Stage2Result
from .tools import build_stage2_tools


def build_stage2_agent(paths: CampaignPaths):
    system_prompt = f"""
You are the Stage2 agent for campaign {paths.campaign_id}.

Your job is to perform the more expensive screening stage:
- read stage1 posterior files rather than stage1 chat history
- allocate most of the budget to exploit high-probability elements
- keep explicit exploration budget for shadow-pool and uncertainty-reduction candidates
- maintain a Pareto archive over stability, diffusion, barrier, and deformation
- when structure paths are available, prefer the real ASE+MACE tools:
  1. multi-temperature MD across four temperatures to extract diffusion and Arrhenius Ea
  2. MLFF relaxation for ΔV/V

Constraints:
- Do not rewrite stage1 outputs except by referencing them in stage2 reports.
- Use stage1 posterior as a soft prior, never as a hard filter.
- Treat the literature pool and background chemistry facts as an important reference source when prioritizing exploit vs explore candidates or interpreting conflicting signals.
- When you need chemistry insight or external background facts, proactively call the researcher subagent to search the web for focused evidence.
- Use web search for literature precedent, synthesis concerns, transport/mechanism context, competing-phase clues, or known diffusion/structural behavior; do not use it for local files or code facts already present in the workspace.
- Use the literature as a soft prior and interpretation aid, not as a replacement for the explicit stage2 MLFF/MD evidence.
- Finish only after the Pareto archive and both queue yaml files exist.

Required artifacts before you finish:
- pareto_archive.csv and, if supported by the environment, pareto_archive.parquet
- recommended_dft_queue.yaml
- recommended_experiment_queue.yaml
- stage2_full_report.md
- decision_log.jsonl

Return a structured Stage2Result once the artifacts exist.
""".strip()

    return create_deep_agent(
        model=build_llm(),
        tools=build_stage2_tools(paths),
        system_prompt=system_prompt,
        middleware=build_default_middleware(
            {
                "load_stage1_prior": 3,
                "propose_stage2_candidates": 4,
                "evaluate_volume_deformation_with_mlff": 3,
                "run_mlff_md_batch": 4,
                "update_pareto_archive": 4,
                "export_stage2_report": 2,
            }
        ),
        subagents=build_search_subagent(),
        response_format=ToolStrategy(Stage2Result),
        checkpointer=build_checkpointer(),
        store=build_store(),
        backend=make_backend,
        name="stage2_agent",
    )
