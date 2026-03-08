from __future__ import annotations

from pathlib import Path

from catmaster.agents.graph import GraphRunPolicy
from catmaster.agents.runner_factory import build_graph_runner
from catmaster.tools.base import system_root
from catmaster.ui.reporters import NullReporter, Reporter

from .models import ExperimentBriefModel, ExperimentRunPack, ResearchArtifactRef, ResearchBoard


def build_experiment_child_request(
    *,
    brief: ExperimentBriefModel,
    research_request,
    board: ResearchBoard,
) -> str:
    hypotheses = {
        item.hypothesis_id: item.text
        for item in board.hypotheses
    }
    lines = [
        "You are executing one bounded experiment inside a larger research campaign.",
        "",
        "Research question:",
        board.question,
        "",
        "Selected hypotheses:",
    ]
    if brief.hypothesis_ids:
        lines.extend(
            [f"- {hid}: {hypotheses.get(hid, '(missing hypothesis text)')}" for hid in brief.hypothesis_ids]
        )
    else:
        lines.append("- (none)")
    lines.extend(
        [
            "",
            f"Experiment title:\n{brief.title}",
            "",
            f"Goal:\n{brief.goal}",
            "",
            f"Detailed instructions:\n{brief.task_detail}",
            "",
            "Expected outputs:",
            *([f"- {item}" for item in brief.expected_outputs] or ["- (none)"]),
            "",
            "Stop condition:",
            brief.stop_condition,
            "",
            "Reference hints:",
            *([f"- {item}" for item in brief.reference_hint] or ["- (none)"]),
            "",
            "Rules:",
            "- This is one bounded experiment only.",
            "- Do not expand scope to adjacent hypotheses.",
            "- Do not run literature research.",
            "- You may update MEMORY/** when the run produces reusable facts, files, or open questions.",
            "- Do not modify the project goal for this bounded child run.",
            "- Return a concise final answer with evidence paths.",
        ]
    )
    if hasattr(research_request, "seed_hypotheses"):
        seed_hypotheses = list(getattr(research_request, "seed_hypotheses", []) or [])
        if seed_hypotheses:
            lines.extend(["", "Seed hypotheses from the outer campaign:"])
            lines.extend([f"- {item}" for item in seed_hypotheses])
    return "\n".join(lines).strip()


class ExperimentLaneRunner:
    CHILD_RUN_POLICY = GraphRunPolicy(
        allow_memory_patch=True,
        allow_human_intervention=False,
        enable_literature_tool=False,
        enable_history_prefetch=True,
    )

    def __init__(
        self,
        *,
        workspace: Path,
        llm_profile,
        project_id: str,
        reporter: Reporter | None = None,
    ) -> None:
        self.workspace = Path(workspace).expanduser().resolve()
        self.llm_profile = llm_profile
        self.project_id = project_id
        self.reporter = reporter or NullReporter()

    async def arun(
        self,
        *,
        brief,
        research_request,
        board: ResearchBoard,
    ) -> ExperimentRunPack:
        brief_model = ExperimentBriefModel.model_validate(
            brief.model_dump() if hasattr(brief, "model_dump") else brief
        )
        built = build_graph_runner(
            workspace=self.workspace,
            llm_profile=self.llm_profile,
            reporter=self.reporter,
            run_control=None,
            project_id=self.project_id,
            run_policy=self.CHILD_RUN_POLICY,
            bind_run_control_id=False,
        )
        child_request = build_experiment_child_request(
            brief=brief_model,
            research_request=research_request,
            board=board,
        )
        result = await built.runner.arun(
            child_request,
            lane=brief_model.lane,
            proposal_review=False,
        )
        observations = list(result.get("observations") or [])
        top_observations: list[str] = []
        key_artifacts: list[ResearchArtifactRef] = []
        seen_paths: set[str] = set()
        open_questions: list[str] = []
        for item in observations:
            if not isinstance(item, dict):
                continue
            summary = " ".join(str(item.get("summary") or "").split()).strip()
            if summary and len(top_observations) < 5:
                top_observations.append(summary)
            structured = item.get("structured_result") or {}
            for question in list(structured.get("open_questions") or []):
                text = " ".join(str(question).split()).strip()
                if text and text not in open_questions:
                    open_questions.append(text)
            for artifact in list(item.get("key_artifacts") or []):
                if not isinstance(artifact, dict):
                    continue
                path = str(artifact.get("path") or "").strip()
                if not path or path in seen_paths:
                    continue
                seen_paths.add(path)
                key_artifacts.append(
                    ResearchArtifactRef(
                        path=path,
                        description=str(artifact.get("description") or ""),
                        kind=str(artifact.get("kind") or "output"),
                    )
                )
        run_dir = Path(str(result.get("run_dir") or built.run_context.run_dir))
        experiment_index = 1 + sum(1 for item in board.action_refs if item.kind == "experiment")
        final_report_path = self._system_relpath(result.get("final_report_path"))
        run_export_path = self._system_relpath(result.get("run_export_path"))
        return ExperimentRunPack(
            experiment_id=f"exp_{experiment_index:03d}",
            brief=brief_model,
            run_id=str(result.get("run_id") or built.run_context.run_id),
            run_dir=self._system_relpath(str(run_dir)),
            lane=brief_model.lane,
            status=str(result.get("status") or "failure"),
            summary=str(result.get("summary") or "").strip(),
            top_observations=top_observations,
            key_artifacts=key_artifacts,
            final_report_path=final_report_path or None,
            run_export_path=run_export_path or None,
            child_memory_candidates=list(result.get("pending_memory_updates") or []),
            open_questions=open_questions[:10],
        )

    def _system_relpath(self, raw: str | Path | None) -> str:
        if raw in (None, ""):
            return ""
        path = Path(str(raw)).expanduser().resolve()
        root = system_root(self.workspace).resolve()
        try:
            return str(path.relative_to(root)).replace("\\", "/")
        except Exception:
            return str(path)


__all__ = ["ExperimentLaneRunner", "build_experiment_child_request"]
