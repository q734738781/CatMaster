# Date: 2026-07-24
# Agent: Codex
# Purpose: Exercise the proposer -> executor -> evidence judge -> controller workflow.
# Input/Output: Reads OpenAlex or an offline fixture and can save the final lean state.

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path

REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
if str(REPOSITORY_ROOT) not in sys.path:
    sys.path.insert(0, str(REPOSITORY_ROOT))

from catmaster.research.hypothesis_engine import (  # noqa: E402
    Band,
    EvidenceEffect,
    EvidenceJudgment,
    ExecutionPacket,
    ExecutionLane,
    Hypothesis,
    HypothesisEngine,
    HypothesisEngineState,
    HypothesisPlan,
    VerificationAction,
)
from catmaster.research.hypothesis_engine.openalex import (  # noqa: E402
    LiteratureRecord,
    search_openalex,
)


TARGET_DOI = "https://doi.org/10.1021/jacs.4c11276"
TARGET_QUERY = (
    '"Atomically Isolated Pd Sites" acetate protonation-regulated mechanism'
)


def initial_proposer_output() -> HypothesisPlan:
    """Deterministic fixture for hypothesis_proposer structured output."""

    return HypothesisPlan(
        hypotheses=[
            {
                "id": "h-water-mediated",
                "claim": (
                    "Atomically isolated Pd promotes acetate mainly by activating water "
                    "and regulating protonation on neighboring Cu."
                ),
                "rationale": (
                    "Isolated Pd can alter interfacial proton transfer without becoming "
                    "the carbon-intermediate adsorption center."
                ),
                "predictions": [
                    "Primary mechanistic evidence assigns water activation to isolated Pd.",
                    "The carbon intermediate remains associated with neighboring Cu.",
                ],
            },
            {
                "id": "h-direct-pd",
                "claim": (
                    "Atomically isolated Pd promotes acetate by directly adsorbing and "
                    "reducing the carbon intermediate."
                ),
                "rationale": (
                    "Pd can bind carbon intermediates and could therefore act as the direct "
                    "reduction center."
                ),
                "predictions": [
                    "Primary mechanistic evidence assigns the carbon intermediate to Pd.",
                    "Water activation is not the decisive promotional role.",
                ],
            },
            {
                "id": "h-system-scope",
                "claim": (
                    "Any supported mechanism is limited to atomically isolated Pd on the "
                    "reported Cu system and should not be generalized to all Pd-Cu catalysts."
                ),
                "rationale": (
                    "Ensemble size and the identity of neighboring Cu sites can change both "
                    "adsorption and proton-transfer behavior."
                ),
                "predictions": [
                    "The resolved primary source describes isolated Pd with neighboring Cu.",
                    "The source does not claim a universal mechanism for all Pd-Cu structures.",
                ],
            },
        ],
        actions=[
            {
                "id": "a-exact-literature",
                "executor": ExecutionLane.LITERATURE,
                "question": (
                    "Does the primary study assign isolated Pd a water-mediated role or "
                    "a direct carbon-reduction role?"
                ),
                "task": (
                    f"Search OpenAlex for {TARGET_QUERY!r} from 2023 onward. Resolve the "
                    "exact primary paper and return its title, abstract mechanism, and DOI."
                ),
                "target_hypotheses": ["h-water-mediated", "h-direct-pd"],
                "decision_rule": (
                    "Water activation on Pd with carbon chemistry on neighboring Cu supports "
                    "h-water-mediated and opposes h-direct-pd. Direct carbon reduction on Pd "
                    "does the reverse. Missing mechanistic assignment is inconclusive."
                ),
                "information_value": Band.HIGH,
                "cost": Band.LOW,
            },
            {
                "id": "a-explicit-water-dft",
                "executor": ExecutionLane.EXPERIMENT,
                "question": "Which atomistic pathway is favored under explicit water?",
                "task": (
                    "Run a managed explicit-water pathway comparison only if cheaper evidence "
                    "leaves the two mechanism hypotheses unresolved."
                ),
                "target_hypotheses": ["h-water-mediated", "h-direct-pd"],
                "decision_rule": (
                    "A lower free-energy water-mediated route supports h-water-mediated; a "
                    "direct Pd-bound carbon route supports h-direct-pd; overlapping uncertainty "
                    "is inconclusive."
                ),
                "prerequisite_action_ids": ["a-exact-literature"],
                "information_value": Band.HIGH,
                "cost": Band.HIGH,
            },
        ],
    )


def scope_revision_output() -> HypothesisPlan:
    """Deterministic fixture for a later hypothesis_proposer revision."""

    return HypothesisPlan(
        actions=[
            {
                "id": "a-scope-check",
                "executor": ExecutionLane.WORKSPACE,
                "question": "What material and mechanism scope does the resolved source support?",
                "task": (
                    "Read the resolved title and abstract, then identify the narrowest material "
                    "scope justified by the source."
                ),
                "target_hypotheses": ["h-system-scope"],
                "decision_rule": (
                    "Language limited to atomically isolated Pd with neighboring Cu supports "
                    "h-system-scope. An explicit universal Pd-Cu claim opposes it. Otherwise the "
                    "scope remains inconclusive."
                ),
                "prerequisite_action_ids": ["a-exact-literature"],
                "information_value": Band.MEDIUM,
                "cost": Band.LOW,
            }
        ]
    )


def offline_record() -> LiteratureRecord:
    return LiteratureRecord(
        title=(
            "Atomically Isolated Pd Sites Promote Electrochemical CO Reduction "
            "to Acetate through a Protonation-Regulated Mechanism"
        ),
        publication_year=2024,
        abstract=(
            "The study assigns isolated Pd a water-activation role. Proton transfer "
            "promotes conversion of CO adsorbed on neighboring Cu into a COH intermediate, "
            "rather than assigning Pd as the direct carbon reduction center."
        ),
        source=TARGET_DOI,
    )


@dataclass(frozen=True)
class DemoExecutionResult:
    """Scientific content handed from an executor to the evidence judge."""

    action_id: str
    summary: str
    source: str
    content: str


class DemoExecutor:
    """No-HPC execution fixture that returns evidence without judging hypotheses."""

    def __init__(self, *, offline: bool) -> None:
        self.offline = offline
        self.record: LiteratureRecord | None = None

    def execute(self, packet: ExecutionPacket) -> DemoExecutionResult:
        if packet.action_id == "a-exact-literature":
            return self._execute_literature(packet.action_id)
        if packet.action_id == "a-scope-check":
            return self._execute_scope_check(packet.action_id)
        raise RuntimeError(f"the no-HPC demo cannot execute {packet.action_id}")

    def _execute_literature(self, action_id: str) -> DemoExecutionResult:
        records = (
            [offline_record()]
            if self.offline
            else search_openalex(TARGET_QUERY, from_year=2023)
        )
        self.record = next(
            (
                record
                for record in records
                if "atomically isolated pd sites" in record.title.lower()
                and "acetate" in record.title.lower()
            ),
            None,
        )
        if self.record is None:
            raise RuntimeError("the targeted primary paper was not returned")
        return DemoExecutionResult(
            action_id=action_id,
            summary=(
                f"Resolved primary record: {self.record.title}. "
                f"Abstract: {self.record.abstract}"
            ),
            source=self.record.source,
            content=f"{self.record.title}\n{self.record.abstract}",
        )

    def _execute_scope_check(self, action_id: str) -> DemoExecutionResult:
        if self.record is None:
            raise RuntimeError("scope check has no resolved source")
        return DemoExecutionResult(
            action_id=action_id,
            summary=(
                "The resolved title and abstract describe atomically isolated Pd acting "
                "with neighboring Cu in the reported catalyst."
            ),
            source=self.record.source,
            content=f"{self.record.title}\n{self.record.abstract}",
        )


class DemoEvidenceJudge:
    """Independent deterministic fixture for the typed evidence-judge role."""

    def judge(
        self,
        packet: ExecutionPacket,
        result: DemoExecutionResult,
    ) -> EvidenceJudgment:
        if result.action_id != packet.action_id:
            raise RuntimeError("executor returned evidence for the wrong verification")
        if packet.action_id == "a-exact-literature":
            return self._judge_literature(result)
        if packet.action_id == "a-scope-check":
            return self._judge_scope(result)
        raise RuntimeError(f"the demo judge does not recognize {packet.action_id}")

    @staticmethod
    def _judge_literature(result: DemoExecutionResult) -> EvidenceJudgment:
        text = result.content.lower()
        water_on_pd = "water" in text and "neighboring cu" in text
        if water_on_pd:
            effects = [
                EvidenceEffect(
                    hypothesis_id="h-water-mediated",
                    verdict="supports",
                    reason="The reported assignment matches both discriminating predictions.",
                ),
                EvidenceEffect(
                    hypothesis_id="h-direct-pd",
                    verdict="opposes",
                    reason="The carbon intermediate is assigned to neighboring Cu rather than Pd.",
                ),
            ]
            summary = (
                "The primary study assigns isolated Pd a water-activation and "
                "protonation-regulation role for carbon chemistry on neighboring Cu."
            )
        else:
            effects = [
                EvidenceEffect(
                    hypothesis_id=hypothesis_id,
                    verdict="inconclusive",
                    reason="The returned title and abstract do not resolve the decision rule.",
                )
                for hypothesis_id in ("h-water-mediated", "h-direct-pd")
            ]
            summary = "The returned literature record does not distinguish the mechanisms."
        return EvidenceJudgment(
            action_id=result.action_id,
            summary=summary,
            source=result.source,
            effects=effects,
        )

    @staticmethod
    def _judge_scope(result: DemoExecutionResult) -> EvidenceJudgment:
        text = result.content.lower()
        scoped_system = "atomically isolated pd" in text and "neighboring cu" in text
        return EvidenceJudgment(
            action_id=result.action_id,
            summary=(
                "The resolved source supports isolated Pd acting with neighboring Cu and "
                "does not establish a universal mechanism for all Pd-Cu catalysts."
                if scoped_system
                else "The returned source does not establish the proposed material scope."
            ),
            source=result.source,
            effects=[
                EvidenceEffect(
                    hypothesis_id="h-system-scope",
                    verdict="supports" if scoped_system else "inconclusive",
                    reason=(
                        "Both material identity and ensemble scope are explicit in the source."
                        if scoped_system
                        else "The available content does not resolve the scope prediction."
                    ),
                )
            ],
        )


class DemoScientificPipeline:
    """Coordinator fixture that keeps execution and evidence judgment separate."""

    def __init__(self, *, offline: bool) -> None:
        self.executor = DemoExecutor(offline=offline)
        self.judge = DemoEvidenceJudge()
        self.packets: list[dict] = []

    def __call__(self, packet: ExecutionPacket) -> EvidenceJudgment:
        self.packets.append(packet.model_dump(mode="json"))
        result = self.executor.execute(packet)
        return self.judge.judge(packet, result)


def build_state(plan: HypothesisPlan) -> HypothesisEngineState:
    return HypothesisEngineState(
        question=(
            "Why do atomically isolated Pd sites promote electrochemical CO reduction "
            "to acetate on Cu?"
        ),
        hypotheses=[
            Hypothesis.model_validate(item.model_dump()) for item in plan.hypotheses
        ],
        actions=[
            VerificationAction.model_validate(item.model_dump()) for item in plan.actions
        ],
    )


def run_ranked_actions(
    engine: HypothesisEngine,
    pipeline: DemoScientificPipeline,
    *,
    max_steps: int,
) -> list[str]:
    attempted: list[str] = []
    for _ in range(max_steps):
        action = engine.select_next()
        if action is None:
            break
        packet = engine.advance(action.id)
        attempted.append(packet.action_id)
        try:
            judgment = pipeline(packet)
            engine.record_result(
                packet.action_id,
                outcome="completed",
                judgment=judgment,
            )
        except Exception as exc:
            engine.record_result(
                packet.action_id,
                outcome="failed",
                failure_reason=str(exc),
            )
    return attempted


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--offline", action="store_true")
    parser.add_argument("--output", type=Path)
    args = parser.parse_args()

    initial_plan = initial_proposer_output()
    engine = HypothesisEngine(build_state(initial_plan))
    pipeline = DemoScientificPipeline(offline=args.offline)

    print("HYPOTHESIS PROPOSER: initial plan")
    print(initial_plan.model_dump_json(indent=2))
    first_actions = run_ranked_actions(engine, pipeline, max_steps=4)
    print("CONTROLLER AFTER FIRST JUDGMENT")
    print(json.dumps(engine.controller_snapshot(), indent=2))
    if engine.controller_snapshot()["status"] != "needs_hypothesis_revision":
        raise RuntimeError("demo expected a separate hypothesis revision stage")

    revision = scope_revision_output()
    print("HYPOTHESIS PROPOSER: evidence-driven revision")
    print(revision.model_dump_json(indent=2))
    engine.extend(
        actions=[
            VerificationAction.model_validate(item.model_dump())
            for item in revision.actions
        ]
    )
    second_actions = run_ranked_actions(engine, pipeline, max_steps=4)

    final = engine.state.model_dump(mode="json")
    print("EXECUTED ACTIONS")
    print(json.dumps(first_actions + second_actions, indent=2))
    print("FINAL CONTROLLER")
    print(json.dumps(engine.controller_snapshot(), indent=2))
    print("FINAL SCIENTIFIC STATE")
    print(json.dumps(final, indent=2))

    if engine.controller_snapshot()["status"] != "complete":
        raise RuntimeError("demo did not resolve the planned hypotheses")
    if any(action.id == "a-explicit-water-dft" and action.status != "planned" for action in engine.state.actions):
        raise RuntimeError("the DFT branch should remain unexecuted after its targets resolve")
    if args.output:
        args.output.parent.mkdir(parents=True, exist_ok=True)
        args.output.write_text(
            json.dumps(final, ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
