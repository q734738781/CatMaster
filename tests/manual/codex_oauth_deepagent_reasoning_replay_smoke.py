"""Pure DeepAgents stress test for tool use and Responses history replay.

This script intentionally makes live model calls. It imports no CatMaster code.
Run it from the repository root with the CatMaster environment:

    conda run -n catmaster python \
      tests/manual/codex_oauth_deepagent_reasoning_replay_smoke.py

The first turn performs a constrained optimization through serial tool calls and
returns a ToolStrategy structured response. The second turn manually replays the
complete first-turn message list and changes one constraint. This exercises the
same Responses-API history path that must preserve encrypted reasoning items when
the model uses ``store=False``.
"""

from __future__ import annotations

import argparse
import json
import time
from typing import Any

from deepagents import create_deep_agent
from langchain.agents.structured_output import ToolStrategy
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langchain_openai.chat_models.codex import _ChatOpenAICodex
from pydantic import BaseModel, Field


CANDIDATES: dict[str, dict[str, int | str]] = {
    "A": {"cost": 4, "score": 8, "risk": 3, "domain": "data"},
    "B": {"cost": 6, "score": 11, "risk": 2, "domain": "runtime"},
    "C": {"cost": 5, "score": 9, "risk": 4, "domain": "data"},
    "D": {"cost": 7, "score": 13, "risk": 3, "domain": "runtime"},
    "E": {"cost": 3, "score": 5, "risk": 1, "domain": "docs"},
    "F": {"cost": 4, "score": 7, "risk": 2, "domain": "tests"},
}

LOOKUPS: list[str] = []
VERIFICATIONS: list[dict[str, Any]] = []


class SequentialCodex(_ChatOpenAICodex):
    """Keep every tool result on a distinct model turn for replay coverage."""

    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)


class SelectionResult(BaseModel):
    """Verified optimal selection for the active constraints."""

    selected: list[str] = Field(description="Exactly three candidate names")
    total_cost: int
    total_score: int
    total_risk: int
    explanation: str = Field(description="A concise explanation of optimality")


@tool
def lookup_candidate(name: str) -> str:
    """Return cost, score, risk, and domain for one named candidate A through F."""

    normalized = name.strip().upper()
    LOOKUPS.append(normalized)
    print(f"TOOL lookup_candidate name={normalized!r}", flush=True)
    if normalized not in CANDIDATES:
        return json.dumps({"error": f"unknown candidate {normalized!r}"})
    return json.dumps(
        {"name": normalized, **CANDIDATES[normalized]},
        ensure_ascii=False,
        sort_keys=True,
    )


@tool
def verify_selection(selected: list[str], budget: int) -> str:
    """Verify a three-candidate selection against budget, risk, and domain rules."""

    normalized = [name.strip().upper() for name in selected]
    known = len(normalized) == 3 and len(set(normalized)) == 3 and all(
        name in CANDIDATES for name in normalized
    )
    if known:
        total_cost = sum(int(CANDIDATES[name]["cost"]) for name in normalized)
        total_score = sum(int(CANDIDATES[name]["score"]) for name in normalized)
        total_risk = sum(int(CANDIDATES[name]["risk"]) for name in normalized)
        domains = {str(CANDIDATES[name]["domain"]) for name in normalized}
    else:
        total_cost = total_score = total_risk = 0
        domains = set()
    payload = {
        "selected": normalized,
        "budget": budget,
        "total_cost": total_cost,
        "total_score": total_score,
        "total_risk": total_risk,
        "valid": known
        and total_cost <= budget
        and total_risk <= 8
        and "data" in domains
        and "runtime" in domains,
    }
    VERIFICATIONS.append(payload)
    print(f"TOOL verify_selection payload={payload!r}", flush=True)
    return json.dumps(payload, ensure_ascii=False, sort_keys=True)


def _message_diagnostics(messages: list[Any]) -> list[dict[str, Any]]:
    diagnostics: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if not isinstance(message, AIMessage):
            continue
        blocks = list(message.content_blocks or [])
        reasoning = [
            block
            for block in blocks
            if isinstance(block, dict) and block.get("type") == "reasoning"
        ]
        diagnostics.append(
            {
                "message_index": index,
                "tool_names": [
                    call.get("name") for call in list(message.tool_calls or [])
                ],
                "content_block_types": [
                    block.get("type")
                    for block in blocks
                    if isinstance(block, dict)
                ],
                "reasoning_items": [
                    {
                        "keys": sorted(block),
                        "has_id": bool(block.get("id")),
                        "direct_encrypted_key": "encrypted_content" in block,
                        "direct_encrypted_length": len(
                            str(block.get("encrypted_content") or "")
                        ),
                        "extras_keys": sorted(block.get("extras") or {}),
                        "extras_encrypted_length": len(
                            str(
                                (block.get("extras") or {}).get(
                                    "encrypted_content"
                                )
                                or ""
                            )
                        ),
                    }
                    for block in reasoning
                ],
                "usage_metadata": message.usage_metadata,
                "response_metadata_keys": sorted(message.response_metadata),
                "additional_kwargs_keys": sorted(message.additional_kwargs),
            }
        )
    return diagnostics


def _invoke(agent: Any, messages: list[Any], label: str) -> dict[str, Any]:
    started = time.perf_counter()
    print(f"PHASE {label} START", flush=True)
    result: dict[str, Any] | None = None
    reported_ai_messages = 0
    for state in agent.stream(
        {"messages": messages},
        config={"recursion_limit": 40},
        stream_mode="values",
    ):
        result = state
        state_messages = list(state.get("messages") or [])
        diagnostics = _message_diagnostics(state_messages)
        if len(diagnostics) > reported_ai_messages:
            for diagnostic in diagnostics[reported_ai_messages:]:
                print(
                    "INTERMEDIATE_AI "
                    + json.dumps(
                        diagnostic,
                        ensure_ascii=False,
                        sort_keys=True,
                    ),
                    flush=True,
                )
            reported_ai_messages = len(diagnostics)
    if result is None:
        raise RuntimeError(f"{label} produced no graph state")
    elapsed = time.perf_counter() - started
    output_messages = list(result.get("messages") or [])
    structured = result.get("structured_response")
    print(
        f"PHASE {label} PASS elapsed_s={elapsed:.3f} messages={len(output_messages)}",
        flush=True,
    )
    print(
        "STRUCTURED "
        + json.dumps(
            structured.model_dump() if isinstance(structured, BaseModel) else structured,
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    print(
        "MESSAGE_DIAGNOSTICS "
        + json.dumps(
            _message_diagnostics(output_messages),
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return result


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default="gpt-5.6-sol")
    parser.add_argument("--reasoning-effort", default="high")
    parser.add_argument(
        "--include-encrypted-content",
        action="store_true",
        help=(
            "Explicitly request reasoning.encrypted_content for the documented "
            "store=False replay path. The default run leaves include unset."
        ),
    )
    args = parser.parse_args()

    model = SequentialCodex(
        model=args.model,
        reasoning={"effort": args.reasoning_effort, "summary": "auto"},
        verbosity="medium",
        include=(
            ["reasoning.encrypted_content"]
            if args.include_encrypted_content
            else None
        ),
    )
    print(
        "MODEL "
        + json.dumps(
            {
                "class": type(model).__name__,
                "model": model.model_name,
                "store": model.store,
                "streaming": model.streaming,
                "include": model.include,
                "output_version": model.output_version,
                "reasoning": model.reasoning,
            },
            sort_keys=True,
        ),
        flush=True,
    )

    agent = create_deep_agent(
        model=model,
        tools=[lookup_candidate, verify_selection],
        system_prompt=(
            "Solve the constrained optimization exactly. Use lookup_candidate to read "
            "candidates and verify_selection to check the solution. After the verifier "
            "confirms the solution, call the bound SelectionResult tool to submit it. "
            "SelectionResult is an authorized output tool, separate from the two domain "
            "tools. Never invent tool results."
        ),
        response_format=ToolStrategy(SelectionResult, handle_errors=False),
        name="pure_deepagent_reasoning_replay_smoke",
    )

    first = _invoke(
        agent,
        [
            HumanMessage(
                content=(
                    "Independently call lookup_candidate exactly once for each of A, B, "
                    "C, D, E, and F, in that order. Then choose exactly three candidates "
                    "with total cost <= 17 and total risk <= 8, including at least one "
                    "data candidate and at least one runtime candidate. Maximize total "
                    "score. Call verify_selection exactly once on the proposed optimum "
                    "before returning the structured result."
                )
            )
        ],
        "complex-first-turn",
    )

    second = _invoke(
        agent,
        [
            *list(first.get("messages") or []),
            HumanMessage(
                content=(
                    "Now lower the budget to 14. Reuse the complete lookup results from "
                    "the preceding turn; do not call lookup_candidate again. Recompute a "
                    "globally score-optimal valid set of exactly three, call "
                    "verify_selection exactly once with budget 14, and return a new "
                    "structured result."
                )
            ),
        ],
        "manual-full-history-replay",
    )

    first_structured = first.get("structured_response")
    second_structured = second.get("structured_response")
    passed = (
        isinstance(first_structured, SelectionResult)
        and first_structured.total_score == 32
        and isinstance(second_structured, SelectionResult)
        and second_structured.total_score == 26
        and LOOKUPS == list(CANDIDATES)
        and len(VERIFICATIONS) == 2
        and all(item["valid"] for item in VERIFICATIONS)
    )
    print(
        "SUMMARY "
        + json.dumps(
            {
                "passed": passed,
                "lookups": LOOKUPS,
                "verification_count": len(VERIFICATIONS),
                "verification_valid": [item["valid"] for item in VERIFICATIONS],
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        flush=True,
    )
    return 0 if passed else 1


if __name__ == "__main__":
    raise SystemExit(main())
