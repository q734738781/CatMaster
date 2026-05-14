from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import typer
from langchain_core.messages import AIMessage, BaseMessage, ToolMessage

from .core.config import (
    CAMPAIGNS_ROOT,
    DEFAULT_GPU_IDS,
    DEFAULT_MACE_DTYPE,
    DEFAULT_MACE_HEAD,
    DEFAULT_MACE_MODEL,
    DEFAULT_MD_TEMPERATURES_K,
    DEFAULT_STAGE1_ANCHOR_TABLE,
    DEFAULT_STAGE1_BASE_STRUCTURE,
    DEFAULT_STAGE1_ROUND_LIMIT,
    get_campaign_paths,
)
from .core.storage import append_jsonl, ensure_campaign_tree, write_markdown, write_yaml
from .stage1.agent import build_stage1_agent
from .stage2.agent import build_stage2_agent


app = typer.Typer(help="Two-stage DeepAgent CLI for HEO screening campaigns.")


def _default_stage1_prompt(campaign_id: str) -> str:
    return (
        f"Run stage1 for campaign {campaign_id}. Use the fixed S416 NFPP base structure, treat x_total as the total x in "
        "Na4Fe3-xMx(PO4)2(P2O7), keep the active element pool within the hard limit, use reasonably large batch sampling, Monte Carlo sample "
        "multi-dopant Fe-site configurations, evaluate them with unified MACE using lightweight relax by default "
        "(20 steps, fmax 0.05 eV/A), iterate up to 15 rounds when useful, update the element posterior, and export all required stage1 artifacts."
    )


def _default_stage2_prompt(campaign_id: str) -> str:
    return (
        f"Run stage2 for campaign {campaign_id}. Load the stage1 prior, propose exploit/explore candidates, "
        "screen them, update the Pareto archive, and export the required stage2 artifacts."
    )


def _normalize_virtual_path(path: str) -> str:
    if path.startswith("/campaigns/"):
        return str(CAMPAIGNS_ROOT / path.removeprefix("/campaigns/"))
    return path


def _normalize_structured_response(structured):
    if not structured or not hasattr(structured, "artifact_paths"):
        return structured
    normalized = {key: _normalize_virtual_path(value) for key, value in structured.artifact_paths.items()}
    return structured.model_copy(update={"artifact_paths": normalized})


def _coerce_message_text(content: Any) -> str:
    if content is None:
        return ""
    if isinstance(content, str):
        return content.strip()
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, dict):
                text = str(item.get("text") or item.get("content") or "").strip()
            else:
                text = str(item).strip()
            if text:
                parts.append(text)
        return "\n".join(parts).strip()
    return str(content).strip()


def _extract_researcher_results(messages: list[BaseMessage]) -> list[dict[str, str]]:
    tool_call_meta: dict[str, dict[str, str]] = {}
    results: list[dict[str, str]] = []
    for message in messages:
        if isinstance(message, AIMessage):
            for tool_call in message.tool_calls or []:
                if tool_call.get("name") != "task":
                    continue
                args = tool_call.get("args") or {}
                if args.get("subagent_type") != "researcher":
                    continue
                tool_call_meta[str(tool_call.get("id") or "")] = {
                    "description": str(args.get("description") or "").strip(),
                }
            continue
        if not isinstance(message, ToolMessage):
            continue
        tool_call_id = str(message.tool_call_id or "")
        meta = tool_call_meta.get(tool_call_id)
        if not meta:
            continue
        text = _coerce_message_text(message.content)
        if not text:
            continue
        results.append(
            {
                "tool_call_id": tool_call_id,
                "description": meta["description"],
                "content": text,
            }
        )
    return results


def _render_literature_review(entries: list[dict[str, str]]) -> str:
    lines = ["# Literature Review Notes", ""]
    for index, entry in enumerate(entries, start=1):
        lines.append(f"## Research Note {index:03d}")
        description = entry.get("description", "").strip()
        if description:
            lines.append(f"Task: {description}")
            lines.append("")
        lines.append(entry["content"].strip())
        lines.append("")
    return "\n".join(lines).rstrip() + "\n"


def _load_existing_literature_entries(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    entries: list[dict[str, str]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            entries.append(json.loads(line))
    return entries


def _entry_signature(entry: dict[str, str]) -> str:
    description = entry.get("description", "").strip()
    content = entry.get("content", "").strip()
    return f"{description}\n---\n{content}"


def _update_literature_review_files(paths, messages: list[BaseMessage], seen_entry_signatures: set[str]) -> set[str]:
    entries = _extract_researcher_results(messages)
    if not entries:
        return seen_entry_signatures
    jsonl_path = paths.stage1 / "literature_review.jsonl"
    existing_entries = _load_existing_literature_entries(jsonl_path)
    known_signatures = set(seen_entry_signatures) | {_entry_signature(entry) for entry in existing_entries}
    new_entries: list[dict[str, str]] = []
    for entry in entries:
        signature = _entry_signature(entry)
        if signature in known_signatures:
            continue
        new_entries.append(entry)
        known_signatures.add(signature)
        append_jsonl(jsonl_path, entry)
    if not new_entries and existing_entries:
        return known_signatures
    rendered = _render_literature_review(existing_entries + new_entries)
    write_markdown(paths.stage1 / "literature_review.md", rendered)
    write_markdown(paths.shared / "literature_review.md", rendered)
    return known_signatures


@app.command("campaign-init")
def campaign_init(
    campaign_id: str = typer.Argument(..., help="Campaign identifier, e.g. HEO_Na_001."),
    description: str = typer.Option("", help="Optional campaign note."),
) -> None:
    paths = ensure_campaign_tree(campaign_id)
    manifest = {
        "campaign_id": campaign_id,
        "description": description,
        "root": str(paths.root),
        "shared": str(paths.shared),
        "stage1_dir": str(paths.stage1),
        "stage2_dir": str(paths.stage2),
        "mace": {
            "model_path": str(DEFAULT_MACE_MODEL),
            "head": DEFAULT_MACE_HEAD,
            "default_dtype": DEFAULT_MACE_DTYPE,
        },
        "stage1": {
            "base_structure_path": str(DEFAULT_STAGE1_BASE_STRUCTURE),
            "anchor_table_path": str(DEFAULT_STAGE1_ANCHOR_TABLE),
            "supercell": "S416",
            "active_pool_limit": 15,
            "round_limit": DEFAULT_STAGE1_ROUND_LIMIT,
            "x_total": 0.5,
            "num_dopants": 5,
            "sample_size": 96,
            "mc_samples": 8,
            "mc_steps": 25,
            "mc_temperature_ev": 0.03,
            "relax_mode": "light_relax",
            "gpu_id": DEFAULT_GPU_IDS[0] if DEFAULT_GPU_IDS else "0",
            "gpu_ids": DEFAULT_GPU_IDS,
            "light_relax_fmax": 0.05,
            "light_relax_steps": 20,
        },
        "md": {
            "temperatures_K": DEFAULT_MD_TEMPERATURES_K,
            "gpu_ids": DEFAULT_GPU_IDS,
            "timestep_fs": 1.0,
            "steps": 4000,
            "sample_interval": 20,
            "friction": 0.02,
        },
    }
    write_yaml(paths.root / "manifest.yaml", manifest)
    typer.echo(f"Initialized campaign at {paths.root}")


@app.command("stage1-run")
def stage1_run(
    campaign_id: str = typer.Argument(..., help="Existing campaign identifier."),
    task: str = typer.Option("", help="Optional override prompt for stage1."),
    quiet: bool = typer.Option(False, "--quiet", help="Disable live agent update/task logs in the terminal."),
) -> None:
    paths = ensure_campaign_tree(campaign_id)
    agent = build_stage1_agent(paths)
    prompt = task or _default_stage1_prompt(campaign_id)
    result = None
    seen_research_entries: set[str] = set()
    for mode, data in agent.stream(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": f"{campaign_id}:stage1"}},
        stream_mode=["values"],
        print_mode=() if quiet else ["updates", "tasks"],
    ):
        if mode != "values":
            continue
        result = data
        messages = data.get("messages") or []
        seen_research_entries = _update_literature_review_files(paths, messages, seen_research_entries)
    if result is None:
        raise typer.BadParameter("Stage1 run produced no final state.")
    structured = _normalize_structured_response(result.get("structured_response"))
    literature_review_path = paths.shared / "literature_review.md"
    if structured and literature_review_path.exists():
        updated_paths = dict(structured.artifact_paths)
        updated_paths.setdefault("literature_review", str(literature_review_path))
        structured = structured.model_copy(update={"artifact_paths": updated_paths})
    typer.echo(structured.model_dump_json(indent=2) if structured else result["messages"][-1].content)


@app.command("stage2-run")
def stage2_run(
    campaign_id: str = typer.Argument(..., help="Existing campaign identifier."),
    task: str = typer.Option("", help="Optional override prompt for stage2."),
    quiet: bool = typer.Option(False, "--quiet", help="Disable live agent update/task logs in the terminal."),
) -> None:
    paths = get_campaign_paths(campaign_id)
    required_inputs = [
        paths.shared / "element_posterior.csv",
        paths.shared / "top10_pool.yaml",
        paths.shared / "shadow_pool.yaml",
        paths.shared / "summary_for_stage2.md",
    ]
    if not required_inputs[-1].exists():
        required_inputs[-1] = paths.stage1 / "summary_for_stage2.md"
    missing = [str(path) for path in required_inputs if not path.exists()]
    if missing:
        raise typer.BadParameter(
            "Stage1 artifacts are missing. Run `stage1-run` first. Missing: " + ", ".join(missing)
        )
    ensure_campaign_tree(campaign_id)
    agent = build_stage2_agent(paths)
    prompt = task or _default_stage2_prompt(campaign_id)
    result = agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": f"{campaign_id}:stage2"}},
        print_mode=() if quiet else ["updates", "tasks"],
    )
    structured = _normalize_structured_response(result.get("structured_response"))
    typer.echo(structured.model_dump_json(indent=2) if structured else result["messages"][-1].content)
