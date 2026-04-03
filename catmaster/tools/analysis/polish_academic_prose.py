from __future__ import annotations

from pathlib import Path
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from pydantic import BaseModel, ConfigDict, Field

from catmaster.agents.llm_utils import llm_text
from catmaster.llm.config import LLMConfig, LLMProfile
from catmaster.llm.factory import build_chat_model
from catmaster.runtime.tool_output_adapter import CatMasterToolExecutionError
from catmaster.tools.base import resolve_workspace_path, workspace_relpath


class PolishAcademicProseInput(BaseModel):
    """[writing/polish] Read a workspace manuscript file, polish its language with the academic polisher model, and write the result back."""

    model_config = ConfigDict(extra="forbid")

    source_path: str = Field(
        ...,
        description="Workspace-relative manuscript path under files/.",
    )
    output_path: str | None = Field(
        None,
        description="Optional workspace-relative output path. Defaults to in-place overwrite.",
    )
    focus: str | None = Field(
        None,
        description="Optional short polishing objective, e.g. make the prose more concise and journal-like.",
    )


def _resolve_polisher_config() -> LLMConfig:
    profile = LLMProfile.from_env_or_file()
    return profile.config_for_role("academic_polisher")


def _format_rules(path: Path) -> str:
    suffix = path.suffix.lower()
    if suffix == ".tex":
        return (
            "The file is LaTeX. Preserve commands, environments, labels, refs, cites, math, comments, and document structure. "
            "Do not break syntax. Improve prose only."
        )
    if suffix in {".md", ".markdown"}:
        return (
            "The file is Markdown. Preserve headings, lists, tables, fenced blocks, references, and section structure. "
            "Improve prose only."
        )
    return (
        "Preserve the document structure and any markup-like syntax. Improve prose only."
    )


def _build_prompt(*, path: Path, text: str, focus: str | None) -> list[Any]:
    focus_line = str(focus or "").strip()
    human_lines = [
        f"File name: {path.name}",
        _format_rules(path),
        "Stay faithful to the original scientific content.",
        "Do not add new claims, facts, references, figure captions, or numerical values.",
        "Do not delete caveats or uncertainty language unless it is purely redundant wording.",
    ]
    if focus_line:
        human_lines.append(f"Extra polishing goal: {focus_line}")
    human_lines.extend(
        [
            "",
            "Return only the full revised document text, with no commentary or fences.",
            "",
            "Original document:",
            text,
        ]
    )
    return [
        SystemMessage(
            content=(
                "You are an academic prose polisher for scientific manuscripts. "
                "Improve clarity, precision, flow, and journal-style phrasing while preserving technical meaning exactly."
            )
        ),
        HumanMessage(content="\n".join(human_lines).strip()),
    ]


def polish_academic_prose(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    """[writing/polish] Rewrite manuscript prose for clearer academic style while preserving technical meaning."""
    tool_name = "polish_academic_prose"
    try:
        params = PolishAcademicProseInput(**payload)
        source_path = resolve_workspace_path(params.source_path, must_exist=True)
        output_path = (
            resolve_workspace_path(params.output_path, must_exist=False)
            if str(params.output_path or "").strip()
            else source_path
        )
        original_text = source_path.read_text(encoding="utf-8")
        cfg = _resolve_polisher_config()
        model = build_chat_model(cfg)
        response = model.invoke(_build_prompt(path=source_path, text=original_text, focus=params.focus))
        polished_text = llm_text(response).strip()
        if not polished_text:
            raise ValueError("polisher returned empty text")
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(polished_text, encoding="utf-8")
        source_ref = workspace_relpath(source_path)
        output_ref = workspace_relpath(output_path)
        content = "\n".join(
            [
                f"Academic polish applied: {output_ref}",
                f"Model: {cfg.model}",
                f"In-place: {'yes' if output_path == source_path else 'no'}",
            ]
        ).strip()
        artifact = {
            "tool_name": tool_name,
            "data": {
                "model_name": cfg.model,
                "source_path": source_ref,
                "output_path": output_ref,
                "focus": str(params.focus or "").strip(),
                "in_place": output_path == source_path,
                "source_chars": len(original_text),
                "output_chars": len(polished_text),
            },
        }
        return content, artifact
    except CatMasterToolExecutionError:
        raise
    except Exception as exc:
        raise CatMasterToolExecutionError(
            tool_name=tool_name,
            public_message=f"{tool_name} failed: {exc}",
            artifact={
                "tool_name": tool_name,
                "data": {
                    "source_path": payload.get("source_path"),
                    "output_path": payload.get("output_path"),
                },
            },
            error_code="polish_academic_prose_failed",
        ) from exc


__all__ = ["PolishAcademicProseInput", "polish_academic_prose"]
