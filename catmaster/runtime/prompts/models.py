from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class PromptFragment:
    id: str
    path: str
    body: str
    frontmatter: dict[str, object] = field(default_factory=dict)
    source_hash: str = ""


@dataclass(frozen=True)
class PromptBundle:
    id: str
    path: str
    fragments: list[str]
    frontmatter: dict[str, object] = field(default_factory=dict)
    source_hash: str = ""


@dataclass(frozen=True)
class RenderedPrompt:
    bundle_id: str
    text: str
    fragment_ids: list[str]
    fragment_hashes: dict[str, str]
    bundle_hash: str
    renderer_version: str = "catmaster.prompt_renderer.v1"
