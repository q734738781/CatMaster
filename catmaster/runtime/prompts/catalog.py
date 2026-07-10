from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any

from .models import PromptBundle, PromptFragment

try:
    import yaml  # type: ignore
except Exception:  # pragma: no cover
    yaml = None


def _hash_text(text: str) -> str:
    return "sha256:" + hashlib.sha256(text.encode("utf-8")).hexdigest()


def _read_frontmatter_markdown(path: Path) -> tuple[dict[str, Any], str, str]:
    text = path.read_text(encoding="utf-8")
    lines = text.splitlines()
    if not lines or lines[0].strip() != "---":
        return {}, text, text
    end = None
    for index, line in enumerate(lines[1:], start=1):
        if line.strip() == "---":
            end = index
            break
    if end is None:
        return {}, text, text
    raw_frontmatter = "\n".join(lines[1:end])
    body = "\n".join(lines[end + 1 :]).strip()
    if yaml is not None:
        loaded = yaml.safe_load(raw_frontmatter) or {}
        frontmatter = dict(loaded) if isinstance(loaded, dict) else {}
    else:
        frontmatter = {}
        for raw in raw_frontmatter.splitlines():
            key, sep, value = raw.partition(":")
            if sep and not raw.startswith((" ", "\t")):
                frontmatter[key.strip()] = value.strip().strip("'\"")
    return frontmatter, body, text


def _read_yaml(path: Path) -> tuple[dict[str, Any], str]:
    text = path.read_text(encoding="utf-8")
    if yaml is not None:
        loaded = yaml.safe_load(text) or {}
    else:
        loaded = json.loads(text)
    if not isinstance(loaded, dict):
        raise ValueError(f"Prompt bundle must be a mapping: {path}")
    return dict(loaded), text


class PromptCatalog:
    def __init__(self, root: Path | str | None = None) -> None:
        self.root = Path(root) if root is not None else Path(__file__).resolve().parents[2] / "prompts"
        self.fragments: dict[str, PromptFragment] = {}
        self.bundles: dict[str, PromptBundle] = {}

    @classmethod
    def load_default(cls) -> "PromptCatalog":
        catalog = cls()
        catalog.load()
        return catalog

    def load(self) -> None:
        self.fragments.clear()
        self.bundles.clear()
        for path in sorted(self.root.rglob("*.md")):
            frontmatter, body, raw = _read_frontmatter_markdown(path)
            if frontmatter.get("kind") != "prompt_fragment":
                continue
            fragment_id = str(frontmatter.get("id") or "").strip()
            if not fragment_id:
                raise ValueError(f"Prompt fragment missing id: {path}")
            self.fragments[fragment_id] = PromptFragment(
                id=fragment_id,
                path=str(path),
                body=body,
                frontmatter=frontmatter,
                source_hash=_hash_text(raw),
            )
        for path in sorted(self.root.rglob("*.yaml")):
            data, raw = _read_yaml(path)
            if data.get("kind") != "prompt_bundle":
                continue
            bundle_id = str(data.get("id") or "").strip()
            fragments = [str(item) for item in list(data.get("fragments") or []) if str(item or "").strip()]
            if not bundle_id or not fragments:
                raise ValueError(f"Prompt bundle missing id or fragments: {path}")
            self.bundles[bundle_id] = PromptBundle(
                id=bundle_id,
                path=str(path),
                fragments=fragments,
                frontmatter=data,
                source_hash=_hash_text(raw),
            )
