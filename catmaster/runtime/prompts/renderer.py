from __future__ import annotations

import hashlib
from collections import UserDict
from typing import Any

from .catalog import PromptCatalog
from .models import RenderedPrompt


class _StrictFormatMap(UserDict):
    def __missing__(self, key: str) -> str:
        raise KeyError(f"Missing prompt render variable: {key}")


class PromptRenderer:
    def __init__(self, catalog: PromptCatalog) -> None:
        self.catalog = catalog

    def render(self, bundle_id: str, variables: dict[str, Any] | None = None) -> RenderedPrompt:
        bundle = self.catalog.bundles[bundle_id]
        variables = dict(variables or {})
        parts: list[str] = []
        fragment_hashes: dict[str, str] = {}
        for fragment_id in bundle.fragments:
            fragment = self.catalog.fragments[fragment_id]
            fragment_hashes[fragment_id] = fragment.source_hash
            parts.append(fragment.body.format_map(_StrictFormatMap(variables)))
        text = "\n\n".join(part.strip() for part in parts if part.strip()).strip() + "\n"
        bundle_hash = "sha256:" + hashlib.sha256(
            (bundle.source_hash + "\n" + "\n".join(fragment_hashes.values()) + "\n" + text).encode("utf-8")
        ).hexdigest()
        return RenderedPrompt(
            bundle_id=bundle.id,
            text=text,
            fragment_ids=list(bundle.fragments),
            fragment_hashes=fragment_hashes,
            bundle_hash=bundle_hash,
        )
