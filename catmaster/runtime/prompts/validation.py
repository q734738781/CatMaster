from __future__ import annotations

from .catalog import PromptCatalog


def validate_prompt_catalog(catalog: PromptCatalog) -> list[str]:
    errors: list[str] = []
    for bundle in catalog.bundles.values():
        for fragment_id in bundle.fragments:
            if fragment_id not in catalog.fragments:
                errors.append(f"{bundle.id} references missing fragment {fragment_id}")
    return errors
