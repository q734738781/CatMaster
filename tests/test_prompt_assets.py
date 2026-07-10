from __future__ import annotations

from pathlib import Path

from catmaster.runtime.prompts import PromptCatalog, PromptRenderer
from catmaster.runtime.prompts.validation import validate_prompt_catalog


def test_package_prompt_assets_render_without_workspace_projection() -> None:
    catalog = PromptCatalog.load_default()

    assert "catmaster.entrypoint.research" in catalog.bundles
    assert validate_prompt_catalog(catalog) == []

    rendered = PromptRenderer(catalog).render("catmaster.entrypoint.research")

    assert rendered.bundle_id == "catmaster.entrypoint.research"
    assert "CatMaster research coordinator" in rendered.text
    assert rendered.bundle_hash.startswith("sha256:")
    assert all("/.deepagents/" not in Path(fragment.path).as_posix() for fragment in catalog.fragments.values())
