from __future__ import annotations

from catmaster.llm.config import LiteratureRuntimeConfig, LLMProfile

from .models import ResearchDepth

_ALLOWED_DEPTHS: tuple[ResearchDepth, ...] = ("none", "quick", "standard", "focused", "deep_report")
_DEPTH_ORDER: dict[ResearchDepth, int] = {
    "none": 0,
    "quick": 1,
    "standard": 2,
    "focused": 3,
    "deep_report": 4,
}
def resolve_depth(
    user_query: str,
    *,
    requested_depth: str = "auto",
    role: str | None = None,
    matched_skills: Iterable[str] | None = None,
    flags: dict | None = None,
    config: LiteratureRuntimeConfig | None = None,
) -> ResearchDepth:
    settings = config or LLMProfile.from_env_or_file().literature
    req = str(requested_depth or "auto").strip().lower()
    role_text = str(role or "").strip().lower()
    _ = user_query, matched_skills
    if req and req != "auto":
        if req not in _ALLOWED_DEPTHS:
            raise ValueError(f"Unsupported literature depth: {requested_depth}")
        return req  # type: ignore[return-value]

    flags = dict(flags or {})
    preferred_depth = str(
        flags.get("preferred_depth")
        or flags.get("auto_depth")
        or settings.auto_default_depth
    ).strip().lower()
    resolved: ResearchDepth
    if preferred_depth not in _ALLOWED_DEPTHS:
        resolved = settings.auto_default_depth  # type: ignore[assignment]
    else:
        resolved = preferred_depth  # type: ignore[assignment]

    role_cap = settings.role_auto_max.get(role_text)
    if role_cap is not None and _DEPTH_ORDER[resolved] > _DEPTH_ORDER[role_cap]:
        return role_cap  # type: ignore[return-value]
    return resolved


__all__ = ["resolve_depth"]
