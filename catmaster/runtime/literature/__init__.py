from .context_pack import LiteratureContextPack
from .depth_policy import resolve_depth
from .models import (
    FindInPageResult,
    InPageMatch,
    LiteratureEvidenceRow,
    PaperRecord,
    PaperSearchHit,
    PublicPageSnapshot,
    PublicWebHit,
    PublicWebSearchResult,
    ResearchDepth,
)
from .openalex_client import OpenAlexClient
from .online_search_adapter import OnlineSearchAdapter
from .semanticscholar_client import SemanticScholarClient
from .store import LiteratureStore
from .subagent import LiteratureSubagent
from .synthesizer import synthesize_deep_report, synthesize_standard
from .tools import RunLiteratureResearchInput, run_literature_research

__all__ = [
    "ResearchDepth",
    "PaperRecord",
    "PaperSearchHit",
    "PublicWebHit",
    "PublicWebSearchResult",
    "PublicPageSnapshot",
    "InPageMatch",
    "FindInPageResult",
    "LiteratureEvidenceRow",
    "LiteratureContextPack",
    "OpenAlexClient",
    "SemanticScholarClient",
    "OnlineSearchAdapter",
    "LiteratureStore",
    "LiteratureSubagent",
    "resolve_depth",
    "synthesize_standard",
    "synthesize_deep_report",
    "RunLiteratureResearchInput",
    "run_literature_research",
]
