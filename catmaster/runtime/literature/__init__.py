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
from .semanticscholar_client import SemanticScholarClient, SemanticScholarRateLimitError
from .store import LiteratureStore
from .subagent import LiteratureSubagent
from .synthesizer import synthesize_deep_report, synthesize_standard
from .tools import (
    FindInPageInput,
    GetOpenAlexRecordInput,
    GetSemanticScholarRecordInput,
    OpenPublicPageInput,
    RecommendSemanticScholarInput,
    RunLiteratureResearchInput,
    SearchOpenAlexInput,
    SearchPublicWebInput,
    SearchSemanticScholarInput,
    find_in_page,
    get_openalex_record,
    get_semantic_scholar_record,
    open_public_page,
    recommend_semantic_scholar,
    run_literature_research,
    search_openalex,
    search_public_web,
    search_semantic_scholar,
)

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
    "SemanticScholarRateLimitError",
    "OnlineSearchAdapter",
    "LiteratureStore",
    "LiteratureSubagent",
    "resolve_depth",
    "synthesize_standard",
    "synthesize_deep_report",
    "RunLiteratureResearchInput",
    "run_literature_research",
    "SearchOpenAlexInput",
    "search_openalex",
    "SearchSemanticScholarInput",
    "search_semantic_scholar",
    "GetOpenAlexRecordInput",
    "get_openalex_record",
    "GetSemanticScholarRecordInput",
    "get_semantic_scholar_record",
    "RecommendSemanticScholarInput",
    "recommend_semantic_scholar",
    "SearchPublicWebInput",
    "search_public_web",
    "OpenPublicPageInput",
    "open_public_page",
    "FindInPageInput",
    "find_in_page",
]
