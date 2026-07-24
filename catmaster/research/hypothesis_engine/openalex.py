from __future__ import annotations

import json
from urllib.parse import urlencode
from urllib.request import Request, urlopen

from pydantic import BaseModel, ConfigDict, Field


class LiteratureRecord(BaseModel):
    model_config = ConfigDict(extra="forbid")

    title: str = Field(..., min_length=1)
    publication_year: int
    abstract: str = ""
    source: str = Field(..., min_length=1)


def _reconstruct_abstract(inverted_index: dict[str, list[int]] | None) -> str:
    if not inverted_index:
        return ""
    positioned_words = [
        (position, word)
        for word, positions in inverted_index.items()
        for position in positions
    ]
    return " ".join(word for _, word in sorted(positioned_words))


def search_openalex(
    query: str,
    *,
    limit: int = 5,
    from_year: int = 2023,
    timeout_seconds: float = 20.0,
) -> list[LiteratureRecord]:
    """Search public OpenAlex metadata without adding a runtime dependency."""

    if not query.strip():
        raise ValueError("query must not be empty")
    if not 1 <= limit <= 25:
        raise ValueError("limit must be between 1 and 25")
    params = urlencode(
        {
            "search": query,
            "filter": f"from_publication_date:{from_year}-01-01",
            "per-page": limit,
        }
    )
    request = Request(
        f"https://api.openalex.org/works?{params}",
        headers={"User-Agent": "CatMaster hypothesis-engine prototype"},
    )
    with urlopen(request, timeout=timeout_seconds) as response:
        payload = json.load(response)

    records: list[LiteratureRecord] = []
    for work in payload.get("results", []):
        title = (work.get("title") or "").strip()
        if not title:
            continue
        doi = (work.get("doi") or "").strip()
        source = doi if doi else (work.get("id") or "").strip()
        if not source:
            continue
        records.append(
            LiteratureRecord(
                title=title,
                publication_year=int(work.get("publication_year") or 0),
                abstract=_reconstruct_abstract(work.get("abstract_inverted_index")),
                source=source,
            )
        )
    return records


__all__ = ["LiteratureRecord", "search_openalex"]
