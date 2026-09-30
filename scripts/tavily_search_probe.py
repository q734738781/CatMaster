#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import statistics
import sys
from dataclasses import dataclass
from typing import Any

from tavily import TavilyClient


DEFAULT_QUERIES: list[tuple[str, str]] = [
    ("materials_narrow", "CO adsorption Fe(110) benchmark adsorption energy DFT surface science"),
    ("sci_broad", "Fe-N-C single atom catalyst ORR review activity stability benchmark"),
    ("docs_practical", "VASP dipole correction slab calculation LDIPOL IDIPOL best practice"),
]


@dataclass(frozen=True)
class ProbeProfile:
    name: str
    search_depth: str | None
    include_answer: bool
    include_raw_content: bool
    max_results: int


PROFILES: dict[str, ProbeProfile] = {
    "current_lightweight": ProbeProfile(
        name="current_lightweight",
        search_depth=None,
        include_answer=False,
        include_raw_content=False,
        max_results=5,
    ),
    "one_shot_rich": ProbeProfile(
        name="one_shot_rich",
        search_depth="advanced",
        include_answer=True,
        include_raw_content=False,
        max_results=5,
    ),
}


def _snippet(value: Any, limit: int) -> str:
    text = " ".join(str(value or "").split()).strip()
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 1)].rstrip() + "…"


def _score_stats(results: list[dict[str, Any]]) -> dict[str, float | None]:
    values = [float(item["score"]) for item in results if item.get("score") is not None]
    if not values:
        return {"min": None, "median": None, "max": None}
    return {
        "min": min(values),
        "median": statistics.median(values),
        "max": max(values),
    }


def _run_probe(
    client: TavilyClient,
    *,
    query_name: str,
    query: str,
    topic: str,
    profile: ProbeProfile,
    snippet_chars: int,
) -> dict[str, Any]:
    kwargs: dict[str, Any] = {
        "topic": topic,
        "max_results": profile.max_results,
        "include_answer": profile.include_answer,
        "include_raw_content": profile.include_raw_content,
        "include_images": False,
    }
    if profile.search_depth:
        kwargs["search_depth"] = profile.search_depth
    response = client.search(query, **kwargs)
    results = response.get("results") if isinstance(response, dict) else []
    rows: list[dict[str, Any]] = []
    for idx, item in enumerate(results or [], start=1):
        if not isinstance(item, dict):
            continue
        rows.append(
            {
                "rank": idx,
                "score": item.get("score"),
                "title": _snippet(item.get("title") or "Untitled result", 160),
                "url": str(item.get("url") or "").strip(),
                "snippet": _snippet(item.get("content") or item.get("raw_content") or "", snippet_chars),
            }
        )
    return {
        "query_name": query_name,
        "query": query,
        "topic": topic,
        "profile": profile.name,
        "answer": _snippet(response.get("answer") or "", 600) if isinstance(response, dict) else "",
        "result_count": len(rows),
        "score_stats": _score_stats(rows),
        "results": rows,
    }


def _print_text(report: list[dict[str, Any]]) -> None:
    for item in report:
        print("=" * 88)
        print(f"[{item['profile']}] {item['query_name']}")
        print(item["query"])
        answer = str(item.get("answer") or "").strip()
        if answer:
            print(f"answer: {answer}")
        print(f"result_count: {item['result_count']}  score_stats: {item['score_stats']}")
        for row in item["results"]:
            print(f"[{row['rank']}] score={row['score']} title={row['title']}")
            print(f"    url={row['url']}")
            print(f"    snippet={row['snippet']}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Probe Tavily one-shot search quality across topics.")
    parser.add_argument(
        "--profile",
        action="append",
        choices=sorted(PROFILES.keys()),
        help="Probe profile to run. Repeat to compare multiple profiles. Defaults to current_lightweight.",
    )
    parser.add_argument(
        "--query",
        action="append",
        help="Custom query. Repeatable. If omitted, built-in benchmark queries are used.",
    )
    parser.add_argument(
        "--topic",
        default="general",
        choices=("general", "news", "finance"),
        help="Tavily topic for all queries.",
    )
    parser.add_argument(
        "--format",
        default="text",
        choices=("text", "json"),
        help="Output format.",
    )
    parser.add_argument(
        "--snippet-chars",
        type=int,
        default=320,
        help="Snippet truncation length for printed results.",
    )
    args = parser.parse_args()

    if not os.environ.get("TAVILY_API_KEY"):
        print("TAVILY_API_KEY is required.", file=sys.stderr)
        return 2

    profiles = [PROFILES[name] for name in (args.profile or ["current_lightweight"])]
    if args.query:
        queries = [(f"custom_{idx+1}", query) for idx, query in enumerate(args.query)]
    else:
        queries = list(DEFAULT_QUERIES)

    client = TavilyClient()
    report: list[dict[str, Any]] = []
    for profile in profiles:
        for query_name, query in queries:
            report.append(
                _run_probe(
                    client,
                    query_name=query_name,
                    query=query,
                    topic=args.topic,
                    profile=profile,
                    snippet_chars=max(80, int(args.snippet_chars)),
                )
            )

    if args.format == "json":
        print(json.dumps(report, ensure_ascii=False, indent=2))
    else:
        _print_text(report)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
