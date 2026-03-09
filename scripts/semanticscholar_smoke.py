from __future__ import annotations

import argparse
import json
import sys

import httpx

from catmaster.runtime.literature import SemanticScholarClient


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Minimal Semantic Scholar smoke test for CatMaster literature module."
    )
    parser.add_argument(
        "--query",
        default="CO adsorption Fe(110)",
        help="Query string to test against Semantic Scholar search.",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=3,
        help="Number of hits requested for the smoke test.",
    )
    args = parser.parse_args(argv)

    client = SemanticScholarClient()
    try:
        hits = client.search_papers(args.query, limit=max(1, int(args.limit)))
    except httpx.HTTPStatusError as exc:
        status = exc.response.status_code
        if status == 429:
            payload = {
                "status": "rate_limited",
                "query": args.query,
                "http_status": status,
                "message": "Semantic Scholar reachable but rate-limited.",
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 2
        if status in {401, 403}:
            payload = {
                "status": "auth_error",
                "query": args.query,
                "http_status": status,
                "message": "Semantic Scholar rejected the request. Check API key or access policy.",
            }
            print(json.dumps(payload, ensure_ascii=False, indent=2))
            return 3
        payload = {
            "status": "http_error",
            "query": args.query,
            "http_status": status,
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 4
    except Exception as exc:
        payload = {
            "status": "error",
            "query": args.query,
            "message": str(exc),
        }
        print(json.dumps(payload, ensure_ascii=False, indent=2))
        return 5

    payload = {
        "status": "ok",
        "query": args.query,
        "hit_count": len(hits),
        "hits": [
            {
                "rank": hit.rank,
                "title": hit.paper.title,
                "year": hit.paper.year,
                "paper_id": hit.paper.paper_id,
                "doi": hit.paper.doi,
                "url": hit.paper.url,
                "landing_page_url": hit.paper.landing_page_url,
                "open_access_pdf_url": hit.paper.open_access_pdf_url,
                "is_open_access": hit.paper.is_open_access,
                "has_abstract": hit.paper.has_abstract,
            }
            for hit in hits
        ],
    }
    print(json.dumps(payload, ensure_ascii=False, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
