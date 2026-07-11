from __future__ import annotations

import json
import os
import re
import time
from concurrent.futures import ThreadPoolExecutor
from html import unescape
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx
from pydantic import BaseModel, Field

from catmaster.tools.base import workspace_root

from .openalex_client import OpenAlexClient


_DOI_PATTERN = re.compile(r"10\.\d{4,9}/[-._;()/:A-Z0-9]+", flags=re.IGNORECASE)


class FinalizeCitationsInput(BaseModel):
    """[literature/citations] Resolve and deduplicate the final selected paper identifiers in one deterministic batch."""

    items: list[str] = Field(
        ...,
        min_length=1,
        max_length=200,
        description="Final selected DOI strings or DOI URLs. Pass only papers intended for the final bibliography.",
    )
    output_stem: str = Field(
        "references",
        min_length=1,
        max_length=80,
        description="Filename stem under `notes/literature/`; omit to use `references`.",
    )


def _extract_doi(value: str) -> str:
    match = _DOI_PATTERN.search(str(value or ""))
    if not match:
        return ""
    return match.group(0).rstrip(".,;:)]}>").lower()


def _crossref_record(doi: str) -> dict[str, Any]:
    headers = {"User-Agent": "CatMaster/1.0 (literature citation finalizer)"}
    mailto = str(os.getenv("CROSSREF_MAILTO") or "").strip()
    params = {"mailto": mailto} if mailto else None
    message: dict[str, Any] = {}
    last_error: Exception | None = None
    for attempt in range(4):
        try:
            with httpx.Client(timeout=20.0, follow_redirects=True, headers=headers) as client:
                response = client.get(f"https://api.crossref.org/works/{quote(doi, safe='/')}", params=params)
            if response.status_code == 429 and attempt < 3:
                retry_after = str(response.headers.get("Retry-After") or "").strip()
                delay = float(retry_after) if retry_after.replace(".", "", 1).isdigit() else float(2**attempt)
                time.sleep(min(15.0, max(0.5, delay)))
                continue
            response.raise_for_status()
            message = response.json().get("message") or {}
            break
        except (httpx.RequestError, httpx.HTTPStatusError) as exc:
            last_error = exc
            if isinstance(exc, httpx.HTTPStatusError):
                status_code = int(exc.response.status_code)
                if status_code != 429 and status_code < 500:
                    raise
            if attempt >= 3:
                raise
            time.sleep(float(2**attempt))
    if not message:
        raise RuntimeError(f"Crossref returned no metadata for {doi}: {last_error}")
    title_values = message.get("title") or []
    container_values = message.get("container-title") or []
    date_parts = ((message.get("published-print") or message.get("published-online") or message.get("issued") or {}).get("date-parts") or [[]])
    year = int(date_parts[0][0]) if date_parts and date_parts[0] else None
    authors = []
    for author in message.get("author") or []:
        full = " ".join(str(author.get(key) or "").strip() for key in ("given", "family")).strip()
        if full:
            authors.append(full)
    canonical_doi = str(message.get("DOI") or doi).lower()
    raw_title = str(title_values[0] if title_values else "").strip()
    clean_title = " ".join(unescape(re.sub(r"<[^>]+>", " ", raw_title)).split())
    return {
        "title": clean_title,
        "authors": authors,
        "venue": str(container_values[0] if container_values else "").strip(),
        "year": year,
        "doi": canonical_doi,
        "url": f"https://doi.org/{canonical_doi}",
        "metadata_source": "crossref",
    }


def _openalex_record(doi: str) -> dict[str, Any]:
    paper = OpenAlexClient().get_work(doi)
    return {
        "title": paper.title,
        "authors": list(paper.authors),
        "venue": paper.venue or "",
        "year": paper.year,
        "doi": (paper.doi or doi).lower(),
        "url": paper.url or f"https://doi.org/{doi}",
        "metadata_source": "openalex_fallback",
    }


def _resolve(doi: str) -> tuple[dict[str, Any] | None, str]:
    try:
        return _crossref_record(doi), ""
    except Exception as crossref_exc:
        openalex_exc: Exception | None = None
        for attempt in range(3):
            try:
                return _openalex_record(doi), ""
            except Exception as exc:
                openalex_exc = exc
                if attempt < 2:
                    time.sleep(float(2**attempt))
        return None, f"Crossref: {crossref_exc}; OpenAlex fallback: {openalex_exc}"


def _safe_stem(value: str) -> str:
    stem = re.sub(r"[^A-Za-z0-9._-]+", "_", str(value or "").strip()).strip("._")
    return (stem or "references")[:80]


def _bib_key(record: dict[str, Any], used: set[str]) -> str:
    author = str((record.get("authors") or ["reference"])[0]).split()[-1]
    author = re.sub(r"[^A-Za-z0-9]+", "", author) or "reference"
    year = str(record.get("year") or "nd")
    title_word = next(iter(re.findall(r"[A-Za-z0-9]+", str(record.get("title") or "paper"))), "paper")
    base = f"{author}{year}{title_word}".lower()
    key = base
    suffix = 2
    while key in used:
        key = f"{base}{suffix}"
        suffix += 1
    used.add(key)
    return key


def _bibtex(records: list[dict[str, Any]]) -> str:
    used: set[str] = set()
    entries = []
    for record in records:
        key = _bib_key(record, used)
        fields = {
            "title": str(record.get("title") or ""),
            "author": " and ".join(record.get("authors") or []),
            "journal": str(record.get("venue") or ""),
            "year": str(record.get("year") or ""),
            "doi": str(record.get("doi") or ""),
            "url": str(record.get("url") or ""),
        }
        lines = [f"@article{{{key},"]
        for name, value in fields.items():
            if value:
                escaped = value.replace("{", "\\{").replace("}", "\\}")
                lines.append(f"  {name} = {{{escaped}}},")
        lines.append("}")
        entries.append("\n".join(lines))
    return "\n\n".join(entries) + ("\n" if entries else "")


def _markdown(records: list[dict[str, Any]], unresolved: list[dict[str, str]]) -> str:
    lines = ["# Finalized references", ""]
    for index, record in enumerate(records, start=1):
        authors = ", ".join(record.get("authors") or [])
        year = record.get("year") or "n.d."
        lines.append(
            f"{index}. {authors}. {record.get('title') or '(untitled)'}. "
            f"*{record.get('venue') or '(venue unresolved)'}* ({year}). "
            f"https://doi.org/{record.get('doi')}"
        )
    if unresolved:
        lines.extend(["", "## Unresolved", ""])
        lines.extend(f"- `{item['input']}`: {item['reason']}" for item in unresolved)
    return "\n".join(lines).rstrip() + "\n"


def finalize_citations(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = FinalizeCitationsInput.model_validate(payload)
    selected: dict[str, str] = {}
    unresolved: list[dict[str, str]] = []
    for item in params.items:
        doi = _extract_doi(item)
        if not doi:
            unresolved.append({"input": item, "reason": "No DOI found in selected identifier."})
            continue
        selected.setdefault(doi, item)

    records: list[dict[str, Any]] = []
    selected_rows = list(selected.items())
    with ThreadPoolExecutor(max_workers=min(3, max(1, len(selected_rows)))) as executor:
        resolved_rows = list(executor.map(lambda row: _resolve(row[0]), selected_rows))
    for (doi, original), (record, error) in zip(selected_rows, resolved_rows, strict=True):
        if record is None:
            unresolved.append({"input": original, "reason": error})
        else:
            records.append(record)
    records.sort(key=lambda item: (-(int(item.get("year") or 0)), str(item.get("title") or "").lower()))

    output_dir = workspace_root() / "notes" / "literature"
    output_dir.mkdir(parents=True, exist_ok=True)
    stem = _safe_stem(params.output_stem)
    json_path = output_dir / f"{stem}.json"
    markdown_path = output_dir / f"{stem}.md"
    bib_path = output_dir / f"{stem}.bib"
    payload_data = {
        "resolved": records,
        "unresolved": unresolved,
        "resolved_count": len(records),
        "unresolved_count": len(unresolved),
        "deduplicated_input_count": len(selected),
    }
    json_path.write_text(json.dumps(payload_data, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    markdown_path.write_text(_markdown(records, unresolved), encoding="utf-8")
    bib_path.write_text(_bibtex(records), encoding="utf-8")
    root = workspace_root().resolve()
    paths = [str(path.relative_to(root)).replace("\\", "/") for path in (markdown_path, bib_path, json_path)]
    data = {**payload_data, "files": paths}
    content = (
        f"Finalized {len(records)} unique citation(s); {len(unresolved)} unresolved.\n"
        + "Files:\n"
        + "\n".join(f"- {path}" for path in paths)
    )
    return content, {
        "tool_name": "finalize_citations",
        "data": data,
        "suppress_content_offload_ref": True,
    }


__all__ = ["FinalizeCitationsInput", "finalize_citations"]
