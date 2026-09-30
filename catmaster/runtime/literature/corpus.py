from __future__ import annotations

import json
import re
import sqlite3
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Literal

from bs4 import BeautifulSoup
from pydantic import BaseModel, ConfigDict, Field

from catmaster.tools.base import resolve_scoped_path, system_root, workspace_root


class IngestLiteratureFilesInput(BaseModel):
    """[literature/corpus] Ingest authorized local papers into the workspace evidence corpus."""

    paths: list[str] = Field(
        ...,
        min_length=1,
        max_length=100,
        description=(
            "Workspace-relative full-text paths to index when durable corpus "
            "search is useful. PDF, HTML, XML/JATS, Markdown, and other readable "
            "text are accepted."
        ),
    )
    doi_by_path: dict[str, str] = Field(
        default_factory=dict,
        description="Optional workspace-path to DOI mapping; omit or pass `{}` when unknown.",
    )


class QueryLiteratureCorpusInput(BaseModel):
    """[literature/corpus] Locate claim-relevant spans in ingested full text with continuation."""

    model_config = ConfigDict(extra="forbid")

    query_mode: Literal["any", "all", "phrase", "fts"] = Field("any", description="any ORs literal tokens; all ANDs them; phrase matches one literal phrase; fts passes native SQLite FTS5 MATCH syntax unchanged.")
    source_paths: list[str] = Field(default_factory=list, description="Restrict to these exact ingested source_path handles; empty searches all documents.")
    query: str = Field(..., min_length=2, description="Focused claim, mechanism, material, or comparison to retrieve evidence for.")
    page_size: int = Field(
        20,
        ge=1,
        le=100,
        description="Number of locator hits in this page; use next_offset to continue.",
    )
    offset: int = Field(
        0,
        ge=0,
        description="Locator offset; leave 0 for the first page and use next_offset to continue.",
    )


@dataclass(frozen=True)
class ExtractedChunk:
    page: int
    section: str
    text: str


def _database_path() -> Path:
    path = system_root() / "literature" / "corpus.sqlite"
    path.parent.mkdir(parents=True, exist_ok=True)
    return path


def _connect() -> sqlite3.Connection:
    connection = sqlite3.connect(_database_path())
    connection.row_factory = sqlite3.Row
    connection.execute("PRAGMA journal_mode=WAL")
    connection.execute(
        """
        CREATE TABLE IF NOT EXISTS documents (
            source_path TEXT PRIMARY KEY,
            doi TEXT NOT NULL DEFAULT '',
            title TEXT NOT NULL DEFAULT '',
            page_count INTEGER NOT NULL DEFAULT 0,
            ingested_at TEXT NOT NULL
        )
        """
    )
    columns = {
        str(row["name"])
        for row in connection.execute("PRAGMA table_info(documents)")
    }
    if "file_hash" in columns or "document_id" in columns:
        with connection:
            connection.execute("DROP TABLE IF EXISTS documents_v2")
            connection.execute(
                """
                CREATE TABLE documents_v2 (
                    source_path TEXT PRIMARY KEY,
                    doi TEXT NOT NULL DEFAULT '',
                    title TEXT NOT NULL DEFAULT '',
                    page_count INTEGER NOT NULL DEFAULT 0,
                    ingested_at TEXT NOT NULL
                )
                """
            )
            connection.execute(
                """
                INSERT OR REPLACE INTO documents_v2(
                    source_path, doi, title, page_count, ingested_at
                )
                SELECT source_path, doi, title, page_count, ingested_at
                FROM documents
                """
            )
            connection.execute("DROP TABLE IF EXISTS chunks_legacy")
            connection.execute(
                """
                CREATE TEMP TABLE chunks_legacy AS
                SELECT source_path, doi, page, section, text FROM chunks
                """
            )
            connection.execute("DROP TABLE chunks")
            connection.execute("DROP TABLE documents")
            connection.execute(
                "ALTER TABLE documents_v2 RENAME TO documents"
            )
            connection.execute(
                """
                CREATE VIRTUAL TABLE chunks USING fts5(
                    source_path UNINDEXED,
                    doi UNINDEXED,
                    page UNINDEXED,
                    section UNINDEXED,
                    text,
                    tokenize='unicode61 remove_diacritics 2'
                )
                """
            )
            connection.execute(
                """
                INSERT INTO chunks(source_path, doi, page, section, text)
                SELECT source_path, doi, page, section, text
                FROM chunks_legacy
                """
            )
            connection.execute("DROP TABLE chunks_legacy")
    connection.execute(
        """
        CREATE VIRTUAL TABLE IF NOT EXISTS chunks USING fts5(
            source_path UNINDEXED,
            doi UNINDEXED,
            page UNINDEXED,
            section UNINDEXED,
            text,
            tokenize='unicode61 remove_diacritics 2'
        )
        """
    )
    return connection


def _normalize_text(text: str) -> str:
    lines = [re.sub(r"\s+", " ", line).strip() for line in str(text or "").splitlines()]
    return "\n".join(line for line in lines if line).strip()


def _split_text(text: str, *, page: int, max_chars: int = 3_500, overlap: int = 300) -> list[ExtractedChunk]:
    normalized = _normalize_text(text)
    if not normalized:
        return []
    chunks: list[ExtractedChunk] = []
    start = 0
    while start < len(normalized):
        end = min(len(normalized), start + max_chars)
        if end < len(normalized):
            boundary = normalized.rfind("\n", start + max_chars // 2, end)
            if boundary > start:
                end = boundary
        body = normalized[start:end].strip()
        if body:
            chunks.append(ExtractedChunk(page=page, section=f"page {page}", text=body))
        if end >= len(normalized):
            break
        start = max(start + 1, end - overlap)
    return chunks


def _extract_pdf(path: Path) -> tuple[str, list[ExtractedChunk]]:
    from pypdf import PdfReader

    reader = PdfReader(str(path))
    title = str((reader.metadata or {}).get("/Title") or path.stem).strip()
    chunks: list[ExtractedChunk] = []
    for page_index, page in enumerate(reader.pages, start=1):
        chunks.extend(_split_text(page.extract_text() or "", page=page_index))
    return title, chunks


def _extract_text_file(path: Path) -> tuple[str, list[ExtractedChunk]]:
    raw = path.read_text(encoding="utf-8", errors="replace")
    stripped = raw.lstrip()
    looks_like_markup = stripped.startswith("<") and ">" in stripped[:500]
    if looks_like_markup:
        parser = (
            "xml"
            if path.suffix.lower() in {".xml", ".nxml", ".jats"}
            or stripped.startswith("<?xml")
            else "html.parser"
        )
        try:
            soup = BeautifulSoup(raw, parser)
        except Exception:
            soup = BeautifulSoup(raw, "html.parser")
        title_node = soup.find(["article-title", "title"])
        title = (
            title_node.get_text(" ", strip=True)
            if title_node is not None
            else path.stem
        )
        raw = soup.get_text("\n")
    else:
        if "\x00" in raw:
            raise ValueError("File is binary rather than readable full text.")
        title = path.stem
    return title, _split_text(raw, page=1)


def _extract(path: Path) -> tuple[str, list[ExtractedChunk]]:
    with path.open("rb") as handle:
        header = handle.read(16)
    if header.startswith(b"%PDF-"):
        return _extract_pdf(path)
    return _extract_text_file(path)


def _write_manifest(connection: sqlite3.Connection) -> str:
    output = workspace_root() / "notes" / "literature" / "acquisition_manifest.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    rows = [
        dict(row)
        for row in connection.execute(
            "SELECT source_path, doi, title, page_count, ingested_at "
            "FROM documents ORDER BY source_path"
        )
    ]
    output.write_text(json.dumps(rows, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    return str(output.relative_to(workspace_root())).replace("\\", "/")


def ingest_literature_files(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = IngestLiteratureFilesInput.model_validate(payload)
    root = workspace_root().resolve()
    results: list[dict[str, Any]] = []
    errors: list[dict[str, str]] = []
    connection = _connect()
    try:
        for raw_path in params.paths:
            source_path = str(raw_path or "").strip()
            try:
                path = resolve_scoped_path(raw_path, "files", must_exist=True)
                source_path = str(path.relative_to(root)).replace("\\", "/")
                title, chunks = _extract(path)
                if not chunks:
                    raise ValueError("No extractable text was found.")
                doi = str(
                    params.doi_by_path.get(raw_path)
                    or params.doi_by_path.get(source_path)
                    or ""
                ).strip()
                with connection:
                    connection.execute(
                        "DELETE FROM chunks WHERE source_path = ?",
                        (source_path,),
                    )
                    connection.execute(
                        "DELETE FROM documents WHERE source_path = ?",
                        (source_path,),
                    )
                    connection.execute(
                        "INSERT INTO documents(source_path, doi, title, page_count, ingested_at) VALUES (?, ?, ?, ?, ?)",
                        (
                            source_path,
                            doi,
                            title,
                            max(chunk.page for chunk in chunks),
                            datetime.now(timezone.utc).isoformat(),
                        ),
                    )
                    connection.executemany(
                        "INSERT INTO chunks(source_path, doi, page, section, text) VALUES (?, ?, ?, ?, ?)",
                        [
                            (
                                source_path,
                                doi,
                                chunk.page,
                                chunk.section,
                                chunk.text,
                            )
                            for chunk in chunks
                        ],
                    )
                results.append(
                    {
                        "path": source_path,
                        "title": title,
                        "page_count": max(chunk.page for chunk in chunks),
                        "chunk_count": len(chunks),
                        "status": "ingested",
                    }
                )
            except Exception as exc:
                errors.append(
                    {
                        "path": source_path,
                        "error": str(exc).strip() or type(exc).__name__,
                    }
                )
        manifest_path = _write_manifest(connection)
    finally:
        connection.close()

    data = {
        "status": "ok" if not errors else ("partial" if results else "error"),
        "documents": results,
        "errors": errors,
        "manifest_path": manifest_path,
    }
    lines = [
        f"Literature corpus: {len(results)} document(s) processed, "
        f"{len(errors)} skipped."
    ]
    lines.extend(f"- {item['status']}: {item['path']}" for item in results)
    lines.extend(f"- skipped: {item['path']} — {item['error']}" for item in errors)
    lines.append(f"Manifest: {manifest_path}")
    return "\n".join(lines), {
        "tool_name": "ingest_literature_files",
        "data": data,
        "suppress_content_offload_ref": True,
    }


def _fts_query(text: str, mode: str = "any") -> str:
    if mode == "fts":
        return text
    if mode == "phrase":
        return '"' + text.replace('"', '""') + '"'
    tokens = re.findall(r"[\w+.-]+", str(text or ""), flags=re.UNICODE)
    return (" AND " if mode == "all" else " OR ").join('"' + token.replace('"', '""') + '"' for token in tokens)


def query_literature_corpus(payload: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    params = QueryLiteratureCorpusInput.model_validate(payload)
    match = _fts_query(params.query, params.query_mode)
    if not match:
        raise ValueError("Corpus query must contain searchable terms.")
    offset = params.offset
    where = "chunks MATCH ?"
    arguments: list[Any] = [match]
    if params.source_paths:
        where += " AND source_path IN (" + ",".join("?" for _ in params.source_paths) + ")"
        arguments.extend(params.source_paths)
    with _connect() as connection:
        total = int(
            connection.execute(
                f"SELECT COUNT(*) AS count FROM chunks WHERE {where}",
                arguments,
            ).fetchone()["count"]
        )
        rows = connection.execute(
            f"""
            SELECT source_path, doi, page, section,
                   snippet(chunks, 4, '[match]', '[/match]', ' ... ', 30) AS snippet
            FROM chunks
            WHERE {where}
            ORDER BY bm25(chunks), source_path, page, section, rowid
            LIMIT ? OFFSET ?
            """,
            (*arguments, params.page_size, offset),
        ).fetchall()

    evidence = []
    for row in rows:
        evidence.append(
            {
                "source_path": row["source_path"],
                "doi": row["doi"] or "",
                "page": int(row["page"] or 0),
                "section": row["section"] or "",
                "partial": True,
                "snippet": _normalize_text(row["snippet"] or ""),
            }
        )
    next_offset = offset + len(evidence)
    next_offset = next_offset if next_offset < total else 0
    data = {
        "query": params.query,
        "partial": True,
        "evidence": evidence,
        "total_count": total,
        "next_offset": next_offset,
    }
    if evidence:
        lines = [f"Partial corpus locators for: {params.query}"]
        for index, item in enumerate(evidence, start=1):
            lines.append(
                f"[{index}] {item['source_path']} p.{item['page']} "
                f"§{item['section']}\n"
                f"Partial locator snippet: {item['snippet']}"
            )
        if next_offset:
            lines.append(f"Continue with offset: {next_offset}")
        content = "\n\n".join(lines)
    else:
        content = f"No ingested full-text evidence matched: {params.query}"
    return content, {
        "tool_name": "query_literature_corpus",
        "data": data,
        "suppress_content_offload_ref": True,
    }


__all__ = [
    "IngestLiteratureFilesInput",
    "QueryLiteratureCorpusInput",
    "ingest_literature_files",
    "query_literature_corpus",
]
