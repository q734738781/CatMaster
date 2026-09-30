from __future__ import annotations

import asyncio
import logging
from dataclasses import replace
from typing import Dict, List, Optional

from catmaster.runtime.run_ledger.models import RunSearchHit
from catmaster.runtime.run_ledger.openrouter_embeddings import OpenRouterEmbeddings
from catmaster.runtime.run_ledger.store import RunLedgerStore
from catmaster.runtime.run_ledger.vector_index import VectorIndex

logger = logging.getLogger(__name__)


class HybridRunLedgerSearcher:
    """Hybrid sparse+dense search with reciprocal-rank fusion."""

    def __init__(
        self,
        *,
        run_ledger_store: RunLedgerStore,
        vector_index: VectorIndex,
        embeddings: OpenRouterEmbeddings,
        rrf_k: int = 60,
    ) -> None:
        self.run_ledger_store = run_ledger_store
        self.vector_index = vector_index
        self.embeddings = embeddings
        self.rrf_k = max(1, int(rrf_k))

    @staticmethod
    def _dedupe_hits(hits: List[RunSearchHit]) -> List[RunSearchHit]:
        out: List[RunSearchHit] = []
        seen: set[str] = set()
        for hit in hits:
            rid = str(hit.run_id or "").strip()
            if not rid or rid in seen:
                continue
            seen.add(rid)
            out.append(hit)
        return out

    async def _search_sparse(
        self,
        *,
        project_id: str,
        query: str,
        limit: int,
        lane: Optional[str],
        status: Optional[str],
    ) -> List[RunSearchHit]:
        return await asyncio.to_thread(
            self.run_ledger_store.search_sparse,
            project_id,
            query,
            limit,
            lane,
            status,
        )

    async def _search_dense(
        self,
        *,
        project_id: str,
        query: str,
        limit: int,
        lane: Optional[str],
        status: Optional[str],
    ) -> List[RunSearchHit]:
        query_vec = await self.embeddings.aembed_query(query)
        return await asyncio.to_thread(
            self.vector_index.search,
            project_id,
            query_vec,
            limit,
            lane,
            status,
        )

    def _fuse_rrf(
        self,
        sparse_hits: List[RunSearchHit],
        dense_hits: List[RunSearchHit],
        *,
        limit: int,
    ) -> List[RunSearchHit]:
        sparse = self._dedupe_hits(sparse_hits)
        dense = self._dedupe_hits(dense_hits)

        fused_score: Dict[str, float] = {}
        chosen: Dict[str, RunSearchHit] = {}

        for rank, hit in enumerate(sparse, start=1):
            rid = hit.run_id
            fused_score[rid] = fused_score.get(rid, 0.0) + 1.0 / float(self.rrf_k + rank)
            chosen.setdefault(rid, hit)
        for rank, hit in enumerate(dense, start=1):
            rid = hit.run_id
            fused_score[rid] = fused_score.get(rid, 0.0) + 1.0 / float(self.rrf_k + rank)
            if rid not in chosen or (chosen[rid].source != "sparse"):
                chosen[rid] = hit

        ranked = sorted(fused_score.items(), key=lambda kv: kv[1], reverse=True)
        out: List[RunSearchHit] = []
        for rid, score in ranked[: max(1, int(limit))]:
            base = chosen.get(rid)
            if base is None:
                continue
            out.append(replace(base, source="hybrid", score=float(score)))
        return out

    async def asearch(
        self,
        query: str,
        project_id: str,
        limit: int = 12,
        sparse_k: int = 24,
        dense_k: int = 24,
        lane: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunSearchHit]:
        q = str(query or "").strip()
        if not q:
            return []
        lim = max(1, int(limit))
        sparse_limit = max(lim, int(sparse_k))
        dense_limit = max(lim, int(dense_k))

        sparse_task = asyncio.create_task(
            self._search_sparse(
                project_id=project_id,
                query=q,
                limit=sparse_limit,
                lane=lane,
                status=status,
            )
        )
        dense_task = asyncio.create_task(
            self._search_dense(
                project_id=project_id,
                query=q,
                limit=dense_limit,
                lane=lane,
                status=status,
            )
        )
        sparse_hits: List[RunSearchHit] = []
        dense_hits: List[RunSearchHit] = []
        try:
            sparse_hits = await sparse_task
        except Exception as exc:
            logger.warning("hybrid sparse search failed: %s", exc)
        try:
            dense_hits = await dense_task
        except Exception as exc:
            logger.warning("hybrid dense search failed: %s", exc)

        if sparse_hits and not dense_hits:
            return self._dedupe_hits(sparse_hits)[:lim]
        if dense_hits and not sparse_hits:
            return self._dedupe_hits(dense_hits)[:lim]
        if not sparse_hits and not dense_hits:
            return []
        return self._fuse_rrf(sparse_hits, dense_hits, limit=lim)


__all__ = ["HybridRunLedgerSearcher"]
