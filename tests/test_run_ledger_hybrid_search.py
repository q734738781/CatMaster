from __future__ import annotations

import asyncio

from catmaster.runtime.run_ledger.hybrid_search import HybridRunLedgerSearcher
from catmaster.runtime.run_ledger.models import RunSearchHit


class _StubStore:
    def search_sparse(self, project_id, query, limit, lane=None, status=None):
        _ = (project_id, query, limit, lane, status)
        return [
            RunSearchHit(
                run_id="run_sparse_top",
                project_id="project_ws_demo",
                lane="standard",
                status="done",
                score=1.0,
                source="sparse",
            ),
            RunSearchHit(
                run_id="run_shared",
                project_id="project_ws_demo",
                lane="standard",
                status="done",
                score=0.8,
                source="sparse",
            ),
        ]


class _StubVector:
    def search(self, project_id, query_vector, limit, lane=None, status=None):
        _ = (project_id, query_vector, limit, lane, status)
        return [
            RunSearchHit(
                run_id="run_shared",
                project_id="project_ws_demo",
                lane="standard",
                status="done",
                score=0.9,
                source="dense",
            ),
            RunSearchHit(
                run_id="run_dense_only",
                project_id="project_ws_demo",
                lane="standard",
                status="done",
                score=0.7,
                source="dense",
            ),
        ]


class _StubEmbeddings:
    async def aembed_query(self, text):
        _ = text
        return [0.1, 0.2, 0.3]


def test_hybrid_rrf_fuses_sparse_and_dense() -> None:
    searcher = HybridRunLedgerSearcher(
        run_ledger_store=_StubStore(),  # type: ignore[arg-type]
        vector_index=_StubVector(),  # type: ignore[arg-type]
        embeddings=_StubEmbeddings(),  # type: ignore[arg-type]
        rrf_k=20,
    )
    hits = asyncio.run(
        searcher.asearch(
            query="find adsorption result",
            project_id="project_ws_demo",
            limit=5,
        )
    )
    assert hits
    run_ids = [item.run_id for item in hits]
    assert "run_shared" in run_ids
    assert "run_sparse_top" in run_ids
    assert "run_dense_only" in run_ids
