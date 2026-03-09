from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np

try:
    import faiss  # type: ignore
except Exception:  # pragma: no cover
    faiss = None

from catmaster.runtime.run_ledger.models import RunSearchHit
from catmaster.tools.base import system_root


def _normalize(vec: List[float]) -> np.ndarray:
    arr = np.array(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= 1e-12:
        return arr
    return arr / norm


class VectorIndex:
    """Simple FAISS-backed run-level vector index."""

    def __init__(
        self,
        *,
        system_root_path: Path,
        index_filename: str = "run_ledger.faiss",
        sidecar_filename: str = "run_ledger_vectors.jsonl",
    ) -> None:
        self.system_root = Path(system_root_path).expanduser().resolve()
        self.index_path = self.system_root / index_filename
        self.sidecar_path = self.system_root / sidecar_filename
        self.system_root.mkdir(parents=True, exist_ok=True)

    @classmethod
    def create_default(cls, *, workspace: Path | str | None = None) -> "VectorIndex":
        return cls(system_root_path=system_root(workspace=workspace))

    def _load_records(self) -> List[Dict[str, Any]]:
        if not self.sidecar_path.exists():
            return []
        out: List[Dict[str, Any]] = []
        try:
            lines = self.sidecar_path.read_text(encoding="utf-8").splitlines()
        except Exception:
            return []
        for raw in lines:
            line = raw.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except Exception:
                continue
            if not isinstance(item, dict):
                continue
            rid = str(item.get("run_id") or "").strip()
            vec = item.get("vector")
            meta = item.get("metadata")
            if not rid or not isinstance(vec, list) or not isinstance(meta, dict):
                continue
            try:
                item["vector"] = [float(v) for v in vec]
            except Exception:
                continue
            item["run_id"] = rid
            out.append(item)
        return out

    def _write_records(self, records: List[Dict[str, Any]]) -> None:
        self.sidecar_path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.sidecar_path.with_suffix(".tmp")
        with tmp.open("w", encoding="utf-8") as f:
            for item in records:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")
        tmp.replace(self.sidecar_path)

    def _rebuild_index(self, records: List[Dict[str, Any]]) -> None:
        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu to enable vector index.")
        if not records:
            if self.index_path.exists():
                self.index_path.unlink()
            return
        first_vec = records[0].get("vector")
        if not isinstance(first_vec, list) or not first_vec:
            raise RuntimeError("Cannot build FAISS index: empty vectors.")
        dim = len(first_vec)
        matrix = np.zeros((len(records), dim), dtype=np.float32)
        for i, item in enumerate(records):
            vector = item.get("vector")
            if not isinstance(vector, list) or len(vector) != dim:
                raise RuntimeError("Cannot build FAISS index: inconsistent vector dimensions.")
            matrix[i, :] = _normalize([float(v) for v in vector])
        index = faiss.IndexFlatIP(dim)
        index.add(matrix)
        faiss.write_index(index, str(self.index_path))

    def upsert(self, run_id: str, vector: List[float], metadata: Dict[str, Any]) -> None:
        rid = str(run_id or "").strip()
        if not rid:
            raise ValueError("run_id is required")
        vec = [float(v) for v in vector]
        if not vec:
            raise ValueError("vector must be non-empty")
        records = self._load_records()
        updated = False
        for item in records:
            if str(item.get("run_id") or "") != rid:
                continue
            item["vector"] = vec
            item["metadata"] = dict(metadata or {})
            updated = True
            break
        if not updated:
            records.append({"run_id": rid, "vector": vec, "metadata": dict(metadata or {})})
        self._write_records(records)
        self._rebuild_index(records)

    def _load_index(self) -> tuple[Any, List[Dict[str, Any]]]:
        records = self._load_records()
        if not records:
            return None, records
        if faiss is None:
            raise RuntimeError("faiss is not available. Install faiss-cpu to enable vector index.")
        if not self.index_path.exists():
            self._rebuild_index(records)
        index = faiss.read_index(str(self.index_path))
        return index, records

    def search(
        self,
        project_id: str,
        query_vector: List[float],
        limit: int,
        lane: Optional[str] = None,
        status: Optional[str] = None,
    ) -> List[RunSearchHit]:
        lim = max(1, int(limit))
        proj = str(project_id or "").strip()
        lane_v = str(lane or "").strip() or None
        status_v = str(status or "").strip() or None

        index, records = self._load_index()
        if index is None or not records:
            return []
        q = _normalize([float(v) for v in query_vector]).reshape(1, -1).astype(np.float32)
        k = len(records)
        scores, ids = index.search(q, k)
        out: List[RunSearchHit] = []
        for idx, score in zip(ids[0].tolist(), scores[0].tolist()):
            if idx < 0 or idx >= len(records):
                continue
            rec = records[idx]
            meta = rec.get("metadata") if isinstance(rec, dict) else {}
            if not isinstance(meta, dict):
                meta = {}
            if str(meta.get("project_id") or "") != proj:
                continue
            if lane_v and str(meta.get("lane") or "") != lane_v:
                continue
            if status_v and str(meta.get("status") or "") != status_v:
                continue
            out.append(
                RunSearchHit(
                    run_id=str(rec.get("run_id") or ""),
                    project_id=str(meta.get("project_id") or ""),
                    lane=str(meta.get("lane") or ""),
                    status=str(meta.get("status") or ""),
                    score=float(score),
                    source="dense",
                    request=str(meta.get("request") or ""),
                    answer_summary=str(meta.get("answer_summary") or ""),
                    final_report_relpath=str(meta.get("final_report_relpath") or ""),
                    run_export_relpath=str(meta.get("run_export_relpath") or ""),
                )
            )
            if len(out) >= lim:
                break
        return out


__all__ = ["VectorIndex"]
