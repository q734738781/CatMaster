from __future__ import annotations

import hashlib
import json
import os
import re
from pathlib import Path
from typing import List

import httpx


class OpenRouterEmbeddings:
    """Thin async embeddings client for OpenRouter /api/v1/embeddings."""

    def __init__(
        self,
        *,
        system_root: Path,
        model: str = "openai/text-embedding-3-small",
        api_key: str | None = None,
        base_url: str | None = None,
        timeout_sec: float = 30.0,
        batch_size: int = 64,
    ) -> None:
        self.model = str(model or "").strip() or "openai/text-embedding-3-small"
        self.api_key = str(api_key or os.getenv("OPENROUTER_API_KEY", "")).strip()
        base = str(base_url or os.getenv("OPENROUTER_BASE_URL", "")).strip()
        self.base_url = base or "https://openrouter.ai/api/v1"
        self.timeout_sec = max(1.0, float(timeout_sec))
        self.batch_size = max(1, int(batch_size))

        safe_model = re.sub(r"[^A-Za-z0-9._-]+", "_", self.model)
        self.cache_dir = Path(system_root).expanduser().resolve() / "embedding_cache" / safe_model
        self.cache_dir.mkdir(parents=True, exist_ok=True)

    @staticmethod
    def _sha256(text: str) -> str:
        return hashlib.sha256(text.encode("utf-8")).hexdigest()

    def _cache_key(self, text: str) -> str:
        return self._sha256(f"{self.model}\n{text}")

    def _cache_path(self, key: str) -> Path:
        return self.cache_dir / f"{key}.json"

    def _load_cache(self, text: str) -> List[float] | None:
        key = self._cache_key(text)
        path = self._cache_path(key)
        if not path.exists():
            return None
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return None
        vec = payload.get("embedding") if isinstance(payload, dict) else None
        if not isinstance(vec, list):
            return None
        try:
            return [float(v) for v in vec]
        except Exception:
            return None

    def _save_cache(self, text: str, embedding: List[float]) -> None:
        key = self._cache_key(text)
        path = self._cache_path(key)
        payload = {"model": self.model, "embedding": [float(v) for v in embedding]}
        tmp = path.with_suffix(".tmp")
        tmp.write_text(json.dumps(payload, ensure_ascii=False), encoding="utf-8")
        os.replace(tmp, path)

    def _endpoint(self) -> str:
        base = self.base_url.rstrip("/")
        if base.endswith("/api/v1"):
            return f"{base}/embeddings"
        return f"{base}/api/v1/embeddings"

    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self.api_key:
            headers["Authorization"] = f"Bearer {self.api_key}"
        referer = os.getenv("OPENROUTER_HTTP_REFERER", "").strip()
        title = os.getenv("OPENROUTER_APP_TITLE", "").strip()
        if referer:
            headers["HTTP-Referer"] = referer
        if title:
            headers["X-Title"] = title
        return headers

    async def _request_embeddings(self, texts: List[str]) -> List[List[float]]:
        if not self.api_key:
            raise RuntimeError("OpenRouter embeddings requires OPENROUTER_API_KEY (or explicit api_key).")
        payload = {"model": self.model, "input": texts}
        timeout = httpx.Timeout(self.timeout_sec)
        async with httpx.AsyncClient(timeout=timeout) as client:
            resp = await client.post(self._endpoint(), headers=self._headers(), json=payload)
        if resp.status_code >= 400:
            snippet = resp.text[:500]
            raise RuntimeError(
                f"OpenRouter embeddings request failed: HTTP {resp.status_code}; body={snippet}"
            )
        try:
            data = resp.json()
        except Exception as exc:
            raise RuntimeError(f"OpenRouter embeddings response is not valid JSON: {exc}") from exc
        rows = data.get("data") if isinstance(data, dict) else None
        if not isinstance(rows, list):
            raise RuntimeError("OpenRouter embeddings response missing data list.")
        pairs: list[tuple[int, list[float]]] = []
        for item in rows:
            if not isinstance(item, dict):
                continue
            idx = int(item.get("index", len(pairs)))
            vec = item.get("embedding")
            if not isinstance(vec, list):
                continue
            try:
                pairs.append((idx, [float(v) for v in vec]))
            except Exception:
                continue
        if not pairs:
            raise RuntimeError("OpenRouter embeddings response contains no usable embedding vectors.")
        pairs.sort(key=lambda x: x[0])
        vectors = [vec for _, vec in pairs]
        if len(vectors) != len(texts):
            raise RuntimeError(
                f"OpenRouter embeddings vector count mismatch: got {len(vectors)} for {len(texts)} inputs."
            )
        return vectors

    async def aembed_documents(self, texts: List[str]) -> List[List[float]]:
        clean = [str(item or "") for item in texts]
        if not clean:
            return []

        out: List[List[float] | None] = [None] * len(clean)
        missing: List[tuple[int, str]] = []
        for idx, text in enumerate(clean):
            cached = self._load_cache(text)
            if cached is None:
                missing.append((idx, text))
            else:
                out[idx] = cached

        if missing:
            for start in range(0, len(missing), self.batch_size):
                batch_items = missing[start : start + self.batch_size]
                batch_texts = [item[1] for item in batch_items]
                vectors = await self._request_embeddings(batch_texts)
                for (row_idx, row_text), vec in zip(batch_items, vectors):
                    out[row_idx] = vec
                    self._save_cache(row_text, vec)

        final: List[List[float]] = []
        for idx, vec in enumerate(out):
            if vec is None:
                raise RuntimeError(f"Embedding generation failed at index {idx}.")
            final.append(vec)
        return final

    async def aembed_query(self, text: str) -> List[float]:
        vectors = await self.aembed_documents([text])
        return vectors[0] if vectors else []


__all__ = ["OpenRouterEmbeddings"]
