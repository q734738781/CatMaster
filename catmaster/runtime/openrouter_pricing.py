from __future__ import annotations

import os
import threading
import time
from typing import Any, Dict, Iterable

try:
    import requests
except Exception:  # pragma: no cover - optional dependency at runtime
    requests = None  # type: ignore[assignment]

_DEFAULT_BASE_URL = "https://openrouter.ai/api/v1"
_CACHE_TTL_SEC = 6 * 60 * 60
_LOCK = threading.Lock()
_MODELS_CACHE: dict[str, Any] = {"ts": 0.0, "models": {}}


def _normalize_model_key(value: str) -> str:
    text = str(value or "").strip()
    return text.lower()


def _candidate_model_ids(model_name: str) -> list[str]:
    raw = str(model_name or "").strip()
    if not raw:
        return []
    out: list[str] = []
    seen: set[str] = set()

    def _push(text: str) -> None:
        key = _normalize_model_key(text)
        if not key or key in seen:
            return
        seen.add(key)
        out.append(text)

    _push(raw)
    if ":" in raw:
        _push(raw.split(":", 1)[0].strip())
    return out


def _base_url() -> str:
    return str(os.getenv("OPENROUTER_BASE_URL") or _DEFAULT_BASE_URL).rstrip("/")


def _headers() -> dict[str, str]:
    headers: dict[str, str] = {"Accept": "application/json"}
    api_key = str(os.getenv("OPENROUTER_API_KEY") or "").strip()
    if api_key:
        headers["Authorization"] = f"Bearer {api_key}"
    return headers


def _to_float(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, bool):
        return float(int(value))
    if isinstance(value, (int, float)):
        return float(value)
    try:
        text = str(value).strip()
        if not text:
            return None
        return float(text)
    except Exception:
        return None


def _extract_pricing(item: dict[str, Any]) -> dict[str, float]:
    pricing_raw = item.get("pricing")
    if not isinstance(pricing_raw, dict):
        return {}
    out: dict[str, float] = {}
    for key, value in pricing_raw.items():
        number = _to_float(value)
        if number is not None:
            out[str(key)] = number
    return out


def _fetch_models_map() -> dict[str, dict[str, float]]:
    if requests is None:
        return {}
    url = f"{_base_url()}/models"
    response = requests.get(url, headers=_headers(), timeout=20)
    response.raise_for_status()
    payload = response.json()
    rows = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        return {}
    models: dict[str, dict[str, float]] = {}
    for item in rows:
        if not isinstance(item, dict):
            continue
        model_id = str(item.get("id") or "").strip()
        if not model_id:
            continue
        pricing = _extract_pricing(item)
        if pricing:
            models[_normalize_model_key(model_id)] = pricing
    return models


def _get_models_map() -> dict[str, dict[str, float]]:
    now = time.time()
    with _LOCK:
        cached = _MODELS_CACHE.get("models")
        cached_ts = float(_MODELS_CACHE.get("ts") or 0.0)
        if isinstance(cached, dict) and cached and now - cached_ts < _CACHE_TTL_SEC:
            return dict(cached)
    try:
        models = _fetch_models_map()
    except Exception:
        models = {}
    with _LOCK:
        if models:
            _MODELS_CACHE["models"] = dict(models)
            _MODELS_CACHE["ts"] = now
        elif isinstance(_MODELS_CACHE.get("models"), dict):
            return dict(_MODELS_CACHE.get("models") or {})
    return dict(models)


def resolve_model_pricing(model_name: str) -> tuple[str | None, dict[str, float]]:
    models = _get_models_map()
    if not models:
        return None, {}
    for candidate in _candidate_model_ids(model_name):
        pricing = models.get(_normalize_model_key(candidate))
        if isinstance(pricing, dict) and pricing:
            return candidate, dict(pricing)
    return None, {}


def pricing_cost(rate: float | None, tokens: int | None) -> float:
    if rate is None or tokens is None:
        return 0.0
    if tokens <= 0:
        return 0.0
    return float(rate) * float(tokens)


def available_pricing_keys(pricing: dict[str, float]) -> list[str]:
    return sorted(k for k, v in pricing.items() if isinstance(v, (int, float)))


__all__ = [
    "available_pricing_keys",
    "pricing_cost",
    "resolve_model_pricing",
]
