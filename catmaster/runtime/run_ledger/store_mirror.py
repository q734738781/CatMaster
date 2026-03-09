from __future__ import annotations

import inspect
import logging
from dataclasses import dataclass
from typing import Any, Optional

from catmaster.runtime.run_ledger.models import RunLedgerEntry

logger = logging.getLogger(__name__)


@dataclass
class RunLedgerStoreMirror:
    """Optional mirror to LangGraph BaseStore-like backends.

    This mirror is compatibility-only and is not part of Phase 1 retrieval.
    """

    backend_store: Any = None
    enabled: bool = False

    async def aupsert(self, entry: RunLedgerEntry) -> None:
        if not self.enabled or self.backend_store is None:
            return
        namespace = ("run_ledger", str(entry.project_id or ""))
        key = str(entry.run_id or "")
        value = entry.to_dict()
        if not key:
            return

        put_fn: Optional[Any] = getattr(self.backend_store, "put", None)
        if not callable(put_fn):
            logger.warning("RunLedgerStoreMirror backend has no put method; mirror skipped.")
            return

        try:
            result = put_fn(namespace, key, value)
            if inspect.isawaitable(result):
                await result
        except Exception as exc:
            logger.warning("RunLedgerStoreMirror put failed for run_id=%s: %s", key, exc)


__all__ = ["RunLedgerStoreMirror"]
