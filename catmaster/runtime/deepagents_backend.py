from __future__ import annotations

import base64
import os
from pathlib import Path

from deepagents.backends import LocalShellBackend
from deepagents.backends.protocol import FileData, ReadResult

from catmaster.runtime.document_reads import (
    SUPPORTED_DOCUMENT_SUFFIXES,
    native_document_input_decision,
)
from catmaster.tools.base import normalize_workspace_virtual_path


_DEEPAGENTS_074_OFFICE_GAPS = frozenset({".doc", ".docx", ".xls", ".xlsx"})


class CatMasterLocalShellBackend(LocalShellBackend):
    """DeepAgents backend for compact native files plus bounded-document handoff.

    DeepAgents 0.7.4 already emits provider-native ``file`` content blocks for
    binary reads, but its extension table omits Word and Excel formats and it
    reads every binary file in full. Compact documents use the native protocol;
    anything that fails CatMaster's bounded preflight is refused here and read
    as paginated text by ``BoundedDocumentReadMiddleware`` instead.
    """

    def _resolve_path(self, key: str) -> Path:
        return super()._resolve_path(normalize_workspace_virtual_path(key))

    def read(
        self,
        file_path: str,
        offset: int = 0,
        limit: int = 2000,
    ) -> ReadResult:
        suffix = Path(str(file_path or "")).suffix.lower()
        if suffix not in SUPPORTED_DOCUMENT_SUFFIXES:
            return super().read(file_path, offset=offset, limit=limit)

        try:
            resolved_path = self._resolve_path(file_path)
        except (OSError, RuntimeError) as exc:
            return ReadResult(error=f"Error reading file '{file_path}': {exc}")

        try:
            if not resolved_path.exists() or not resolved_path.is_file():
                return ReadResult(error=f"File '{file_path}' not found")
            decision = native_document_input_decision(resolved_path)
            if not decision.allowed:
                return ReadResult(
                    error=(
                        f"Document '{file_path}' requires CatMaster's bounded "
                        "read_file path instead of a native inline file block"
                    )
                )
            if suffix not in _DEEPAGENTS_074_OFFICE_GAPS:
                return super().read(file_path, offset=offset, limit=limit)
            descriptor = os.open(
                resolved_path,
                os.O_RDONLY | getattr(os, "O_NOFOLLOW", 0),
            )
            try:
                with os.fdopen(descriptor, "rb") as handle:
                    descriptor = -1
                    encoded = base64.standard_b64encode(handle.read()).decode("ascii")
            finally:
                if descriptor >= 0:
                    os.close(descriptor)
            return ReadResult(
                file_data=FileData(content=encoded, encoding="base64")
            )
        except OSError as exc:
            return ReadResult(error=f"Error reading file '{file_path}': {exc}")


__all__ = ["CatMasterLocalShellBackend"]
