from __future__ import annotations

import queue
from typing import Any


SENTINEL = object()


class UIBus:
    def __init__(self, maxsize: int = 200):
        self._queue: queue.Queue[Any] = queue.Queue(maxsize=maxsize)
        self._closed = False

    def emit(self, event: Any) -> bool:
        if self._closed:
            return False
        try:
            self._queue.put_nowait(event)
            return True
        except queue.Full:
            level = getattr(event, "level", "info")
            if level in {"warning", "error"}:
                try:
                    self._queue.get_nowait()
                except queue.Empty:
                    return False
                try:
                    self._queue.put_nowait(event)
                    return True
                except queue.Full:
                    return False
            return False

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        try:
            self._queue.put_nowait(SENTINEL)
        except queue.Full:
            try:
                self._queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._queue.put_nowait(SENTINEL)
            except queue.Full:
                pass

    def get(self, timeout: float = 0.1) -> Any:
        try:
            return self._queue.get(timeout=timeout)
        except queue.Empty:
            return None


__all__ = ["UIBus", "SENTINEL"]
