"""Opt-in, run-scoped capture of Codex requests after SDK serialization."""

from __future__ import annotations

from contextlib import contextmanager
from contextvars import ContextVar
import logging
import json
import os
from threading import Lock
from typing import Any

import httpx

logger = logging.getLogger(__name__)
_active_run: ContextVar[Any] = ContextVar("provider_request_capture_run", default=None)


def _selection(name: str) -> set[str]:
    return {value.strip() for value in os.getenv(name, "").split(",") if value.strip()}


class RequestCaptureSelection:
    """A small HTTP-attempt budget owned by one run's observation handler."""

    def __init__(self, run_id: str) -> None:
        self.enabled = bool(run_id) and run_id in _selection("CATMASTER_CAPTURE_REQUEST_RUN_IDS")
        self.agents = _selection("CATMASTER_CAPTURE_REQUEST_AGENTS")
        self.remaining = 8
        if self.enabled:
            try:
                self.remaining = int(os.getenv("CATMASTER_CAPTURE_REQUEST_LIMIT", "8"))
                if self.remaining <= 0:
                    raise ValueError
            except ValueError:
                logger.warning("Request capture disabled: CATMASTER_CAPTURE_REQUEST_LIMIT must be positive.")
                self.enabled = False
        self._lock = Lock()

    def accepts(self, agent: str) -> bool:
        return self.enabled and self.remaining > 0 and (not self.agents or agent in self.agents)

    def claim(self, agent: str) -> bool:
        with self._lock:
            if not self.accepts(agent):
                return False
            self.remaining -= 1
            return True


@contextmanager
def capture_run(run_manager: Any):
    token = _active_run.set(run_manager)
    try:
        yield
    finally:
        _active_run.reset(token)


def _capture_request(request: Any) -> None:
    """HTTPX request hooks see the prepared body, including extra_body merges.

    Never inspect credentials or query data. Response observation tees chunks.
    Preserve the body as text because the observation store sorts JSON keys.
    """
    manager = _active_run.get()
    if manager is None or request.method != "POST" or not request.url.path.endswith("/responses"):
        return
    for handler in manager.handlers:
        capture = getattr(handler, "capture_provider_request", None)
        if callable(capture):
            try:
                if capture(manager.run_id, request):
                    request.extensions.setdefault("catmaster_response_observers", []).append(
                        (handler, manager.run_id)
                    )
            except Exception as exc:
                # Observation failure must not fail or retry a scientific call.
                logger.warning("Provider request capture failed (%s).", type(exc).__name__)


async def _acapture_request(request: Any) -> None:
    _capture_request(request)


class _ResponseObserver:
    def __init__(self, callbacks, content_decoder=None):
        # Use the installed SDK's SSE grammar, including multi-line data.
        from openai._streaming import SSEDecoder
        self.decoder = SSEDecoder()
        self.pending = b""
        self.callbacks = callbacks
        self.content_decoder = content_decoder

    def feed(self, chunk: bytes) -> None:
        try:
            if self.content_decoder is not None:
                chunk = self.content_decoder.decode(chunk)
            self.pending += chunk
            while b"\n" in self.pending:
                line, self.pending = self.pending.split(b"\n", 1)
                event = self.decoder.decode(line.decode("utf-8").rstrip("\r"))
                if event is None or not event.data or event.data == "[DONE]":
                    continue
                data = json.loads(event.data)
                if data.get("type") in {"response.completed", "response.incomplete", "response.failed"}:
                    for handler, callback_id in self.callbacks:
                        try:
                            handler.capture_provider_response(callback_id, data)
                        except Exception as exc:
                            logger.warning("Provider response capture failed (%s).", type(exc).__name__)
        except Exception as exc:
            logger.warning("Provider response decoding failed (%s).", type(exc).__name__)
            self.pending = b""


class _ObservedSyncStream(httpx.SyncByteStream):
    def __init__(self, stream, observer):
        self.stream, self.observer = stream, observer

    def __iter__(self):
        for chunk in self.stream:
            self.observer.feed(chunk)
            yield chunk

    def close(self):
        self.stream.close()


class _ObservedAsyncStream(httpx.AsyncByteStream):
    def __init__(self, stream, observer):
        self.stream, self.observer = stream, observer

    async def __aiter__(self):
        async for chunk in self.stream:
            self.observer.feed(chunk)
            yield chunk

    async def aclose(self):
        await self.stream.aclose()


def _capture_response(response, *, asynchronous=False):
    callbacks = response.request.extensions.get("catmaster_response_observers", [])
    # The live Codex endpoint can omit Content-Type. This hook is installed
    # only on Codex clients, whose Responses requests always stream.
    if callbacks and response.is_success:
        try:
            _observe_response_stream(response, callbacks, asynchronous=asynchronous)
        except Exception as exc:
            logger.warning("Provider response observation failed (%s).", type(exc).__name__)


def _observe_response_stream(response, callbacks, *, asynchronous):
    if response.is_stream_consumed:
        # Mock/custom transports may return an already buffered response.
        _ResponseObserver(callbacks).feed(response.content)
        return
    stream_type = _ObservedAsyncStream if asynchronous else _ObservedSyncStream
    # HTTPX 0.28 stream wrappers sit before decompression. Construct an
    # independent decoder through its own response so gzip/br streams are
    # observed without advancing the SDK consumer's decoder or buffering it.
    decoder = httpx.Response(response.status_code, headers=response.headers)._get_content_decoder()
    response.stream = stream_type(response.stream, _ResponseObserver(callbacks, decoder))


async def _acapture_response(response):
    _capture_response(response, asynchronous=True)


def configure_capture_clients(kwargs: dict[str, Any]) -> None:
    """Keep OpenAI's HTTP defaults and append hooks to explicitly supplied clients."""
    if not _selection("CATMASTER_CAPTURE_REQUEST_RUN_IDS"):
        return
    from openai import DefaultAsyncHttpxClient, DefaultHttpxClient

    for key, client_type, hook, response_hook in (
        ("http_client", DefaultHttpxClient, _capture_request, _capture_response),
        ("http_async_client", DefaultAsyncHttpxClient, _acapture_request, _acapture_response),
    ):
        client = kwargs.get(key)
        if client is None:
            client = client_type()
            kwargs[key] = client
        hooks = client.event_hooks.setdefault("request", [])
        if hook not in hooks:
            hooks.append(hook)
        hooks = client.event_hooks.setdefault("response", [])
        if response_hook not in hooks:
            hooks.append(response_hook)


class ProviderRequestCaptureMixin:
    """Associate HTTP attempts with native LangChain callback IDs.

    LangChain's invoke/ainvoke path does not always forward run_manager into
    _stream. Native stream_events(version='v3') instead drives _iter_v2_events.
    Bind at these two verified boundaries; reset before yielding to consumers.
    """

    def _generate_with_cache(self, messages, stop=None, run_manager=None, **kwargs):
        with capture_run(run_manager):
            return super()._generate_with_cache(messages, stop=stop, run_manager=run_manager, **kwargs)

    async def _agenerate_with_cache(self, messages, stop=None, run_manager=None, **kwargs):
        with capture_run(run_manager):
            return await super()._agenerate_with_cache(messages, stop=stop, run_manager=run_manager, **kwargs)

    def _iter_v2_events(self, messages, *, run_manager, **kwargs):
        events = super()._iter_v2_events(messages, run_manager=run_manager, **kwargs)
        try:
            while True:
                with capture_run(run_manager):
                    try:
                        event = next(events)
                    except StopIteration:
                        return
                yield event
        finally:
            events.close()

    async def _aiter_v2_events(self, messages, *, run_manager, **kwargs):
        events = super()._aiter_v2_events(messages, run_manager=run_manager, **kwargs)
        try:
            while True:
                with capture_run(run_manager):
                    try:
                        event = await anext(events)
                    except StopAsyncIteration:
                        return
                yield event
        finally:
            await events.aclose()
