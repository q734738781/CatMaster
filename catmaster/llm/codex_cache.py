"""Stable Codex cache affinity at the Responses payload boundary."""

from __future__ import annotations

import json
from uuid import NAMESPACE_URL, uuid5

from catmaster.llm.request_capture import _active_run


def apply_cache_affinity(payload: dict, fallback_session: str) -> dict:
    """Keep the key stable across model calls, reconstruction and compaction.

    LangChain OpenAI 1.6.0 forwards explicit keys but creates none for Codex.
    The gateway also uses the HTTP session-id (openai/codex client.rs);
    a body key alone is insufficient in third-party reproductions, including
    NousResearch/hermes-agent#47126. Headers remain SDK transport arguments,
    never fields inside extra_body.
    """
    manager = _active_run.get()
    metadata = getattr(manager, "metadata", None) or {}
    thread = metadata.get("catmaster_thread_id") or metadata.get("thread_id")
    if thread:
        # The final model node UUID changes every graph step. Ancestor task
        # namespaces distinguish concurrently delegated conversations.
        namespace = str(metadata.get("langgraph_checkpoint_ns") or "")
        namespace = namespace.rsplit("|", 1)[0] if "|" in namespace else ""
        scope = json.dumps([str(thread), namespace, metadata.get("lc_agent_name", "")])
        session = str(uuid5(NAMESPACE_URL, "catmaster:codex:" + scope))
    else:
        # Direct model users without a graph keep affinity for this instance.
        session = fallback_session

    extra_body = payload.get("extra_body") or {}
    # SDK extra_body wins at serialization. Respect that precedence, and
    # explicit top-level keys, instead of silently replacing caller policy.
    key = extra_body.get("prompt_cache_key", payload.get("prompt_cache_key"))
    if not key:
        key = session
        payload["prompt_cache_key"] = key
        if "prompt_cache_key" in extra_body:
            payload["extra_body"] = {**extra_body, "prompt_cache_key": key}
    headers = dict(payload.get("extra_headers") or {})
    if not any(name.lower() == "session-id" for name in headers):
        headers["session-id"] = str(key)
    if not any(name.lower() == "thread-id" for name in headers):
        # Match the conversation header, including an explicit caller override.
        # This is a compatibility hint used by third-party Codex clients, not
        # a server guarantee of cache reuse.
        headers["thread-id"] = next(
            value for name, value in headers.items() if name.lower() == "session-id"
        )
    payload["extra_headers"] = headers
    return payload
