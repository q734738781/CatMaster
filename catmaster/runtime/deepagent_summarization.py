"""Restore retained summaries and count native media context before compaction."""
from copy import copy
from math import ceil
from deepagents.middleware.summarization import SummarizationMiddleware
from langchain.agents.middleware.types import ModelRequest
from langchain_core.messages import AIMessage, HumanMessage


class RestoringSummarizationMiddleware(SummarizationMiddleware):
    """Keep DeepAgents compaction, including for already imported checkpoints.

    DeepAgents 0.7.11 declares SummarizationEvent.summary_message as a
    HumanMessage, but its LastValue channel retains a dict after Agent Server
    Command.update JSON transport. Unlike the root messages channel, this
    nested field has no message reducer. Restore it at the model-call boundary
    so input=None resumes work too, without re-importing or rewriting history.
    """

    @property
    def name(self) -> str:
        # DeepAgents replaces core middleware by name. Keep exactly one
        # summarizer in the original stack position, not a second layer.
        return "SummarizationMiddleware"

    def _count_tokens(self, messages, system_message, tools):
        estimated = super()._count_tokens(messages, system_message, tools)
        # LangChain's approximate counter treats native file.data/base64 as
        # prose. A pair of small XLSX inputs in ExampleResearch consequently cost
        # 204K estimated tokens even after the provider reported 56K for the
        # entire request. Conversely, fixed per-image estimates miss actual
        # high-detail usage, and the openai/openai-codex provider alias mismatch
        # bypasses upstream reported-usage triggers. Use actual same-model
        # prefix usage plus the unconsumed tail as a lower bound for text, or
        # replace the estimate when consumed media distorts its prefix count.
        # Keep current system and tools as additional headroom because they
        # can change between calls.
        # Unmeasured files and model changes retain upstream estimation. This
        # supplies the trigger and the retention calibration below.
        latest = next((i for i in range(len(messages) - 1, -1, -1)
                       if isinstance(messages[i], AIMessage)), None)
        if latest is None:
            return estimated
        previous = messages[latest]
        usage = previous.usage_metadata or {}
        reported = usage.get("total_tokens")
        model_name = getattr(self.model, "model_name", None) or getattr(self.model, "model", None)
        previous_model = previous.response_metadata.get("model_name") or previous.response_metadata.get("model")
        if not model_name or previous_model != model_name or not isinstance(reported, int) or reported <= 0:
            return estimated
        has_consumed_media = any(
            block.get("type") in {"image", "image_url"} or (block.get("type") == "file" and (
                block.get("source_type") == "base64" or block.get("base64")
                or (isinstance(block.get("file"), dict) and block["file"].get("file_data"))
            ))
            for message in messages[:latest]
            if isinstance(message.content, list)
            for block in message.content if isinstance(block, dict)
        )
        measured = reported + super()._count_tokens(messages[latest + 1:], system_message, tools)
        return measured if has_consumed_media else max(estimated, measured)

    def _determine_cutoff_index(self, messages):
        """Use one token scale for the trigger and the retained suffix.

        LangChain 1.3's full counter scales reported usage (capped at 1.25),
        but its suffix counter disables that scaling. A measured 277K-token
        history can consequently keep its entire 104K-estimated tail under a
        105K retention budget, summarizing only the old summary on every step.
        Calibrate suffix estimates against this history's trigger count. This
        is an estimate, not provider tokenization of each individual image.
        Keep native safe AI/tool boundaries and history/media offloading.
        A local helper copy avoids sharing per-request calibration across
        concurrent invocations of this middleware.
        """
        helper = copy(self._lc_helper)
        if helper.keep[0] not in {"tokens", "fraction"} or not messages:
            return helper._determine_cutoff_index(messages)
        partial = helper._partial_token_counter
        baseline = partial(messages)
        if baseline <= 0:
            return helper._determine_cutoff_index(messages)
        scale = self._count_tokens(messages, None, None) / baseline

        def calibrated(suffix):
            return ceil(partial(suffix) * scale)

        helper.token_counter = calibrated
        helper._partial_token_counter = calibrated
        return helper._determine_cutoff_index(messages)

    @staticmethod
    def _restore_request(request: ModelRequest) -> ModelRequest:
        event = request.state.get("_summarization_event")
        if not isinstance(event, dict) or not isinstance(event.get("summary_message"), dict):
            return request
        return request.override(state={
            **request.state,
            "_summarization_event": {
                **event,
                "summary_message": HumanMessage.model_validate(event["summary_message"]),
            },
        })

    def wrap_model_call(self, request, handler):
        return super().wrap_model_call(self._restore_request(request), handler)

    async def awrap_model_call(self, request, handler):
        return await super().awrap_model_call(self._restore_request(request), handler)
