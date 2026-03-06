from __future__ import annotations

import logging
from typing import Any

from langchain.agents.middleware.tool_selection import (
    LLMToolSelectorMiddleware,
    _SelectionRequest,
    _create_tool_selection_response,
)
from langchain.agents.middleware.types import ContextT, ModelRequest, ModelResponse, ResponseT
from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import BaseTool

logger = logging.getLogger(__name__)


class SafeLLMToolSelectorMiddleware(LLMToolSelectorMiddleware):
    """LLM tool selector with graceful fallback on invalid or empty selections."""

    def _effective_always_include(self, request: ModelRequest[ContextT]) -> list[str]:
        if not request.tools:
            return []
        available = {
            str(getattr(tool, "name", "") or "")
            for tool in request.tools
            if not isinstance(tool, dict)
        }
        return [name for name in self.always_include if name in available]

    def _prepare_selection_request(
        self, request: ModelRequest[ContextT]
    ) -> _SelectionRequest | None:
        if not request.tools:
            return None

        base_tools = [tool for tool in request.tools if not isinstance(tool, dict)]
        effective_always_include = self._effective_always_include(request)
        available_tools = [
            tool for tool in base_tools if str(getattr(tool, "name", "") or "") not in effective_always_include
        ]
        if not available_tools:
            return None

        system_message = self.system_prompt
        if effective_always_include:
            system_message += (
                "\nIMPORTANT: The following core tools are already available and do not need to be selected: "
                + ", ".join(effective_always_include)
                + "."
            )
        if self.max_tools is not None:
            system_message += (
                f"\nIMPORTANT: List the tool names in order of relevance, "
                f"with the most relevant first. "
                f"If you exceed the maximum number of tools, "
                f"only the first {self.max_tools} will be used."
            )

        last_user_message: HumanMessage
        for message in reversed(request.messages):
            if isinstance(message, HumanMessage):
                last_user_message = message
                break
        else:
            msg = "No user message found in request messages"
            raise AssertionError(msg)

        model = self.model or request.model
        valid_tool_names = [tool.name for tool in available_tools]

        return _SelectionRequest(
            available_tools=available_tools,
            system_message=system_message,
            last_user_message=last_user_message,
            model=model,
            valid_tool_names=valid_tool_names,
        )

    def _process_selection_response(
        self,
        response: dict[str, Any],
        available_tools: list[BaseTool],
        valid_tool_names: list[str],
        request: ModelRequest[ContextT],
    ) -> ModelRequest[ContextT]:
        effective_always_include = set(self._effective_always_include(request))
        selected_tool_names: list[str] = []
        invalid_tool_selections: list[str] = []

        for tool_name in response.get("tools", []):
            if tool_name in effective_always_include:
                continue
            if tool_name not in valid_tool_names:
                invalid_tool_selections.append(str(tool_name))
                continue
            if tool_name not in selected_tool_names and (
                self.max_tools is None or len(selected_tool_names) < self.max_tools
            ):
                selected_tool_names.append(tool_name)

        if invalid_tool_selections:
            logger.warning(
                "Tool selector returned invalid tools %s; falling back to original tool set.",
                invalid_tool_selections,
            )
            return request

        if not selected_tool_names:
            logger.warning(
                "Tool selector returned no usable tool names; falling back to original tool set."
            )
            return request

        selected_tools: list[Any] = [
            tool for tool in available_tools if str(getattr(tool, "name", "") or "") in selected_tool_names
        ]
        always_included_tools: list[Any] = [
            tool
            for tool in (request.tools or [])
            if not isinstance(tool, dict) and str(getattr(tool, "name", "") or "") in effective_always_include
        ]
        provider_tools = [tool for tool in (request.tools or []) if isinstance(tool, dict)]

        combined: list[Any] = []
        seen_names: set[str] = set()
        for tool in [*selected_tools, *always_included_tools, *provider_tools]:
            if isinstance(tool, dict):
                combined.append(tool)
                continue
            name = str(getattr(tool, "name", "") or "")
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            combined.append(tool)
        return request.override(tools=combined)

    def wrap_model_call(self, request, handler):
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return handler(request)
        try:
            type_adapter = _create_tool_selection_response(selection_request.available_tools)
            schema = type_adapter.json_schema()
            structured_model = selection_request.model.with_structured_output(schema)
            response = structured_model.invoke(
                [
                    {"role": "system", "content": selection_request.system_message},
                    selection_request.last_user_message,
                ]
            )
            if not isinstance(response, dict):
                msg = f"Expected dict response, got {type(response)}"
                raise AssertionError(msg)
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
        except Exception as exc:
            logger.warning("Tool selector failed; falling back to original tool set: %s", exc)
            return handler(request)
        return handler(modified_request)

    async def awrap_model_call(self, request, handler):
        selection_request = self._prepare_selection_request(request)
        if selection_request is None:
            return await handler(request)
        try:
            type_adapter = _create_tool_selection_response(selection_request.available_tools)
            schema = type_adapter.json_schema()
            structured_model = selection_request.model.with_structured_output(schema)
            response = await structured_model.ainvoke(
                [
                    {"role": "system", "content": selection_request.system_message},
                    selection_request.last_user_message,
                ]
            )
            if not isinstance(response, dict):
                msg = f"Expected dict response, got {type(response)}"
                raise AssertionError(msg)
            modified_request = self._process_selection_response(
                response,
                selection_request.available_tools,
                selection_request.valid_tool_names,
                request,
            )
        except Exception as exc:
            logger.warning("Tool selector failed; falling back to original tool set: %s", exc)
            return await handler(request)
        return await handler(modified_request)


__all__ = ["SafeLLMToolSelectorMiddleware"]
