from __future__ import annotations

import argparse
import os
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepagents import create_deep_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI


WEATHER_CALLS: list[str] = []


class SequentialToolChatOpenAI(ChatOpenAI):
    def bind_tools(self, tools: Any, **kwargs: Any) -> Any:
        kwargs["parallel_tool_calls"] = False
        return super().bind_tools(tools, **kwargs)


@tool
def get_weather(city: str) -> str:
    """Return a fixed weather report for a city."""
    WEATHER_CALLS.append(city)
    print(f"tool_call[{len(WEATHER_CALLS)}]: get_weather(city={city!r})")
    return f"{city}: sunny, 22 C, light wind."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Minimal DeepAgents weather-tool smoke test for the PPCHAT "
            "OpenAI-compatible endpoint."
        )
    )
    parser.add_argument(
        "--model",
        default="gpt-5.5",
        help="Model name exposed by PPCHAT. Default: gpt-5.5",
    )
    parser.add_argument(
        "--reasoning-effort",
        default="xhigh",
        help="Reasoning effort to pass to ChatOpenAI. Default: xhigh",
    )
    parser.add_argument(
        "--prompt",
        default=None,
        help=(
            "Prompt to send once to the DeepAgent. If omitted, the script asks "
            "for sequential tool calls for the configured cities."
        ),
    )
    parser.add_argument(
        "--cities",
        default="北京,上海,广州,深圳,杭州,南京,成都,重庆,武汉,西安",
        help=(
            "Comma-separated city list used by the default prompt. "
            "Default contains 10 cities."
        ),
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.0,
        help="Sampling temperature. Default: 0.0",
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=120.0,
        help="Request timeout in seconds. Default: 120",
    )
    parser.add_argument(
        "--max-retries",
        type=int,
        default=1,
        help="HTTP retry count. Default: 1",
    )
    return parser.parse_args()


def _require_env(name: str) -> str:
    value = os.environ.get(name, "").strip()
    if not value:
        raise RuntimeError(f"{name} must be set")
    return value


def _print_messages(result: Any) -> None:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        print(result)
        return

    for index, message in enumerate(messages):
        msg_type = getattr(message, "type", type(message).__name__)
        content = getattr(message, "content", "")
        tool_calls = getattr(message, "tool_calls", None)
        additional_kwargs = getattr(message, "additional_kwargs", None)
        print(f"[{index}] {msg_type}: {content}")
        if tool_calls:
            print(f"    tool_calls: {tool_calls}")
        if additional_kwargs:
            print(f"    additional_kwargs: {additional_kwargs}")


def _count_tool_calls(result: Any) -> int:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return 0

    count = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            count += len(tool_calls)
    return count


def _max_tool_calls_per_assistant_message(result: Any) -> int:
    messages = result.get("messages") if isinstance(result, dict) else None
    if not isinstance(messages, list):
        return 0

    max_count = 0
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if isinstance(tool_calls, list):
            max_count = max(max_count, len(tool_calls))
    return max_count


def _default_prompt(cities: list[str]) -> str:
    city_text = "、".join(cities)
    return (
        f"请查询以下 {len(cities)} 个城市的天气：{city_text}。\n"
        "必须在这一次任务中完成全部城市，但工具调用必须严格串行：\n"
        "1. 每个 assistant message 最多只能包含一个 get_weather tool call。\n"
        "2. 先调用第 1 个城市，等待工具结果返回后，再调用第 2 个城市。\n"
        "3. 以此类推，直到所有城市都完成。\n"
        "4. 不要在同一个 assistant message 里发出多个 tool calls。\n"
        "5. 每个城市必须恰好调用一次 get_weather。\n"
        "全部工具结果拿到以后，用中文按城市逐行汇总。"
    )


def _invoke_agent(agent: Any, prompt: str, thread_id: str) -> Any:
    return agent.invoke(
        {"messages": [{"role": "user", "content": prompt}]},
        config={"configurable": {"thread_id": thread_id}},
    )


def main() -> int:
    args = parse_args()
    api_base = _require_env("PPCHAT_API_BASE")
    api_key = _require_env("PPCHAT_API_KEY")
    cities = [city.strip() for city in args.cities.split(",") if city.strip()]
    prompt = args.prompt if args.prompt is not None else _default_prompt(cities)
    thread_id = f"ppchat-weather-{uuid.uuid4().hex}"

    print(f"model: {args.model}")
    print(f"base_url: {api_base}")
    print("api_key_env: PPCHAT_API_KEY")
    print(f"reasoning_effort: {args.reasoning_effort}")
    print(f"expected_tool_calls: {len(cities) if args.prompt is None else 'custom prompt'}")
    print("parallel_tool_calls: false")
    print(f"thread_id: {thread_id}")

    model = SequentialToolChatOpenAI(
        model=args.model,
        api_key=api_key,
        base_url=api_base,
        temperature=args.temperature,
        reasoning_effort=args.reasoning_effort,
        streaming=False,
        disable_streaming=True,
        use_responses_api=False,
        timeout=args.timeout,
        max_retries=args.max_retries,
    )

    agent = create_deep_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "You are a minimal weather assistant. "
            "Use the weather tool when the user asks about weather. "
            "Never issue more than one tool call in a single assistant message."
        ),
        name="ppchat_weather_smoke",
    )

    result = _invoke_agent(agent, prompt, thread_id)
    print("result:")
    _print_messages(result)
    print("\nsummary:")
    print(f"expected_tool_calls: {len(cities) if args.prompt is None else 'custom prompt'}")
    print(f"observed_tool_calls_from_messages: {_count_tool_calls(result)}")
    print(
        "max_tool_calls_per_assistant_message: "
        f"{_max_tool_calls_per_assistant_message(result)}"
    )
    print(f"actual_tool_calls: {len(WEATHER_CALLS)}")
    print(f"actual_tool_call_cities: {', '.join(WEATHER_CALLS)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
