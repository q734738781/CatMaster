from __future__ import annotations

import argparse
import sys
import uuid
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from deepagents import create_deep_agent
from langchain_core.tools import tool

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model


@tool
def get_weather(city: str) -> str:
    """Return a fixed weather report for a city."""
    return f"{city}: sunny, 22 C, light wind."


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Minimal DeepAgents weather-tool smoke test for ModelE DeepSeek config."
    )
    parser.add_argument(
        "--llm-config",
        default="configs/llm_modele.yaml",
        help="LLM config path. Default: configs/llm_modele.yaml",
    )
    parser.add_argument(
        "--role",
        default="experiment_specialist",
        help="Role to test from the LLM config. Default: experiment_specialist",
    )
    parser.add_argument(
        "--prompt",
        default="请调用天气工具查询巴黎天气，然后用一句中文回答。",
        help="Prompt to send to the deep agent.",
    )
    parser.add_argument(
        "--print-http-raw-post",
        action="store_true",
        help="Print raw POST/response bodies through CatMaster's ChatOpenAI httpx hooks.",
    )
    parser.add_argument(
        "--chatdeepseek",
        action="store_true",
        help="Bypass CatMaster's factory and use langchain_deepseek.ChatDeepSeek directly.",
    )
    parser.add_argument(
        "--disable-reasoning",
        action="store_true",
        help="Clear reasoning_effort before constructing the model.",
    )
    return parser.parse_args()


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


def main() -> int:
    args = parse_args()
    config_path = Path(args.llm_config).expanduser()
    profile = LLMProfile.from_env_or_file(str(config_path))
    cfg = profile.config_for_role(args.role)

    print(f"config: {config_path}")
    print(f"role: {args.role}")
    print(f"model_label: {profile.label_for_role(args.role)}")
    print(f"provider: {cfg.provider}")
    print(f"model: {cfg.model}")
    print(f"base_url: {cfg.base_url}")
    print(f"api_key_env: {cfg.api_key_env}")
    print(f"reasoning_effort: {cfg.reasoning_effort}")
    print(f"provider_options: {cfg.provider_options}")
    if args.disable_reasoning:
        cfg.reasoning_effort = None
    if args.chatdeepseek:
        from langchain_deepseek import ChatDeepSeek

        model = ChatDeepSeek(
            model=cfg.model,
            api_key=cfg.api_key or __import__("os").environ[str(cfg.api_key_env)],
            base_url=cfg.base_url,
            temperature=cfg.temperature,
            reasoning_effort=cfg.reasoning_effort,
            streaming=False,
            disable_streaming=True,
            use_responses_api=False,
        )
    else:
        if args.print_http_raw_post:
            cfg.print_http_raw_post = True
        model = build_chat_model(cfg)

    agent = create_deep_agent(
        model=model,
        tools=[get_weather],
        system_prompt=(
            "You are a minimal weather assistant. "
            "Use the weather tool when the user asks about weather."
        ),
        name="deepseek_weather_smoke",
    )

    result = agent.invoke(
        {"messages": [{"role": "user", "content": args.prompt}]},
        config={"configurable": {"thread_id": f"deepseek-weather-{uuid.uuid4().hex}"}},
    )
    print("result:")
    _print_messages(result)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
