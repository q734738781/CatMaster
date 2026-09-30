from __future__ import annotations

import argparse
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from catmaster.llm.config import LLMProfile
from catmaster.llm.factory import build_chat_model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Smoke-test the ModelE DeepSeek OpenAI-compatible LLM config."
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
        default="巴黎的首都在哪里？请只回答一个短句。",
        help="Prompt to send.",
    )
    return parser.parse_args()


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

    llm = build_chat_model(cfg)
    response = llm.invoke(args.prompt)
    print("response:")
    print(getattr(response, "content", response))
    additional_kwargs = getattr(response, "additional_kwargs", None)
    if additional_kwargs:
        print("additional_kwargs:")
        print(additional_kwargs)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
