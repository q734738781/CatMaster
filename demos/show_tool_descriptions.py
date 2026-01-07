#!/usr/bin/env python3
"""
Print tool descriptions (full or short) from the registry.

Usage:
  python -m tests.show_tool_descriptions --short
"""
from __future__ import annotations

import argparse

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    parser = argparse.ArgumentParser(description="Show tool descriptions for LLM prompts")
    parser.add_argument("--short", action="store_true", help="Show short descriptions (name + docstring only)")
    args = parser.parse_args()

    registry = get_tool_registry()
    if args.short:
        print(registry.get_short_tool_descriptions_for_llm())
    else:
        print(registry.get_tool_descriptions_for_llm())


if __name__ == "__main__":
    main()
