#!/usr/bin/env python3
"""
WebUI entry point for CatMaster.
"""
from __future__ import annotations

import argparse
import logging
from pathlib import Path

from catmaster.webui import launch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s %(levelname)s %(name)s | %(message)s",
    datefmt="%H:%M:%S",
)


def main() -> None:
    parser = argparse.ArgumentParser(description="CatMaster WebUI")
    default_project_space_root = str(Path.cwd() / "project_space")
    parser.add_argument(
        "--project-space-root",
        default=default_project_space_root,
        help="Project space root (default: ./project_space)",
    )
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument(
        "--no-login",
        action="store_true",
        help="Disable login and open the built-in admin workspace.",
    )
    args = parser.parse_args()
    launch(host=args.host, port=args.port, project_space_root=args.project_space_root, no_login=args.no_login)


if __name__ == "__main__":
    main()
