#!/usr/bin/env python3
"""
WebUI entry point for CatMaster.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from catmaster.webui import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="CatMaster WebUI")
    default_workspace = str(Path.cwd() / "workspace")
    parser.add_argument("--workspace", default=default_workspace, help="Workspace root (default: ./workspace)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(host=args.host, port=args.port, workspace=args.workspace)


if __name__ == "__main__":
    main()
