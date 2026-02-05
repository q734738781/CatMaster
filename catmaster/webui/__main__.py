from __future__ import annotations

import argparse

from .app import launch


def main() -> None:
    parser = argparse.ArgumentParser(description="CatMaster WebUI")
    parser.add_argument("--workspace", default=None, help="Workspace root (or set CATMASTER_WORKSPACE)")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    args = parser.parse_args()
    launch(host=args.host, port=args.port, workspace=args.workspace)


if __name__ == "__main__":
    main()
