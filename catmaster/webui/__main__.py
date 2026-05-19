from __future__ import annotations

import argparse
from pathlib import Path

from .server import launch


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
    parser.add_argument(
        "--timeout-keep-alive",
        type=int,
        default=0,
        help="Uvicorn keep-alive timeout in seconds (default: 0 for immediate shutdown).",
    )
    parser.add_argument(
        "--timeout-graceful-shutdown",
        type=int,
        default=0,
        help="Uvicorn graceful shutdown timeout in seconds (default: 0 for immediate shutdown).",
    )
    args = parser.parse_args()
    launch(
        host=args.host,
        port=args.port,
        project_space_root=args.project_space_root,
        no_login=args.no_login,
        timeout_keep_alive=args.timeout_keep_alive,
        timeout_graceful_shutdown=args.timeout_graceful_shutdown,
    )


if __name__ == "__main__":
    main()
