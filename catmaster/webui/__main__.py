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
    args = parser.parse_args()
    launch(host=args.host, port=args.port, project_space_root=args.project_space_root)


if __name__ == "__main__":
    main()
