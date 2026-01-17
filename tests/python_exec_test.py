#!/usr/bin/env python3
from pathlib import Path
import os

from catmaster.tools.registry import get_tool_registry


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    os.environ.setdefault("CATMASTER_WORKSPACE", str(root))

    reg = get_tool_registry()
    tool = reg.get_tool_function("python_exec")

    code = (
        "import math\n"
        "from pathlib import Path\n"
        "def f():\n"
        "    return Path('a')\n"
        "print(f())\n"
        f"expected = Path({str(root)!r}).resolve()\n"
        "print(Path.cwd().resolve() == expected)\n"
        "print(math.sqrt(9))\n"
    )

    res = tool({"code": code})
    assert res["status"] == "success", f"python_exec failed: {res}"
    output = res["data"].get("run_result", "")
    print(output)


if __name__ == "__main__":
    main()
