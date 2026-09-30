from __future__ import annotations

from typing import Optional
import os
import signal
import time
import shutil
import subprocess


def build_no_network_prefix() -> Optional[list[str]]:
    """
    Return argv prefix that disables network, or None if not available.

    Preference: unshare -Urn (no docker).
    """
    if shutil.which("unshare"):
        # -U: user ns, -r: map root user, -n: net ns
        return ["unshare", "--user", "--map-root-user", "--net", "--"]
    return None


def kill_process_tree(proc: subprocess.Popen) -> None:
    try:
        if os.name == "posix":
            try:
                os.killpg(proc.pid, signal.SIGTERM)
            except Exception:
                pass
            time.sleep(0.2)
            try:
                os.killpg(proc.pid, signal.SIGKILL)
            except Exception:
                pass
        else:
            proc.kill()
    except Exception:
        pass


__all__ = ["build_no_network_prefix", "kill_process_tree"]
