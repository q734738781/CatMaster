from __future__ import annotations

from typing import Any


def create_app(*args: Any, **kwargs: Any):
    from .server import create_app as _create_app

    return _create_app(*args, **kwargs)


def launch(*args: Any, **kwargs: Any):
    from .server import launch as _launch

    return _launch(*args, **kwargs)


__all__ = ["create_app", "launch"]
