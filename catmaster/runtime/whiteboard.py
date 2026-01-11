#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Whiteboard store with stable anchors and hash support.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, Optional
import hashlib

from catmaster.tools.base import ensure_system_root, system_root


DEFAULT_WHITEBOARD = """# Whiteboard
## Current State
### Goal
- (empty)
### Key Facts
- (none)
### Key Files
- (none)
### Constraints
- (none)
### Open Questions
- (none)
## Journal
- (empty)
"""


@dataclass(frozen=True)
class WhiteboardStore:
    path: Path

    @staticmethod
    def default_path() -> Path:
        return system_root() / "whiteboard.md"

    @classmethod
    def create_default(cls) -> "WhiteboardStore":
        ensure_system_root()
        return cls(path=cls.default_path())

    def ensure_exists(self) -> None:
        if self.path.exists():
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        legacy = system_root() / "whiteboard.md"
        if legacy.exists():
            legacy.replace(self.path)
            return
        self.path.write_text(DEFAULT_WHITEBOARD, encoding="utf-8")

    def read(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Whiteboard not found: {self.path}")
        return self.path.read_text(encoding="utf-8")

    def get_hash(self) -> str:
        if not self.path.exists():
            raise FileNotFoundError(f"Whiteboard not found: {self.path}")
        data = self.path.read_bytes()
        return hashlib.sha256(data).hexdigest()

    def read_sections(self, sections: Iterable[str], *, max_chars: Optional[int] = None) -> str:
        content = self.read()
        section_map = _extract_sections(content)
        chunks = []
        for section in sections:
            if section not in section_map:
                raise ValueError(f"Missing whiteboard section: {section}")
            header = "## " + section if section == "Journal" else "### " + section
            body = section_map[section].strip()
            if body:
                chunks.append(f"{header}\n{body}")
            else:
                chunks.append(f"{header}")
        text = "\n\n".join(chunks).strip()
        if max_chars is not None and len(text) > max_chars:
            text = text[:max_chars]
        return text


def _extract_sections(content: str) -> Dict[str, str]:
    lines = content.splitlines()
    sections: Dict[str, list[str]] = {}
    current: Optional[str] = None
    for line in lines:
        if line.startswith("### "):
            current = line[4:].strip()
            sections[current] = []
            continue
        if line.startswith("## "):
            current = line[3:].strip()
            sections[current] = []
            continue
        if current is not None:
            sections[current].append(line)
    return {key: "\n".join(value).rstrip() for key, value in sections.items()}


__all__ = ["WhiteboardStore"]
