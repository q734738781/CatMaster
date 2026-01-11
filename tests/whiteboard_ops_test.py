#!/usr/bin/env python3
from catmaster.runtime.whiteboard_ops import apply_whiteboard_ops_text, validate_whiteboard_ops


WHITEBOARD = """# Whiteboard
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


def main() -> None:
    ops = [
        {"op": "UPSERT", "section": "Goal", "text": "Run VASP relaxation"},
        {"op": "UPSERT", "section": "Key Facts", "record_type": "FACT", "id": "F1", "text": "Energy = -1.23 eV"},
        {"op": "UPSERT", "section": "Key Files", "record_type": "FILE", "id": "K1", "path": "o2/OUTCAR", "kind": "output", "description": "VASP output"},
    ]
    validation = validate_whiteboard_ops(ops)
    assert validation["ok"], f"Validation failed: {validation}"

    updated = apply_whiteboard_ops_text(WHITEBOARD, ops, "task_01")["updated_text"]
    assert "- Run VASP relaxation" in updated
    assert "FACT[F1]:" in updated
    assert "FILE[K1]:" in updated
    assert "- (none)" not in updated

    ops_replace = [
        {"op": "UPSERT", "section": "Key Facts", "record_type": "FACT", "id": "F1", "text": "Energy = -2.34 eV"},
    ]
    updated2 = apply_whiteboard_ops_text(updated, ops_replace, "task_02")["updated_text"]
    assert "Energy = -2.34 eV" in updated2

    ops_deprecate = [
        {"op": "DEPRECATE", "section": "Key Facts", "record_type": "FACT", "id": "F1", "reason": "Superseded"},
    ]
    updated3 = apply_whiteboard_ops_text(updated2, ops_deprecate, "task_03")["updated_text"]
    assert "status=deprecated" in updated3
    assert "deprecated_at=" in updated3

    ops_missing = [
        {"op": "DEPRECATE", "section": "Key Facts", "record_type": "FACT", "id": "MISSING"},
    ]
    res = apply_whiteboard_ops_text(updated3, ops_missing, "task_04")
    assert res["warnings"], "Expected warning for missing record"


if __name__ == "__main__":
    main()
