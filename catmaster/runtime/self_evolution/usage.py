"""Skills Evo uses the same full observation scope as other standalone agents."""
from __future__ import annotations

from pathlib import Path
from typing import Any

from catmaster.runtime.observed_invocation import observed_invocation


def usage_invocation(*, workspace: Path, stage: str, model_label: str,
                     context: dict[str, Any]):
    return observed_invocation(
        workspace=workspace, entrypoint="self_evolution", stage=stage,
        model_label=model_label, context=context,
    )
