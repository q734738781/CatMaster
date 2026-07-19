# Code writing date: 2026-07-17
# Responsible agent: Codex, for the CatMaster MLFF runtime maintainers.
# Implementation principle: select the provider through the common adapter and
# pass its ASE calculator into one provider-independent dynamics runner.
# Purpose: managed MLFF MD entry point for every enabled backend.
from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

try:  # Package import in tests; flat import after DPDispatcher staging.
    from .mlff_common import _ADAPTER_TYPES, _load_run_config
    from .mlff_dynamics import run_single
except ImportError:  # pragma: no cover - exercised by remote staged scripts
    from mlff_common import _ADAPTER_TYPES, _load_run_config
    from mlff_dynamics import run_single


def run(run_config: str) -> dict[str, Any]:
    config = _load_run_config(run_config)
    if config["operation"] != "md":
        raise ValueError(f"Managed MLFF MD received operation={config['operation']!r}.")
    items = dict(config["items"])
    if len(items) != 1:
        raise ValueError("MLFF MD requires exactly one trajectory source per stage.")
    backend = str(config["backend"])
    adapter = _ADAPTER_TYPES[backend]()
    rel, item_config = next(iter(sorted(items.items())))
    output_root = Path("output")
    output_dir = output_root / Path(rel).with_suffix("")
    output_root.mkdir(parents=True, exist_ok=True)
    errors: list[dict[str, str]] = []
    results: list[dict[str, Any]] = []
    try:
        summary = run_single(
            source=Path("input") / rel,
            output_dir=output_dir,
            config=config,
            item_config=dict(item_config),
            adapter=adapter,
        )
        results.append(
            {
                "input_rel": rel,
                "output_rel": output_dir.relative_to(output_root).as_posix(),
                "summary": summary,
            }
        )
    except Exception as exc:
        errors.append({"input_rel": rel, "error": f"{type(exc).__name__}: {exc}"})
    batch = {
        "schema_version": 1,
        "config_digest": str(config["config_digest"]),
        "operation": "md",
        "backend": backend,
        "provider_version": adapter.provider_version,
        "results": results,
        "errors": errors,
    }
    (output_root / "batch_summary.json").write_text(
        json.dumps(batch, ensure_ascii=False, indent=2) + "\n",
        encoding="utf-8",
    )
    if errors:
        raise RuntimeError("Managed MLFF MD failed; see output/batch_summary.json.")
    return batch


def main() -> None:
    parser = argparse.ArgumentParser(description="Run managed calculator-independent MLFF MD.")
    parser.add_argument("--run_config", required=True)
    args = parser.parse_args()
    print(json.dumps(run(args.run_config), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
