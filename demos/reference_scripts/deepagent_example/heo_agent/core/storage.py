from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
import yaml

from .config import CAMPAIGNS_ROOT, MEMORIES_ROOT, RUNTIME_ROOT, SCRATCH_ROOT, CampaignPaths, get_campaign_paths


def ensure_runtime_tree() -> None:
    for path in (RUNTIME_ROOT, CAMPAIGNS_ROOT, MEMORIES_ROOT, SCRATCH_ROOT):
        path.mkdir(parents=True, exist_ok=True)


def ensure_campaign_tree(campaign_id: str) -> CampaignPaths:
    ensure_runtime_tree()
    paths = get_campaign_paths(campaign_id)
    for path in (paths.root, paths.shared, paths.stage1, paths.stage2):
        path.mkdir(parents=True, exist_ok=True)
    return paths


def write_yaml(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(yaml.safe_dump(payload, sort_keys=False, allow_unicode=False), encoding="utf-8")
    return path


def write_json(path: Path, payload: Any) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
    return path


def append_jsonl(path: Path, payload: dict[str, Any]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(payload, ensure_ascii=True) + "\n")
    return path


def write_markdown(path: Path, text: str) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")
    return path


def write_rows_csv(path: Path, rows: Iterable[dict[str, Any]]) -> Path:
    rows = list(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        path.write_text("", encoding="utf-8")
        return path
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    return path


def write_dataframe_bundle(stem: Path, rows: Iterable[dict[str, Any]]) -> dict[str, Path]:
    frame = pd.DataFrame(list(rows))
    results = {"csv": stem.with_suffix(".csv")}
    results["csv"].parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(results["csv"], index=False)
    parquet_path = stem.with_suffix(".parquet")
    try:
        frame.to_parquet(parquet_path, index=False)
    except Exception:
        pass
    else:
        results["parquet"] = parquet_path
    return results


def read_yaml(path: Path) -> Any:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def read_csv_records(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))
