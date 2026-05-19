from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path


REPO_ROOT = Path(__file__).resolve().parents[3]
MODULE_ROOT = REPO_ROOT / "Agent_Optimization"
AGENT_ROOT = MODULE_ROOT / "heo_agent"
ASSETS_ROOT = AGENT_ROOT / "assets"
RUNTIME_ROOT = MODULE_ROOT / "runtime"
CAMPAIGNS_ROOT = RUNTIME_ROOT / "campaigns"
MEMORIES_ROOT = RUNTIME_ROOT / "memories"
SCRATCH_ROOT = RUNTIME_ROOT / "scratch"

DEFAULT_MODEL = os.environ.get("BATTERY_AGENT_MODEL", "openai/gpt-5.4")
DEFAULT_SEARCH_MODEL = os.environ.get("BATTERY_SEARCH_MODEL", DEFAULT_MODEL)
DEFAULT_REASONING_EFFORT = os.environ.get("BATTERY_REASONING_EFFORT", "high")
DEFAULT_SEARCH_REASONING_EFFORT = os.environ.get("BATTERY_SEARCH_REASONING_EFFORT", "medium")
DEFAULT_MACE_MODEL = Path(
    os.environ.get(
        "BATTERY_MACE_MODEL",
        str(ASSETS_ROOT / "battery_mh1_replay_run-42.model"),
    )
)
DEFAULT_MACE_HEAD = os.environ.get("BATTERY_MACE_HEAD", "Default")
DEFAULT_MACE_DTYPE = os.environ.get("BATTERY_MACE_DTYPE", "float32")
DEFAULT_GPU_IDS = [gpu.strip() for gpu in os.environ.get("BATTERY_GPU_IDS", "0,1,2,3").split(",") if gpu.strip()]
DEFAULT_MD_TEMPERATURES_K = [
    float(value.strip())
    for value in os.environ.get("BATTERY_MD_TEMPERATURES_K", "600,800,1000,1200").split(",")
    if value.strip()
]
DEFAULT_STAGE1_BASE_STRUCTURE = Path(
    os.environ.get(
        "BATTERY_STAGE1_BASE_STRUCTURE",
        str(REPO_ROOT / "NFPP_Structure" / "Na4Fe3P4O15_S416.vasp"),
    )
)
DEFAULT_STAGE1_ACTIVE_POOL_LIMIT = int(os.environ.get("BATTERY_STAGE1_ACTIVE_POOL_LIMIT", "15"))
DEFAULT_STAGE1_ROUND_LIMIT = int(os.environ.get("BATTERY_STAGE1_ROUND_LIMIT", "15"))
DEFAULT_STAGE1_ANCHOR_ROOT = Path(
    os.environ.get(
        "BATTERY_STAGE1_ANCHOR_ROOT",
        str(REPO_ROOT / "MACE_Training" / "DFT_files" / "v0" / "ABC"),
    )
)
DEFAULT_STAGE1_ANCHOR_TABLE = Path(
    os.environ.get(
        "BATTERY_STAGE1_ANCHOR_TABLE",
        str(ASSETS_ROOT / "s208_nfpp_anchor_table.json"),
    )
)

DEFAULT_ELEMENT_POOL = [
    "Ti",
    "V",
    "Cr",
    "Mn",
    "Co",
    "Ni",
    "Mo",
    "Ru",
    "Pd",
    "Mg",
    "Al",
    "Sc",
    "Cu",
    "Zn",
    "Ga",
    "Sr",
    "Y",
    "Zr",
    "Nb",
    "Rh",
    "In",
    "Sn",
    "Ca",
]


@dataclass(frozen=True)
class CampaignPaths:
    campaign_id: str
    root: Path
    shared: Path
    stage1: Path
    stage2: Path


def get_campaign_paths(campaign_id: str) -> CampaignPaths:
    root = CAMPAIGNS_ROOT / campaign_id
    return CampaignPaths(
        campaign_id=campaign_id,
        root=root,
        shared=root / "shared",
        stage1=root / "stage1",
        stage2=root / "stage2",
    )
