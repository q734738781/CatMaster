"""
Generate candidate adsorbate placements with a JSON-friendly API.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Mapping

from catmaster.core.adsorbate import generator as adsorbate_gen


def generate_placements(config: Mapping[str, object]) -> Dict[str, object]:
    parser = adsorbate_gen.build_argparser()
    args = parser.parse_args([])
    for key, value in config.items():
        setattr(args, key, value)
    adsorbate_gen.generate(args)
    outdir = Path(args.outdir)
    seeds = sorted(p.as_posix() for p in outdir.glob("**/*.vasp"))
    return {
        "schema_version": "v2",
        "output_dir": outdir.as_posix(),
        "num_candidates": len(seeds),
        "candidate_paths": seeds,
    }


