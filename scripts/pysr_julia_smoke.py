#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import shutil
from pathlib import Path


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Prewarm and verify the PySR Julia backend used by CatMaster. "
            "Importing PySR may install Julia/SymbolicRegression through juliapkg "
            "when no system Julia is configured."
        )
    )
    parser.add_argument(
        "--julia-bindir",
        default="",
        help="Optional directory containing the julia binary; exported as PYTHON_JULIACALL_BINDIR before import.",
    )
    parser.add_argument(
        "--fit",
        action="store_true",
        help="Run a tiny deterministic symbolic-regression fit after import/precompile.",
    )
    return parser.parse_args()


def main() -> int:
    args = _parse_args()
    if args.julia_bindir:
        bindir = Path(args.julia_bindir).expanduser().resolve()
        if not (bindir / "julia").exists():
            raise SystemExit(f"--julia-bindir does not contain julia: {bindir}")
        os.environ["PYTHON_JULIACALL_BINDIR"] = str(bindir)

    import pysr
    from juliacall import Main as jl

    result: dict[str, object] = {
        "pysr_version": getattr(pysr, "__version__", "unknown"),
        "system_julia": shutil.which("julia") or "",
        "juliacall_bindir_env": os.environ.get("PYTHON_JULIACALL_BINDIR", ""),
        "julia_version": jl.seval("string(VERSION)"),
        "julia_bindir": jl.seval("Sys.BINDIR"),
        "julia_project": jl.seval("Base.active_project()"),
    }

    if args.fit:
        import numpy as np
        from pysr import PySRRegressor

        x = np.linspace(-2, 2, 25).reshape(-1, 1)
        y = x[:, 0] ** 2 + 2 * x[:, 0] + 1
        model = PySRRegressor(
            niterations=3,
            populations=3,
            population_size=40,
            tournament_selection_n=8,
            binary_operators=["+", "*"],
            unary_operators=[],
            progress=False,
            verbosity=0,
            temp_equation_file=True,
            random_state=0,
            deterministic=True,
            parallelism="serial",
        )
        model.fit(x, y)
        result["fit_expression"] = str(model.sympy())

    print(json.dumps(result, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
