#!/usr/bin/env python3
"""Run iStratDE (NumPy). Outputs -> project_2/out/iStratDE/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    args = build_argparser("iStratDE on CEC2014").parse_args()
    run_experiment(
        module="methods.iStratDE.istratde",
        optimize_name="optimize",
        algorithm_name="iStratDE",
        param_line=(
            "Parameters: D=10, NP=max(200,50*D), individual-level fixed "
            "F/CR/base_prim/base_sec/num_diff/crossover (official IStratDE logic), "
            "MaxFEs=100000, Runs=20 (NumPy port of EMI-Group/istratde)"
        ),
        out_dir=ROOT / "out" / "iStratDE",
        workers=args.workers,
        funcs=args.funcs,
    )


if __name__ == "__main__":
    main()
