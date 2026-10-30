#!/usr/bin/env python3
"""Run NICE-DE. Outputs -> project_2/out/NICE_DE/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    p = build_argparser("NICE-DE on CEC2014")
    args = p.parse_args()
    seeds = args.seeds or "1001:1020"
    out = ROOT / "out" / "NICE_DE"
    run_experiment(
        module="methods.NICE_DE.nice_de",
        optimize_name="optimize",
        algorithm_name="NICE-DE",
        param_line=(
            f"Parameters: D=10, NP=100 fixed SHADE current-to-pbest/1 (H=6), "
            f"archive=100, binomial CR + <=2 persistent niche CMA (pycma ask/tell), "
            f"cycle=1000 DE FE then <=100 CMA FE (global CMA cap 12000), "
            f"sigma0 from nearest-10 median, inject<=2/gen <=4/cycle, "
            f"MaxFEs=100000, seeds={seeds}"
        ),
        out_dir=out,
        workers=args.workers,
        funcs=args.funcs,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
