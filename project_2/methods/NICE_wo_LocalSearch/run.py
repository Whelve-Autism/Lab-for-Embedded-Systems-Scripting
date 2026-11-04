#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    p = build_argparser("NICE w/o Local Search on CEC2014")
    args = p.parse_args()
    seeds = args.seeds or "1001:1020"
    run_experiment(
        module="methods.NICE_wo_LocalSearch.nice_wo_localsearch",
        optimize_name="optimize",
        algorithm_name="NICE-wo-LocalSearch",
        param_line=(
            f"Formal ablation A4: disable niche/CMA entirely; former CMA FE "
            f"budget (~9000) returned to global DE. MaxFEs=100000, seeds={seeds}"
        ),
        out_dir=ROOT / "out" / "NICE_wo_LocalSearch",
        workers=args.workers,
        funcs=args.funcs,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
