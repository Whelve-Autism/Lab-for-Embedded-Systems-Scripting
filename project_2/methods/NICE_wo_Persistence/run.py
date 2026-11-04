#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    p = build_argparser("NICE w/o Persistence on CEC2014")
    args = p.parse_args()
    seeds = args.seeds or "1001:1020"
    run_experiment(
        module="methods.NICE_wo_Persistence.nice_wo_persistence",
        optimize_name="optimize",
        algorithm_name="NICE-wo-Persistence",
        param_line=(
            f"Hard ablation A3: re-seed local search after every CMA generation "
            f"(no cross-generation state). Same FE schedule. seeds={seeds}"
        ),
        out_dir=ROOT / "out" / "NICE_wo_Persistence",
        workers=args.workers,
        funcs=args.funcs,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
