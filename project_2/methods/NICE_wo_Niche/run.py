#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    p = build_argparser("NICE w/o Niche on CEC2014")
    args = p.parse_args()
    seeds = args.seeds or "1001:1020"
    run_experiment(
        module="methods.NICE_wo_Niche.nice_wo_niche",
        optimize_name="optimize",
        algorithm_name="NICE-wo-Niche",
        param_line=(
            f"Hard ablation A1: single randomly seeded local island "
            f"(no fitness/spatial niche guidance). Same FE schedule. seeds={seeds}"
        ),
        out_dir=ROOT / "out" / "NICE_wo_Niche",
        workers=args.workers,
        funcs=args.funcs,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
