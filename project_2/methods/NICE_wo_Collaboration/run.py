#!/usr/bin/env python3
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    p = build_argparser("NICE w/o Collaboration on CEC2014")
    args = p.parse_args()
    seeds = args.seeds or "1001:1020"
    run_experiment(
        module="methods.NICE_wo_Collaboration.nice_wo_collaboration",
        optimize_name="optimize",
        algorithm_name="NICE-wo-Collaboration",
        param_line=(
            f"Hard ablation A4: no injection and CMA cannot update reported "
            f"global best/curve. Same FE schedule. seeds={seeds}"
        ),
        out_dir=ROOT / "out" / "NICE_wo_Collaboration",
        workers=args.workers,
        funcs=args.funcs,
        seeds=seeds,
    )


if __name__ == "__main__":
    main()
