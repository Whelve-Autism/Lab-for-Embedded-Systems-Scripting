#!/usr/bin/env python3
"""Run BIDE (DE/rand/1/bin). Outputs go to project_2/out/BIDE/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    args = build_argparser("BIDE on CEC2014").parse_args()
    run_experiment(
        module="methods.BIDE.de",
        optimize_name="optimize",
        algorithm_name="BIDE (DE/rand/1/bin)",
        param_line=(
            "Parameters: D=10, NP=100, F=0.5, CR=0.9, MaxFEs=100000, "
            "Runs=20, Bounds=[-100,100]"
        ),
        out_dir=ROOT / "out" / "BIDE",
        workers=args.workers,
        funcs=args.funcs,
    )


if __name__ == "__main__":
    main()
