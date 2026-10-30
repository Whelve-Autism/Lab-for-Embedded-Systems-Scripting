#!/usr/bin/env python3
"""Run ECL-SPACMA. Outputs -> project_2/out/ECL_SPACMA/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    args = build_argparser("ECL-SPACMA on CEC2014").parse_args()
    run_experiment(
        module="methods.ECL_SPACMA.ecl_spacma",
        optimize_name="optimize",
        algorithm_name="ECL-SPACMA",
        param_line=(
            "Parameters: D=10, NPinit=18*D, NPmin=4, p=0.11, |A|=1.4*NP, "
            "H=5, c=0.9, basek=1.1, alpha=0.05, NFELS=0.75, "
            "PLSmin=0.01, PLSmax=0.4, MaxFEs=100000, Runs=20 (SLSQP LS)"
        ),
        out_dir=ROOT / "out" / "ECL_SPACMA",
        workers=args.workers,
        funcs=args.funcs,
    )


if __name__ == "__main__":
    main()
