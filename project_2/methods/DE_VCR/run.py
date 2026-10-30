#!/usr/bin/env python3
"""Run DE-VCR. Outputs -> project_2/out/DE_VCR/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    args = build_argparser("DE-VCR on CEC2014").parse_args()
    run_experiment(
        module="methods.DE_VCR.de_vcr",
        optimize_name="optimize",
        algorithm_name="DE-VCR",
        param_line=(
            "Parameters: D=10, PS=25*ln(D)*sqrt(D)~4, muF=0.6, muCR=0.6, "
            "p=0.11, H=4, k=0.2, C=7D, theta=0.9, MaxFEs=100000, Runs=20"
        ),
        out_dir=ROOT / "out" / "DE_VCR",
        workers=args.workers,
        funcs=args.funcs,
    )


if __name__ == "__main__":
    main()
