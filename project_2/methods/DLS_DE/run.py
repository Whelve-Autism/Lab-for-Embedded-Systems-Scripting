#!/usr/bin/env python3
"""Run DLS-DE. Outputs -> project_2/out/DLS_DE/."""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from common.experiment import build_argparser, run_experiment


def main():
    args = build_argparser("DLS-DE on CEC2014").parse_args()
    run_experiment(
        module="methods.DLS_DE.dls_de",
        optimize_name="optimize",
        algorithm_name="DLS-DE",
        param_line=(
            "Parameters: D=10, PS=25*ln(D)*sqrt(D)~4, H=6, p=0.11, "
            "rarc=1.6, ST=30, tau=0.3, MaxFEs=100000, Runs=20"
        ),
        out_dir=ROOT / "out" / "DLS_DE",
        workers=args.workers,
        funcs=args.funcs,
    )


if __name__ == "__main__":
    main()
