#!/usr/bin/env python3
"""Project 1: DE/rand/1/bin on CEC2014 (10D), 20 runs per function."""

from __future__ import annotations

import argparse
import importlib
import warnings
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

from de import de_rand1_bin
from schwefel_patch import apply_schwefel_patch

ROOT = Path(__file__).resolve().parent
OUT = ROOT / "out"

# Frozen experiment config
DIM = 10
NPOP = 100
F = 0.5
CR = 0.9
MAX_FES = 100_000
SEEDS = list(range(1001, 1021))  # 20 seeds
LOWER, UPPER = -100.0, 100.0
N_FUNCS = 30

# Convergence checkpoints: FE=100, then every 1000 until 100000
CHECKPOINTS = np.array(
    [100] + list(range(1000, MAX_FES + 1, 1000)), dtype=int
)


def _make_func(fid: int):
    mod = importlib.import_module("opfunu.cec_based.cec2014")
    cls = getattr(mod, f"F{fid}2014")
    return cls(ndim=DIM)


def _run_one(fid: int, seed: int) -> dict:
    """One independent run for (function, seed)."""
    # Each worker process must apply the Schwefel fix independently.
    apply_schwefel_patch()
    rng = np.random.default_rng(seed)
    func = _make_func(fid)
    optimum = float(func.f_bias)

    def evaluate(x):
        return float(func.evaluate(np.asarray(x, dtype=float)))

    t0 = time.perf_counter()
    best_f, best_x, actual_fes, curve_f = de_rand1_bin(
        evaluate=evaluate,
        lower=LOWER,
        upper=UPPER,
        dim=DIM,
        npop=NPOP,
        F=F,
        CR=CR,
        max_fes=MAX_FES,
        rng=rng,
        checkpoints=CHECKPOINTS,
    )
    elapsed = time.perf_counter() - t0
    error = best_f - optimum
    curve_err = curve_f - optimum

    result = {
        "function_id": fid,
        "function": f"C-14-f{fid}",
        "seed": seed,
        "dimension": DIM,
        "population": NPOP,
        "F": F,
        "CR": CR,
        "max_fes": MAX_FES,
        "actual_fes": int(actual_fes),
        "optimum": optimum,
        "best_fitness": float(best_f),
        "error": float(error),
        "elapsed_sec": float(elapsed),
        "best_x": best_x.tolist(),
        "curve_fes": CHECKPOINTS.tolist(),
        "curve_error": curve_err.tolist(),
    }
    return result


def _sci(x: float) -> str:
    """Scientific notation like 6.39E-02."""
    if not np.isfinite(x):
        return str(x)
    return f"{x:.2E}"


def export_outputs(runs: list[dict]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)

    # --- one table: every run final result + packed curve ---
    run_rows = []
    for r in runs:
        row = {k: r[k] for k in (
            "function_id", "function", "seed", "dimension", "population",
            "F", "CR", "max_fes", "actual_fes", "optimum",
            "best_fitness", "error", "elapsed_sec",
        )}
        for d, v in enumerate(r["best_x"], start=1):
            row[f"x{d}"] = v
        row["curve_fes"] = ";".join(str(int(fe)) for fe in r["curve_fes"])
        row["curve_error"] = ";".join(f"{e:.16g}" for e in r["curve_error"])
        run_rows.append(row)
    df_runs = pd.DataFrame(run_rows).sort_values(
        ["function_id", "seed"]
    ).reset_index(drop=True)

    # --- paper summary ---
    lines = []
    lines.append("Algorithm: DE/rand/1/bin")
    lines.append(
        f"Parameters: D={DIM}, NP={NPOP}, F={F}, CR={CR}, "
        f"MaxFEs={MAX_FES}, Runs={len(SEEDS)}, Bounds=[{LOWER},{UPPER}]"
    )
    lines.append(f"Seeds: {SEEDS[0]}-{SEEDS[-1]}")
    lines.append("Metric: error = f(x_best) - f*")
    lines.append("")
    header = f"{'F':<10}{'Best':>12}{'Median':>12}{'Ave':>12}{'Worst':>12}{'Std':>12}"
    lines.append(header)
    lines.append("-" * len(header))

    present = sorted(df_runs["function_id"].unique())
    summary_rows = []
    for fid in present:
        errs = df_runs.loc[df_runs["function_id"] == fid, "error"].to_numpy()
        best = float(np.min(errs))
        median = float(np.median(errs))
        ave = float(np.mean(errs))
        worst = float(np.max(errs))
        std = float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0
        name = f"C-14-f{fid}"
        lines.append(
            f"{name:<10}{_sci(best):>12}{_sci(median):>12}"
            f"{_sci(ave):>12}{_sci(worst):>12}{_sci(std):>12}"
        )
        summary_rows.append({
            "F": name, "Best": best, "Median": median,
            "Ave": ave, "Worst": worst, "Std": std,
        })

    table_path = OUT / "results.csv"
    df_runs.to_csv(table_path, index=False)

    txt_path = OUT / "results.txt"
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    print(f"[OK] wrote {table_path} ({len(df_runs)} runs)")
    print(f"[OK] wrote {txt_path}")


def main():
    parser = argparse.ArgumentParser(description="Project 1 CEC2014 DE experiment")
    parser.add_argument("--workers", type=int, default=32,
                        help="parallel worker processes")
    parser.add_argument("--funcs", type=str, default="1-30",
                        help="function range, e.g. 1-30 or 1,2,3")
    parser.add_argument("--smoke", action="store_true",
                        help="quick smoke: F1 only, MaxFEs override via env not used")
    args = parser.parse_args()

    if "-" in args.funcs:
        a, b = args.funcs.split("-", 1)
        func_ids = list(range(int(a), int(b) + 1))
    else:
        func_ids = [int(x) for x in args.funcs.split(",") if x.strip()]

    tasks = [(fid, seed) for fid in func_ids for seed in SEEDS]
    total = len(tasks)
    print(
        f"Start DE/rand/1/bin | funcs={func_ids[0]}..{func_ids[-1]} "
        f"| runs/func={len(SEEDS)} | tasks={total} | workers={args.workers}",
        flush=True,
    )

    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futs = {ex.submit(_run_one, fid, seed): (fid, seed) for fid, seed in tasks}
        for fut in as_completed(futs):
            fid, seed = futs[fut]
            try:
                res = fut.result()
                results.append(res)
                done += 1
                if done % 10 == 0 or done == total:
                    elapsed = time.perf_counter() - t0
                    print(
                        f"[{done}/{total}] last=F{fid}/seed{seed} "
                        f"error={res['error']:.6e} "
                        f"elapsed={elapsed / 60:.1f}min",
                        flush=True,
                    )
            except Exception as e:
                print(f"[FAIL] F{fid} seed{seed}: {e}", flush=True)
                raise

    export_outputs(results)
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


if __name__ == "__main__":
    main()
