#!/usr/bin/env python3
"""
Project 3 thin entry: run NICE (from Project 2) under a chosen FE budget.

Does not modify NICE core. Adds project_2 to sys.path, runs CEC2014 F1–F30
(or a subset) with exactly one seed, writes results under --out, and prints
one P3_RESULT line only after a fully successful configuration:

    E_mean = (1/n) * sum_i |f_i_obtained - f_i*|
"""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import re
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

P3_ROOT = Path(__file__).resolve().parent
P2_ROOT = P3_ROOT.parent / "project_2"
if not P2_ROOT.is_dir():
    raise SystemExit(f"Project 2 root not found: {P2_ROOT}")
if str(P2_ROOT) not in sys.path:
    sys.path.insert(0, str(P2_ROOT))

from common.schwefel_patch import apply_schwefel_patch  # noqa: E402

DIM = 10
LOWER, UPPER = -100.0, 100.0
NPOP = 100


class CountingEvaluate:
    def __init__(self, fn):
        self._fn = fn
        self.n_calls = 0

    def __call__(self, x):
        self.n_calls += 1
        return float(self._fn(x))


def _make_func(fid: int, dim: int = DIM):
    mod = importlib.import_module("opfunu.cec_based.cec2014")
    return getattr(mod, f"F{fid}2014")(ndim=dim)


def _checkpoints(max_fes: int) -> np.ndarray:
    max_fes = int(max_fes)
    pts = [100] if max_fes >= 100 else []
    pts.extend(range(1000, max_fes + 1, 1000))
    if not pts or pts[-1] != max_fes:
        pts.append(max_fes)
    # unique + sorted
    return np.array(sorted(set(int(x) for x in pts if 0 < x <= max_fes)), dtype=int)


def _parse_funcs(spec: str) -> list[int]:
    spec = spec.strip()
    interval = re.fullmatch(r"([0-9]+)\s*[-:]\s*([0-9]+)", spec)
    if interval:
        first, last = map(int, interval.groups())
        if not 1 <= first <= last <= 30:
            raise ValueError("function range must be ascending and within 1-30")
        return list(range(first, last + 1))
    if not re.fullmatch(r"[0-9]+(?:\s*,\s*[0-9]+)*", spec):
        raise ValueError("use 1-30, 1:30, or a comma-separated list such as 1,2,3")
    func_ids = [int(x) for x in spec.split(",")]
    if any(fid < 1 or fid > 30 for fid in func_ids):
        raise ValueError("function ids must be within 1-30")
    if len(set(func_ids)) != len(func_ids):
        raise ValueError("duplicate function ids are not allowed")
    return func_ids


def _worker(payload: dict) -> dict:
    apply_schwefel_patch()
    from methods.NICE_DE.nice_de import optimize

    fid = int(payload["fid"])
    seed = int(payload["seed"])
    max_fes = int(payload["max_fes"])
    rng = np.random.default_rng(seed)
    func = _make_func(fid)
    optimum = float(func.f_bias)
    raw = lambda x: float(func.evaluate(np.asarray(x, dtype=float)))
    evaluate = CountingEvaluate(raw)
    checkpoints = _checkpoints(max_fes)

    t0 = time.perf_counter()
    status = "ok"
    err_msg = ""
    try:
        best_f, best_x, actual_fes, curve_f, meta = optimize(
            evaluate=evaluate,
            lower=LOWER,
            upper=UPPER,
            dim=DIM,
            max_fes=max_fes,
            rng=rng,
            checkpoints=checkpoints,
        )
        best_f = float(best_f)
        best_x = np.asarray(best_x, dtype=float)
        curve_f = np.asarray(curve_f, dtype=float)
        if evaluate.n_calls != int(actual_fes):
            status = "fe_mismatch"
            err_msg = f"eval_calls={evaluate.n_calls} actual_fes={actual_fes}"
        if evaluate.n_calls > max_fes:
            status = "over_budget"
            err_msg = f"eval_calls={evaluate.n_calls}>{max_fes}"
        elif evaluate.n_calls < max_fes:
            status = "under_budget"
            err_msg = f"eval_calls={evaluate.n_calls}<{max_fes}"
        if not np.isfinite(best_f):
            status = "invalid_result"
            err_msg = "optimizer returned a nonfinite best fitness"
    except Exception as exc:  # noqa: BLE001 — surface as failed run row
        status = "failed"
        err_msg = repr(exc)
        best_f = float("nan")
        best_x = np.full(DIM, np.nan)
        actual_fes = evaluate.n_calls
        curve_f = np.full(len(checkpoints), np.nan)
        meta = {"population": NPOP}

    elapsed = time.perf_counter() - t0
    abs_err = float(abs(best_f - optimum)) if np.isfinite(best_f) else float("nan")
    return {
        "function_id": fid,
        "function": f"C-14-f{fid}",
        "seed": seed,
        "dimension": DIM,
        "population": int((meta or {}).get("population", NPOP)),
        "max_fes": max_fes,
        "actual_fes": int(actual_fes),
        "eval_calls": int(evaluate.n_calls),
        "optimum": optimum,
        "best_fitness": float(best_f) if np.isfinite(best_f) else float("nan"),
        "error": float(best_f - optimum) if np.isfinite(best_f) else float("nan"),
        "abs_error": abs_err,
        "elapsed_sec": float(elapsed),
        "status": status,
        "error_msg": err_msg,
        "best_x": best_x.tolist(),
        "curve_fes": checkpoints.tolist(),
        "curve_error": (curve_f - optimum).tolist()
        if np.isfinite(curve_f).any()
        else [float("nan")] * len(checkpoints),
    }


def _sci(x: float) -> str:
    if not np.isfinite(x):
        return str(x)
    return f"{x:.2E}"


def export_outputs(runs: list[dict], out_dir: Path, run_id: str, param_line: str) -> float:
    if not runs:
        raise ValueError("cannot export an empty configuration")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = []
    for r in runs:
        row = {k: r[k] for k in (
            "function_id", "function", "seed", "dimension", "population",
            "max_fes", "actual_fes", "eval_calls", "optimum",
            "best_fitness", "error", "abs_error", "elapsed_sec", "status", "error_msg",
        )}
        for d, v in enumerate(r["best_x"], start=1):
            row[f"x{d}"] = v
        row["curve_fes"] = ";".join(str(int(fe)) for fe in r["curve_fes"])
        row["curve_error"] = ";".join(f"{e:.16g}" for e in r["curve_error"])
        rows.append(row)
    df = pd.DataFrame(rows).sort_values(["function_id", "seed"]).reset_index(drop=True)
    if df["seed"].nunique() != 1:
        raise ValueError("Project 3 scalar summaries require exactly one seed")
    if df["function_id"].duplicated().any():
        raise ValueError("each function must appear exactly once per configuration")
    n_fail = int((df["status"] != "ok").sum())

    lines = [
        f"Algorithm: NICE-DE (Project 3)",
        param_line,
        f"Run-ID: {run_id}",
        f"Seeds: {int(df['seed'].min())}-{int(df['seed'].max())}" if len(df) else "Seeds: n/a",
        "Metric: abs_error = |f(x_best) - f*| ; E_mean = mean abs_error over functions",
        f"Failed runs: {n_fail}/{len(df)}",
        "",
        f"{'F':<10}{'Best':>12}{'Median':>12}{'Ave':>12}{'Worst':>12}{'Std':>12}",
        "-" * 82,
    ]
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    for fid in sorted(df["function_id"].unique()):
        errs = ok.loc[ok["function_id"] == fid, "abs_error"].to_numpy()
        name = f"C-14-f{fid}"
        if len(errs) == 0:
            lines.append(f"{name:<10}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}")
            continue
        best = float(np.min(errs))
        median = float(np.median(errs))
        ave = float(np.mean(errs))
        worst = float(np.max(errs))
        std = float(np.std(errs, ddof=1)) if len(errs) > 1 else 0.0
        lines.append(
            f"{name:<10}{_sci(best):>12}{_sci(median):>12}"
            f"{_sci(ave):>12}{_sci(worst):>12}{_sci(std):>12}"
        )

    # Every requested function must succeed; never average a partial function set.
    errors = df["abs_error"].to_numpy(dtype=float)
    mean_abs = (
        float(np.mean(errors))
        if n_fail == 0 and np.isfinite(errors).all()
        else float("nan")
    )

    df.to_csv(out_dir / "results.csv", index=False)
    (out_dir / "results.txt").write_text("\n".join(lines) + "\n", encoding="utf-8")

    protocol = {
        "run_id": run_id,
        "algorithm": "NICE-DE",
        "dimension": DIM,
        "population": NPOP,
        "max_fes": int(df["max_fes"].iloc[0]) if len(df) else None,
        "functions": sorted(int(x) for x in df["function_id"].unique()),
        "seeds": sorted(int(x) for x in df["seed"].unique()),
        "n_runs": int(len(df)),
        "n_fail": n_fail,
        "mean_abs_error": mean_abs if np.isfinite(mean_abs) else None,
        "status": "ok" if n_fail == 0 and np.isfinite(mean_abs) else "fail",
        "param_line": param_line,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "numpy": np.__version__,
        "pandas": pd.__version__,
        "project_2": str(P2_ROOT),
    }
    (out_dir / "protocol.json").write_text(
        json.dumps(protocol, indent=2, ensure_ascii=False, allow_nan=False) + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {out_dir / 'results.csv'} ({len(df)} runs, fail={n_fail})", flush=True)
    print(f"[OK] wrote {out_dir / 'results.txt'}", flush=True)
    print(f"[OK] wrote {out_dir / 'protocol.json'}", flush=True)
    return mean_abs


def main() -> None:
    p = argparse.ArgumentParser(
        description="Project 3 runner: NICE on CEC2014 with configurable FE budget"
    )
    p.add_argument("--run-id", type=str, required=True, help="Unique run identifier")
    p.add_argument("--max-fes", type=int, required=True, help="Function-evaluation budget")
    p.add_argument("--funcs", type=str, default="1-30", help="Function ids, e.g. 1-30 or 1,2,3")
    p.add_argument("--seeds", type=str, default="1001", help="Exactly one non-negative integer seed (no lists or ranges)")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--out", type=str, required=True, help="Output directory for this run")
    args = p.parse_args()

    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", args.run_id):
        p.error("--run-id must contain only letters, digits, dots, underscores or hyphens")
    if args.max_fes < NPOP:
        p.error(f"--max-fes must be >= NP={NPOP}, got {args.max_fes}")
    if args.workers < 1:
        p.error("--workers must be >= 1")

    try:
        func_ids = _parse_funcs(args.funcs)
    except ValueError as exc:
        p.error(f"invalid --funcs: {exc}")
    seeds_spec = args.seeds.strip()
    if not re.fullmatch(r"[0-9]+", seeds_spec):
        p.error("--seeds requires exactly one non-negative integer, not a list or range")
    seed_list = [int(seeds_spec)]

    out_dir = Path(args.out).expanduser().resolve()
    if out_dir.exists() and (not out_dir.is_dir() or any(out_dir.iterdir())):
        p.error("--out must be a new or empty directory; existing results will not be overwritten")
    out_dir.mkdir(parents=True, exist_ok=True)

    param_line = (
        f"Parameters: D={DIM}, NP={NPOP} fixed (NICE), MaxFEs={args.max_fes}, "
        f"funcs={func_ids[0]}..{func_ids[-1]}, seeds={seed_list[0]}..{seed_list[-1]}, "
        f"workers={args.workers}"
    )
    print(
        f"Start P3 NICE | run_id={args.run_id} | max_fes={args.max_fes} | "
        f"funcs={func_ids[0]}..{func_ids[-1]} | seeds={seed_list} | "
        f"workers={args.workers} | out={out_dir}",
        flush=True,
    )

    tasks = [
        {"fid": fid, "seed": seed, "max_fes": int(args.max_fes)}
        for fid in func_ids
        for seed in seed_list
    ]
    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=int(args.workers)) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            results.append(res)
            done += 1
            if done % 5 == 0 or done == len(tasks):
                print(
                    f"[{done}/{len(tasks)}] last=F{t['fid']}/seed{t['seed']} "
                    f"status={res['status']} abs_error={res['abs_error']:.6e} "
                    f"elapsed={(time.perf_counter() - t0) / 60:.1f}min",
                    flush=True,
                )

    mean_abs = export_outputs(results, out_dir, args.run_id, param_line)
    n_fail = sum(1 for r in results if r["status"] != "ok")
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)
    if n_fail or not np.isfinite(mean_abs):
        # Bash writes the single failed P3_RESULT record. Keep partial results
        # for diagnosis, but do not advertise them as a valid scalar summary.
        raise SystemExit(f"configuration failed: failed_tasks={n_fail}; no valid scalar summary")
    # Machine-readable line for Bash grep|awk (do not change field order lightly)
    print(
        "P3_RESULT"
        f"|run_id={args.run_id}"
        f"|max_fes={int(args.max_fes)}"
        f"|seed={seed_list[0]}"
        f"|n_funcs={len(func_ids)}"
        f"|mean_abs_error={mean_abs:.16e}"
        f"|n_fail={n_fail}"
        f"|out={out_dir}"
        "|status=ok",
        flush=True,
    )


if __name__ == "__main__":
    main()
