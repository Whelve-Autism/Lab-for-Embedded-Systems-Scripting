"""Shared CEC2014 experiment harness for Project 2 methods."""

from __future__ import annotations

import argparse
import importlib
import json
import platform
import sys
import time
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore", message="pkg_resources is deprecated")

# project_2 root on sys.path
_COMMON_DIR = Path(__file__).resolve().parent
_P2_ROOT = _COMMON_DIR.parent
if str(_P2_ROOT) not in sys.path:
    sys.path.insert(0, str(_P2_ROOT))

from common.schwefel_patch import apply_schwefel_patch

DIM = 10
MAX_FES = 100_000
SEEDS = list(range(1001, 1021))
LOWER, UPPER = -100.0, 100.0
CHECKPOINTS = np.array([100] + list(range(1000, MAX_FES + 1, 1000)), dtype=int)

OptimizeFn = Callable[..., tuple]


class CountingEvaluate:
    """Wrap objective: count every real call (no silent skips)."""

    def __init__(self, fn, max_fes: int):
        self._fn = fn
        self.max_fes = int(max_fes)
        self.n_calls = 0

    def __call__(self, x):
        self.n_calls += 1
        return float(self._fn(x))


def _make_func(fid: int, dim: int = DIM):
    mod = importlib.import_module("opfunu.cec_based.cec2014")
    return getattr(mod, f"F{fid}2014")(ndim=dim)


def _sci(x: float) -> str:
    if not np.isfinite(x):
        return str(x)
    return f"{x:.2E}"


def export_outputs(
    runs: list[dict],
    out_dir: Path,
    algorithm_name: str,
    param_line: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    run_rows = []
    for r in runs:
        row = {k: r[k] for k in (
            "function_id", "function", "seed", "dimension", "population",
            "max_fes", "actual_fes", "eval_calls", "optimum",
            "best_fitness", "error", "elapsed_sec", "status",
        )}
        for d, v in enumerate(r["best_x"], start=1):
            row[f"x{d}"] = v
        row["curve_fes"] = ";".join(str(int(fe)) for fe in r["curve_fes"])
        row["curve_error"] = ";".join(f"{e:.16g}" for e in r["curve_error"])
        run_rows.append(row)
    df = pd.DataFrame(run_rows).sort_values(
        ["function_id", "seed"]
    ).reset_index(drop=True)

    n_fail = int((df["status"] != "ok").sum()) if "status" in df.columns else 0
    lines = [
        f"Algorithm: {algorithm_name}",
        param_line,
        f"Seeds: {int(df['seed'].min())}-{int(df['seed'].max())}"
        if len(df) else f"Seeds: {SEEDS[0]}-{SEEDS[-1]}",
        "Metric: error = f(x_best) - f*",
        f"Failed runs: {n_fail}/{len(df)}",
        "",
    ]
    header = f"{'F':<10}{'Best':>12}{'Median':>12}{'Ave':>12}{'Worst':>12}{'Std':>12}"
    lines.append(header)
    lines.append("-" * len(header))
    ok = df[df["status"] == "ok"] if "status" in df.columns else df
    for fid in sorted(df["function_id"].unique()):
        errs = ok.loc[ok["function_id"] == fid, "error"].to_numpy()
        if len(errs) == 0:
            name = f"C-14-f{fid}"
            lines.append(f"{name:<10}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}{'FAIL':>12}")
            continue
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

    csv_path = out_dir / "results.csv"
    txt_path = out_dir / "results.txt"
    meta_path = out_dir / "run_meta.txt"
    df.to_csv(csv_path, index=False)
    txt_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    meta_path.write_text(
        "\n".join(
            [
                f"algorithm={algorithm_name}",
                f"param={param_line}",
                f"dim={DIM}",
                f"max_fes={MAX_FES}",
                (
                    f"seeds={int(df['seed'].min())}-{int(df['seed'].max())}"
                    if len(df) else f"seeds={SEEDS[0]}-{SEEDS[-1]}"
                ),
                f"checkpoints={CHECKPOINTS.tolist()[:5]}...({len(CHECKPOINTS)})",
                f"python={sys.version.split()[0]}",
                f"platform={platform.platform()}",
                f"numpy={np.__version__}",
                f"pandas={pd.__version__}",
                f"n_runs={len(df)}",
                f"n_fail={n_fail}",
            ]
        )
        + "\n",
        encoding="utf-8",
    )
    print(f"[OK] wrote {csv_path} ({len(df)} runs, fail={n_fail})")
    print(f"[OK] wrote {txt_path}")

    def _jsonable(obj):
        if isinstance(obj, dict):
            return {str(k): _jsonable(v) for k, v in obj.items()}
        if isinstance(obj, (list, tuple)):
            return [_jsonable(v) for v in obj]
        if isinstance(obj, np.ndarray):
            return obj.tolist()
        if isinstance(obj, (np.floating, np.integer)):
            return obj.item()
        if isinstance(obj, float) and not np.isfinite(obj):
            return None
        return obj

    # Optional per-run diagnostics (PGDE game / SRADE state / ORDE direction).
    game_rows = [r for r in runs if r.get("game_trace")]
    if game_rows:
        trace_path = out_dir / "game_traces.jsonl"
        with trace_path.open("w", encoding="utf-8") as fh:
            for r in sorted(game_rows, key=lambda x: (x["function_id"], x["seed"])):
                fh.write(json.dumps(_jsonable({
                    "function_id": r["function_id"],
                    "seed": r["seed"],
                    "strategy_fes": r.get("strategy_fes"),
                    "game_trace": r["game_trace"],
                }), ensure_ascii=False) + "\n")
        print(f"[OK] wrote {trace_path} ({len(game_rows)} runs)")

    state_rows = [r for r in runs if r.get("state_traces")]
    if state_rows:
        state_path = out_dir / "state_traces.jsonl"
        with state_path.open("w", encoding="utf-8") as fh:
            for r in sorted(state_rows, key=lambda x: (x["function_id"], x["seed"])):
                for record in r["state_traces"]:
                    fh.write(json.dumps(_jsonable({
                        "function_id": r["function_id"],
                        "seed": r["seed"],
                        **record,
                    }), ensure_ascii=False) + "\n")
        print(f"[OK] wrote {state_path} ({len(state_rows)} runs)")

    dir_rows = [r for r in runs if r.get("direction_traces")]
    if dir_rows:
        dir_path = out_dir / "direction_traces.jsonl"
        with dir_path.open("w", encoding="utf-8") as fh:
            for r in sorted(dir_rows, key=lambda x: (x["function_id"], x["seed"])):
                for record in r["direction_traces"]:
                    fh.write(json.dumps(_jsonable({
                        "function_id": r["function_id"],
                        "seed": r["seed"],
                        "mode": r.get("orde_mode"),
                        **record,
                    }), ensure_ascii=False) + "\n")
        print(f"[OK] wrote {dir_path} ({len(dir_rows)} runs)")

    pair_rows = [r for r in runs if r.get("pair_traces")]
    if pair_rows:
        pair_path = out_dir / "pair_traces.jsonl"
        with pair_path.open("w", encoding="utf-8") as fh:
            for r in sorted(pair_rows, key=lambda x: (x["function_id"], x["seed"])):
                for record in r["pair_traces"]:
                    fh.write(json.dumps(_jsonable({
                        "function_id": r["function_id"],
                        "seed": r["seed"],
                        "mode": r.get("cape_mode"),
                        **record,
                    }), ensure_ascii=False) + "\n")
        print(f"[OK] wrote {pair_path} ({len(pair_rows)} runs)")

    diag_rows = [r for r in runs if r.get("nice_diagnostics")]
    if diag_rows:
        diag_path = out_dir / "diagnostics.jsonl"
        with diag_path.open("w", encoding="utf-8") as fh:
            for r in sorted(diag_rows, key=lambda x: (x["function_id"], x["seed"])):
                fh.write(json.dumps(_jsonable({
                    "function_id": r["function_id"],
                    "seed": r["seed"],
                    **r["nice_diagnostics"],
                }), ensure_ascii=False) + "\n")
        print(f"[OK] wrote {diag_path} ({len(diag_rows)} runs)")


def _worker(payload: dict) -> dict:
    """Run one (function, seed) experiment. payload carries method import path."""
    apply_schwefel_patch()
    method_mod = importlib.import_module(payload["module"])
    optimize = getattr(method_mod, payload["optimize_name"])

    fid = payload["fid"]
    seed = payload["seed"]
    rng = np.random.default_rng(seed)
    func = _make_func(fid)
    optimum = float(func.f_bias)

    raw_evaluate = lambda x: float(func.evaluate(np.asarray(x, dtype=float)))
    evaluate = CountingEvaluate(raw_evaluate, MAX_FES)

    t0 = time.perf_counter()
    status = "ok"
    err_msg = ""
    try:
        opt_kwargs = dict(payload.get("optimize_kwargs") or {})
        best_f, best_x, actual_fes, curve_f, meta = optimize(
            evaluate=evaluate,
            lower=LOWER,
            upper=UPPER,
            dim=DIM,
            max_fes=MAX_FES,
            rng=rng,
            checkpoints=CHECKPOINTS,
            **opt_kwargs,
        )
        best_f = float(best_f)
        best_x = np.asarray(best_x, dtype=float)
        curve_f = np.asarray(curve_f, dtype=float)
        # Consistency checks (soft flags)
        if evaluate.n_calls != int(actual_fes):
            status = "fe_mismatch"
            err_msg = f"eval_calls={evaluate.n_calls} actual_fes={actual_fes}"
        if evaluate.n_calls > MAX_FES:
            status = "over_budget"
            err_msg = f"eval_calls={evaluate.n_calls}>{MAX_FES}"
        if len(curve_f) and np.isfinite(curve_f[-1]) and abs(curve_f[-1] - best_f) > 1e-6 * max(1.0, abs(best_f)):
            # curve end should match returned best (observed)
            if status == "ok":
                status = "curve_mismatch"
            err_msg = (err_msg + f"; curve_end={curve_f[-1]} best={best_f}").strip("; ")
    except Exception as exc:
        status = "failed"
        err_msg = repr(exc)
        best_f = np.nan
        best_x = np.full(DIM, np.nan)
        actual_fes = evaluate.n_calls
        curve_f = np.full(len(CHECKPOINTS), np.nan)
        meta = {"population": -1}

    elapsed = time.perf_counter() - t0
    out = {
        "function_id": fid,
        "function": f"C-14-f{fid}",
        "seed": seed,
        "dimension": DIM,
        "population": int(meta.get("population", -1)),
        "max_fes": MAX_FES,
        "actual_fes": int(actual_fes),
        "eval_calls": int(evaluate.n_calls),
        "optimum": optimum,
        "best_fitness": float(best_f) if np.isfinite(best_f) else np.nan,
        "error": float(best_f - optimum) if np.isfinite(best_f) else np.nan,
        "elapsed_sec": float(elapsed),
        "best_x": best_x.tolist(),
        "curve_fes": CHECKPOINTS.tolist(),
        "curve_error": (curve_f - optimum).tolist(),
        "status": status,
        "error_msg": err_msg,
    }
    if isinstance(meta, dict):
        if meta.get("game_trace") is not None:
            out["game_trace"] = meta["game_trace"]
        if meta.get("strategy_fes") is not None:
            out["strategy_fes"] = meta["strategy_fes"]
        if meta.get("state_traces") is not None:
            out["state_traces"] = meta["state_traces"]
        if meta.get("direction_traces") is not None:
            out["direction_traces"] = meta["direction_traces"]
        if meta.get("orde_mode") is not None:
            out["orde_mode"] = meta["orde_mode"]
        if meta.get("pair_traces") is not None:
            out["pair_traces"] = meta["pair_traces"]
        if meta.get("cape_mode") is not None:
            out["cape_mode"] = meta["cape_mode"]
        if meta.get("nice_diagnostics") is not None:
            out["nice_diagnostics"] = meta["nice_diagnostics"]
    return out


def _parse_seed_spec(seeds: str | None) -> list[int]:
    if not seeds:
        return list(SEEDS)
    seeds = seeds.strip()
    if ":" in seeds:
        a, b = seeds.split(":", 1)
        return list(range(int(a), int(b) + 1))
    if "-" in seeds and "," not in seeds:
        a, b = seeds.split("-", 1)
        # Avoid treating negative numbers; seed ranges are positive.
        if a.isdigit() and b.isdigit():
            return list(range(int(a), int(b) + 1))
    return [int(x) for x in seeds.split(",") if x.strip()]


def run_experiment(
    *,
    module: str,
    optimize_name: str,
    algorithm_name: str,
    param_line: str,
    out_dir: Path,
    workers: int = 32,
    funcs: str = "1-30",
    seeds: str | None = None,
    optimize_kwargs: dict | None = None,
) -> None:
    if "-" in funcs:
        a, b = funcs.split("-", 1)
        func_ids = list(range(int(a), int(b) + 1))
    else:
        func_ids = [int(x) for x in funcs.split(",") if x.strip()]

    seed_list = _parse_seed_spec(seeds)
    kw = dict(optimize_kwargs or {})
    tasks = []
    for fid in func_ids:
        for seed in seed_list:
            tasks.append({
                "module": module,
                "optimize_name": optimize_name,
                "fid": fid,
                "seed": seed,
                "optimize_kwargs": kw,
            })

    total = len(tasks)
    out_dir.mkdir(parents=True, exist_ok=True)
    print(
        f"Start {algorithm_name} | funcs={func_ids[0]}..{func_ids[-1]} "
        f"| seeds={seed_list[0]}..{seed_list[-1]} "
        f"| tasks={total} | workers={workers} | out={out_dir}",
        flush=True,
    )
    results: list[dict] = []
    done = 0
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=workers) as ex:
        futs = {ex.submit(_worker, t): t for t in tasks}
        for fut in as_completed(futs):
            t = futs[fut]
            res = fut.result()
            results.append(res)
            done += 1
            if done % 10 == 0 or done == total:
                st = res.get("status", "?")
                err = res.get("error", float("nan"))
                print(
                    f"[{done}/{total}] last=F{t['fid']}/seed{t['seed']} "
                    f"status={st} error={err:.6e} "
                    f"elapsed={(time.perf_counter() - t0) / 60:.1f}min",
                    flush=True,
                )
    export_outputs(results, out_dir, algorithm_name, param_line)
    print(f"All done in {(time.perf_counter() - t0) / 60:.1f} min", flush=True)


def build_argparser(description: str) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=description)
    p.add_argument("--workers", type=int, default=32)
    p.add_argument("--funcs", type=str, default="1-30")
    p.add_argument(
        "--seeds",
        type=str,
        default=None,
        help="Seed list/range, e.g. 2101:2120 or 1001-1020 (default: 1001-1020)",
    )
    return p
