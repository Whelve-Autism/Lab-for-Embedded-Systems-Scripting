"""
Export F1–F30 figure PDFs under out/plot/:

  out/plot/curve/   — median + IQR convergence curves
  out/plot/box/     — final-error box plots (20 seeds)

Shared 1:1 canvas / fonts / axes geometry.
Curve & box colors share one global log-scale rainbow:
  high error -> #d62728 ... low error -> #9467bd.
"""

from __future__ import annotations

import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm, to_hex
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "out" / "results.csv"
PLOT_ROOT = ROOT / "out" / "plot"
CURVE_DIR = PLOT_ROOT / "curve"
BOX_DIR = PLOT_ROOT / "box"

FIG_INCH = 6.5

# Absolute axes box in figure fraction.
AXES_RECT = (0.215, 0.150, 0.763, 0.835)

# Curve keeps the original anchor; box "Algorithm" is slightly higher
# so the descender of "g" is not clipped.
XLABEL_FIG = (0.596, 0.034)
BOX_XLABEL_FIG = (0.596, 0.040)
YLABEL_FIG = (0.038, 0.567)

RAINBOW_HIGH_TO_LOW = (
    "#d62728",
    "#ff7f0e",
    "#f1c40f",
    "#2ca02c",
    "#1f77b4",
    "#9467bd",
)
BAND_ALPHA = 0.34
LINEWIDTH = 8.5
MARKER_SIZE = 280
MARKER_EDGEWIDTH = 4.4

ERROR_FLOOR = 1e-12
X_MAX = 100_000
X_TICKS = np.array([0, 20_000, 40_000, 60_000, 80_000, 100_000], dtype=float)

FONT_LABEL = 38
FONT_TICK = 32
LEGEND_FONTSIZE = 18
LEGEND_FRAME_ALPHA = 0.78
METHOD_LABEL = "BIDE"

ERROR_CMAP = LinearSegmentedColormap.from_list(
    "err_rainbow", RAINBOW_HIGH_TO_LOW
).reversed()


def set_style() -> None:
    mpl.rcParams["font.family"] = "Times New Roman"
    mpl.rcParams["axes.unicode_minus"] = False
    mpl.rcParams["pdf.fonttype"] = 42
    mpl.rcParams["font.size"] = FONT_TICK
    mpl.rcParams["axes.labelsize"] = FONT_LABEL
    mpl.rcParams["xtick.labelsize"] = FONT_TICK
    mpl.rcParams["ytick.labelsize"] = FONT_TICK
    mpl.rcParams["axes.linewidth"] = 6.4
    mpl.rcParams["xtick.major.width"] = 5.2
    mpl.rcParams["ytick.major.width"] = 5.2
    mpl.rcParams["xtick.minor.width"] = 3.6
    mpl.rcParams["ytick.minor.width"] = 3.6
    mpl.rcParams["xtick.major.size"] = 13.5
    mpl.rcParams["ytick.major.size"] = 13.5
    mpl.rcParams["xtick.minor.size"] = 7.5
    mpl.rcParams["ytick.minor.size"] = 7.5
    mpl.rcParams["grid.linewidth"] = 2.7
    mpl.rcParams["xtick.direction"] = "in"
    mpl.rcParams["ytick.direction"] = "in"
    mpl.rcParams["xtick.top"] = False
    mpl.rcParams["ytick.right"] = False
    mpl.rcParams["patch.linewidth"] = 1.4


def parse_curves(df_func: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    fes = np.array(
        [int(x) for x in str(df_func.iloc[0]["curve_fes"]).split(";")],
        dtype=float,
    )
    rows = []
    for _, row in df_func.iterrows():
        errs = np.array(
            [float(x) for x in str(row["curve_error"]).split(";")],
            dtype=float,
        )
        rows.append(errs)
    return fes, np.vstack(rows)


def display_error(err: np.ndarray) -> np.ndarray:
    out = np.asarray(err, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return np.maximum(out, ERROR_FLOOR)


def global_error_range(df: pd.DataFrame) -> tuple[float, float]:
    vals: list[np.ndarray] = []
    for _, row in df.iterrows():
        e = np.array([float(x) for x in str(row["curve_error"]).split(";")], dtype=float)
        vals.append(display_error(e))
        vals.append(display_error(np.array([float(row["error"])], dtype=float)))
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def color_for_error(value: float, norm: LogNorm) -> str:
    rgba = ERROR_CMAP(norm(float(max(value, ERROR_FLOOR))))
    return to_hex(rgba, keep_alpha=False)


def fid_seed_key(out_path: Path) -> int:
    """Stable RNG seed from f{id}.pdf so jitter is reproducible per function."""
    stem = out_path.stem.lower()
    if stem.startswith("f") and stem[1:].isdigit():
        return 10_000 + int(stem[1:])
    return 10_000


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, format="pdf", bbox_inches=None, pad_inches=0)


def _seg_values(y: np.ndarray) -> np.ndarray:
    y0 = np.asarray(y, dtype=float)
    return np.sqrt(np.clip(y0[:-1], ERROR_FLOOR, None) * np.clip(y0[1:], ERROR_FLOOR, None))


def draw_gradient_band(ax, x, ylo, yhi, yref, norm: LogNorm) -> None:
    seg_v = _seg_values(yref)
    for i, v in enumerate(seg_v):
        color = ERROR_CMAP(norm(float(v)))
        ax.fill_between(
            x[i : i + 2],
            ylo[i : i + 2],
            yhi[i : i + 2],
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2,
        )


def draw_gradient_line(ax, x, y, norm: LogNorm) -> None:
    points = np.column_stack([x, y]).reshape(-1, 1, 2)
    segs = np.concatenate([points[:-1], points[1:]], axis=1)
    lc = LineCollection(
        segs,
        cmap=ERROR_CMAP,
        norm=norm,
        linewidths=LINEWIDTH,
        capstyle="round",
        joinstyle="round",
        zorder=3,
    )
    lc.set_array(_seg_values(y))
    ax.add_collection(lc)

    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)
    mx, my = [], []
    for tick in X_TICKS:
        idx = int(np.argmin(np.abs(x - tick)))
        mx.append(float(x[idx]))
        my.append(float(y[idx]))
    mx_a = np.asarray(mx, dtype=float)
    my_a = np.asarray(my, dtype=float)
    rim = ERROR_CMAP(norm(np.clip(my_a, ERROR_FLOOR, None)))
    ax.scatter(
        mx_a,
        my_a,
        s=MARKER_SIZE,
        marker="o",
        facecolors="white",
        edgecolors=rim,
        linewidths=MARKER_EDGEWIDTH,
        zorder=4,
        clip_on=True,
    )


def _add_axis_labels(fig) -> None:
    fig.text(
        XLABEL_FIG[0],
        XLABEL_FIG[1],
        "Function evaluations",
        ha="center",
        va="center",
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )
    fig.text(
        YLABEL_FIG[0],
        YLABEL_FIG[1],
        "Optimization error",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )


def _top_major_y_tick(ax):
    ymin, ymax = ax.get_ylim()
    top_tick = None
    top_loc = None
    for tick in ax.yaxis.get_major_ticks():
        loc = float(tick.get_loc())
        if loc < ymin or loc > ymax * (1.0 + 1e-12):
            continue
        if not tick.label1.get_visible():
            continue
        if top_loc is None or loc > top_loc:
            top_loc = loc
            top_tick = tick
    return top_tick, top_loc


def _label_clips_figure_top(fig, label, margin_px: float = 0.5) -> bool:
    renderer = fig.canvas.get_renderer()
    bbox = label.get_window_extent(renderer=renderer)
    return float(bbox.y1) > float(fig.bbox.y1) - float(margin_px)


def _nudge_edge_tick_labels(fig, ax, *, nudge_xmax: bool) -> bool:
    if nudge_xmax:
        x_off = mpl.transforms.ScaledTranslation(-22.0 / 72.0, 0.0, fig.dpi_scale_trans)
        for tick in ax.xaxis.get_major_ticks():
            if abs(float(tick.get_loc()) - float(X_MAX)) > 1e-9:
                continue
            tick.label1.set_clip_on(False)
            tick.label1.set_transform(tick.label1.get_transform() + x_off)
            break

    top_tick, _ = _top_major_y_tick(ax)
    if top_tick is None:
        return False

    fig.canvas.draw()
    if not _label_clips_figure_top(fig, top_tick.label1):
        return False

    y_off = mpl.transforms.ScaledTranslation(0.0, -12.0 / 72.0, fig.dpi_scale_trans)
    top_tick.label1.set_clip_on(False)
    top_tick.label1.set_transform(top_tick.label1.get_transform() + y_off)
    fig.canvas.draw()
    return True


def plot_curve(
    fes: np.ndarray,
    errors: np.ndarray,
    out_path: Path,
    norm: LogNorm,
) -> bool:
    disp = display_error(errors)
    q25 = np.nanpercentile(disp, 25, axis=0)
    q50 = np.nanpercentile(disp, 50, axis=0)
    q75 = np.nanpercentile(disp, 75, axis=0)

    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))
    ax.set_position(AXES_RECT)

    ax.set_yscale("log")
    ax.set_xlim(0.0, float(X_MAX))

    ymin = float(np.nanmin(q25))
    ymax = float(np.nanmax(q75))
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = ERROR_FLOOR, 1.0
    ymin = max(ERROR_FLOOR, ymin / 1.8)
    ymax = max(ymax * 1.8, ymin * 10.0)
    ax.set_ylim(ymin, ymax)

    draw_gradient_band(ax, fes, q25, q75, q50, norm)
    draw_gradient_line(ax, fes, q50, norm)

    ax.set_xticks(list(X_TICKS))
    ax.set_xticklabels(["0", "2e4", "4e4", "6e4", "8e4", "1e5"])
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="both", which="major", labelsize=FONT_TICK, pad=5)
    ax.tick_params(axis="both", which="minor", labelsize=FONT_TICK)
    ax.grid(True, which="major", alpha=0.25)
    ax.grid(True, which="minor", alpha=0.12)

    _add_axis_labels(fig)

    fig.canvas.draw()
    nudged_top = _nudge_edge_tick_labels(fig, ax, nudge_xmax=True)

    save_fig(out_path)
    plt.close(fig)
    return nudged_top


def plot_box(
    final_errors: np.ndarray,
    out_path: Path,
    norm: LogNorm,
) -> bool:
    """Single-method final-error box plot; color matches curve rainbow at median."""
    disp = display_error(final_errors)
    disp = disp[np.isfinite(disp)]
    if len(disp) == 0:
        raise SystemExit(f"no finite final errors for {out_path.name}")

    median = float(np.median(disp))
    color = color_for_error(median, norm)

    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))
    ax.set_position(AXES_RECT)
    ax.set_yscale("log")

    bp = ax.boxplot(
        [disp],
        positions=[1.0],
        widths=0.45,
        patch_artist=True,
        showfliers=False,  # all 20 runs drawn as scatter below
        whis=1.5,
        medianprops={"color": "black", "linewidth": 3.2},
        whiskerprops={"color": color, "linewidth": 3.6},
        capprops={"color": color, "linewidth": 3.6},
        boxprops={"linewidth": 3.6, "edgecolor": color},
    )
    for patch in bp["boxes"]:
        patch.set_facecolor(color)
        patch.set_alpha(0.38)
        patch.set_edgecolor(color)

    # One marker per independent run (20 seeds), lightly jittered on x.
    rng = np.random.default_rng(fid_seed_key(out_path))
    jitter = rng.uniform(-0.08, 0.08, size=len(disp))
    ax.scatter(
        np.full(len(disp), 1.0) + jitter,
        disp,
        s=110,
        marker="o",
        facecolors="white",
        edgecolors=color,
        linewidths=2.6,
        zorder=5,
        clip_on=True,
    )

    ymin = float(np.min(disp))
    ymax = float(np.max(disp))
    ymin = max(ERROR_FLOOR, ymin / 1.8)
    ymax = max(ymax * 1.8, ymin * 10.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.35, 1.65)

    ax.set_xticks([1.0])
    ax.set_xticklabels([])  # names only in legend; keep Algorithm + legend
    ax.yaxis.set_major_locator(LogLocator(base=10.0))
    ax.yaxis.set_minor_locator(LogLocator(base=10.0, subs=np.arange(2, 10) * 0.1))
    ax.yaxis.set_minor_formatter(NullFormatter())

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.tick_params(axis="y", which="major", labelsize=FONT_TICK, pad=5)
    ax.tick_params(axis="y", which="minor", labelsize=FONT_TICK)
    ax.tick_params(axis="x", which="major", length=7.5, pad=2)
    ax.grid(True, which="major", axis="y", alpha=0.25)
    ax.grid(True, which="minor", axis="y", alpha=0.12)

    fig.text(
        BOX_XLABEL_FIG[0],
        BOX_XLABEL_FIG[1],
        "Algorithm",
        ha="center",
        va="center",
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )
    fig.text(
        YLABEL_FIG[0],
        YLABEL_FIG[1],
        "Optimization error",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )

    leg = ax.legend(
        handles=[
            Patch(
                facecolor=color,
                edgecolor=color,
                alpha=0.38,
                linewidth=3.0,
                label=METHOD_LABEL,
            )
        ],
        loc="upper right",
        fontsize=LEGEND_FONTSIZE,
        framealpha=LEGEND_FRAME_ALPHA,
        edgecolor="0.35",
        fancybox=False,
        borderpad=0.45,
    )
    leg.set_zorder(120)

    fig.canvas.draw()
    nudged_top = _nudge_edge_tick_labels(fig, ax, nudge_xmax=False)

    save_fig(out_path)
    plt.close(fig)
    return nudged_top


def migrate_legacy_curve_pdfs() -> int:
    """Move out/plot/f*.pdf -> out/plot/curve/ if leftover from older layout."""
    if not PLOT_ROOT.is_dir():
        return 0
    moved = 0
    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    for path in sorted(PLOT_ROOT.glob("f*.pdf")):
        dest = CURVE_DIR / path.name
        if dest.exists():
            path.unlink()
        else:
            shutil.move(str(path), str(dest))
        moved += 1
    return moved


def main() -> None:
    set_style()
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing results CSV: {CSV_PATH}")

    n_migrated = migrate_legacy_curve_pdfs()
    if n_migrated:
        print(f"Moved {n_migrated} legacy curve PDFs -> {CURVE_DIR}")

    df = pd.read_csv(CSV_PATH)
    vmin, vmax = global_error_range(df)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    print(f"Global error color scale (log): [{vmin:.3e}, {vmax:.3e}]")
    print("Curve/box share this mapping (high=#d62728, low=#9467bd).")

    CURVE_DIR.mkdir(parents=True, exist_ok=True)
    BOX_DIR.mkdir(parents=True, exist_ok=True)

    curve_nudged: list[int] = []
    box_nudged: list[int] = []
    for fid in range(1, 31):
        sub = df[df["function_id"] == fid].sort_values("seed")
        if len(sub) != 20:
            raise SystemExit(f"F{fid}: expected 20 runs, got {len(sub)}")
        fes, errors = parse_curves(sub)
        finals = sub["error"].to_numpy(dtype=float)

        if plot_curve(fes, errors, CURVE_DIR / f"f{fid}.pdf", norm):
            curve_nudged.append(fid)
        if plot_box(finals, BOX_DIR / f"f{fid}.pdf", norm):
            box_nudged.append(fid)

    print(f"Wrote 30 curve PDFs under {CURVE_DIR}")
    print(f"Wrote 30 box PDFs under {BOX_DIR}")
    print(
        "Curve top y-tick nudged: "
        + (", ".join(f"F{i}" for i in curve_nudged) if curve_nudged else "(none)")
    )
    print(
        "Box top y-tick nudged: "
        + (", ".join(f"F{i}" for i in box_nudged) if box_nudged else "(none)")
    )


if __name__ == "__main__":
    main()
