"""
Export F1–F30 convergence PDFs to out/plot/ (1:1 canvas, Times New Roman).

Each figure: median error curve + IQR band over 20 runs.
Curve / band colors follow a global log-scale rainbow:
  high error -> #d62728 ... low error -> #9467bd.
All 30 PDFs share identical layout geometry (no title).
"""

from __future__ import annotations

from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.collections import LineCollection
from matplotlib.colors import LinearSegmentedColormap, LogNorm
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
CSV_PATH = ROOT / "out" / "results.csv"
OUT_DIR = ROOT / "out" / "plot"

FIG_INCH = 6.5

# Absolute axes box in figure fraction.
# right = 0.215+0.763 = 0.978, top = 0.150+0.835 = 0.985
AXES_RECT = (0.215, 0.150, 0.763, 0.835)

XLABEL_FIG = (0.596, 0.034)
YLABEL_FIG = (0.038, 0.567)

# Conceptual high->low: red -> orange -> yellow -> green -> blue -> purple.
# Matplotlib cmap maps 0=low / 1=high, so build then reverse.
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
# Major x-axis ticks only — markers are placed at these abscissa nodes.
X_TICKS = np.array([0, 20_000, 40_000, 60_000, 80_000, 100_000], dtype=float)

FONT_LABEL = 38
FONT_TICK = 32

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
    out = np.maximum(out, ERROR_FLOOR)
    return out


def global_error_range(df: pd.DataFrame) -> tuple[float, float]:
    """Min/max display error over all 600 runs × checkpoints (for shared color scale)."""
    vals: list[np.ndarray] = []
    for _, row in df.iterrows():
        e = np.array([float(x) for x in str(row["curve_error"]).split(";")], dtype=float)
        vals.append(display_error(e))
    arr = np.concatenate(vals)
    arr = arr[np.isfinite(arr)]
    vmin = float(np.min(arr))
    vmax = float(np.max(arr))
    if vmax <= vmin:
        vmax = vmin * 10.0
    return vmin, vmax


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, format="pdf", bbox_inches=None, pad_inches=0)


def _seg_values(y: np.ndarray) -> np.ndarray:
    """Log-space midpoint of each consecutive pair (for segment coloring)."""
    y0 = np.asarray(y, dtype=float)
    return np.sqrt(np.clip(y0[:-1], ERROR_FLOOR, None) * np.clip(y0[1:], ERROR_FLOOR, None))


def draw_gradient_band(
    ax,
    x: np.ndarray,
    ylo: np.ndarray,
    yhi: np.ndarray,
    yref: np.ndarray,
    norm: LogNorm,
) -> None:
    """IQR ribbon: same rainbow as the curve, lighter alpha, colored by error."""
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


def draw_gradient_line(ax, x: np.ndarray, y: np.ndarray, norm: LogNorm) -> None:
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

    # Circle markers only at major x-axis tick nodes: white fill, colored rim.
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


def plot_one(
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

    # Shared global log color scale across all 30 figures.
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

    fig.canvas.draw()
    nudged_top = _nudge_edge_tick_labels(fig, ax)

    save_fig(out_path)
    plt.close(fig)
    return nudged_top


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
    """True if tick text extends past / into the top edge of the figure canvas."""
    renderer = fig.canvas.get_renderer()
    bbox = label.get_window_extent(renderer=renderer)
    return float(bbox.y1) > float(fig.bbox.y1) - float(margin_px)


def _nudge_edge_tick_labels(fig, ax) -> bool:
    """
    Always shift rightmost x tick text left a bit.
    Top y tick text: only shift down when it would be clipped by the canvas.
    """
    x_off = mpl.transforms.ScaledTranslation(-22.0 / 72.0, 0.0, fig.dpi_scale_trans)
    for tick in ax.xaxis.get_major_ticks():
        if abs(float(tick.get_loc()) - float(X_MAX)) > 1e-9:
            continue
        tick.label1.set_clip_on(False)
        tick.label1.set_transform(tick.label1.get_transform() + x_off)
        break

    top_tick, top_loc = _top_major_y_tick(ax)
    if top_tick is None:
        return False

    # Default: no downward shift. Check first; nudge only if clipped.
    fig.canvas.draw()
    if not _label_clips_figure_top(fig, top_tick.label1):
        return False

    y_off = mpl.transforms.ScaledTranslation(0.0, -12.0 / 72.0, fig.dpi_scale_trans)
    top_tick.label1.set_clip_on(False)
    top_tick.label1.set_transform(top_tick.label1.get_transform() + y_off)
    fig.canvas.draw()
    return True


def main() -> None:
    set_style()
    if not CSV_PATH.exists():
        raise SystemExit(f"Missing results CSV: {CSV_PATH}")

    df = pd.read_csv(CSV_PATH)
    vmin, vmax = global_error_range(df)
    norm = LogNorm(vmin=vmin, vmax=vmax)
    print(f"Global error color scale (log): [{vmin:.3e}, {vmax:.3e}]")
    print("All 30 figures share this single mapping (high=#d62728, low=#9467bd).")

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    written: list[Path] = []
    nudged_ids: list[int] = []
    for fid in range(1, 31):
        sub = df[df["function_id"] == fid].sort_values("seed")
        if len(sub) != 20:
            raise SystemExit(f"F{fid}: expected 20 runs, got {len(sub)}")
        fes, errors = parse_curves(sub)
        out_path = OUT_DIR / f"f{fid}.pdf"
        nudged = plot_one(fes, errors, out_path, norm)
        written.append(out_path)
        if nudged:
            nudged_ids.append(fid)

    print(f"Wrote {len(written)} PDF files under {OUT_DIR}")
    print(
        "Top y-tick nudged down (clipped only): "
        + (", ".join(f"F{i}" for i in nudged_ids) if nudged_ids else "(none)")
    )


if __name__ == "__main__":
    main()
