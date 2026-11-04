"""
Export F1–F30 multi-method convergence PDFs.

  out/plot/main/      — main experiment (6 algorithms)
  out/plot/ablation/  — NICE ablation suite

Layout / canvas / fonts / markers match across both packs.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out"
PLOT_ROOT = OUT_ROOT / "plot"

FIG_INCH = 6.5

AXES_RECT = (0.215, 0.150, 0.763, 0.835)
XLABEL_FIG = (0.596, 0.034)
YLABEL_FIG = (0.045, 0.567)

BAND_ALPHA = 0.22
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
LEGEND_ZORDER = 120

# folder, legend label, color
MAIN_METHODS = (
    ("NICE_DE", "NICE", "#d62728"),
    ("DLS_DE", "DLS-DE", "#ff7f0e"),
    ("DE_VCR", "DE-VCR", "#f1c40f"),
    ("ECL_SPACMA", "ECL-SPACMA", "#2ca02c"),
    ("iStratDE", "iStratDE", "#1f77b4"),
    ("BIDE", "BIDE", "#9467bd"),
)

ABLATION_METHODS = (
    ("NICE_DE", "NICE", "#d62728"),
    ("NICE_wo_Persistence", "w/o Persistence", "#c9a227"),
    ("NICE_wo_LocalSearch", "w/o Local Search", "#0d9488"),
    ("NICE_wo_Niche", "w/o Niche", "#5ba3d9"),
    ("NICE_wo_Collaboration", "w/o Collaboration", "#c026d3"),
)

PACKS = {
    "main": {
        "methods": MAIN_METHODS,
        "out_dir": PLOT_ROOT / "main",
        "legend_order": [m[1] for m in MAIN_METHODS],
    },
    "ablation": {
        "methods": ABLATION_METHODS,
        "out_dir": PLOT_ROOT / "ablation",
        "legend_order": [m[1] for m in ABLATION_METHODS],
    },
}


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


def display_error(err: np.ndarray) -> np.ndarray:
    out = np.asarray(err, dtype=float).copy()
    out[~np.isfinite(out)] = np.nan
    return np.maximum(out, ERROR_FLOOR)


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


def load_method_curves(folder: str) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    path = OUT_ROOT / folder / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    out: dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = {}
    for fid in range(1, 31):
        sub = df[df["function_id"] == fid].sort_values("seed")
        if len(sub) == 0:
            raise SystemExit(f"{folder}: no runs for F{fid}")
        fes, errors = parse_curves(sub)
        disp = display_error(errors)
        q25 = np.nanpercentile(disp, 25, axis=0)
        q50 = np.nanpercentile(disp, 50, axis=0)
        q75 = np.nanpercentile(disp, 75, axis=0)
        out[fid] = (fes, q25, q50, q75)
    return out


def save_fig(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(path, format="pdf", bbox_inches=None, pad_inches=0)


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


def _nudge_edge_tick_labels(fig, ax) -> bool:
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


def _add_legend(ax, label_to_color: dict[str, str], legend_order: list[str]) -> None:
    handles, labels = [], []
    for name in legend_order:
        color = label_to_color[name]
        handles.append(
            Line2D(
                [0],
                [0],
                color=color,
                linewidth=LINEWIDTH,
                solid_capstyle="round",
            )
        )
        labels.append(name)
    leg = ax.legend(
        handles,
        labels,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
        frameon=True,
        ncol=1,
        handlelength=1.8,
        handletextpad=0.55,
        borderpad=0.45,
        labelspacing=0.35,
        fontsize=LEGEND_FONTSIZE,
        prop={"family": "Times New Roman", "size": LEGEND_FONTSIZE},
    )
    leg.set_zorder(LEGEND_ZORDER)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(LEGEND_FRAME_ALPHA)
    frame.set_edgecolor("black")
    frame.set_linewidth(1.4)


def plot_one(
    fid: int,
    methods: tuple,
    method_data: dict[str, dict[int, tuple]],
    legend_order: list[str],
    out_path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))
    ax.set_position(AXES_RECT)
    ax.set_yscale("log")
    ax.set_xlim(0.0, float(X_MAX))

    y_all_lo: list[float] = []
    y_all_hi: list[float] = []
    label_to_color: dict[str, str] = {}

    # Paint reverse so first-listed method (NICE) ends on top.
    paint_order = list(reversed(methods))
    for zi, (folder, label, color) in enumerate(paint_order):
        fes, q25, q50, q75 = method_data[folder][fid]
        label_to_color[label] = color
        y_all_lo.append(float(np.nanmin(q25)))
        y_all_hi.append(float(np.nanmax(q75)))
        ax.fill_between(
            fes,
            q25,
            q75,
            color=color,
            alpha=BAND_ALPHA,
            linewidth=0,
            zorder=2 + 0.01 * zi,
        )

    for zi, (folder, label, color) in enumerate(paint_order):
        fes, q25, q50, q75 = method_data[folder][fid]
        ax.plot(
            fes,
            q50,
            color=color,
            linewidth=LINEWIDTH,
            solid_capstyle="round",
            solid_joinstyle="round",
            zorder=10 + zi,
            label=label,
        )
        mx, my = [], []
        for tick in X_TICKS:
            idx = int(np.argmin(np.abs(fes - tick)))
            mx.append(float(fes[idx]))
            my.append(float(q50[idx]))
        ax.scatter(
            mx,
            my,
            s=MARKER_SIZE,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=MARKER_EDGEWIDTH,
            zorder=20 + zi,
            clip_on=True,
        )

    ymin = min(y_all_lo) if y_all_lo else ERROR_FLOOR
    ymax = max(y_all_hi) if y_all_hi else 1.0
    if not np.isfinite(ymin) or not np.isfinite(ymax):
        ymin, ymax = ERROR_FLOOR, 1.0
    ymin = max(ERROR_FLOOR, ymin / 1.8)
    ymax = max(ymax * 1.8, ymin * 10.0)
    ax.set_ylim(ymin, ymax)

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

    _add_legend(ax, label_to_color, legend_order)

    fig.canvas.draw()
    nudged = _nudge_edge_tick_labels(fig, ax)
    save_fig(out_path)
    plt.close(fig)
    return nudged


def run_pack(pack_name: str) -> None:
    pack = PACKS[pack_name]
    methods = pack["methods"]
    out_dir: Path = pack["out_dir"]
    legend_order = pack["legend_order"]

    method_data: dict[str, dict[int, tuple]] = {}
    for folder, label, color in methods:
        print(f"[{pack_name}] Loading {folder} ({label}, {color}) ...")
        method_data[folder] = load_method_curves(folder)

    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    nudged_ids: list[int] = []
    for fid in range(1, 31):
        out_path = out_dir / f"f{fid}.pdf"
        if plot_one(fid, methods, method_data, legend_order, out_path):
            nudged_ids.append(fid)
        written.append(out_path)

    print(f"[{pack_name}] Wrote {len(written)} PDF files under {out_dir}")
    print(
        f"[{pack_name}] Top y-tick nudged: "
        + (", ".join(f"F{i}" for i in nudged_ids) if nudged_ids else "(none)")
    )


def migrate_legacy_plot_dir() -> None:
    """Move out/plot/f*.pdf into out/plot/main/ if still at the old location."""
    main_dir = PLOT_ROOT / "main"
    main_dir.mkdir(parents=True, exist_ok=True)
    moved = 0
    for pdf in sorted(PLOT_ROOT.glob("f*.pdf")):
        dest = main_dir / pdf.name
        pdf.replace(dest)
        moved += 1
    if moved:
        print(f"Moved {moved} legacy PDFs -> {main_dir}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2 convergence plots")
    parser.add_argument(
        "--pack",
        choices=("main", "ablation", "all"),
        default="all",
        help="which figure pack to export",
    )
    args = parser.parse_args()

    set_style()
    migrate_legacy_plot_dir()

    packs = ("main", "ablation") if args.pack == "all" else (args.pack,)
    for name in packs:
        run_pack(name)


if __name__ == "__main__":
    main()
