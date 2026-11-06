"""
Export F1–F30 multi-method figure PDFs.

  out/plot/main/curve/       — main convergence curves
  out/plot/main/box/         — main final-error box plots
  out/plot/ablation/curve/   — ablation curves
  out/plot/ablation/box/     — ablation boxes

Curve and box share canvas size, axes rect, and x/y label figure anchors.
Box legend / box / point colors match the corresponding curve colors.
"""

from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
from matplotlib.ticker import LogLocator, NullFormatter

ROOT = Path(__file__).resolve().parent
OUT_ROOT = ROOT / "out"
PLOT_ROOT = OUT_ROOT / "plot"

FIG_INCH = 6.5

AXES_RECT = (0.215, 0.150, 0.763, 0.835)
# Curve x/y label anchors (P2 historical).
XLABEL_FIG = (0.596, 0.034)
YLABEL_FIG = (0.045, 0.567)
# Box label anchors — match project_1 box (Algorithm slightly above curve x-label).
BOX_XLABEL_FIG = (0.596, 0.040)
BOX_YLABEL_FIG = (0.038, 0.567)

BAND_ALPHA = 0.22
BOX_FACE_ALPHA = 0.38
LINEWIDTH = 8.5
MARKER_SIZE = 280
MARKER_EDGEWIDTH = 4.4
POINT_SIZE = 70
POINT_EDGEWIDTH = 2.0

ERROR_FLOOR = 1e-12
X_MAX = 100_000
X_TICKS = np.array([0, 20_000, 40_000, 60_000, 80_000, 100_000], dtype=float)

FONT_LABEL = 38
FONT_TICK = 32
LEGEND_FONTSIZE = 18
LEGEND_FRAME_ALPHA = 0.78
LEGEND_ZORDER = 120

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


def load_method_curves(
    folder: str,
) -> dict[int, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
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


def load_method_finals(folder: str) -> dict[int, np.ndarray]:
    path = OUT_ROOT / folder / "results.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    df = pd.read_csv(path)
    if "status" in df.columns:
        df = df[df["status"] == "ok"]
    out: dict[int, np.ndarray] = {}
    for fid in range(1, 31):
        sub = df[df["function_id"] == fid].sort_values("seed")
        if len(sub) == 0:
            raise SystemExit(f"{folder}: no runs for F{fid}")
        out[fid] = display_error(sub["error"].to_numpy(dtype=float))
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


def _add_xy_labels(
    fig,
    xlabel: str,
    *,
    xlabel_fig: tuple[float, float] | None = None,
    ylabel_fig: tuple[float, float] | None = None,
) -> None:
    x_anchor = xlabel_fig if xlabel_fig is not None else XLABEL_FIG
    y_anchor = ylabel_fig if ylabel_fig is not None else YLABEL_FIG
    fig.text(
        x_anchor[0],
        x_anchor[1],
        xlabel,
        ha="center",
        va="center",
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )
    fig.text(
        y_anchor[0],
        y_anchor[1],
        "Optimization error",
        ha="center",
        va="center",
        rotation=90,
        fontsize=FONT_LABEL,
        fontfamily="Times New Roman",
    )


def _style_legend(leg) -> None:
    leg.set_zorder(LEGEND_ZORDER)
    frame = leg.get_frame()
    frame.set_facecolor("white")
    frame.set_alpha(LEGEND_FRAME_ALPHA)
    frame.set_edgecolor("black")
    frame.set_linewidth(1.4)


def _add_curve_legend(ax, label_to_color: dict[str, str], legend_order: list[str]) -> None:
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
    _style_legend(leg)


def _add_box_legend(
    ax,
    label_to_color: dict[str, str],
    legend_order: list[str],
) -> None:
    handles = [
        Patch(
            facecolor=label_to_color[name],
            edgecolor=label_to_color[name],
            alpha=BOX_FACE_ALPHA,
            linewidth=2.4,
            label=name,
        )
        for name in legend_order
    ]
    leg = ax.legend(
        handles=handles,
        loc="upper right",
        bbox_to_anchor=(0.98, 0.98),
        bbox_transform=ax.transAxes,
        frameon=True,
        ncol=1,
        handlelength=1.4,
        handletextpad=0.55,
        borderpad=0.45,
        labelspacing=0.35,
        fontsize=LEGEND_FONTSIZE,
        prop={"family": "Times New Roman", "size": LEGEND_FONTSIZE},
    )
    _style_legend(leg)


def plot_curve(
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

    _add_xy_labels(fig, "Function evaluations")
    _add_curve_legend(ax, label_to_color, legend_order)

    fig.canvas.draw()
    nudged = _nudge_edge_tick_labels(fig, ax, nudge_xmax=True)
    save_fig(out_path)
    plt.close(fig)
    return nudged


def plot_box(
    fid: int,
    methods: tuple,
    finals: dict[str, dict[int, np.ndarray]],
    legend_order: list[str],
    out_path: Path,
) -> bool:
    fig, ax = plt.subplots(figsize=(FIG_INCH, FIG_INCH))
    ax.set_position(AXES_RECT)
    ax.set_yscale("log")

    n = len(methods)
    positions = np.arange(1, n + 1, dtype=float)
    series: list[np.ndarray] = []
    colors: list[str] = []
    label_to_color: dict[str, str] = {}
    y_all: list[float] = []

    for folder, label, color in methods:
        vals = finals[folder][fid]
        vals = vals[np.isfinite(vals)]
        if len(vals) == 0:
            raise SystemExit(f"{folder} F{fid}: no finite final errors")
        series.append(vals)
        colors.append(color)
        label_to_color[label] = color
        y_all.extend(float(v) for v in vals)

    bp = ax.boxplot(
        series,
        positions=positions,
        widths=0.55,
        patch_artist=True,
        showfliers=False,
        whis=1.5,
        medianprops={"color": "black", "linewidth": 2.8},
        whiskerprops={"linewidth": 2.8},
        capprops={"linewidth": 2.8},
        boxprops={"linewidth": 2.8},
    )
    for i, (patch, color) in enumerate(zip(bp["boxes"], colors)):
        patch.set_facecolor(color)
        patch.set_alpha(BOX_FACE_ALPHA)
        patch.set_edgecolor(color)
        bp["whiskers"][2 * i].set_color(color)
        bp["whiskers"][2 * i + 1].set_color(color)
        bp["caps"][2 * i].set_color(color)
        bp["caps"][2 * i + 1].set_color(color)

    for pos, vals, color in zip(positions, series, colors):
        rng = np.random.default_rng(1000 * fid + int(pos * 17))
        jitter = rng.uniform(-0.12, 0.12, size=len(vals))
        ax.scatter(
            np.full(len(vals), pos) + jitter,
            vals,
            s=POINT_SIZE,
            marker="o",
            facecolors="white",
            edgecolors=color,
            linewidths=POINT_EDGEWIDTH,
            zorder=5,
            clip_on=True,
        )

    ymin = min(y_all) if y_all else ERROR_FLOOR
    ymax = max(y_all) if y_all else 1.0
    ymin = max(ERROR_FLOOR, ymin / 1.8)
    ymax = max(ymax * 1.8, ymin * 10.0)
    ax.set_ylim(ymin, ymax)
    ax.set_xlim(0.4, n + 0.6)

    # Keep tick marks for alignment; drop method-name labels (legend only).
    ax.set_xticks(list(positions))
    ax.set_xticklabels([])
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

    _add_xy_labels(
        fig,
        "Algorithm",
        xlabel_fig=BOX_XLABEL_FIG,
        ylabel_fig=BOX_YLABEL_FIG,
    )
    _add_box_legend(ax, label_to_color, legend_order)

    fig.canvas.draw()
    nudged = _nudge_edge_tick_labels(fig, ax, nudge_xmax=False)
    save_fig(out_path)
    plt.close(fig)
    return nudged


def migrate_pack_layout(pack_dir: Path) -> int:
    """
    Move pack_dir/f*.pdf -> pack_dir/curve/f*.pdf if still at the old location.
    """
    curve_dir = pack_dir / "curve"
    curve_dir.mkdir(parents=True, exist_ok=True)
    (pack_dir / "box").mkdir(parents=True, exist_ok=True)
    moved = 0
    for pdf in sorted(pack_dir.glob("f*.pdf")):
        dest = curve_dir / pdf.name
        if dest.exists():
            pdf.unlink()
        else:
            shutil.move(str(pdf), str(dest))
        moved += 1
    return moved


def migrate_legacy_plot_dir() -> None:
    """Move out/plot/f*.pdf into out/plot/main/curve/ if still at repo-root plot/."""
    main_curve = PLOT_ROOT / "main" / "curve"
    main_curve.mkdir(parents=True, exist_ok=True)
    (PLOT_ROOT / "main" / "box").mkdir(parents=True, exist_ok=True)
    moved = 0
    for pdf in sorted(PLOT_ROOT.glob("f*.pdf")):
        dest = main_curve / pdf.name
        if dest.exists():
            pdf.unlink()
        else:
            shutil.move(str(pdf), str(dest))
        moved += 1
    if moved:
        print(f"Moved {moved} legacy PDFs -> {main_curve}")


def run_pack(pack_name: str) -> None:
    pack = PACKS[pack_name]
    methods = pack["methods"]
    out_dir: Path = pack["out_dir"]
    legend_order = pack["legend_order"]

    n_moved = migrate_pack_layout(out_dir)
    if n_moved:
        print(f"[{pack_name}] Moved {n_moved} PDFs -> {out_dir / 'curve'}")

    curve_data: dict[str, dict[int, tuple]] = {}
    final_data: dict[str, dict[int, np.ndarray]] = {}
    for folder, label, color in methods:
        print(f"[{pack_name}] Loading {folder} ({label}, {color}) ...")
        curve_data[folder] = load_method_curves(folder)
        final_data[folder] = load_method_finals(folder)

    curve_dir = out_dir / "curve"
    box_dir = out_dir / "box"
    curve_dir.mkdir(parents=True, exist_ok=True)
    box_dir.mkdir(parents=True, exist_ok=True)

    curve_nudged: list[int] = []
    box_nudged: list[int] = []
    for fid in range(1, 31):
        if plot_curve(
            fid, methods, curve_data, legend_order, curve_dir / f"f{fid}.pdf"
        ):
            curve_nudged.append(fid)
        if plot_box(
            fid, methods, final_data, legend_order, box_dir / f"f{fid}.pdf"
        ):
            box_nudged.append(fid)

    print(f"[{pack_name}] Wrote 30 curve PDFs under {curve_dir}")
    print(f"[{pack_name}] Wrote 30 box PDFs under {box_dir}")
    print(
        f"[{pack_name}] Curve top y-tick nudged: "
        + (", ".join(f"F{i}" for i in curve_nudged) if curve_nudged else "(none)")
    )
    print(
        f"[{pack_name}] Box top y-tick nudged: "
        + (", ".join(f"F{i}" for i in box_nudged) if box_nudged else "(none)")
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Project 2 convergence / box plots")
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
