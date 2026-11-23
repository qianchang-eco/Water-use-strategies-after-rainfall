"""Regenerate Fig. 1 and Extended Data Fig. 2 with unique-event frequencies.

Fig. 1b and Extended Data Fig. 2a display occurrence frequencies calculated at
the unique-event grain and annotate percentages rather than row/event counts.
Other panels retain the source, sampling, styling, and inferential annotations
used by the existing submission figures.  Unique events are identified by
(file, event_id).
"""

from importlib.util import module_from_spec, spec_from_file_location
import os
from pathlib import Path
import sys

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


ROOT = Path(os.environ.get("WATER_USE_DATA_ROOT", Path.cwd()))
PLOTS = ROOT / "plots"
SUBMISSION = ROOT / "提交" / "Figures"
LEGACY_SOURCE = Path(__file__).with_name("make_detection_strategy_shift_split.py")
LEGACY_SUMMARY = PLOTS / "rewet_window_ecosystem_dq_summary.csv"
UNIQUE_SUMMARY = PLOTS / "rewet_window_ecosystem_unique_event_summary.csv"



mpl.rcParams.update(
    {
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans", "sans-serif"],
        "pdf.fonttype": 42,
        "svg.fonttype": "none",
    }
)


def load_legacy_module():
    spec = spec_from_file_location("fig1_legacy", LEGACY_SOURCE)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import plotting source: {LEGACY_SOURCE}")
    module = module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def plot_frequency_bars(ax, legacy, summary, windows, big=False):
    """Draw occurrence-frequency bars and label percentages, never counts."""
    x = np.arange(len(windows))
    width = 0.23 if len(windows) > 1 else 0.22
    offsets = np.array([-1, 0, 1]) * width
    values = []
    for i, ecosystem in enumerate(legacy.ECOS):
        rows = [legacy.lookup(summary, ecosystem, window) for window in windows]
        heights = [row.occurrence_frequency * 100 for row in rows]
        values.extend(heights)
        ax.bar(
            x + offsets[i],
            heights,
            width=width * 0.9,
            color=legacy.COLOR[ecosystem],
            label=ecosystem,
            edgecolor="white",
            linewidth=0.6,
        )
        for position, height in zip(x + offsets[i], heights):
            ax.text(
                position,
                height + (0.55 if big else 0.24),
                f"{height:.1f}%",
                ha="center",
                va="bottom",
                fontsize=17 if big else 8.2,
                color="#333333",
            )
    ax.set_xlim(-0.52, len(windows) - 0.48)
    ax.set_ylim(0, max(values) + (5.5 if big else 2.2))
    ax.set_ylabel("Occurrence frequency (%)", fontsize=20 if big else 10)
    ax.set_xticks(x, windows, fontsize=19 if big else 10)
    ax.legend(
        loc="upper left",
        ncol=3,
        frameon=False,
        fontsize=17 if big else 8.5,
        handlelength=0.9,
        columnspacing=0.7,
        bbox_to_anchor=(0, 1.01),
    )
    legacy.frame(ax, tick_fontsize=18 if big else 10)


def save_figure(fig, basename):
    """Save submission PNG plus inspectable vector and raster plot outputs."""
    submission_png = SUBMISSION / f"{basename}.png"
    plot_png = PLOTS / f"{basename}.png"
    plot_pdf = PLOTS / f"{basename}.pdf"
    plot_svg = PLOTS / f"{basename}.svg"
    fig.savefig(submission_png, dpi=600, bbox_inches="tight")
    fig.savefig(plot_png, dpi=600, bbox_inches="tight")
    fig.savefig(plot_pdf, bbox_inches="tight")
    fig.savefig(plot_svg, bbox_inches="tight")
    for path in (submission_png, plot_png, plot_pdf, plot_svg):
        print(path)
    plt.close(fig)


def fixed_panel_label(fig, ax, letter, fontsize=16, x_offset_pt=-36.5, y_offset_pt=6.65):
    """Place a panel letter at a fixed physical offset from the plot area."""
    width_pt, height_pt = fig.get_size_inches() * 72
    position = ax.get_position()
    ax.text(
        position.x0 + x_offset_pt / width_pt,
        position.y1 + y_offset_pt / height_pt,
        letter,
        transform=fig.transFigure,
        ha="left",
        va="bottom",
        fontsize=fontsize,
        fontweight="bold",
    )


def main() -> None:
    legacy = load_legacy_module()
    legacy_summary = pd.read_csv(LEGACY_SUMMARY)
    unique_summary = pd.read_csv(UNIQUE_SUMMARY)

    # Keep panel c unchanged in this targeted revision.
    dq = legacy.load_dq_sample(legacy_summary)

    fig = plt.figure(figsize=(14.2, 8.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1.57, 1],
        left=0.035,
        right=0.985,
        top=0.96,
        bottom=0.09,
        hspace=0.43,
        wspace=0.27,
    )

    ax_a = fig.add_subplot(grid[:, 0])
    schematic = legacy.clean_schematic_y_label(plt.imread(legacy.SOURCE_IMAGE))
    ax_a.imshow(schematic, aspect="equal")
    ax_a.text(
        0.038,
        0.50,
        r"Soil moisture, $\theta$",
        transform=ax_a.transAxes,
        rotation=90,
        ha="center",
        va="center",
        fontsize=15,
        fontfamily="serif",
        color="black",
        rotation_mode="anchor",
    )
    ax_a.axis("off")
    legacy.panel(ax_a, "a", fontsize=22)

    ax_b = fig.add_subplot(grid[0, 1])
    plot_frequency_bars(ax_b, legacy, unique_summary, ("21d",), big=True)
    legacy.panel(ax_b, "b", fontsize=22)

    ax_c = fig.add_subplot(grid[1, 1])
    legacy.plot_dq_21(ax_c, dq)
    legacy.panel(ax_c, "c", fontsize=22)

    fig.canvas.draw()

    SUBMISSION.mkdir(parents=True, exist_ok=True)
    save_figure(fig, "Fig. 1")

    # Extended Data Fig. 2: replace only panel a with unique-event frequencies.
    paradox = legacy.load_paradox()
    fig = plt.figure(figsize=(13.2, 10.0))
    grid = fig.add_gridspec(
        2,
        2,
        width_ratios=[1, 1.08],
        left=0.085,
        right=0.98,
        top=0.96,
        bottom=0.085,
        hspace=0.37,
        wspace=0.24,
    )
    ax_a = fig.add_subplot(grid[0, 0])
    plot_frequency_bars(ax_a, legacy, unique_summary, legacy.WINDOWS)
    ax_b = fig.add_subplot(grid[0, 1])
    legacy.plot_dq_all(ax_b, dq)
    ax_c = fig.add_subplot(grid[1, :])
    legacy.plot_paradox(ax_c, paradox)

    fig.canvas.draw()
    fixed_panel_label(fig, ax_a, "a")
    fixed_panel_label(fig, ax_b, "b")
    fixed_panel_label(fig, ax_c, "c")
    fig.canvas.draw()
    save_figure(fig, "Extended Data Fig. 2")


if __name__ == "__main__":
    main()
