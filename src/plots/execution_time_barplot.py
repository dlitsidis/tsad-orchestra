"""Bar plot of average execution time grouped by detection method.

Includes base detectors, ensemble, agentic, and agentic-no-validator.
Data is pulled directly from the execution_time table in TimescaleDB.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.utils.db import execute_query

# ── Academic font configuration ──────────────────────────────────────────────
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": [
        "Computer Modern Roman",
        "CMU Serif",
        "Times New Roman",
        "DejaVu Serif",
    ],
    "mathtext.fontset": "cm",
    "font.size": 18,
    "axes.labelsize": 20,
    "axes.titlesize": 22,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ── Configuration ────────────────────────────────────────────────────────────

DISPLAY_NAMES: dict[str, str] = {
    "hbos": "HBOS",
    "iforest": "IForest",
    "lof": "LOF",
    "pca": "PCA",
    "poly": "POLY",
    "ensemble": "Ensemble",
    "new_new_agentic": "Agentic",
    "no_validator_4o": "Agentic\n(w/o valid.)",
}

BASE_DETECTORS = {"hbos", "iforest", "lof", "pca", "poly"}

CATEGORY_COLORS: dict[str, str] = {
    "base": "#6C8EBF",
    "ensemble": "#D4A03C",
    "agentic": "#82B366",
}


def _category_color(method: str) -> str:
    if method in BASE_DETECTORS:
        return CATEGORY_COLORS["base"]
    if method == "ensemble":
        return CATEGORY_COLORS["ensemble"]
    return CATEGORY_COLORS["agentic"]


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_avg_execution_time() -> tuple[list[str], list[float], list[str]]:
    """Query the database for average execution time per method.

    Returns:
        Tuple of (method_labels, avg_seconds, bar_colors) sorted ascending.
    """
    query = """
        SELECT method, AVG(time) AS avg_time
        FROM execution_time
        GROUP BY method
        ORDER BY avg_time ASC
    """
    df = execute_query(query)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for _, row in df.iterrows():
        method = row["method"]
        labels.append(DISPLAY_NAMES.get(method, method))
        values.append(row["avg_time"])
        colors.append(_category_color(method))

    return labels, values, colors


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_execution_time(save_path: str | None = None) -> None:
    """Create and display (or save) the average execution-time bar plot.

    Args:
        save_path: If provided, save the figure to this path instead of showing.
    """
    labels, values, colors = fetch_avg_execution_time()

    fig, ax = plt.subplots(figsize=(10, 6))

    x = np.arange(len(labels))
    bar_width = 0.65
    bars = ax.bar(
        x,
        values,
        width=bar_width,
        color=colors,
        edgecolor="white",
        linewidth=0.8,
    )

    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + max(values) * 0.015,
            f"{val:.2f}s",
            ha="center",
            va="bottom",
            fontsize=15,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=16, ha="center")
    ax.set_ylabel("Execution Time (s)", fontsize=20)
    ax.set_title(
        "Average Execution Time by Detection Method",
        fontsize=22,
        fontweight="bold",
        pad=14,
    )

    # Y-axis formatting
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact x-axis
    ax.set_xlim(-0.5, len(labels) - 0.5)

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS["base"], label="Base Detector"),
        Patch(facecolor=CATEGORY_COLORS["ensemble"], label="Ensemble"),
        Patch(facecolor=CATEGORY_COLORS["agentic"], label="Agentic"),
    ]
    ax.legend(handles=legend_handles, loc="upper left", framealpha=0.9, fontsize=16)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
        svg_path = save_path.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Figure saved to {svg_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_execution_time(save_path=str(output_dir / "execution_time.png"))
