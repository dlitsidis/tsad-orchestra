"""Bar plot of average VUS-ROC grouped by detection method.

Includes base detectors, ensemble, and new_new_agentic.
Data is pulled directly from the experiments table in TimescaleDB.
"""

from __future__ import annotations

import matplotlib
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import numpy as np

from src.utils.db import execute_query

# Use serif / Computer Modern font for an academic look
matplotlib.rcParams.update({
    "font.family": "serif",
    "font.serif": ["Computer Modern Roman", "CMU Serif", "Times New Roman", "DejaVu Serif"],
    "mathtext.fontset": "cm",
    "font.size": 14,
})

# --- Configuration -----------------------------------------------------------

BASE_DETECTORS = ["hbos", "iforest", "lof", "pca", "poly"]
METHODS_OF_INTEREST = BASE_DETECTORS + ["ensemble", "new_new_agentic"]

DISPLAY_NAMES: dict[str, str] = {
    "hbos": "HBOS",
    "iforest": "IForest",
    "lof": "LOF",
    "pca": "PCA",
    "poly": "POLY",
    "ensemble": "Basic Ensemble",
    "new_new_agentic": "Agentic",
}

CATEGORY_COLORS: dict[str, str] = {
    "base": "#6C8EBF",
    "ensemble": "#D4A03C",
    "agentic": "#82B366",
}


# --- Data fetching -----------------------------------------------------------


def fetch_avg_vus_roc() -> tuple[list[str], list[float], list[str]]:
    """Query the database for average VUS-ROC per method.

    Returns:
        Tuple of (method_labels, avg_values, bar_colors) sorted descending.
    """
    placeholders = ", ".join(f"'{m}'" for m in METHODS_OF_INTEREST)
    query = f"""
        SELECT method, AVG(vus_roc) AS avg_vus_roc
        FROM experiments
        WHERE method IN ({placeholders})
        GROUP BY method
        ORDER BY avg_vus_roc DESC
    """
    df = execute_query(query)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for _, row in df.iterrows():
        method = row["method"]
        labels.append(DISPLAY_NAMES.get(method, method))
        values.append(row["avg_vus_roc"])

        if method in BASE_DETECTORS:
            colors.append(CATEGORY_COLORS["base"])
        elif method == "ensemble":
            colors.append(CATEGORY_COLORS["ensemble"])
        else:
            colors.append(CATEGORY_COLORS["agentic"])

    return labels, values, colors


# --- Plotting ----------------------------------------------------------------


def plot_avg_vus_roc(save_path: str | None = None) -> None:
    """Create and display (or save) the average VUS-ROC bar plot.

    Args:
        save_path: If provided, save the figure to this path instead of showing.
    """
    labels, values, colors = fetch_avg_vus_roc()

    fig, ax = plt.subplots(figsize=(9, 6))

    x = np.arange(len(labels))
    bar_width = 0.7
    bars = ax.bar(x, values, width=bar_width, color=colors, edgecolor="white", linewidth=0.8)

    # Value annotations
    for bar, val in zip(bars, values):
        ax.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.005,
            f"{val:.3f}",
            ha="center",
            va="bottom",
            fontsize=13,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=14, rotation=15, ha="right")
    ax.set_ylabel("VUS-ROC", fontsize=16)
    ax.set_title("Average VUS-ROC by Detection Method", fontsize=18, fontweight="bold")

    # Y-axis formatting
    ax.set_ylim(0, max(values) * 1.12)
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.2f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Legend
    from matplotlib.patches import Patch

    legend_handles = [
        Patch(facecolor=CATEGORY_COLORS["base"], label="Base Detector"),
        Patch(facecolor=CATEGORY_COLORS["ensemble"], label="Ensemble"),
        Patch(facecolor=CATEGORY_COLORS["agentic"], label="Agentic"),
    ]
    ax.legend(handles=legend_handles, loc="upper right", framealpha=0.9, fontsize=13)

    fig.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=150, bbox_inches="tight")
        print(f"Figure saved to {save_path}")
        # Export as SVG
        svg_path = save_path.rsplit(".", 1)[0] + ".svg"
        fig.savefig(svg_path, format="svg", bbox_inches="tight")
        print(f"Figure saved to {svg_path}")
    else:
        plt.show()


if __name__ == "__main__":
    from pathlib import Path

    output_dir = Path(__file__).resolve().parent / "results"
    output_dir.mkdir(parents=True, exist_ok=True)
    plot_avg_vus_roc(save_path=str(output_dir / "performance.png"))
