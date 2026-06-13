"""Bar plot comparing average token usage: Agentic with vs without validator.

Data is pulled directly from the token_usage table in TimescaleDB.
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
    "xtick.labelsize": 17,
    "ytick.labelsize": 16,
    "legend.fontsize": 16,
})

# ── Configuration ────────────────────────────────────────────────────────────

DISPLAY_NAMES: dict[str, str] = {
    "agentic_with_validator": "Agentic\n(w/ validator)",
    "agentic_no_validator": "Agentic\n(w/o validator)",
}

METHOD_COLORS: dict[str, str] = {
    "agentic_with_validator": "#82B366",
    "agentic_no_validator": "#6C8EBF",
}


# ── Data fetching ────────────────────────────────────────────────────────────


def fetch_avg_token_usage() -> tuple[list[str], list[float], list[str]]:
    """Query the database for average token usage per method.

    Returns:
        Tuple of (method_labels, avg_tokens, bar_colors).
    """
    query = """
        SELECT method, AVG(token_used) AS avg_tokens
        FROM token_usage
        WHERE method IN ('agentic_with_validator', 'agentic_no_validator')
        GROUP BY method
        ORDER BY avg_tokens DESC
    """
    df = execute_query(query)

    labels: list[str] = []
    values: list[float] = []
    colors: list[str] = []

    for _, row in df.iterrows():
        method = row["method"]
        labels.append(DISPLAY_NAMES.get(method, method))
        values.append(row["avg_tokens"])
        colors.append(METHOD_COLORS.get(method, "#999999"))

    return labels, values, colors


# ── Plotting ─────────────────────────────────────────────────────────────────


def plot_token_usage(save_path: str | None = None) -> None:
    """Create and display (or save) the average token-usage bar plot.

    Args:
        save_path: If provided, save the figure to this path instead of showing.
    """
    labels, values, colors = fetch_avg_token_usage()

    fig, ax = plt.subplots(figsize=(7, 6))

    x = np.arange(len(labels))
    bar_width = 0.55
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
            f"{val:,.0f}",
            ha="center",
            va="bottom",
            fontsize=17,
            fontweight="bold",
        )

    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=17, ha="center")
    ax.set_ylabel("Token Usage", fontsize=20)
    ax.set_title(
        "Average Token Usage: Validator Ablation",
        fontsize=22,
        fontweight="bold",
        pad=14,
    )

    # Y-axis formatting
    ax.set_ylim(0, max(values) * 1.18)
    ax.yaxis.set_major_formatter(mticker.FuncFormatter(lambda v, _: f"{v:,.0f}"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Compact x-axis
    ax.set_xlim(-0.6, len(labels) - 0.4)

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
    plot_token_usage(save_path=str(output_dir / "token_usage.png"))
