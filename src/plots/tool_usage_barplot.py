"""Grouped bar plot comparing average tool usage per tool: with vs without validator.

Data is pulled directly from the tool_usage table in TimescaleDB.
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
    "agentic_tool_usage": "Agentic (w/ validator)",
    "no_validator_tool_usage": "Agentic (w/o validator)",
}

METHOD_COLORS: dict[str, str] = {
    "agentic_tool_usage": "#82B366",
    "no_validator_tool_usage": "#6C8EBF",
}

def format_tool_name(name: str) -> str:
    """Format tool names for better display (e.g. wrapping or cleaning)."""
    return name.replace("_", "\n")


# ── Data fetching ────────────────────────────────────────────────────────────

def fetch_tool_usage_data() -> tuple[list[str], dict[str, list[float]]]:
    """Query the database for average tool usage per method and tool.

    We divide the SUM of counts by the COUNT of distinct datasets per method
    to correctly average even if a tool is entirely unused (missing row) for some datasets.

    Returns:
        tool_names (list), method_data (dict mapping method to list of averages)
    """
    query = """
        WITH MethodDatasetCount AS (
            SELECT method, COUNT(DISTINCT dataset_name) as num_datasets
            FROM tool_usage
            WHERE method IN ('agentic_tool_usage', 'no_validator_tool_usage')
            GROUP BY method
        )
        SELECT 
            t.method, 
            t.tool_name, 
            SUM(t.count)::FLOAT / m.num_datasets AS avg_count
        FROM tool_usage t
        JOIN MethodDatasetCount m ON t.method = m.method
        WHERE t.method IN ('agentic_tool_usage', 'no_validator_tool_usage')
        GROUP BY t.method, t.tool_name, m.num_datasets
    """
    df = execute_query(query)

    # Pivot the data
    # Rows: tool_name, Columns: method, Values: avg_count
    df_pivot = df.pivot(index="tool_name", columns="method", values="avg_count").fillna(0)

    # Sort tools
    df_pivot["total"] = df_pivot.sum(axis=1)
    df_pivot = df_pivot.sort_values("total", ascending=False).drop(columns=["total"])

    tool_names = [format_tool_name(t) for t in df_pivot.index]
    
    method_data = {}
    for method in ['agentic_tool_usage', 'no_validator_tool_usage']:
        if method in df_pivot.columns:
            method_data[method] = df_pivot[method].tolist()
        else:
            method_data[method] = [0.0] * len(tool_names)

    return tool_names, method_data


# ── Plotting ─────────────────────────────────────────────────────────────────

def plot_tool_usage(save_path: str | None = None) -> None:
    tool_names, method_data = fetch_tool_usage_data()

    fig, ax = plt.subplots(figsize=(12, 6))

    x = np.arange(len(tool_names))
    width = 0.35  # width of each bar

    # We have exactly two methods to plot side-by-side
    methods = ['agentic_tool_usage', 'no_validator_tool_usage']
    
    # Calculate positions for the two groups of bars
    offsets = [-width/2, width/2]

    bars_list = []
    for i, method in enumerate(methods):
        values = method_data[method]
        color = METHOD_COLORS.get(method, "#999999")
        label = DISPLAY_NAMES.get(method, method)
        
        bars = ax.bar(
            x + offsets[i], 
            values, 
            width, 
            label=label, 
            color=color, 
            edgecolor="white", 
            linewidth=0.8
        )
        bars_list.append(bars)

        # Value annotations
        for bar, val in zip(bars, values):
            if val > 0:  # Only show text if there is some usage
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    bar.get_height() + 0.1,
                    f"{val:.1f}",
                    ha="center",
                    va="bottom",
                    fontsize=12,
                    fontweight="bold"
                )

    ax.set_xticks(x)
    ax.set_xticklabels(tool_names, fontsize=14)
    ax.set_ylabel("Average Tool Calls per Dataset", fontsize=18)
    ax.set_title(
        "Average Tool Usage: With vs Without Validator",
        fontsize=22,
        fontweight="bold",
        pad=14,
    )

    # Calculate max height for ylim
    max_val = max([max(vals) for vals in method_data.values()]) if method_data else 1
    ax.set_ylim(0, max_val * 1.15)
    
    ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.1f"))
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.set_xlim(-0.6, len(tool_names) - 0.4)

    ax.legend(loc="upper right", framealpha=0.9, fontsize=15)

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
    plot_tool_usage(save_path=str(output_dir / "tool_usage.png"))
