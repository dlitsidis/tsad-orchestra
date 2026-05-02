"""FastMCP server module for TSAD Orchestra tool discovery."""

from __future__ import annotations
from fastmcp import FastMCP

from src.agent.models import Anomaly, DetectionStubResult, TimeSeriesProfile
from src.utils.db import read_time_series_by_id, read_time_series, execute_query, get_time_series_name, list_tables

import numpy as np
import pandas as pd
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.pca import PCA

mcp = FastMCP("tsad-orchestra")

@mcp.tool()  # type: ignore[untyped-decorator]
def profile_time_series(
    name: str,
) -> dict:
    """Profile a time series with statistical summary.

    Computes comprehensive statistics about a time series including:
    - Count: number of data points
    - Min/Max: minimum and maximum values
    - Mean: average value
    - Median: middle value
    - Std Dev: standard deviation
    - Time range: first and last timestamp, duration in seconds

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        dict: TimeSeriesProfile with statistical summary, or error details.
    """
    try:
        # Get the table name
        if name.isdigit():
            table_name = get_time_series_name(name)
        else:
            table_name = name

        # Execute comprehensive profiling query
        query = f"""
        SELECT
            COUNT(*) as count,
            MIN(data) as min_value,
            MAX(data) as max_value,
            AVG(data) as mean_value,
            PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY data) as median_value,
            STDDEV_POP(data) as std_dev,
            MIN(time) as first_timestamp,
            MAX(time) as last_timestamp,
            EXTRACT(EPOCH FROM (MAX(time) - MIN(time))) as duration_seconds
        FROM "{table_name}"
        """

        result_df = execute_query(query)

        if result_df.empty:
            return {
                "error": f"No data found in table '{table_name}'",
                "series_name": table_name,
            }

        row = result_df.iloc[0]

        profile = TimeSeriesProfile(
            series_name=table_name,
            count=int(row["count"]),
            min_value=float(row["min_value"]),
            max_value=float(row["max_value"]),
            mean_value=float(row["mean_value"]),
            median_value=float(row["median_value"]),
            std_dev=float(row["std_dev"]) if pd.notna(row["std_dev"]) else 0.0,
            first_timestamp=str(row["first_timestamp"]),
            last_timestamp=str(row["last_timestamp"]),
            duration_seconds=float(row["duration_seconds"]),
        )

        return {
            "series_name": profile.series_name,
            "count": profile.count,
            "min_value": profile.min_value,
            "max_value": profile.max_value,
            "mean_value": profile.mean_value,
            "median_value": profile.median_value,
            "std_dev": profile.std_dev,
            "first_timestamp": profile.first_timestamp,
            "last_timestamp": profile.last_timestamp,
            "duration_seconds": profile.duration_seconds,
        }

    except ValueError as e:
        return {
            "error": f"Failed to profile series '{name}': {e}",
            "series_name": name,
        }
    except Exception as e:
        return {
            "error": f"Profiling error: {e}",
            "series_name": name,
        }


@mcp.tool()  # type: ignore[untyped-decorator]
def list_time_series() -> dict:
    """List all available time series in the database.

    Returns a list of all tables in the public schema.

    Returns:
        dict: Contains 'tables' list with table names, or error details.
    """
    try:
        tables = list_tables()
        return {
            "tables": tables,
            "count": len(tables),
        }
    except Exception as e:
        return {
            "error": f"Failed to list tables: {e}",
            "tables": [],
        }

@mcp.tool()  # type: ignore[untyped-decorator]
def lof_detector(
    name: str,
) -> DetectionStubResult:
    """Detect anomalies using Local Outlier Factor (LOF).

    Best for identifying local anomalies in time series with non-uniform or varying densities.
    It compares the local density of each point to that of its k nearest neighbours, flagging
    points that are significantly less dense than their surroundings as anomalies.
    Use this when anomalies are expected to be contextual (i.e. deviations relative to
    a local neighbourhood) rather than global outliers against the full distribution.

    Complexity:
        Time  — O(n²) with brute-force k-NN (default for small n); O(n log n) with
                ball-tree / KD-tree indexing for larger n.
        Space — O(n · k) to store neighbourhood distances and reachability scores.

    Args:
        name: Name or pattern of the time series CSV file (e.g., '001_NAB_id_1').

    Returns:
        DetectionStubResult: Detected anomalies with indices and reasons.
    """
    try:
        if name.isdigit():
            df = read_time_series_by_id(name)
        else:
            df = read_time_series(name)
        
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return DetectionStubResult(
                    anomalies=[],
                    notes=f"LOF detector: No data found for series '{name}'.",
                )
            series = df.iloc[:, 1].values
        else:
            series = df
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"LOF detector: Error loading series '{name}': {e}",
        )
    
    if len(series) < 20:
        return DetectionStubResult(
            anomalies=[],
            notes="LOF detector: insufficient data (need at least 20 points).",
        )

    array = np.array(series).reshape(-1, 1)
    model = LOF(n_neighbors=20)
    predictions = model.fit_predict(array)

    anomalies = [
        Anomaly(
            index=int(i),
            value=float(series[i]),
            reason=f"LOF anomaly: local outlier factor detected.",
        )
        for i in range(len(series))
        if predictions[i] == 1
    ]

    return DetectionStubResult(
        anomalies=anomalies,
        notes=f"LOF detector ({name}): found {len(anomalies)} anomalies.",
    )


@mcp.tool()  # type: ignore[untyped-decorator]
def hbos_detector(
    name: str,
) -> DetectionStubResult:
    """Detect anomalies using Histogram-based Outlier Score (HBOS).

    Best for identifying global outliers by modelling the marginal distribution of the
    series with histograms and scoring each point by how unlikely its bin is.
    Use this when speed is a priority and anomalies are expected to be value-based
    global deviations (e.g. extreme sensor readings) rather than contextual ones.
    Not well-suited for anomalies that are only abnormal relative to their local context.

    Complexity:
        Time  — O(n · b) to build and score histograms, where b is the number of bins;
                effectively O(n) for the default automatic bin count.
        Space — O(b) to store the histogram; O(b) ≪ O(n) for typical bin counts.

    Args:
        name: Name or pattern of the time series CSV file (e.g., '001_NAB_id_1').

    Returns:
        DetectionStubResult: Detected anomalies with indices and reasons.
    """
    try:
        if name.isdigit():
            df = read_time_series_by_id(name)
        else:
            df = read_time_series(name)
        
        # Extract values from DataFrame if needed
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return DetectionStubResult(
                    anomalies=[],
                    notes=f"HBOS detector: No data found for series '{name}'.",
                )
            series = df.iloc[:, 1].values
        else:
            series = df
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"HBOS detector: Error loading series '{name}': {e}",
        )

    array = np.array(series).reshape(-1, 1)
    model = HBOS()
    predictions = model.fit_predict(array)

    anomalies = [
        Anomaly(
            index=int(i),
            value=float(series[i]),
            reason="HBOS anomaly: histogram-based outlier detected.",
        )
        for i in range(len(series))
        if predictions[i] == 1
    ]

    return DetectionStubResult(
        anomalies=anomalies,
        notes=f"HBOS detector ({name}): found {len(anomalies)} anomalies.",
    )


@mcp.tool()  # type: ignore[untyped-decorator]
def iforest_detector(
    name: str,
) -> DetectionStubResult:
    """Detect anomalies using Isolation Forest (IForest).

    Best for general-purpose anomaly detection where anomalies are assumed to be
    'few and different'. Recursively partitions the data with random splits; anomalous
    points are isolated in fewer splits on average and receive a higher anomaly score.
    Use this when you lack prior knowledge about the anomaly structure. It is robust
    across a wide range of datasets but may miss anomalies that cluster together or
    follow a repeating rhythmic pattern.

    Complexity:
        Time  — O(t · ψ · log ψ) to build t trees over a sub-sample of size ψ
                (defaults: t=100, ψ=256); O(t · log ψ · n) to score n points.
        Space — O(t · ψ) to store the ensemble of isolation trees.

    Args:
        name: Name or pattern of the time series CSV file (e.g., '001_NAB_id_1').

    Returns:
        DetectionStubResult: Detected anomalies with indices and reasons.
    """
    try:
        if name.isdigit():
            df = read_time_series_by_id(name)
        else:
            df = read_time_series(name)
        
        # Extract values from DataFrame if needed
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return DetectionStubResult(
                    anomalies=[],
                    notes=f"IForest detector: No data found for series '{name}'.",
                )
            series = df.iloc[:, 1].values
        else:
            series = df
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"IForest detector: Error loading series '{name}': {e}",
        )

    array = np.array(series).reshape(-1, 1)
    model = IForest()
    predictions = model.fit_predict(array)

    anomalies = [
        Anomaly(
            index=int(i),
            value=float(series[i]),
            reason="IForest anomaly: isolation forest detected.",
        )
        for i in range(len(series))
        if predictions[i] == 1
    ]

    return DetectionStubResult(
        anomalies=anomalies,
        notes=f"IForest detector ({name}): found {len(anomalies)} anomalies.",
    )


@mcp.tool()  # type: ignore[untyped-decorator]
def pca_detector(
    name: str,
) -> DetectionStubResult:
    """Detect anomalies using Principal Component Analysis (PCA).

    Best for detecting anomalies that break the dominant linear structure of the data.
    Projects observations onto the principal components and flags points with a high
    reconstruction error (i.e. points that cannot be well-explained by the learned
    linear subspace) as anomalies.
    Note: this implementation reshapes the input to a single feature column, so PCA
    operates on a 1-D signal and primarily captures variance along the value axis.
    It is most effective when used directly on genuinely multi-dimensional feature
    matrices where inter-feature correlations encode the 'normal' state.

    Complexity:
        Time  — O(n · d² + d³) for SVD decomposition, where d is the number of
                features; reduces to O(n) here because d=1.
        Space — O(d²) for the component matrix; O(1) when d=1.

    Args:
        name: Name or pattern of the time series CSV file (e.g., '001_NAB_id_1').

    Returns:
        DetectionStubResult: Detected anomalies with indices and reasons.
    """
    try:
        if name.isdigit():
            df = read_time_series_by_id(name)
        else:
            df = read_time_series(name)
        
        # Extract values from DataFrame if needed
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return DetectionStubResult(
                    anomalies=[],
                    notes=f"PCA detector: No data found for series '{name}'.",
                )
            series = df.iloc[:, 1].values
        else:
            series = df
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"PCA detector: Error loading series '{name}': {e}",
        )
    
    array = np.array(series).reshape(-1, 1)
    model = PCA()
    predictions = model.fit_predict(array)

    anomalies = [
        Anomaly(
            index=int(i),
            value=float(series[i]),
            reason="PCA anomaly: high reconstruction error detected.",
        )
        for i in range(len(series))
        if predictions[i] == 1
    ]

    return DetectionStubResult(
        anomalies=anomalies,
        notes=f"PCA detector ({name}): found {len(anomalies)} anomalies.",
    )


@mcp.tool()  # type: ignore[untyped-decorator]
def poly_detector(
    name: str,
    threshold_percentile: float = 90.0,
) -> DetectionStubResult:
    """Detect anomalies using Polynomial Fitting.

    Best for detecting point anomalies and contextual anomalies in non-stationary
    data that exhibits a smooth global trend. Fits a degree-3 polynomial to the
    entire series via least squares and returns the absolute residual for each
    point; larger residuals indicate greater deviation from the expected trend.
    Use this when the series has a clear underlying trend or drift and anomalies
    manifest as sudden spikes or dips relative to that trend.

    Complexity:
        Time  — O(n · deg²) for the least-squares fit via QR decomposition,
                where deg=3 (constant), so effectively O(n); O(n · deg) = O(n)
                for polynomial evaluation.
        Space — O(deg) = O(1) for the coefficient vector; O(n) for residuals.

    Args:
        name: Name or pattern of the time series CSV file (e.g., '001_NAB_id_1').
        threshold_percentile: Percentile threshold for flagging anomalies (0-100).
                              Points with residuals above this percentile are flagged.

    Returns:
        DetectionStubResult: Detected anomalies with indices and reasons.
    """
    try:
        if name.isdigit():
            df = read_time_series_by_id(name)
        else:
            df = read_time_series(name)
        
        # Extract values from DataFrame if needed
        if isinstance(df, pd.DataFrame):
            if df.empty:
                return DetectionStubResult(
                    anomalies=[],
                    notes=f"Poly detector: No data found for series '{name}'.",
                )
            series = df.iloc[:, 1].values
        else:
            series = df
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"Poly detector: Error loading series '{name}': {e}",
        )

    array = np.array(series)
    x = np.arange(len(series))
    coeffs = np.polyfit(x, array, deg=3)
    prediction = np.polyval(coeffs, x)
    residuals = np.abs(array - prediction)

    threshold = np.percentile(residuals, threshold_percentile)
    anomalies = [
        Anomaly(
            index=int(i),
            value=float(series[i]),
            reason=f"Poly anomaly: residual {residuals[i]:.2f} exceeds threshold {threshold:.2f}.",
        )
        for i in range(len(series))
        if residuals[i] >= threshold
    ]

    return DetectionStubResult(
        anomalies=anomalies,
        notes=f"Poly detector ({name}): found {len(anomalies)} anomalies (threshold={threshold:.2f}).",
    )


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()