"""FastMCP server module for TSAD Orchestra tool discovery."""

from __future__ import annotations
from fastmcp import FastMCP

from src.agent.models import Anomaly, DetectionStubResult, TimeSeriesData
from src.utils.db import read_time_series_by_id, read_time_series

import numpy as np
import pandas as pd
from pyod.models.lof import LOF
from pyod.models.hbos import HBOS
from pyod.models.iforest import IForest
from pyod.models.pca import PCA

mcp = FastMCP("tsad-orchestra")

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
            series = read_time_series_by_id(name)
        else:
            series = read_time_series(name)
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"LOF detector: Error loading series '{name}': {e}",
        )
    
    if not series or len(series) < 20:
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
            series = read_time_series_by_id(name)
        else:
            series = read_time_series(name)
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"HBOS detector: Error loading series '{name}': {e}",
        )
    
    if not series:
        return DetectionStubResult(
            anomalies=[],
            notes="HBOS detector: empty series.",
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
            series = read_time_series_by_id(name)
        else:
            series = read_time_series(name)
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"IForest detector: Error loading series '{name}': {e}",
        )
    
    if not series:
        return DetectionStubResult(
            anomalies=[],
            notes="IForest detector: empty series.",
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
            series = read_time_series_by_id(name)
        else:
            series = read_time_series(name)
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"PCA detector: Error loading series '{name}': {e}",
        )
    
    if not series:
        return DetectionStubResult(
            anomalies=[],
            notes="PCA detector: empty series.",
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
            series = read_time_series_by_id(name)
        else:
            series = read_time_series(name)
    except ValueError as e:
        return DetectionStubResult(
            anomalies=[],
            notes=f"Poly detector: Error loading series '{name}': {e}",
        )
    
    if not series or len(series) < 4:
        return DetectionStubResult(
            anomalies=[],
            notes="Poly detector: insufficient data (need at least 4 points).",
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