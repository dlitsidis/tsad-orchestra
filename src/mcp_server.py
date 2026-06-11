import json
import math
import os
import tempfile

import numpy as np
import pandas as pd
from fastmcp import FastMCP
from loguru import logger
from sklearn.preprocessing import MinMaxScaler
from TSB_UAD.models.distance import Fourier
from TSB_UAD.models.feature import Window
from TSB_UAD.models.hbos import HBOS
from TSB_UAD.models.iforest import IForest
from TSB_UAD.models.lof import LOF
from TSB_UAD.models.pca import PCA
from TSB_UAD.models.poly import POLY
from TSB_UAD.utils.slidingWindows import find_length

from src.agent.models import Anomaly, DetectionStubResult, DetectorSummary, TimeSeriesProfile
from src.logging_config import setup_logging
from src.utils.db import execute_query, get_time_series_name, list_tables, read_time_series, read_time_series_by_id

setup_logging(process_name="mcp_server")

mcp = FastMCP("tsad-orchestra")


# Keyed by (series_name, detector_name).  Populated on first call; reused by
# drill_down_range and store_ensemble_scores so each detector runs at most
# once per series per server lifetime.
_score_cache: dict[tuple[str, str], np.ndarray] = {}


def _get_or_compute_raw(
    name: str,
    detector_name: str,
) -> np.ndarray | None:
    """Return cached raw scores, computing and caching them if necessary.

    Args:
        name: Series name or ID.
        detector_name: Short detector name (e.g. ``'iforest'``).

    Returns:
        Numpy array of [0, 1] scores, or ``None`` on failure.
    """
    key = (name, detector_name)
    if key in _score_cache:
        logger.debug("Cache hit: {} / {}", name, detector_name)
        return _score_cache[key]

    # Map short name → detector function (avoids a forward-reference problem)
    _fn_map = {
        "lof": lambda n: _run_lof_raw(n),
        "hbos": lambda n: _run_hbos_raw(n),
        "iforest": lambda n: _run_iforest_raw(n),
        "pca": lambda n: _run_pca_raw(n),
        "poly": lambda n: _run_poly_raw(n),
    }
    fn = _fn_map.get(detector_name)
    if fn is None:
        logger.warning("_get_or_compute_raw: unknown detector '{}'", detector_name)
        return None
    try:
        scores = fn(name)
        _score_cache[key] = scores
        logger.debug("Cached {} scores for {} / {}", len(scores), name, detector_name)
        return scores
    except Exception as exc:  # noqa: BLE001
        logger.error("_get_or_compute_raw: {} / {} failed: {}", name, detector_name, exc)
        return None

def _compute_detector_summary(
    series: np.ndarray,
    scores: np.ndarray,
    detector_name: str,
    series_id: str,
    n_segments: int = 20,
) -> DetectorSummary:
    """Build a compact stat-block from a raw [0, 1] score array.

    The stat-block replaces the full anomaly list in the MCP tool response so
    that the LLM context stays small regardless of series length.  Hot segments
    point the agent toward suspicious regions for ``drill_down_range``.
    """
    n = len(scores)

    seg_size = max(1, n // n_segments)
    hot_segments: list[dict] = []
    for seg_start in range(0, n, seg_size):
        seg_end = min(seg_start + seg_size, n)
        seg_scores = scores[seg_start:seg_end]
        seg_max = float(np.max(seg_scores))
        if seg_max > 0.5:
            hot_segments.append(
                {
                    "start": seg_start,
                    "end": seg_end - 1,
                    "max_score": round(seg_max, 3),
                    "count_above_0.7": int(np.sum(seg_scores > 0.7)),
                }
            )

    return DetectorSummary(
        detector=detector_name,
        series=series_id,
        n_points=n,
        anomaly_candidates={
            "above_0.5": int(np.sum(scores > 0.5)),
            "above_0.7": int(np.sum(scores > 0.7)),
            "above_0.9": int(np.sum(scores > 0.9)),
        },
        top_score=round(float(np.max(scores)), 4),
        score_percentiles={
            "p50": round(float(np.percentile(scores, 50)), 4),
            "p90": round(float(np.percentile(scores, 90)), 4),
            "p95": round(float(np.percentile(scores, 95)), 4),
            "p99": round(float(np.percentile(scores, 99)), 4),
        },
        hot_segments=hot_segments,
    )


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


def _load_series(name: str) -> np.ndarray:
    """Load the value column of a time series from the database."""
    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    return df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df




def _run_lof_raw(name: str) -> np.ndarray:

    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    series = df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df

    if len(series) < 20:
        raise ValueError("LOF requires at least 20 data points.")

    sw = find_length(series)
    X = Window(window=sw).convert(series).to_numpy()
    model = LOF(n_neighbors=20, n_jobs=-1)
    model.fit(X)
    raw = model.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    scores = np.array(
        [scores[0]] * math.ceil((sw - 1) / 2) + list(scores) + [scores[-1]] * ((sw - 1) // 2)
    )
    return scores


def _run_hbos_raw(name: str) -> np.ndarray:

    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    series = df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df

    sw = find_length(series)
    X = Window(window=sw).convert(series).to_numpy()
    model = HBOS(alpha=np.float64(0.1), tol=np.float64(0.5), contamination=np.float64(0.1))
    model.fit(X)
    raw = model.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    scores = np.array(
        [scores[0]] * math.ceil((sw - 1) / 2) + list(scores) + [scores[-1]] * ((sw - 1) // 2)
    )
    return scores


def _run_iforest_raw(name: str) -> np.ndarray:

    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    series = df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df

    sw = find_length(series)
    X = Window(window=sw).convert(series).to_numpy()
    model = IForest(n_jobs=1)
    model.fit(X)
    raw = model.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    scores = np.array(
        [scores[0]] * math.ceil((sw - 1) / 2) + list(scores) + [scores[-1]] * ((sw - 1) // 2)
    )
    return scores


def _run_pca_raw(name: str) -> np.ndarray:

    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    series = df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df

    sw = find_length(series)
    X = Window(window=sw).convert(series).to_numpy()
    model = PCA()
    model.fit(X)
    raw = model.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    scores = np.array(
        [scores[0]] * math.ceil((sw - 1) / 2) + list(scores) + [scores[-1]] * ((sw - 1) // 2)
    )
    return scores


def _run_poly_raw(name: str) -> np.ndarray:

    if name.isdigit():
        df = read_time_series_by_id(name)
    else:
        df = read_time_series(name)
    series = df.iloc[:, 1].values if isinstance(df, pd.DataFrame) else df

    sw = find_length(series)
    model = POLY(power=3, window=sw)
    model.fit(series)
    measure = Fourier()
    measure.detector = model
    measure.set_param()
    model.decision_function(measure=measure)
    raw = model.decision_scores_
    scores = MinMaxScaler(feature_range=(0, 1)).fit_transform(raw.reshape(-1, 1)).ravel()
    return scores




@mcp.tool()  # type: ignore[untyped-decorator]
def lof_detector(
    name: str,
    _return_raw: bool = False,
) -> DetectorSummary | np.ndarray:
    """Detect anomalies using Local Outlier Factor (LOF).

    Best for identifying local anomalies in time series with non-uniform or varying densities.
    It compares the local density of each point to that of its k nearest neighbours, flagging
    points that are significantly less dense than their surroundings as anomalies.
    This detector is most effective when anomalies are contextual (i.e., deviations relative to
    a local neighborhood) rather than global outliers against the full distribution.
    It requires a minimum of 20 data points for effective operation, as it uses `n_neighbors=20`.
    Complexity:
        Time  — O(n²) with brute-force k-NN (default for small n); O(n log n) with
                ball-tree / KD-tree indexing for larger n.
        Space — O(n · k) to store neighbourhood distances and reachability scores.

    When called without _return_raw, returns a compact DetectorSummary stat block with
    hot_segments to guide drill_down_range. Raw scores are cached server-side.

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        DetectorSummary stat block (default) or raw [0,1] score array (_return_raw=True).
    """
    scores = _get_or_compute_raw(name, "lof")
    if scores is None:
        return DetectorSummary(
            detector="lof", series=name, n_points=0,
            anomaly_candidates={"above_0.5": 0, "above_0.7": 0, "above_0.9": 0},
            top_score=0.0, score_percentiles={"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
            hot_segments=[], hint="LOF failed — check series name or data size (min 20 points).",
        )
    if _return_raw:
        return scores
    series = _load_series(name)
    return _compute_detector_summary(series, scores, "lof", name)


@mcp.tool()  # type: ignore[untyped-decorator]
def hbos_detector(
    name: str,
    _return_raw: bool = False,
) -> DetectorSummary | np.ndarray:
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

    When called without _return_raw, returns a compact DetectorSummary stat block with
    hot_segments to guide drill_down_range. Raw scores are cached server-side.

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        DetectorSummary stat block (default) or raw [0,1] score array (_return_raw=True).
    """
    scores = _get_or_compute_raw(name, "hbos")
    if scores is None:
        return DetectorSummary(
            detector="hbos", series=name, n_points=0,
            anomaly_candidates={"above_0.5": 0, "above_0.7": 0, "above_0.9": 0},
            top_score=0.0, score_percentiles={"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
            hot_segments=[], hint="HBOS failed — check series name.",
        )
    if _return_raw:
        return scores
    series = _load_series(name)
    return _compute_detector_summary(series, scores, "hbos", name)


@mcp.tool()  # type: ignore[untyped-decorator]
def iforest_detector(
    name: str,
    _return_raw: bool = False,
) -> DetectorSummary | np.ndarray:
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

    When called without _return_raw, returns a compact DetectorSummary stat block with
    hot_segments to guide drill_down_range. Raw scores are cached server-side.

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        DetectorSummary stat block (default) or raw [0,1] score array (_return_raw=True).
    """
    scores = _get_or_compute_raw(name, "iforest")
    if scores is None:
        return DetectorSummary(
            detector="iforest", series=name, n_points=0,
            anomaly_candidates={"above_0.5": 0, "above_0.7": 0, "above_0.9": 0},
            top_score=0.0, score_percentiles={"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
            hot_segments=[], hint="IForest failed — check series name.",
        )
    if _return_raw:
        return scores
    series = _load_series(name)
    return _compute_detector_summary(series, scores, "iforest", name)


@mcp.tool()  # type: ignore[untyped-decorator]
def pca_detector(
    name: str,
    _return_raw: bool = False,
) -> DetectorSummary | np.ndarray:
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

    When called without _return_raw, returns a compact DetectorSummary stat block with
    hot_segments to guide drill_down_range. Raw scores are cached server-side.

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        DetectorSummary stat block (default) or raw [0,1] score array (_return_raw=True).
    """
    scores = _get_or_compute_raw(name, "pca")
    if scores is None:
        return DetectorSummary(
            detector="pca", series=name, n_points=0,
            anomaly_candidates={"above_0.5": 0, "above_0.7": 0, "above_0.9": 0},
            top_score=0.0, score_percentiles={"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
            hot_segments=[], hint="PCA failed — check series name.",
        )
    if _return_raw:
        return scores
    series = _load_series(name)
    return _compute_detector_summary(series, scores, "pca", name)


@mcp.tool()  # type: ignore[untyped-decorator]
def poly_detector(
    name: str,
    _return_raw: bool = False,
) -> DetectorSummary | np.ndarray:
    """Detect anomalies using Polynomial Approximation (POLY).

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

    When called without _return_raw, returns a compact DetectorSummary stat block with
    hot_segments to guide drill_down_range. Raw scores are cached server-side.

    Args:
        name: Name or ID of the time series (e.g., '001_NAB_id_1' or '1').

    Returns:
        DetectorSummary stat block (default) or raw [0,1] score array (_return_raw=True).
    """
    scores = _get_or_compute_raw(name, "poly")
    if scores is None:
        return DetectorSummary(
            detector="poly", series=name, n_points=0,
            anomaly_candidates={"above_0.5": 0, "above_0.7": 0, "above_0.9": 0},
            top_score=0.0, score_percentiles={"p50": 0.0, "p90": 0.0, "p95": 0.0, "p99": 0.0},
            hot_segments=[], hint="POLY failed — check series name.",
        )
    if _return_raw:
        return scores
    series = _load_series(name)
    return _compute_detector_summary(series, scores, "poly", name)



@mcp.tool()  # type: ignore[untyped-decorator]
def drill_down_range(
    name: str,
    start: int,
    end: int,
    detectors: list[str],
) -> dict:
    """Zoom into a specific index range for detailed per-point anomaly inspection.

    Use this AFTER reviewing the hot_segments in the detector stat summaries.
    It runs each requested detector on the full series, slices the score array
    to [start, end], and reports:
    - Per-detector top anomaly points within the range.
    - Consensus points: indices flagged (score > 0.5) by ALL detectors simultaneously
      (strongest possible signal).

    Args:
        name: Series name or ID (same format as other tools).
        start: Inclusive start index of the range to inspect.
        end: Inclusive end index of the range to inspect.
        detectors: Short detector names to use, e.g. ["iforest", "lof", "hbos"].

    Returns:
        dict with 'range', 'per_detector' results, and 'consensus_points'.
    """
    _VALID = {"lof", "hbos", "iforest", "pca", "poly"}

    try:
        series = _load_series(name)
    except Exception as e:  # noqa: BLE001
        return {"error": f"Failed to load series '{name}': {e}"}

    n = len(series)
    start = max(0, min(start, n - 1))
    end = max(start, min(end, n - 1))
    range_len = end - start + 1

    per_detector_raw: dict[str, np.ndarray] = {}
    per_detector_results: dict[str, dict] = {}

    for raw_name in detectors:
        key = raw_name.lower().replace("_detector", "")
        if key not in _VALID:
            per_detector_results[raw_name] = {
                "error": f"Unknown detector '{raw_name}'. Valid: {sorted(_VALID)}"
            }
            continue

        # Use cache — detector has already been run during the stat-block phase.
        raw = _get_or_compute_raw(name, key)
        if raw is None:
            per_detector_results[key] = {"error": f"{key} failed to compute scores."}
            continue

        per_detector_raw[key] = raw
        r_scores = raw[start : end + 1]
        r_values = series[start : end + 1]

        # Top anomalies in range by score (cap at 10)
        sorted_local = np.argsort(r_scores)[::-1]
        top_anomalies = []
        for local_idx in sorted_local:
            score = float(r_scores[local_idx])
            if score <= 0.4:
                break
            top_anomalies.append(
                {
                    "index": int(start + local_idx),
                    "value": round(float(r_values[local_idx]), 4),
                    "score": round(score, 4),
                }
            )
            if len(top_anomalies) >= 10:
                break
        top_anomalies.sort(key=lambda x: x["index"])

        per_detector_results[key] = {
            "max_score": round(float(np.max(r_scores)), 4),
            "mean_score": round(float(np.mean(r_scores)), 4),
            "top_anomalies": top_anomalies,
        }

    # Consensus: indices where ALL detectors score > 0.5
    consensus: list[dict] = []
    if len(per_detector_raw) >= 2:
        votes = np.zeros(range_len, dtype=int)
        score_sums = np.zeros(range_len, dtype=float)
        for sc_arr in per_detector_raw.values():
            r = sc_arr[start : end + 1]
            votes += (r > 0.5).astype(int)
            score_sums += r
        n_det = len(per_detector_raw)
        unanimous = np.where(votes == n_det)[0]
        for local_idx in unanimous:
            consensus.append(
                {
                    "index": int(start + local_idx),
                    "value": round(float(series[start + local_idx]), 4),
                    "mean_score": round(float(score_sums[local_idx] / n_det), 4),
                    "detector_votes": int(votes[local_idx]),
                }
            )
        consensus.sort(key=lambda x: x["mean_score"], reverse=True)
        consensus = consensus[:20]

    return {
        "range": {"start": start, "end": end, "length": range_len},
        "per_detector": per_detector_results,
        "consensus_points": consensus,
        "tip": (
            "consensus_points are flagged by ALL detectors simultaneously — "
            "treat these as the strongest anomaly evidence in this range."
        ),
    }


@mcp.tool()  # type: ignore[untyped-decorator]
def store_ensemble_scores(
    name: str,
    detectors: list[str],
) -> dict:
    """Fuse cached detector scores and persist them for the finalize node.

    Call this as your LAST tool action before finishing, once you have decided
    which detectors to include in the ensemble.  It:
      1. Retrieves the already-computed raw score arrays from the server cache
         (no re-running of detectors).
      2. Mean-fuses the arrays into a single [0, 1] score vector.
      3. Writes the vector to a temporary file keyed by series name.
      4. Returns a lightweight confirmation (NOT the full score array).

    Args:
        name: Series name or ID (must match the name used with the detectors).
        detectors: Short names of detectors to fuse, e.g. ["iforest", "lof"].

    Returns:
        dict with status, n_points, and the temp file path.
    """
    _VALID = {"lof", "hbos", "iforest", "pca", "poly"}
    score_arrays: list[np.ndarray] = []
    failed: list[str] = []

    for raw_name in detectors:
        key = raw_name.lower().replace("_detector", "")
        if key not in _VALID:
            logger.warning("store_ensemble_scores: unknown detector '{}', skipping.", key)
            continue
        scores = _get_or_compute_raw(name, key)
        if scores is None:
            failed.append(key)
        else:
            score_arrays.append(scores)

    if not score_arrays:
        return {
            "status": "error",
            "message": f"No valid scores available. Failed detectors: {failed}",
        }

    min_len = min(len(s) for s in score_arrays)
    fused: np.ndarray = np.mean([s[:min_len] for s in score_arrays], axis=0)

    # Persist to a temp file readable by the finalize node in the parent process.
    out_path = os.path.join(tempfile.gettempdir(), f"tsad_ensemble_{name}.json")
    with open(out_path, "w") as f:
        json.dump(fused.tolist(), f)

    logger.info(
        "store_ensemble_scores: fused {} detector(s) → {} points → {}",
        len(score_arrays),
        fused.size,
        out_path,
    )
    return {
        "status": "ok",
        "n_points": int(fused.size),
        "detectors_fused": [d.lower().replace("_detector", "") for d in detectors if d.lower().replace("_detector", "") in _VALID],
        "failed_detectors": failed,
        "message": "Ensemble scores stored. The finalize node will read them automatically.",
    }

def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
