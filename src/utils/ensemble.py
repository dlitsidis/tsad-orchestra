"""Deterministic ensemble scoring utilities for TSAD Orchestra.

Called post-reasoning in the finalize node to compute the final per-point
anomaly score array.  The LLM never sees the N-length score vector; it only
decides *which* detectors to use, then this module does the math.
"""

from __future__ import annotations

import numpy as np
from loguru import logger

# Maps both short names ('iforest') and full names ('iforest_detector') to the
# canonical short key used to look up the detector function.
_NAME_MAP: dict[str, str] = {
    "lof": "lof",
    "hbos": "hbos",
    "iforest": "iforest",
    "pca": "pca",
    "poly": "poly",
    "lof_detector": "lof",
    "hbos_detector": "hbos",
    "iforest_detector": "iforest",
    "pca_detector": "pca",
    "poly_detector": "poly",
}


def _normalize(name: str) -> str:
    """Resolve any detector name variant to its canonical short key."""
    return _NAME_MAP.get(name.lower(), name.lower().replace("_detector", ""))


def compute_ensemble_scores(series_name: str, detector_names: list[str]) -> list[float]:
    """Run each detector in raw mode and return a mean-fused [0, 1] score vector.

    This is a pure Python call (no MCP round-trip) intended to be executed
    from the ``finalize`` graph node *after* the LLM has decided which
    detectors to use.  The result is stored in ``AnomalyReport.point_scores``
    and is never injected into the LLM context.

    Args:
        series_name: The series name or ID forwarded to each detector.
        detector_names: Detector identifiers in any recognised form
            (short or ``*_detector`` variants).

    Returns:
        List of ``float`` in ``[0, 1]``, one per time-series point.
        Returns an empty list if no detector produces valid scores.
    """
    # Local import avoids a circular dependency at module load time.
    from src.mcp_server import (
        hbos_detector,
        iforest_detector,
        lof_detector,
        pca_detector,
        poly_detector,
    )

    fn_map = {
        "lof": lof_detector,
        "hbos": hbos_detector,
        "iforest": iforest_detector,
        "pca": pca_detector,
        "poly": poly_detector,
    }

    score_arrays: list[np.ndarray] = []

    for raw_name in detector_names:
        key = _normalize(raw_name)
        fn = fn_map.get(key)
        if fn is None:
            logger.warning(
                "compute_ensemble_scores: unknown detector '{}' (normalised: '{}'), skipping.",
                raw_name,
                key,
            )
            continue
        try:
            result = fn(series_name, _return_raw=True)
            if isinstance(result, np.ndarray) and result.size > 0:
                score_arrays.append(result.astype(float))
                logger.debug(
                    "ensemble: collected {} raw scores from '{}'.",
                    result.size,
                    key,
                )
        except Exception as exc:  # noqa: BLE001
            logger.warning("compute_ensemble_scores: '{}' raised {}: {}", key, type(exc).__name__, exc)

    if not score_arrays:
        logger.error(
            "compute_ensemble_scores: no valid score arrays for series '{}' with detectors {}.",
            series_name,
            detector_names,
        )
        return []

    # Align lengths (edge case: padding artefacts can differ by a few indices)
    min_len = min(len(s) for s in score_arrays)
    trimmed = [s[:min_len] for s in score_arrays]

    fused: np.ndarray = np.mean(trimmed, axis=0)
    logger.info(
        "compute_ensemble_scores: fused {} detector(s) → {} points (mean strategy).",
        len(score_arrays),
        fused.size,
    )
    return fused.tolist()
