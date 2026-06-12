"""Pydantic output models for TSAD Orchestra."""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    index: int = Field(..., description="Index of the anomalous value in the series.")
    value: float = Field(..., description="Anomalous value at the index.")
    score: float = Field(default=0.0, description="Severity score or confidence of the anomaly.")


class AnomalyReport(BaseModel):
    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Confirmed anomalies identified through drill-down. High-confidence points only.",
    )
    anomaly_count: int = Field(default=0, description="Total number of confirmed anomalies.", exclude=True)
    detectors_used: list[str] = Field(
        ...,
        description=(
            "Short names of detectors whose outputs were aggregated (e.g. ['iforest', 'lof']). "
            "These determine which detectors are fused into the final per-point score vector."
        ),
    )
    tools_called: list[str] = Field(default_factory=list, description="All tool names called during analysis.", exclude=True)
    tool_counts: dict[str, int] = Field(
        default_factory=dict,
        description="Counts of each tool called during analysis.",
        exclude=True,
    )
    summary: str = Field(..., description="Summary of anomaly detection results and ensemble reasoning.")
    point_scores: list[float] = Field(
        default_factory=list,
        exclude=True,
        description=(
            "Per-point ensemble anomaly score in [0, 1], one value per series point. "
            "Computed deterministically post-reasoning by fusing the detectors_used score arrays. "
            "Never set by the LLM — populated automatically in the finalize node."
        ),
    )
    prompt_tokens: int | None = Field(
        default=None,
        exclude=True,
        description="Total prompt tokens consumed during this run.",
    )
    completion_tokens: int | None = Field(
        default=None,
        exclude=True,
        description="Total completion tokens consumed during this run.",
    )
    total_tokens: int | None = Field(
        default=None,
        exclude=True,
        description="Total tokens consumed during this run.",
    )


class LLMFinalReport(BaseModel):
    detectors_used: list[str] = Field(
        ...,
        description=(
            "Short names of detectors whose outputs were aggregated (e.g. ['iforest', 'lof']). "
            "These determine which detectors are fused into the final per-point score vector."
        ),
    )
    summary: str = Field(..., description="Summary of anomaly detection results and ensemble reasoning.")


class ValidationResult(BaseModel):
    """Structured output produced by the validator agent.

    Attributes:
        accepted: True if the primary report passes validation; False if it
            must be refined.
        critique: Human-readable summary of the issues found.  Empty string
            when accepted is True.
        severity: One of "minor", "major", or "critical" - guides whether the
            primary agent should do a light touch-up or a full re-analysis.
    """

    accepted: bool
    critique: str
    severity: str = "minor"  # "minor" | "major" | "critical"


class TimeSeriesData(BaseModel):
    """Tool output for loading a time series."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    series: pd.DataFrame = Field(..., description="Loaded time series as DataFrame with time and data columns.")  # noqa: E501
    source: str = Field(..., description="Source identifier for the series.")

    @field_serializer("series", when_used="json")
    def serialize_series(self, value: pd.DataFrame) -> dict:
        """Serialize DataFrame to dict for JSON output."""
        return value.to_dict(orient="list")


class DetectionStubResult(BaseModel):
    """Tool output for stub anomaly detection (kept for backward compat)."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Detected anomalies (stubbed).",
    )
    notes: str = Field(..., description="Stub detector notes.")


class DetectorSummary(BaseModel):
    """Compact stat-block returned by a detector tool when not in raw mode.

    The LLM sees only these aggregate statistics — not the full score array —
    keeping the context window small regardless of series length.
    """

    detector: str = Field(..., description="Short detector name (e.g. 'iforest').")
    series: str = Field(..., description="Series name or ID that was analysed.")
    n_points: int = Field(..., description="Total number of points in the series.")
    anomaly_candidates: dict[str, int] = Field(
        ...,
        description="Point counts exceeding score thresholds: keys are 'above_0.5', 'above_0.7', 'above_0.9'.",
    )
    top_score: float = Field(..., description="Highest anomaly score produced by this detector.")
    score_percentiles: dict[str, float] = Field(
        ...,
        description="Score distribution: p50, p90, p95, p99.",
    )
    hot_segments: list[dict] = Field(
        default_factory=list,
        description=(
            "Time-index ranges with max_score > 0.5. "
            "Each entry has: start, end, max_score, count_above_0.7. "
            "Use these as targets for drill_down_range."
        ),
    )
    hint: str = Field(
        default="Use drill_down_range(name, start, end, detectors) to inspect suspicious segments.",
        description="Guidance for the next step.",
    )


@dataclass
class TimeSeriesProfile:
    """Data profile for a time series."""

    series_name: str
    count: int
    min_value: float
    max_value: float
    mean_value: float
    median_value: float
    std_dev: float
    first_timestamp: str
    last_timestamp: str
    duration_seconds: float
