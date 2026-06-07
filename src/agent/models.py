"""Pydantic output models for TSAD Orchestra."""

from dataclasses import dataclass

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field, field_serializer


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    index: int = Field(..., description="Index of the anomalous value in the series.")
    value: float = Field(..., description="Anomalous value at the index.")


class AnomalyReport(BaseModel):
    """Describes anomaly detection results for a time series."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="List of detected anomalies.",
    )
    detector_used: str = Field(
        ..., description="Name of the detector tool selected for the final report."
    )  # noqa: E501
    summary: str = Field(..., description="Summary of anomaly detection results.")


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

    series: pd.DataFrame = Field(
        ..., description="Loaded time series as DataFrame with time and data columns."
    )  # noqa: E501
    source: str = Field(..., description="Source identifier for the series.")

    @field_serializer("series", when_used="json")
    def serialize_series(self, value: pd.DataFrame) -> dict:
        """Serialize DataFrame to dict for JSON output."""
        return value.to_dict(orient="list")


class DetectionStubResult(BaseModel):
    """Tool output for stub anomaly detection."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Detected anomalies (stubbed).",
    )
    notes: str = Field(..., description="Stub detector notes.")


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
