"""Pydantic output models for TSAD Orchestra."""

from pydantic import BaseModel, Field


class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    index: int = Field(..., description="Index of the anomalous value in the series.")
    value: float = Field(..., description="Anomalous value at the index.")
    reason: str = Field(..., description="Why the value is considered anomalous.")


class AnomalyReport(BaseModel):
    """Describes anomaly detection results for a time series."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="List of detected anomalies.",
    )
    summary: str = Field(..., description="Summary of anomaly detection results.")


class TimeSeriesData(BaseModel):
    """Tool output for loading a time series."""

    series: list[float] = Field(..., description="Loaded time series values.")
    source: str = Field(..., description="Source identifier for the series.")


class DetectionStubResult(BaseModel):
    """Tool output for stub anomaly detection."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="Detected anomalies (stubbed).",
    )
    notes: str = Field(..., description="Stub detector notes.")
