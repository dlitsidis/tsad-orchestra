"""Pydantic output models for TSAD Orchestra."""

from pydantic import BaseModel, ConfigDict, Field, field_serializer
import pandas as pd
from dataclasses import dataclass

class Anomaly(BaseModel):
    """Represents a single anomaly in a time series."""

    index: int = Field(..., description="Index of the anomalous value in the series.")
    value: float = Field(..., description="Anomalous value at the index.")
    # reason: str = Field(..., description="Why the value is considered anomalous.")


class AnomalyReport(BaseModel):
    """Describes anomaly detection results for a time series."""

    anomalies: list[Anomaly] = Field(
        default_factory=list,
        description="List of detected anomalies.",
    )
    summary: str = Field(..., description="Summary of anomaly detection results.")


class TimeSeriesData(BaseModel):
    """Tool output for loading a time series."""

    model_config = ConfigDict(arbitrary_types_allowed=True)

    series: pd.DataFrame = Field(..., description="Loaded time series as DataFrame with time and data columns.")
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
