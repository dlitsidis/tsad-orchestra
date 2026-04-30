"""FastMCP server module for TSAD Orchestra tool discovery."""

from __future__ import annotations

from fastmcp import FastMCP

from src.agent.models import Anomaly, DetectionStubResult, TimeSeriesData

mcp = FastMCP("tsad-orchestra")


@mcp.tool()  # type: ignore[misc]
def load_time_series(series: list[float], source: str | None = None) -> TimeSeriesData:
    """Load a time series payload.

    Args:
        series: Numeric readings for the time series.
        source: Optional source identifier for the series.

    Returns:
        TimeSeriesData: Loaded series payload with source metadata.
    """
    return TimeSeriesData(series=series, source=source or "inline")


@mcp.tool()  # type: ignore[misc]
def detect_anomalies(series: list[float]) -> DetectionStubResult:
    """Detect anomalies in a time series (stub implementation).

    Args:
        series: Numeric readings for the time series.

    Returns:
        DetectionStubResult: Stub detection results.
    """
    if not series:
        return DetectionStubResult(
            anomalies=[],
            notes="Stub detector: empty series.",
        )
    peak_index = max(range(len(series)), key=lambda i: abs(series[i]))
    anomaly = Anomaly(
        index=peak_index,
        value=series[peak_index],
        reason="Mock anomaly: largest magnitude value.",
    )
    return DetectionStubResult(
        anomalies=[anomaly],
        notes="Stub detector: returned a mock anomaly.",
    )


def main() -> None:
    """Run the MCP server over stdio."""
    mcp.run()


if __name__ == "__main__":
    main()
