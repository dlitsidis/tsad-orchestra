"""Streamlit UI for TSAD Orchestra time series anomaly detection."""

import asyncio
import os
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from dotenv import load_dotenv
from loguru import logger

from src.logging_config import setup_logging

setup_logging(process_name="tsad_orchestra_ui")

from src.agent.client import run
from src.utils.db import list_tables, read_time_series, read_time_series_by_id

load_dotenv()

st.set_page_config(
    page_title="TSAD Orchestra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

if "selected_series" not in st.session_state:
    st.session_state.selected_series = None
if "series_data" not in st.session_state:
    st.session_state.series_data = None
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False


def load_available_series() -> list[str]:
    """Load all available time series from the database."""
    try:
        tables = list_tables()
        return sorted(tables) if tables else []
    except Exception as e:
        st.error(f"Error loading time series list: {e}")
        return []


def load_time_series_data(series_name: str) -> pd.DataFrame | None:
    """Load a specific time series from the database."""
    try:
        # Check if it's a numeric ID or table name
        if series_name.isdigit():
            df = read_time_series_by_id(series_name)
        else:
            df = read_time_series(series_name)

        return df
    except Exception as e:
        st.error(f"Error loading time series '{series_name}': {e}")
        return None


def plot_time_series(df: pd.DataFrame, anomalies: list[dict] | None = None) -> go.Figure:
    """Create an interactive Plotly chart of the time series with anomalies highlighted."""
    fig = go.Figure()

    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.iloc[:, 1],
            mode="lines",
            name="Time Series",
            line={"color": "steelblue", "width": 2},
            hovertemplate="<b>Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>",
        )
    )

    if anomalies:
        anomaly_indices = [a["index"] for a in anomalies]
        anomaly_values = [df.iloc[a["index"], 1] for a in anomalies if a["index"] < len(df)]
        anomaly_indices_valid = [i for i in anomaly_indices if i < len(df)]

        if anomaly_indices_valid:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_indices_valid,
                    y=anomaly_values,
                    mode="markers",
                    name="Anomalies",
                    marker={"color": "red", "size": 10, "symbol": "x"},
                    hovertemplate="<b>Anomaly at Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>",
                )
            )

    fig.update_layout(
        title="Time Series with Detected Anomalies",
        xaxis_title="Index",
        yaxis_title="Value",
        hovermode="x unified",
        height=400,
        template="plotly_white",
    )

    return fig


def run_anomaly_detection_sync(series_id: str) -> Any:
    """Run the agent's anomaly detection on the selected time series (sync wrapper).

    Args:
        series_id: The ID or name of the time series to analyze.

    Returns:
        AnomalyReport with detected anomalies.
    """
    try:
        # Run the agent asynchronously using asyncio.run
        # This properly handles child watcher initialization for subprocesses
        return asyncio.run(run(series_id))

    except Exception as e:
        st.error(f"Error running anomaly detection: {str(e)}")
        logger.error(f"Anomaly detection failed: {e}")
        raise


def main():
    """Main Streamlit application."""
    st.title("📊 TSAD Orchestra — Time Series Anomaly Detection")
    st.markdown("An intelligent agent-powered tool for detecting anomalies in time series data from your database.")

    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Missing OPENAI_API_KEY environment variable. " "Please set it in your .env file before running this app.")
        st.stop()

    if not os.getenv("POSTGRES_USER"):
        st.error(
            "❌ Missing database configuration. "
            "Please ensure POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, and POSTGRES_DB are set in your .env file."
        )
        st.stop()

    with st.sidebar:
        st.header("⚙️ Configuration")

        available_series = load_available_series()

        if not available_series:
            st.warning("No time series found in the database. Please check your database connection.")
            return

        selected_series = st.selectbox(
            "Select a Time Series",
            options=available_series,
            index=0,
            help="Choose a time series from the database",
        )

        if selected_series != st.session_state.selected_series:
            st.session_state.selected_series = selected_series
            st.session_state.series_data = load_time_series_data(selected_series)
            st.session_state.detection_result = None

        if st.session_state.series_data is not None:
            df = st.session_state.series_data
            st.markdown("---")
            st.subheader("📈 Series Statistics")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Data Points", len(df))
            with col2:
                st.metric("Mean", f"{df.iloc[:, 1].mean():.4f}")
            with col3:
                st.metric("Std Dev", f"{df.iloc[:, 1].std():.4f}")

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Min", f"{df.iloc[:, 1].min():.4f}")
            with col2:
                st.metric("Max", f"{df.iloc[:, 1].max():.4f}")
            with col3:
                st.metric("Range", f"{df.iloc[:, 1].max() - df.iloc[:, 1].min():.4f}")

    if st.session_state.series_data is not None:
        df = st.session_state.series_data

        if st.button(
            "🔍 Run Detection",
            disabled=st.session_state.is_running,
            help="Automatically analyze the time series for anomalies",
            use_container_width=True,
        ):
            st.session_state.is_running = True

            with st.spinner("🚀 Running anomaly detection agent... This may take a minute."):
                try:
                    report = run_anomaly_detection_sync(st.session_state.selected_series)

                    if report:
                        st.session_state.detection_result = report
                        st.session_state.is_running = False
                        st.success(f"✅ Detection complete! Found {len(report.anomalies)} anomalies.")

                except Exception as e:
                    st.error(f"Failed to run detection: {e}")
                    logger.error(f"Detection error: {e}")
                finally:
                    st.session_state.is_running = False

        st.markdown("---")

        st.subheader("📊 Time Series Visualization")
        if st.session_state.detection_result:
            anomalies = [{"index": a.index, "value": a.value} for a in st.session_state.detection_result.anomalies]
            fig = plot_time_series(df, anomalies)
        else:
            fig = plot_time_series(df)

        st.plotly_chart(fig, use_container_width=True)

        if st.session_state.detection_result:
            report = st.session_state.detection_result

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Anomalies Found", len(report.anomalies))
            with col2:
                st.metric("Detectors Used", ", ".join(report.detectors_used))

            if st.toggle("🔧 Show Tools Used By The Agent"):
                with st.expander("🛠️ Agent Tool Calls", expanded=True):
                    for tool in report.tools_called:
                        prefix = "✅" if tool in report.detectors_used else "🔍"
                        st.markdown(f"{prefix} `{tool}`")

            with st.expander("📋 View Summary"):
                st.text(report.summary)

            if report.anomalies:
                with st.expander("📊 View All Anomalies"):
                    anomaly_df = pd.DataFrame(
                        [
                            {
                                "Index": a.index,
                                "Value": f"{a.value:.4f}",
                                "Score": f"{a.score:.2f}",
                            }
                            for a in report.anomalies
                        ]
                    )
                    st.dataframe(anomaly_df, use_container_width=True)

    else:
        st.warning("Unable to load the selected time series. Please check the database connection.")


if __name__ == "__main__":
    main()
