"""Streamlit UI for TSAD Orchestra time series anomaly detection."""

import asyncio
import concurrent.futures
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

# ── Tables to hide from the series selector ──────────────────────────────────
_EXCLUDED_TABLES = {"experiments"}

st.set_page_config(
    page_title="TSAD Orchestra",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Custom CSS ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <style>
    /* ── Google Font ───────────────────────────────────────────────────── */
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');
    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    /* ── Accent colour tokens ─────────────────────────────────────────── */
    :root {
        --red:           #DC3545;
        --red-light:     #FFF0F1;
        --red-mid:       #F8D7DA;
        --blue:          #2563EB;
        --blue-dark:     #1E3A5F;
        --blue-light:    #EFF6FF;
        --navy:          #1E293B;
        --slate:         #64748B;
        --grey-bg:       #F4F6FA;
    }

    /* ── Header banner ────────────────────────────────────────────────── */
    .hero-banner {
        background: linear-gradient(135deg, #7F1D1D 0%, #DC3545 60%, #EF4444 100%);
        border-radius: 16px;
        padding: 2rem 2.5rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 4px 24px rgba(220, 53, 69, .18);
    }
    .hero-banner h1 {
        font-size: 2rem;
        font-weight: 700;
        color: #FFFFFF;
        margin-bottom: .35rem;
    }
    .hero-banner p {
        color: rgba(255, 255, 255, .75);
        font-size: .95rem;
        margin: 0;
    }

    /* ── Sidebar polish ───────────────────────────────────────────────── */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #F8FAFC 0%, #F1F5F9 100%);
        border-right: 1px solid #E2E8F0;
    }
    section[data-testid="stSidebar"] .stSelectbox label {
        color: #1E3A5F !important;
        font-weight: 600;
    }

    /* ── Metric cards ─────────────────────────────────────────────────── */
    [data-testid="stMetric"] {
        background: #FFFFFF;
        border: 1px solid #E2E8F0;
        border-radius: 12px;
        padding: 1rem 1.25rem;
        box-shadow: 0 1px 4px rgba(0, 0, 0, .04);
        transition: border-color .2s, box-shadow .2s;
    }
    [data-testid="stMetric"]:hover {
        border-color: #2563EB;
        box-shadow: 0 2px 12px rgba(37, 99, 235, .1);
    }
    [data-testid="stMetric"] label {
        color: #64748B !important;
        font-size: .8rem;
        text-transform: uppercase;
        letter-spacing: .03em;
    }
    [data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #1E293B !important;
        font-weight: 700;
    }

    /* ── Buttons ──────────────────────────────────────────────────────── */
    .stButton > button {
        background: linear-gradient(135deg, #DC3545 0%, #B91C2C 100%);
        color: #FFFFFF !important;
        font-weight: 600;
        border: none;
        border-radius: 10px;
        padding: .65rem 1.5rem;
        transition: transform .15s, box-shadow .2s;
    }
    .stButton > button:hover {
        transform: translateY(-1px);
        box-shadow: 0 6px 20px rgba(220, 53, 69, .3);
    }
    .stButton > button:active { transform: translateY(0); }
    .stButton > button:disabled {
        background: #CBD5E1 !important;
        box-shadow: none;
        transform: none;
    }

    /* ── Expanders ────────────────────────────────────────────────────── */
    .streamlit-expanderHeader {
        background: #F8FAFC !important;
        border: 1px solid #E2E8F0 !important;
        border-radius: 10px !important;
        color: #1E293B !important;
        font-weight: 500;
    }

    /* ── Dataframe ────────────────────────────────────────────────────── */
    .stDataFrame { border-radius: 10px; overflow: hidden; }

    /* ── Dividers ─────────────────────────────────────────────────────── */
    hr {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #E2E8F0, transparent);
        margin: 1.5rem 0;
    }

    /* ── Success toast ────────────────────────────────────────────────── */
    .stSuccess {
        background: #F0FDF4 !important;
        border-left: 4px solid #22C55E !important;
        border-radius: 8px;
    }

    /* ── Toggle ───────────────────────────────────────────────────────── */
    .stToggle label span { color: #64748B !important; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Session state ─────────────────────────────────────────────────────────────
if "selected_series" not in st.session_state:
    st.session_state.selected_series = None
if "series_data" not in st.session_state:
    st.session_state.series_data = None
if "detection_result" not in st.session_state:
    st.session_state.detection_result = None
if "is_running" not in st.session_state:
    st.session_state.is_running = False
if "detection_just_completed" not in st.session_state:
    st.session_state.detection_just_completed = False


# ── Helpers ───────────────────────────────────────────────────────────────────

def load_available_series() -> list[str]:
    """Load all available time series from the database, excluding non-data tables."""
    try:
        tables = list_tables()
        if not tables:
            return []
        return sorted(t for t in tables if t not in _EXCLUDED_TABLES)
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

    # ── Main time series line ─────────────────────────────────────────────
    fig.add_trace(
        go.Scatter(
            x=df.index,
            y=df.iloc[:, 1],
            mode="lines",
            name="Time Series",
            line={"color": "#2563EB", "width": 2.2},
            fill="tozeroy",
            fillcolor="rgba(37, 99, 235, 0.06)",
            hovertemplate="<b>Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>",
        )
    )

    # ── Anomaly markers ───────────────────────────────────────────────────
    if anomalies:
        valid_anomalies = [a for a in anomalies if a["index"] < len(df)]
        anomaly_x = [df.index[a["index"]] for a in valid_anomalies]
        anomaly_values = [df.iloc[a["index"], 1] for a in valid_anomalies]

        if anomaly_x:
            fig.add_trace(
                go.Scatter(
                    x=anomaly_x,
                    y=anomaly_values,
                    mode="markers",
                    name="Anomalies",
                    marker={
                        "color": "#DC3545",
                        "size": 11,
                        "symbol": "x",
                        "line": {"width": 2, "color": "#B91C2C"},
                    },
                    hovertemplate="<b>Anomaly at Index:</b> %{x}<br><b>Value:</b> %{y:.4f}<extra></extra>",
                )
            )

    # ── Layout ────────────────────────────────────────────────────────────
    fig.update_layout(
        xaxis_title="Index",
        yaxis_title="Value",
        hovermode="x unified",
        height=420,
        template="plotly_white",
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="#FFFFFF",
        font={"family": "Inter, sans-serif", "color": "#1E293B"},
        xaxis={
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "linecolor": "#E2E8F0",
        },
        yaxis={
            "gridcolor": "#F1F5F9",
            "zerolinecolor": "#E2E8F0",
            "linecolor": "#E2E8F0",
        },
        legend={
            "bgcolor": "rgba(255,255,255,0.8)",
            "bordercolor": "#E2E8F0",
            "borderwidth": 1,
            "font": {"color": "#64748B"},
        },
        margin={"l": 50, "r": 20, "t": 30, "b": 50},
    )

    return fig


def run_anomaly_detection_sync(series_id: str) -> Any:
    """Run the agent's anomaly detection on the selected time series (sync wrapper).

    Runs asyncio.run() in a dedicated thread to avoid conflicts with
    Streamlit's own event loop.

    Args:
        series_id: The ID or name of the time series to analyze.

    Returns:
        AnomalyReport with detected anomalies.
    """
    try:
        with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
            future = pool.submit(asyncio.run, run(series_id))
            return future.result()

    except Exception as e:
        st.error(f"Error running anomaly detection: {str(e)}")
        logger.error(f"Anomaly detection failed: {e}")
        raise


# ── Main app ──────────────────────────────────────────────────────────────────

def main():
    """Main Streamlit application."""

    # ── Hero banner ───────────────────────────────────────────────────────
    st.markdown(
        """
        <div class="hero-banner">
            <h1>📊 TSAD Orchestra</h1>
            <p>Agent-powered anomaly detection for time series data — profile, detect, validate.</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not os.getenv("OPENAI_API_KEY"):
        st.error("❌ Missing OPENAI_API_KEY environment variable. " "Please set it in your .env file before running this app.")
        st.stop()

    if not os.getenv("POSTGRES_USER"):
        st.error(
            "❌ Missing database configuration. "
            "Please ensure POSTGRES_USER, POSTGRES_PASSWORD, POSTGRES_HOST, and POSTGRES_DB are set in your .env file."
        )
        st.stop()

    # ── Sidebar ───────────────────────────────────────────────────────────
    with st.sidebar:
        st.markdown("### 🎛️ Configuration")

        available_series = load_available_series()

        if not available_series:
            st.warning("No time series found in the database. Please check your database connection.")
            return

        selected_series = st.selectbox(
            "🗂️ Select a Time Series",
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
            st.markdown("### 📈 Series Statistics")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Data Points", f"{len(df):,}")
            with col2:
                st.metric("Mean", f"{df.iloc[:, 1].mean():.2f}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Std Dev", f"{df.iloc[:, 1].std():.2f}")
            with col2:
                st.metric("Range", f"{df.iloc[:, 1].max() - df.iloc[:, 1].min():.2f}")

            col1, col2 = st.columns(2)
            with col1:
                st.metric("Min", f"{df.iloc[:, 1].min():.2f}")
            with col2:
                st.metric("Max", f"{df.iloc[:, 1].max():.2f}")

    # ── Main content ──────────────────────────────────────────────────────
    if st.session_state.series_data is not None:
        df = st.session_state.series_data

        if st.button(
            "🔍  Run Anomaly Detection",
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
                        st.session_state.detection_just_completed = True

                except Exception as e:
                    st.error(f"Failed to run detection: {e}")
                    logger.error(f"Detection error: {e}")
                finally:
                    st.session_state.is_running = False

        if st.session_state.detection_just_completed:
            st.success(f"✅ Detection complete! Found {len(st.session_state.detection_result.anomalies)} anomalies.")
            st.session_state.detection_just_completed = False

        st.markdown("---")

        # ── Chart ─────────────────────────────────────────────────────────
        st.markdown("### 📊 Time Series Visualization")
        if st.session_state.detection_result:
            anomalies = [{"index": a.index, "value": a.value} for a in st.session_state.detection_result.anomalies]
            fig = plot_time_series(df, anomalies)
        else:
            fig = plot_time_series(df)

        st.plotly_chart(fig, use_container_width=True)

        # ── Results ───────────────────────────────────────────────────────
        if st.session_state.detection_result:
            report = st.session_state.detection_result

            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("🚨 Anomalies Found", len(report.anomalies))
            with col2:
                st.metric("🧪 Detectors Used", len(report.detectors_used))
            with col3:
                st.metric("🔧 Tools Called", len(report.tools_called))

            # Detector names as coloured tags
            detector_tags = " ".join(
                f"<code style='background:#EFF6FF;color:#2563EB;padding:4px 10px;"
                f"border-radius:6px;border:1px solid #BFDBFE;"
                f"font-size:.85rem;font-weight:500'>{d}</code>"
                for d in report.detectors_used
            )
            st.markdown(f"**Detectors:** {detector_tags}", unsafe_allow_html=True)

            st.markdown("")  # spacing

            if st.toggle("🔧 Show All Tool Calls"):
                with st.expander("🛠️ Agent Tool Calls", expanded=True):
                    for tool in report.tools_called:
                        if tool in report.detectors_used:
                            st.markdown(f"✅ `{tool}`")
                        else:
                            st.markdown(f"🔍 `{tool}`")

            with st.expander("📋 View Summary"):
                st.markdown(report.summary)

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
        st.info("👈 Select a time series from the sidebar to get started.")


if __name__ == "__main__":
    main()
