"""
Smart Environmental KPI Dashboard — Main Application
====================================================
Streamlit entry point.  Run with:  streamlit run app.py

Pages
-----
  Dashboard       — KPI cards, alerts, trend charts, ML predictions, anomalies
  Data Management — CSV upload, database viewer, data export
  Settings        — Customise KPI targets and thresholds
"""

import logging
import time
import urllib.request

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
from datetime import datetime, timedelta

# Local modules
import database
import data_loader
import kpi_engine
import ml_model
from generate_sample_data import generate_sample_data

# ---------------------------------------------------------------------------
# Logging
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Page configuration — must be first Streamlit call
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Smart Environmental KPI Dashboard",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ---------------------------------------------------------------------------
# Custom CSS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
/* ── Unified KPI Cards ── */
.kpi-card {
    padding: 20px;
    border-radius: 12px;
    box-shadow: 0 4px 10px rgba(0,0,0,0.07);
    display: flex;
    flex-direction: column;
    margin-bottom: 1rem;
}

.kpi-title {
    font-size: 1.05rem;
    font-weight: 600;
    margin-bottom: 8px;
    display: flex;
    align-items: center;
    gap: 8px;
}

.kpi-value {
    font-size: 2.4rem;
    font-weight: 700;
    line-height: 1.1;
    margin-bottom: 4px;
}

.kpi-delta {
    font-size: 0.95rem;
    font-weight: 500;
    margin-bottom: 16px;
}

.kpi-footer {
    display: flex;
    justify-content: space-between;
    align-items: center;
    padding-top: 12px;
    font-size: 0.9rem;
}

.kpi-target {
    font-weight: 600;
}

/* Status: OK */
.kpi-card-ok { background-color: rgba(76,175,80,0.08); border-left: 4px solid #4caf50; }
.kpi-card-ok .kpi-title { color: #33691e; }
.kpi-card-ok .kpi-value { color: #2e7d32; }
.kpi-card-ok .kpi-delta { color: #388e3c; }
.kpi-card-ok .kpi-footer { border-top: 1px solid rgba(76,175,80,0.2); }
.kpi-card-ok .kpi-target { color: #33691e; }
.kpi-card-ok .status-badge { background-color: #4caf50; color: white; }

/* Status: Warning */
.kpi-card-warning { background-color: rgba(255,152,0,0.08); border-left: 4px solid #ff9800; }
.kpi-card-warning .kpi-title { color: #e65100; }
.kpi-card-warning .kpi-value { color: #bf360c; }
.kpi-card-warning .kpi-delta { color: #f57c00; }
.kpi-card-warning .kpi-footer { border-top: 1px solid rgba(255,152,0,0.2); }
.kpi-card-warning .kpi-target { color: #e65100; }
.kpi-card-warning .status-badge { background-color: #ff9800; color: white; }

/* Status: Critical */
.kpi-card-critical { background-color: rgba(244,67,54,0.08); border-left: 4px solid #f44336; }
.kpi-card-critical .kpi-title { color: #c62828; }
.kpi-card-critical .kpi-value { color: #b71c1c; }
.kpi-card-critical .kpi-delta { color: #c62828; }
.kpi-card-critical .kpi-footer { border-top: 1px solid rgba(244,67,54,0.2); }
.kpi-card-critical .kpi-target { color: #c62828; }
.kpi-card-critical .status-badge { background-color: #f44336; color: white; }

/* Status badge pill */
.status-badge {
    display: inline-block; 
    padding: 4px 12px; 
    border-radius: 12px;
    font-size: 0.75rem; 
    font-weight: 700;
    text-transform: uppercase; 
    letter-spacing: 0.6px;
}

h1 { color: #1f77b4; }
.stAlert { margin-top: 1rem; }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# App initialisation — runs once per session
# ---------------------------------------------------------------------------

def initialize_app() -> None:
    """
    Initialise DB and auto-seed with 90 days of sample data when the
    database is empty.  This ensures the dashboard always shows data on
    Streamlit Cloud (ephemeral filesystem) and in fresh local setups.
    """
    database.init_db()

    summary = database.get_data_summary()
    if summary["total_records"] == 0:
        logger.info("Empty database detected — seeding with 90 days of sample data.")
        seed_df = generate_sample_data(days=90, output_file=None)  # returns df, no file write
        database.save_data(seed_df)
        logger.info("Sample data loaded.")


# ---------------------------------------------------------------------------
# Weather helper
# ---------------------------------------------------------------------------

@st.cache_data(ttl=3600)
def get_current_weather() -> str:
    """
    Fetch one-line weather from wttr.in.  Cached for 1 hour to avoid
    hammering the free API on every rerun.

    Returns:
        A short weather string, or a fallback message on failure.
    """
    try:
        req = urllib.request.Request(
            "https://wttr.in/?format=1",
            headers={"User-Agent": "curl/7.68.0"},
        )
        with urllib.request.urlopen(req, timeout=5) as resp:
            raw = resp.read().decode("utf-8", errors="replace").strip()
        if any(kw in raw for kw in ("Unknown", "unreachable", "Sorry")):
            return " Weather unavailable"
        return raw
    except Exception:
        return " Weather unavailable"


# ---------------------------------------------------------------------------
# KPI card component
# ---------------------------------------------------------------------------

def display_kpi_card(metric_name: str, kpi_info: dict, col) -> None:
    """
    Render a single colour-coded KPI metric card inside *col*.

    Args:
        metric_name: Snake-case metric identifier (e.g. 'aqi').
        kpi_info:    Dict returned by kpi_engine.get_latest_kpis().
        col:         Streamlit column object.
    """
    with col:
        display_name = metric_name.replace("_", " ").title()

        current = kpi_info["current_value"]
        target  = kpi_info["target_value"]
        change  = kpi_info["pct_change"]
        status  = kpi_info["status"]
        icon    = kpi_info["icon"]

        # CSS classes based on status
        status_map = {
            "OK":       "kpi-card-ok",
            "Warning":  "kpi-card-warning",
            "Critical": "kpi-card-critical",
        }
        card_class = status_map.get(status, "kpi-card-warning")

        value_str = f"{current:.1f}" if current is not None else "N/A"
        delta_str = f"{change:+.1f}% vs last month"

        html_content = f"""
        <div class="kpi-card {card_class}">
            <div class="kpi-title">{icon} {display_name}</div>
            <div class="kpi-value">{value_str}</div>
            <div class="kpi-delta">{delta_str}</div>
            <div class="kpi-footer">
                <span class="status-badge">{status}</span>
                <span class="kpi-target">Target: {target:.1f}</span>
            </div>
        </div>
        """
        st.markdown(html_content, unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# Chart builders
# ---------------------------------------------------------------------------

def create_trend_chart(df: pd.DataFrame, metric_name: str) -> go.Figure:
    """
    Build an interactive line chart showing a metric's historical trend plus
    target and threshold reference lines.
    """
    df_sorted    = df.sort_values("date")
    targets_df   = database.fetch_targets()
    target_row   = targets_df[targets_df["metric_name"] == metric_name]

    target_value = threshold_warning = threshold_critical = None
    if not target_row.empty:
        target_value       = float(target_row.iloc[0]["target_value"])
        threshold_warning  = float(target_row.iloc[0]["threshold_warning"])
        threshold_critical = float(target_row.iloc[0]["threshold_critical"])

    fig = go.Figure()

    # Actual values
    fig.add_trace(go.Scatter(
        x=df_sorted["date"],
        y=df_sorted[metric_name],
        mode="lines+markers",
        name="Actual",
        line=dict(color="#1f77b4", width=3),
        marker=dict(size=5),
    ))

    n = len(df_sorted)
    x = df_sorted["date"]

    if target_value is not None:
        fig.add_trace(go.Scatter(x=x, y=[target_value] * n, mode="lines",
                                 name="Target", line=dict(color="green", width=2, dash="dash")))
    if threshold_warning is not None:
        fig.add_trace(go.Scatter(x=x, y=[threshold_warning] * n, mode="lines",
                                 name="Warning", line=dict(color="orange", width=1, dash="dot")))
    if threshold_critical is not None:
        fig.add_trace(go.Scatter(x=x, y=[threshold_critical] * n, mode="lines",
                                 name="Critical", line=dict(color="red", width=1, dash="dot")))

    display_name = metric_name.replace("_", " ").title()
    fig.update_layout(
        title=f"{display_name} — Trend Over Time",
        xaxis_title="Date",
        yaxis_title=display_name,
        hovermode="x unified",
        height=400,
        showlegend=True,
    )
    return fig


def create_comparison_chart(kpis_dict: dict) -> go.Figure:
    """
    Grouped bar chart comparing current values vs targets for all metrics.
    """
    metrics       = list(kpis_dict.keys())
    display_names = [m.replace("_", " ").title() for m in metrics]
    actual_values = [kpis_dict[m]["current_value"] or 0 for m in metrics]
    target_values = [kpis_dict[m]["target_value"] for m in metrics]

    fig = go.Figure()
    fig.add_trace(go.Bar(name="Actual", x=display_names, y=actual_values, marker_color="#1f77b4"))
    fig.add_trace(go.Bar(name="Target", x=display_names, y=target_values, marker_color="#2ca02c"))

    fig.update_layout(
        title="Current Values vs Targets",
        xaxis_title="Metric",
        yaxis_title="Value",
        barmode="group",
        height=400,
    )
    return fig


# ---------------------------------------------------------------------------
# Section renderers
# ---------------------------------------------------------------------------

def display_predictions(df: pd.DataFrame) -> None:
    """Render ML next-month predictions section."""
    st.subheader(" Next-Month Predictions (ML Forecast)")

    predictions = ml_model.generate_all_predictions(df)
    cols = st.columns(5)

    for idx, (metric, pred) in enumerate(predictions.items()):
        with cols[idx % 5]:
            display_name = metric.replace("_", " ").title()
            if pred["predicted_value"] is not None:
                st.metric(
                    label=display_name,
                    value=f"{pred['predicted_value']:.1f}",
                    delta=f"Confidence: {pred['confidence']}",
                )
                date_str = pred["prediction_date"].strftime("%Y-%m-%d")
                r2_str   = f"{pred['r2_score']:.2f}" if pred["r2_score"] is not None else "N/A"
                st.caption(f" {date_str} · R²={r2_str}")
            else:
                st.metric(label=display_name, value="N/A", delta="Insufficient data")


def display_anomalies(df: pd.DataFrame) -> None:
    """Render the anomaly detection summary section."""
    st.subheader(" Anomaly Detection")

    summary = ml_model.get_anomaly_summary(df)

    if summary["total_anomalies"] > 0:
        st.warning(
            f" Detected **{summary['total_anomalies']}** anomalies "
            f"across **{len(summary['metrics_affected'])}** metric(s)."
        )
        for metric in summary["metrics_affected"]:
            detail = summary["details"][metric]
            with st.expander(
                f" {metric.replace('_', ' ').title()} — {detail['count']} anomaly(s)"
            ):
                if detail["records"]:
                    adf = pd.DataFrame(detail["records"])
                    adf["date"] = pd.to_datetime(adf["date"]).dt.strftime("%Y-%m-%d")
                    st.dataframe(adf, use_container_width=True)
    else:
        st.success(" No anomalies detected in the current dataset.")


def display_alerts(df: pd.DataFrame, kpis_dict: dict) -> None:
    """
    Display active alerts for critical/warning KPI statuses and recent spikes.
    Shows at most 10 alerts to keep the UI clean.
    """
    alerts: list[dict] = []

    # Threshold breaches
    for metric, info in kpis_dict.items():
        if info["status"] == "Critical":
            alerts.append({
                "type":    "Critical",
                "metric":  metric.replace("_", " ").title(),
                "message": f"exceeds critical threshold ({info['threshold_critical']:.1f})",
                "value":   info["current_value"],
            })
        elif info["status"] == "Warning":
            alerts.append({
                "type":    "Warning",
                "metric":  metric.replace("_", " ").title(),
                "message": f"exceeds warning threshold ({info['threshold_warning']:.1f})",
                "value":   info["current_value"],
            })

    # Spike alerts (show last 3 per metric)
    for metric in ml_model.METRICS:
        for spike in ml_model.get_spike_alerts(df, metric, spike_threshold=1.3)[-3:]:
            alerts.append({
                "type":    "Spike",
                "metric":  metric.replace("_", " ").title(),
                "message": (
                    f"sudden +{spike['increase_pct']:.1f}% on "
                    f"{spike['date'].strftime('%Y-%m-%d')}"
                ),
                "value":   spike["current_value"],
            })

    if alerts:
        st.subheader("⚠️ Active Alerts")
        for alert in alerts[:10]:
            if alert["type"] == "Critical":
                st.error(
                    f" **{alert['metric']}**: {alert['message']} "
                    f"(Current: {alert['value']:.1f})"
                )
            elif alert["type"] == "Warning":
                st.warning(
                    f" **{alert['metric']}**: {alert['message']} "
                    f"(Current: {alert['value']:.1f})"
                )
            else:
                st.info(f" **{alert['metric']}**: {alert['message']}")


# ---------------------------------------------------------------------------
# Pages
# ---------------------------------------------------------------------------

def dashboard_page() -> None:
    """Main dashboard: KPI cards, charts, predictions, and anomaly section."""
    st.title(" Smart Environmental KPI Dashboard")

    # Date + weather header row
    col_date, col_weather = st.columns([3, 1])
    with col_date:
        st.markdown(f"####  {datetime.now().strftime('%A, %B %d, %Y')}")
    with col_weather:
        st.markdown(f"#### {get_current_weather()}")

    # Live auto-refresh toggle
    col_div, col_toggle = st.columns([3, 1])
    with col_div:
        st.markdown("---")
    with col_toggle:
        live_mode = st.toggle(
            " Live Monitoring",
            value=False,
            help="Auto-refreshes the page every 30 seconds",
        )

    # Auto-refresh using meta-refresh (no extra package required)
    if live_mode:
        refresh_interval = 30  # seconds
        st.markdown(
            f'<meta http-equiv="refresh" content="{refresh_interval}">',
            unsafe_allow_html=True,
        )
        st.caption(f" Page will auto-refresh every {refresh_interval} s")

    # Fetch data
    df         = database.fetch_data()
    targets_df = database.fetch_targets()

    if df.empty:
        st.warning(" No data available.")
        st.info(" Go to ** Data Management** to upload a CSV, or wait for the auto-seed.")
        return

    kpis = kpi_engine.get_latest_kpis(df, targets_df)

    # Alerts
    display_alerts(df, kpis)
    st.markdown("---")

    # KPI cards
    st.subheader(" Current KPI Status")
    cols = st.columns(5)
    for idx, metric in enumerate(kpi_engine.METRICS):
        if metric in kpis:
            display_kpi_card(metric, kpis[metric], cols[idx])

    st.markdown("---")

    # Comparison chart
    st.subheader(" Actual vs Target Comparison")
    st.plotly_chart(create_comparison_chart(kpis), use_container_width=True)
    st.markdown("---")

    # Trend charts (tabbed)
    st.subheader(" Trend Analysis")
    tabs = st.tabs([m.replace("_", " ").title() for m in kpi_engine.METRICS])
    for idx, metric in enumerate(kpi_engine.METRICS):
        with tabs[idx]:
            st.plotly_chart(create_trend_chart(df, metric), use_container_width=True)

            stats = kpi_engine.get_metric_statistics(df, metric)
            if stats:
                c1, c2, c3, c4 = st.columns(4)
                c1.metric("Average", f"{stats['mean']:.2f}")
                c2.metric("Median",  f"{stats['median']:.2f}")
                c3.metric("Min",     f"{stats['min']:.2f}")
                c4.metric("Max",     f"{stats['max']:.2f}")

    st.markdown("---")

    # ML predictions
    display_predictions(df)
    st.markdown("---")

    # Anomaly detection
    display_anomalies(df)


def data_management_page() -> None:
    """CSV upload, database viewer, and data export page."""
    st.title("📁 Data Management")
    st.markdown("---")

    # Download template
    st.subheader(" CSV Template")
    st.markdown("Download a sample template to see the expected column format.")
    st.download_button(
        label=" Download Template CSV",
        data=data_loader.download_template_csv(),
        file_name="environmental_data_template.csv",
        mime="text/csv",
    )

    st.markdown("---")

    # Upload section
    st.subheader(" Upload New Data")
    uploaded_file = st.file_uploader(
        "Choose a CSV file",
        type=["csv"],
        help="Must contain: date, aqi, water_usage, energy_consumption, waste_generated, co2_emissions",
    )

    if uploaded_file is not None:
        success, df_clean, message, range_warnings = data_loader.process_uploaded_file(uploaded_file)

        if success:
            st.success(f" {message}")

            # Show range warnings (from pure data_loader — displayed here in UI layer)
            for w in range_warnings:
                st.warning(f" {w}")

            st.subheader(" Data Preview (10 most recent rows)")
            st.dataframe(
                df_clean.sort_values("date", ascending=False).head(10),
                use_container_width=True,
            )

            if st.button(" Upload to Database", type="primary"):
                ok, upload_msg = data_loader.upload_data_to_database(df_clean)
                if ok:
                    st.success(upload_msg)
                    st.balloons()
                else:
                    st.error(upload_msg)
        else:
            st.error(f"❌ {message}")

    st.markdown("---")

    # Current database viewer
    st.subheader(" Current Database")
    df = database.fetch_data()

    if not df.empty:
        summary = database.get_data_summary()
        c1, c2, c3 = st.columns(3)
        c1.metric("Total Records", summary["total_records"])
        c2.metric("Date From",     summary["date_from"])
        c3.metric("Date To",       summary["date_to"])

        st.markdown("---")
        st.dataframe(
            df.sort_values("date", ascending=False),
            use_container_width=True,
            height=400,
        )

        st.download_button(
            label=" Download Current Data as CSV",
            data=df.to_csv(index=False),
            file_name=f"environmental_data_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
        )

        st.markdown("---")
        st.subheader(" Reset Database")
        st.warning("This will permanently delete all stored data and reload sample data.")
        if st.button(" Reset & Reload Sample Data", type="secondary"):
            database.clear_all_data()
            seed_df = generate_sample_data(days=90, output_file=None)
            database.save_data(seed_df)
            st.success(" Database reset with fresh 90-day sample data.")
            st.rerun()
    else:
        st.info("No data in database yet. Upload a CSV or reset to reload sample data.")


def settings_page() -> None:
    """KPI target configuration page."""
    st.title("⚙️ Settings")
    st.markdown("---")

    st.subheader(" Customise KPI Targets")
    st.markdown(
        "Adjust target values and thresholds for each environmental metric. "
        "Changes take effect immediately on the Dashboard."
    )

    targets_df = database.fetch_targets()

    with st.form("targets_form"):
        for _, row in targets_df.iterrows():
            metric       = row["metric_name"]
            display_name = metric.replace("_", " ").title()

            st.markdown(f"**{display_name}** — *{row['description']}*")
            c1, c2, c3 = st.columns(3)
            with c1:
                st.number_input("Target Value",      value=float(row["target_value"]),      key=f"{metric}_target",   step=1.0)
            with c2:
                st.number_input("Warning Threshold", value=float(row["threshold_warning"]),  key=f"{metric}_warning",  step=1.0)
            with c3:
                st.number_input("Critical Threshold",value=float(row["threshold_critical"]), key=f"{metric}_critical", step=1.0)
            st.markdown("---")

        if st.form_submit_button("💾 Save All Targets", type="primary"):
            for _, row in targets_df.iterrows():
                metric = row["metric_name"]
                database.update_target(
                    metric,
                    st.session_state[f"{metric}_target"],
                    st.session_state[f"{metric}_warning"],
                    st.session_state[f"{metric}_critical"],
                )
            st.success("✅ Targets updated! Return to the Dashboard to see the changes.")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    """Application entry point — called by Streamlit on every run."""
    initialize_app()

    # Sidebar navigation
    st.sidebar.title("🌍 Navigation")
    st.sidebar.markdown("---")

    page = st.sidebar.radio(
        "Select page",
        ["📊 Dashboard", "📁 Data Management", "⚙️ Settings"],
        label_visibility="collapsed",
    )

    st.sidebar.markdown("---")
    st.sidebar.info("""
**Smart Environmental KPI Dashboard**

Monitor and analyse 5 environmental metrics:

 Air Quality Index (AQI)
 Water Usage
 Energy Consumption
 Waste Generated
 CO₂ Emissions

**Features**
- Live KPI monitoring
- Trend analysis charts
- ML-based forecasting
- Anomaly & spike detection
- Alert system
- CSV import/export
""")

    st.sidebar.markdown("---")
    st.sidebar.caption("v1.0.0 · Built with Streamlit")

    # Route to page
    if page == "📊 Dashboard":
        dashboard_page()
    elif page == "📁 Data Management":
        data_management_page()
    elif page == "⚙️ Settings":
        settings_page()


if __name__ == "__main__":
    main()
