"""
KPI Engine Module — Smart Environmental KPI Dashboard
=====================================================
Handles all KPI calculations, trend analysis, and status determination.
Pure computation module — no UI dependencies.
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd

logger = logging.getLogger(__name__)

# Metrics tracked throughout the application
METRICS: list[str] = [
    "aqi",
    "water_usage",
    "energy_consumption",
    "waste_generated",
    "co2_emissions",
]


# ---------------------------------------------------------------------------
# Aggregation helpers
# ---------------------------------------------------------------------------

def calculate_monthly_kpis(df: pd.DataFrame) -> pd.DataFrame:
    """
    Resample daily data to monthly averages.

    Args:
        df: DataFrame with a ``date`` column and numeric metric columns.

    Returns:
        DataFrame indexed by month-end date with mean values per month,
        or an empty DataFrame if *df* is empty.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    df_indexed = df.set_index("date")

    # 'ME' = Month-End frequency (replaces deprecated 'M' in pandas ≥ 2.2)
    monthly_df = df_indexed.resample("ME").mean(numeric_only=True)
    return monthly_df.reset_index()


def calculate_daily_averages(df: pd.DataFrame) -> pd.DataFrame:
    """
    Average multiple readings per day into a single daily value.

    Args:
        df: DataFrame with a ``date`` column.

    Returns:
        DataFrame with one row per day.
    """
    if df.empty:
        return pd.DataFrame()

    df = df.copy()
    df["date"] = pd.to_datetime(df["date"])
    return df.groupby("date").mean(numeric_only=True).reset_index()


# ---------------------------------------------------------------------------
# Trend analysis
# ---------------------------------------------------------------------------

def calculate_trend(df: pd.DataFrame, metric_name: str) -> dict:
    """
    Calculate overall trend for a metric across the full date range.

    Args:
        df:          DataFrame with ``date`` and metric columns.
        metric_name: Column name of the metric.

    Returns:
        Dict with keys: trend (str), change (float), direction (str),
        first_value (float), last_value (float).
    """
    if df.empty or metric_name not in df.columns:
        return {"trend": "N/A", "change": 0.0, "direction": "neutral"}

    df_sorted = df.sort_values("date")
    first_value = df_sorted[metric_name].iloc[0]
    last_value  = df_sorted[metric_name].iloc[-1]

    pct_change = calculate_percentage_change(last_value, first_value)

    if pct_change > 5:
        direction = "increasing"
    elif pct_change < -5:
        direction = "decreasing"
    else:
        direction = "stable"

    return {
        "trend":       f"{pct_change:.2f}%",
        "change":      pct_change,
        "direction":   direction,
        "first_value": first_value,
        "last_value":  last_value,
    }


def calculate_percentage_change(current: float, previous: float) -> float:
    """
    Calculate percentage change from *previous* to *current*.

    Returns 0 if *previous* is zero or either value is NaN.
    """
    if previous == 0 or pd.isna(previous) or pd.isna(current):
        return 0.0
    return ((current - previous) / previous) * 100.0


# ---------------------------------------------------------------------------
# Status determination
# ---------------------------------------------------------------------------

def check_status(
    actual_value: float,
    target_value: float,
    threshold_warning: float,
    threshold_critical: float,
    lower_is_better: bool = True,
) -> dict:
    """
    Determine KPI status relative to its target and thresholds.

    For all five metrics (AQI, water, energy, waste, CO2) lower values are
    better, so ``lower_is_better=True`` (default).

    Args:
        actual_value:       Measured value.
        target_value:       Desired target.
        threshold_warning:  Value at which status becomes Warning.
        threshold_critical: Value at which status becomes Critical.
        lower_is_better:    True for pollution-type metrics.

    Returns:
        Dict with keys: status, color, icon.
    """
    if pd.isna(actual_value):
        return {"status": "Unknown", "color": "gray", "icon": "❓"}

    if lower_is_better:
        if actual_value <= target_value:
            return {"status": "OK",       "color": "green",  "icon": "✅"}
        elif actual_value <= threshold_warning:
            return {"status": "Warning",  "color": "orange", "icon": "⚠️"}
        else:
            return {"status": "Critical", "color": "red",    "icon": "🚨"}
    else:
        if actual_value >= target_value:
            return {"status": "OK",       "color": "green",  "icon": "✅"}
        elif actual_value >= threshold_warning:
            return {"status": "Warning",  "color": "orange", "icon": "⚠️"}
        else:
            return {"status": "Critical", "color": "red",    "icon": "🚨"}


# ---------------------------------------------------------------------------
# KPI snapshot
# ---------------------------------------------------------------------------

def get_latest_kpis(df: pd.DataFrame, targets_df: pd.DataFrame) -> dict:
    """
    Build a full KPI snapshot for the most recent date in *df*.

    Compares the latest reading against the value from ~30 days earlier to
    compute a month-over-month percentage change.

    Args:
        df:         DataFrame with environmental data (must have a ``date``
                    column already parsed as datetime).
        targets_df: DataFrame from ``database.fetch_targets()``.

    Returns:
        Dict keyed by metric name, each value being a dict with:
        current_value, target_value, pct_change, status, color, icon,
        threshold_warning, threshold_critical.
    """
    if df.empty:
        return {}

    latest_date  = df["date"].max()
    latest_row   = df[df["date"] == latest_date].iloc[0]

    # Previous reference: closest row at or before 30 days ago
    cutoff = latest_date - timedelta(days=30)
    prev_rows = df[df["date"] <= cutoff]
    prev_row  = prev_rows.iloc[-1] if not prev_rows.empty else None

    kpis: dict = {}

    for metric in METRICS:
        # Resolve target information
        t_row = targets_df[targets_df["metric_name"] == metric]
        if not t_row.empty:
            t_row            = t_row.iloc[0]
            target_value     = float(t_row["target_value"])
            thresh_warning   = float(t_row["threshold_warning"])
            thresh_critical  = float(t_row["threshold_critical"])
        else:
            target_value = thresh_warning = thresh_critical = 0.0

        # Current value
        current_value = float(latest_row[metric]) if metric in latest_row.index else None

        # Month-over-month change
        if prev_row is not None and metric in prev_row.index and current_value is not None:
            pct_change = calculate_percentage_change(current_value, float(prev_row[metric]))
        else:
            pct_change = 0.0

        # Status
        status_info = check_status(
            current_value, target_value, thresh_warning, thresh_critical
        )

        kpis[metric] = {
            "current_value":      current_value,
            "target_value":       target_value,
            "pct_change":         pct_change,
            "status":             status_info["status"],
            "color":              status_info["color"],
            "icon":               status_info["icon"],
            "threshold_warning":  thresh_warning,
            "threshold_critical": thresh_critical,
        }

    return kpis


# ---------------------------------------------------------------------------
# Statistics
# ---------------------------------------------------------------------------

def get_metric_statistics(df: pd.DataFrame, metric_name: str) -> dict:
    """
    Compute descriptive statistics for a single metric.

    Args:
        df:          DataFrame containing the metric column.
        metric_name: Column name.

    Returns:
        Dict with keys: mean, median, std, min, max, count.
        Returns empty dict if data is unavailable.
    """
    if df.empty or metric_name not in df.columns:
        return {}

    series = df[metric_name].dropna()
    return {
        "mean":   series.mean(),
        "median": series.median(),
        "std":    series.std(),
        "min":    series.min(),
        "max":    series.max(),
        "count":  series.count(),
    }
