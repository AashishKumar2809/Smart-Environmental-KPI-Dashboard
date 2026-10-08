"""
Machine Learning Module — Smart Environmental KPI Dashboard
===========================================================
Provides next-month forecasting and anomaly / spike detection for all five
environmental metrics using lightweight scikit-learn models.

ML Approach
-----------
* Forecasting    : Linear regression on ordinal date feature.
                   R² score is used to indicate prediction confidence.
* Anomaly detect : Z-score method (default threshold = 2.5 σ).
* Spike detect   : Day-over-day ratio threshold (default = 1.3 × previous).
"""

import logging
from datetime import timedelta

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

logger = logging.getLogger(__name__)

# Canonical list of tracked metrics — single source of truth
METRICS: list[str] = [
    "aqi",
    "water_usage",
    "energy_consumption",
    "waste_generated",
    "co2_emissions",
]


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _prepare_time_series(
    df: pd.DataFrame, metric_name: str
) -> tuple[np.ndarray | None, np.ndarray | None]:
    """
    Convert a time-series DataFrame into (X, y) arrays suitable for sklearn.

    Args:
        df:          DataFrame with ``date`` (datetime) and metric columns.
        metric_name: Column to use as the regression target.

    Returns:
        (X, y) where X is ordinal dates shaped (n, 1) and y is the metric
        values, or (None, None) if there are fewer than 3 clean rows.
    """
    if df.empty or metric_name not in df.columns:
        return None, None

    df_s = df[["date", metric_name]].dropna().sort_values("date").copy()

    if len(df_s) < 3:
        return None, None

    X = df_s["date"].map(lambda d: d.toordinal()).values.reshape(-1, 1)
    y = df_s[metric_name].values
    return X, y


# ---------------------------------------------------------------------------
# Forecasting
# ---------------------------------------------------------------------------

def predict_next_month(
    df: pd.DataFrame, metric_name: str, days_ahead: int = 30
) -> dict:
    """
    Predict the metric value *days_ahead* days into the future using a trained
    linear regression model.

    Args:
        df:          Historical data DataFrame.
        metric_name: Metric to forecast.
        days_ahead:  How many days ahead to predict (default 30).

    Returns:
        Dict with keys:
            predicted_value  (float | None)
            prediction_date  (datetime | None)
            confidence       ('High' | 'Medium' | 'Low')
            r2_score         (float | None)
            model_type       (str)
    """
    X, y = _prepare_time_series(df, metric_name)

    if X is None:
        return {
            "predicted_value": None,
            "prediction_date": None,
            "confidence":      "Low",
            "r2_score":        None,
            "model_type":      "N/A — insufficient data",
        }

    # Train linear regression on the full history
    model = LinearRegression()
    model.fit(X, y)

    # Build future date ordinal
    latest_date  = df["date"].max()
    future_date  = latest_date + timedelta(days=days_ahead)
    future_ord   = np.array([[future_date.toordinal()]])

    predicted_value: float = float(model.predict(future_ord)[0])

    # R² on training data — used as a confidence proxy
    r2: float = float(model.score(X, y))

    # A negative R² means the model is worse than predicting the mean
    if r2 > 0.7:
        confidence = "High"
    elif r2 > 0.4:
        confidence = "Medium"
    else:
        confidence = "Low"

    logger.debug(
        "Prediction for %s: %.2f (R²=%.3f, confidence=%s)",
        metric_name, predicted_value, r2, confidence,
    )

    return {
        "predicted_value": predicted_value,
        "prediction_date": future_date,
        "confidence":      confidence,
        "r2_score":        r2,
        "model_type":      "Linear Regression",
    }


def generate_all_predictions(df: pd.DataFrame) -> dict:
    """
    Generate next-month predictions for every tracked metric.

    Args:
        df: Historical data DataFrame.

    Returns:
        Dict keyed by metric name, values are prediction dicts from
        :func:`predict_next_month`.
    """
    return {
        metric: predict_next_month(df, metric)
        for metric in METRICS
        if metric in df.columns
    }


# ---------------------------------------------------------------------------
# Anomaly detection — Z-score method
# ---------------------------------------------------------------------------

def detect_anomalies_zscore(
    df: pd.DataFrame, metric_name: str, threshold: float = 2.5
) -> pd.DataFrame:
    """
    Flag rows whose metric value deviates more than *threshold* standard
    deviations from the column mean (Z-score method).

    Args:
        df:          Input DataFrame.
        metric_name: Column to analyse.
        threshold:   Z-score magnitude above which a point is anomalous.

    Returns:
        Copy of *df* with additional columns ``z_score`` and ``is_anomaly``.
    """
    if df.empty or metric_name not in df.columns:
        return df

    df_out = df.copy()
    series  = df_out[metric_name]
    mean    = series.mean()
    std     = series.std()

    if std == 0 or pd.isna(std):
        df_out["z_score"]   = 0.0
        df_out["is_anomaly"] = False
        return df_out

    df_out["z_score"]   = (series - mean) / std
    df_out["is_anomaly"] = df_out["z_score"].abs() > threshold
    return df_out


def detect_anomalies_all_metrics(
    df: pd.DataFrame, threshold: float = 2.5
) -> dict:
    """
    Run Z-score anomaly detection for every tracked metric.

    Args:
        df:        Input DataFrame.
        threshold: Z-score threshold (default 2.5).

    Returns:
        Dict keyed by metric name, each value is:
        ``{'count': int, 'records': list[dict]}``.
    """
    anomalies: dict = {}

    for metric in METRICS:
        if metric not in df.columns:
            continue
        df_flagged = detect_anomalies_zscore(df, metric, threshold)
        bad_rows   = df_flagged[df_flagged["is_anomaly"]]

        anomalies[metric] = {
            "count":   len(bad_rows),
            "records": (
                bad_rows[["date", metric, "z_score"]].to_dict("records")
                if not bad_rows.empty
                else []
            ),
        }

    return anomalies


def get_anomaly_summary(df: pd.DataFrame) -> dict:
    """
    Produce a dashboard-ready summary of all detected anomalies.

    Returns:
        Dict with keys:
            total_anomalies   (int)
            metrics_affected  (list[str])
            details           (dict — same structure as detect_anomalies_all_metrics)
    """
    details = detect_anomalies_all_metrics(df)
    total   = sum(v["count"] for v in details.values())
    affected = [m for m, v in details.items() if v["count"] > 0]

    return {
        "total_anomalies":  total,
        "metrics_affected": affected,
        "details":          details,
    }


# ---------------------------------------------------------------------------
# Spike detection — day-over-day ratio
# ---------------------------------------------------------------------------

def get_spike_alerts(
    df: pd.DataFrame, metric_name: str, spike_threshold: float = 1.3
) -> list[dict]:
    """
    Detect sudden day-over-day spikes in a metric.

    A spike is defined as: current_value / previous_value >= *spike_threshold*.

    Args:
        df:              Input DataFrame (must have ``date`` and metric columns).
        metric_name:     Column to check.
        spike_threshold: Multiplier threshold (1.3 → 30 % increase triggers alert).

    Returns:
        List of spike dicts with keys: date, metric, current_value,
        previous_value, increase_pct.
    """
    if df.empty or metric_name not in df.columns:
        return []

    df_s = (
        df[["date", metric_name]]
        .dropna()
        .sort_values("date")
        .reset_index(drop=True)
    )

    if len(df_s) < 2:
        return []

    spikes: list[dict] = []

    for i in range(1, len(df_s)):
        prev = float(df_s.at[i - 1, metric_name])
        curr = float(df_s.at[i, metric_name])

        if prev > 0 and (curr / prev) >= spike_threshold:
            spikes.append({
                "date":           df_s.at[i, "date"],
                "metric":         metric_name,
                "current_value":  curr,
                "previous_value": prev,
                "increase_pct":   round((curr / prev - 1) * 100, 1),
            })

    return spikes
