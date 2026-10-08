"""
Data Loader Module — Smart Environmental KPI Dashboard
=======================================================
Handles CSV upload, validation, cleaning, and insertion into the database.

Design note
-----------
This module is intentionally free of Streamlit imports so that it can be
used and tested independently of the web layer.  All user-facing messages
are returned as strings — the caller (app.py) is responsible for displaying
them.
"""

import logging
from datetime import datetime

import pandas as pd

import database

logger = logging.getLogger(__name__)

# Required columns and their valid numeric ranges
REQUIRED_COLUMNS: list[str] = [
    "date",
    "aqi",
    "water_usage",
    "energy_consumption",
    "waste_generated",
    "co2_emissions",
]

NUMERIC_RANGES: dict[str, tuple[float, float]] = {
    "aqi":                (0.0,   500.0),
    "water_usage":        (0.0, 10000.0),
    "energy_consumption": (0.0,  5000.0),
    "waste_generated":    (0.0,  1000.0),
    "co2_emissions":      (0.0,  1000.0),
}


# ---------------------------------------------------------------------------
# Validation helpers
# ---------------------------------------------------------------------------

def validate_csv_columns(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Verify that *df* contains all required columns (case-insensitive).

    Returns:
        (True, success_message) or (False, error_message).
    """
    # Normalise column names for comparison
    missing = [c for c in REQUIRED_COLUMNS if c not in df.columns]
    if missing:
        return False, f"Missing required columns: {', '.join(missing)}"
    return True, "All required columns present."


def validate_data_types(df: pd.DataFrame) -> tuple[bool, pd.DataFrame | None, str]:
    """
    Parse and coerce column types:
    - ``date`` → datetime
    - All numeric columns → float (invalid strings become NaN)

    Rejects any column where > 50 % of values are NaN after coercion.

    Returns:
        (True, cleaned_df, message) or (False, None, error_message).
    """
    df_clean = df.copy()

    # Parse date
    try:
        df_clean["date"] = pd.to_datetime(df_clean["date"], infer_datetime_format=True)
    except Exception as exc:
        return False, None, f"Cannot parse 'date' column: {exc}"

    # Coerce numeric columns
    numeric_cols = [c for c in REQUIRED_COLUMNS if c != "date"]
    for col in numeric_cols:
        df_clean[col] = pd.to_numeric(df_clean[col], errors="coerce")
        nan_ratio = df_clean[col].isna().mean()
        if nan_ratio > 0.5:
            n_nan = df_clean[col].isna().sum()
            return (
                False,
                None,
                f"Column '{col}' has too many missing values ({n_nan}/{len(df_clean)}).",
            )

    return True, df_clean, "Data types validated successfully."


def validate_data_ranges(df: pd.DataFrame) -> tuple[bool, list[str]]:
    """
    Check that numeric values fall within expected physical ranges.

    Returns:
        (all_ok, list_of_warning_strings).
        ``list_of_warning_strings`` is empty when all values are in range.
    """
    warnings: list[str] = []
    for col, (lo, hi) in NUMERIC_RANGES.items():
        if col not in df.columns:
            continue
        out = df[(df[col] < lo) | (df[col] > hi)]
        if not out.empty:
            warnings.append(
                f"'{col}': {len(out)} value(s) outside expected range "
                f"[{lo:.0f} – {hi:.0f}]."
            )
    return len(warnings) == 0, warnings


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def process_uploaded_file(
    uploaded_file,
) -> tuple[bool, pd.DataFrame | None, str, list[str]]:
    """
    Full validation pipeline for a Streamlit UploadedFile object.

    Pipeline:
        1. Read CSV
        2. Validate columns
        3. Validate / coerce data types
        4. Check numeric ranges (yields warnings, not hard errors)

    Args:
        uploaded_file: ``streamlit.runtime.uploaded_file_manager.UploadedFile``

    Returns:
        (success, cleaned_df, message, range_warnings)
        * range_warnings is a list of strings (may be empty).
    """
    try:
        df = pd.read_csv(uploaded_file)
    except Exception as exc:
        return False, None, f"Cannot read CSV file: {exc}", []

    # Step 1 — columns
    ok, msg = validate_csv_columns(df)
    if not ok:
        return False, None, msg, []

    # Step 2 — types
    ok, df_clean, msg = validate_data_types(df)
    if not ok:
        return False, None, msg, []

    # Step 3 — ranges (soft warnings)
    _, range_warnings = validate_data_ranges(df_clean)

    logger.info(
        "File '%s' processed: %d rows, %d range warning(s).",
        getattr(uploaded_file, "name", "unknown"),
        len(df_clean),
        len(range_warnings),
    )
    return True, df_clean, "File processed successfully.", range_warnings


def upload_data_to_database(df: pd.DataFrame) -> tuple[bool, str]:
    """
    Save a validated DataFrame to the database (duplicates are skipped).

    Args:
        df: Cleaned DataFrame from :func:`process_uploaded_file`.

    Returns:
        (True, success_message) or (False, error_message).
    """
    try:
        cols = [c for c in REQUIRED_COLUMNS]  # explicit column selection
        rows_inserted = database.save_data(df[cols].copy())
        msg = (
            f"✓ Inserted {rows_inserted} new record(s). "
            "Duplicate dates were skipped automatically."
        )
        return True, msg
    except Exception as exc:
        logger.error("upload_data_to_database failed: %s", exc)
        return False, f"Database error: {exc}"


# ---------------------------------------------------------------------------
# CSV template
# ---------------------------------------------------------------------------

def get_csv_template() -> pd.DataFrame:
    """
    Build a 3-row example DataFrame matching the required upload format.

    Returns:
        DataFrame with sample values for the last 3 days.
    """
    today = datetime.now()
    dates = [(today - pd.Timedelta(days=i)).strftime("%Y-%m-%d") for i in range(2, -1, -1)]

    return pd.DataFrame({
        "date":               dates,
        "aqi":                [45.5,  52.3,  48.1],
        "water_usage":        [950.0, 1020.5, 980.0],
        "energy_consumption": [480.0,  510.0, 495.0],
        "waste_generated":    [ 95.0,  102.0,  98.5],
        "co2_emissions":      [185.0,  195.0, 190.0],
    })


def download_template_csv() -> str:
    """
    Return the CSV template as a string ready for ``st.download_button``.
    """
    return get_csv_template().to_csv(index=False)
