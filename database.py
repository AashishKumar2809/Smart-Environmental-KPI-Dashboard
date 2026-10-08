"""
Database Module — Smart Environmental KPI Dashboard
====================================================
Handles SQLite database connection, table creation, and all data operations.
Designed to work both locally and on Streamlit Cloud (writable path fallback).
"""

import sqlite3
import logging
import os
import pandas as pd
from datetime import datetime

# ---------------------------------------------------------------------------
# Logging setup
# ---------------------------------------------------------------------------
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
)
logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Database path — prefer the project directory; fall back to a writable
# temp location when running on Streamlit Cloud (read-only source tree).
# ---------------------------------------------------------------------------
_DEFAULT_DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environmental_kpi.db")
DB_PATH: str = os.environ.get("KPI_DB_PATH", _DEFAULT_DB_PATH)


# ---------------------------------------------------------------------------
# Connection helper
# ---------------------------------------------------------------------------

def get_connection() -> sqlite3.Connection:
    """Return a new SQLite connection with WAL journal mode for concurrency."""
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    conn.execute("PRAGMA journal_mode=WAL;")
    return conn


# ---------------------------------------------------------------------------
# Schema initialisation
# ---------------------------------------------------------------------------

def init_db() -> None:
    """
    Initialise the database — create tables if they don't exist and insert
    default targets when the targets table is empty.
    Safe to call multiple times (idempotent).
    """
    conn = get_connection()
    cursor = conn.cursor()

    # --- environmental_data table -------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS environmental_data (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            date                 TEXT    NOT NULL UNIQUE,
            aqi                  REAL,
            water_usage          REAL,
            energy_consumption   REAL,
            waste_generated      REAL,
            co2_emissions        REAL,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        )
    """)

    # --- targets table -------------------------------------------------------
    cursor.execute("""
        CREATE TABLE IF NOT EXISTS targets (
            id                 INTEGER PRIMARY KEY AUTOINCREMENT,
            metric_name        TEXT    UNIQUE NOT NULL,
            target_value       REAL    NOT NULL,
            threshold_warning  REAL,
            threshold_critical REAL,
            description        TEXT
        )
    """)

    # Seed default targets once
    cursor.execute("SELECT COUNT(*) FROM targets")
    if cursor.fetchone()[0] == 0:
        default_targets = [
            ("aqi",                50.0,  100.0, 150.0, "Air Quality Index — lower is better"),
            ("water_usage",      1000.0, 1200.0, 1500.0, "Water usage in litres"),
            ("energy_consumption", 500.0,  600.0,  750.0, "Energy consumption in kWh"),
            ("waste_generated",    100.0,  150.0,  200.0, "Waste generated in kg"),
            ("co2_emissions",      200.0,  250.0,  300.0, "CO₂ emissions in kg"),
        ]
        cursor.executemany(
            """INSERT INTO targets
               (metric_name, target_value, threshold_warning, threshold_critical, description)
               VALUES (?, ?, ?, ?, ?)""",
            default_targets,
        )
        logger.info("Default KPI targets inserted.")

    conn.commit()
    conn.close()
    logger.info("Database initialised at: %s", DB_PATH)


# ---------------------------------------------------------------------------
# Data operations
# ---------------------------------------------------------------------------

def save_data(df: pd.DataFrame) -> int:
    """
    Insert rows from *df* into environmental_data, skipping dates that already
    exist (upsert-ignore strategy to prevent duplicate entries on re-upload).

    Args:
        df: DataFrame with columns: date, aqi, water_usage,
            energy_consumption, waste_generated, co2_emissions

    Returns:
        Number of rows actually inserted.
    """
    if df.empty:
        return 0

    # Normalise date to string YYYY-MM-DD
    df = df.copy()
    df["date"] = pd.to_datetime(df["date"]).dt.strftime("%Y-%m-%d")

    conn = get_connection()
    cursor = conn.cursor()

    # Fetch existing dates once for efficient deduplication that works on both
    # old schemas (no UNIQUE constraint) and new schemas (with UNIQUE).
    cursor.execute("SELECT date FROM environmental_data")
    existing_dates: set[str] = {r[0] for r in cursor.fetchall()}

    columns = ["date", "aqi", "water_usage", "energy_consumption",
                "waste_generated", "co2_emissions"]
    inserted = 0

    for _, row in df[columns].iterrows():
        date_str = row["date"]
        if date_str in existing_dates:
            continue  # Skip duplicate
        try:
            cursor.execute(
                """INSERT INTO environmental_data
                   (date, aqi, water_usage, energy_consumption, waste_generated, co2_emissions)
                   VALUES (?, ?, ?, ?, ?, ?)""",
                (date_str, row["aqi"], row["water_usage"],
                 row["energy_consumption"], row["waste_generated"], row["co2_emissions"]),
            )
            existing_dates.add(date_str)
            inserted += 1
        except sqlite3.Error as exc:
            logger.warning("Skipped row %s: %s", date_str, exc)

    conn.commit()
    conn.close()
    logger.info("Inserted %d new records into environmental_data.", inserted)
    return inserted


def fetch_data() -> pd.DataFrame:
    """
    Fetch all environmental data ordered by date ascending.

    Returns:
        DataFrame with columns: id, date, aqi, water_usage,
        energy_consumption, waste_generated, co2_emissions
    """
    conn = get_connection()
    df = pd.read_sql_query(
        """SELECT id, date, aqi, water_usage, energy_consumption,
                  waste_generated, co2_emissions
           FROM environmental_data
           ORDER BY date ASC""",
        conn,
    )
    conn.close()

    if not df.empty:
        df["date"] = pd.to_datetime(df["date"])

    return df


def fetch_targets() -> pd.DataFrame:
    """
    Fetch all KPI target definitions.

    Returns:
        DataFrame with target rows.
    """
    conn = get_connection()
    df = pd.read_sql_query("SELECT * FROM targets", conn)
    conn.close()
    return df


def update_target(
    metric_name: str,
    target_value: float,
    threshold_warning: float,
    threshold_critical: float,
) -> bool:
    """
    Update target thresholds for a metric.

    Args:
        metric_name:        Metric identifier (e.g. 'aqi').
        target_value:       Desired target value.
        threshold_warning:  Value at which a warning is raised.
        threshold_critical: Value at which a critical alert is raised.

    Returns:
        True if the row was updated, False if metric not found.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute(
        """UPDATE targets
           SET target_value = ?, threshold_warning = ?, threshold_critical = ?
           WHERE metric_name = ?""",
        (target_value, threshold_warning, threshold_critical, metric_name),
    )
    updated = cursor.rowcount > 0
    conn.commit()
    conn.close()
    if updated:
        logger.info("Target updated: %s", metric_name)
    else:
        logger.warning("update_target: metric '%s' not found.", metric_name)
    return updated


def clear_all_data() -> None:
    """Delete all rows from environmental_data (useful for testing / reset)."""
    conn = get_connection()
    conn.execute("DELETE FROM environmental_data")
    conn.commit()
    conn.close()
    logger.info("All environmental data cleared.")


def get_data_summary() -> dict:
    """
    Return high-level statistics about the stored data.

    Returns:
        Dict with keys: total_records, date_from, date_to.
    """
    conn = get_connection()
    cursor = conn.cursor()
    cursor.execute("SELECT COUNT(*) FROM environmental_data")
    total = cursor.fetchone()[0]
    cursor.execute("SELECT MIN(date), MAX(date) FROM environmental_data")
    date_row = cursor.fetchone()
    conn.close()

    return {
        "total_records": total,
        "date_from": date_row[0] if date_row[0] else "N/A",
        "date_to":   date_row[1] if date_row[1] else "N/A",
    }


# ---------------------------------------------------------------------------
# Standalone entry point
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    init_db()
    summary = get_data_summary()
    print(f"Database ready — {summary['total_records']} records "
          f"({summary['date_from']} → {summary['date_to']})")
