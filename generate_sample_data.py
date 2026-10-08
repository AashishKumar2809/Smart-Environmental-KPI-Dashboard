"""
Generate Sample Environmental Data
===================================
Generates synthetic environmental KPI data for the last N days.
Can be called from app.py (returns DataFrame) or run standalone
(also writes a CSV file to data/).

Usage (standalone):
    python generate_sample_data.py
"""

import os
from datetime import datetime, timedelta

import numpy as np
import pandas as pd


def generate_sample_data(days: int = 90, output_file: str | None = "data/sample_environment_data.csv") -> pd.DataFrame:
    """
    Generate *days* rows of synthetic environmental data with a realistic
    upward trend and random daily noise.

    Args:
        days:        Number of days of history to generate (default 90).
        output_file: Path to write a CSV file, or ``None`` to skip file I/O.

    Returns:
        DataFrame with columns: date, aqi, water_usage, energy_consumption,
        waste_generated, co2_emissions.
    """
    np.random.seed(42)  # Reproducible sample data

    end_date   = datetime.now().date()
    start_date = end_date - timedelta(days=days - 1)
    dates      = pd.date_range(start=start_date, end=end_date, freq="D")

    # Base values (realistic starting point, all below default targets)
    base = {
        "aqi":                45.0,
        "water_usage":        980.0,
        "energy_consumption": 485.0,
        "waste_generated":     92.0,
        "co2_emissions":      178.0,
    }

    rows = []
    for i, date in enumerate(dates):
        # Gradual upward trend (60 % increase over the full period)
        trend = 1.0 + (i / days) * 0.6
        # Daily noise ±5 %
        noise = np.random.uniform(0.95, 1.05)

        rows.append({
            "date":               date.strftime("%Y-%m-%d"),
            "aqi":                round(base["aqi"]                * trend * noise, 1),
            "water_usage":        round(base["water_usage"]        * trend * noise, 1),
            "energy_consumption": round(base["energy_consumption"] * trend * noise, 1),
            "waste_generated":    round(base["waste_generated"]    * trend * noise, 1),
            "co2_emissions":      round(base["co2_emissions"]      * trend * noise, 1),
        })

    df = pd.DataFrame(rows)

    if output_file:
        os.makedirs(os.path.dirname(output_file) or ".", exist_ok=True)
        df.to_csv(output_file, index=False)
        print(f"✅ {len(df)} records written to {output_file}")
        print(f"   Date range: {df['date'].min()} → {df['date'].max()}")

    return df


if __name__ == "__main__":
    df = generate_sample_data(days=90)
    latest = df.iloc[-1]
    print("\n📊 Latest values:")
    print(f"  AQI:    {latest['aqi']}")
    print(f"  Water:  {latest['water_usage']} L")
    print(f"  Energy: {latest['energy_consumption']} kWh")
    print(f"  Waste:  {latest['waste_generated']} kg")
    print(f"  CO₂:    {latest['co2_emissions']} kg")
