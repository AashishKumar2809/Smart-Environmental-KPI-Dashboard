# 🌍 Smart Environmental KPI Dashboard

> A production-ready, ML-powered environmental monitoring dashboard built with Python and Streamlit.  
> Tracks Air Quality, Water Usage, Energy Consumption, Waste Generation, and CO₂ Emissions — with real-time KPI status, trend charts, anomaly detection, and next-month forecasting.

[![Open in Streamlit](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://streamlit.io/cloud)

---

## 📋 Table of Contents

- [Overview](#overview)
- [Features](#features)
- [Project Structure](#project-structure)
- [Tech Stack](#tech-stack)
- [Local Setup](#local-setup)
- [Streamlit Cloud Deployment](#streamlit-cloud-deployment)
- [CSV Data Format](#csv-data-format)
- [Database Schema](#database-schema)
- [Machine Learning](#machine-learning)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Smart Environmental KPI Dashboard** is designed to support data-driven sustainability governance by:

- **Automating KPI computation** from raw daily measurements
- **Colour-coded status cards** (OK / Warning / Critical) based on configurable thresholds
- **Interactive trend charts** with target and threshold reference lines
- **ML-based forecasting** (next-month prediction with confidence score)
- **Anomaly & spike detection** using statistical Z-score analysis
- **CSV import** with full validation pipeline, deduplication, and range checks
- **Auto-seeding** — the dashboard loads 90 days of sample data automatically on first run (great for demos and Streamlit Cloud)

---

## ✨ Features

### Dashboard
| Feature | Details |
|---|---|
| **5 KPI Cards** | AQI, Water Usage, Energy, Waste, CO₂ — each with status badge & MoM % change |
| **Active Alerts** | Critical / Warning threshold breaches and sudden spikes shown prominently |
| **Actual vs Target Bar Chart** | Side-by-side grouped bar chart for all metrics |
| **Trend Charts** | Per-metric line charts with target, warning, and critical reference lines |
| **Statistical Summary** | Mean, Median, Min, Max for each metric |
| **ML Predictions** | Next-month value with `High / Medium / Low` confidence and R² score |
| **Anomaly Detection** | Z-score anomaly table with expandable drill-down per metric |
| **Live Mode** | Optional 30-second auto-refresh via HTTP meta-refresh |
| **Weather Widget** | Current weather line from wttr.in (cached 1 hour) |

### Data Management
- Upload and validate CSV files
- Preview data before inserting
- Auto-deduplication (re-uploading the same file won't create duplicates)
- Download all stored data as CSV
- One-click database reset with fresh 90-day sample data

### Settings
- Real-time KPI target and threshold configuration
- Changes propagate to the Dashboard immediately

---

## 📁 Project Structure

```
kpi-dashboard/
│
├── app.py                       # Main Streamlit application (entry point)
├── database.py                  # SQLite operations — CRUD, init, deduplication
├── kpi_engine.py                # KPI calculations, trend analysis, status logic
├── ml_model.py                  # Linear regression forecasting & anomaly detection
├── data_loader.py               # CSV validation and upload pipeline
├── generate_sample_data.py      # 90-day synthetic data generator
│
├── requirements.txt             # Pinned Python dependencies
├── README.md                    # This file
│
├── .streamlit/
│   └── config.toml              # Streamlit theme and server configuration
│
└── data/
    └── sample_environment_data.csv   # Generated on first standalone run
```

---

## 🛠️ Tech Stack

| Layer | Technology |
|---|---|
| Language | Python 3.10+ |
| Web framework | Streamlit 1.40 |
| Database | SQLite (via `sqlite3` stdlib) |
| Data processing | Pandas 2.2, NumPy 1.26 |
| Visualisation | Plotly 5.24 |
| Machine Learning | scikit-learn 1.5 |

---

## 🚀 Local Setup

### Prerequisites

- Python **3.10** or higher
- `pip` (bundled with Python)

### Step-by-step

```bash
# 1. Navigate to the project folder
cd "KPI dashboard"

# 2. Create and activate a virtual environment
python -m venv .venv

# Windows
.venv\Scripts\activate

# macOS / Linux
source .venv/bin/activate

# 3. Install dependencies
pip install -r requirements.txt

# 4. Run the app
streamlit run app.py
```

The app will open automatically at **http://localhost:8501**.  
On first launch it auto-loads 90 days of sample data — no manual seeding required.

---

## ☁️ Streamlit Cloud Deployment

1. Push the project to a **public GitHub repository**.
2. Go to [share.streamlit.io](https://share.streamlit.io/) → **New app**.
3. Select your repository, branch (`main`), and set the **Main file path** to `app.py`.
4. Click **Deploy**.

> **Note on data persistence**: Streamlit Cloud uses an ephemeral filesystem — the SQLite database is reset on each new deployment. The app handles this gracefully by auto-seeding 90 days of sample data on startup. For persistent storage across deployments, consider replacing the SQLite backend with a hosted database (e.g., Supabase, Turso, or PlanetScale).

### Required files for cloud

All files below must be present in your repository root:

```
app.py
database.py
kpi_engine.py
ml_model.py
data_loader.py
generate_sample_data.py
requirements.txt
.streamlit/config.toml
```

---

## 📄 CSV Data Format

Your upload file must contain exactly these columns:

```csv
date,aqi,water_usage,energy_consumption,waste_generated,co2_emissions
2026-01-01,45.2,980.5,485.3,92.1,178.4
2026-01-02,48.7,1005.2,492.8,95.3,182.6
```

| Column | Type | Unit | Valid range |
|---|---|---|---|
| `date` | Date | YYYY-MM-DD | Any valid date |
| `aqi` | Float | index (0–500) | 0 – 500 |
| `water_usage` | Float | Litres | 0 – 10 000 |
| `energy_consumption` | Float | kWh | 0 – 5 000 |
| `waste_generated` | Float | kg | 0 – 1 000 |
| `co2_emissions` | Float | kg | 0 – 1 000 |

A pre-filled template is available for download from the **Data Management** page.

---

## 🗄️ Database Schema

### `environmental_data`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key, auto-increment |
| `date` | TEXT | `YYYY-MM-DD`, **unique** — prevents duplicates |
| `aqi` | REAL | Air Quality Index |
| `water_usage` | REAL | Litres |
| `energy_consumption` | REAL | kWh |
| `waste_generated` | REAL | kg |
| `co2_emissions` | REAL | kg |
| `created_at` | TIMESTAMP | Auto-set by SQLite |

### `targets`

| Column | Type | Notes |
|---|---|---|
| `id` | INTEGER | Primary key |
| `metric_name` | TEXT | Unique identifier |
| `target_value` | REAL | Desired value to achieve |
| `threshold_warning` | REAL | Above this → Warning status |
| `threshold_critical` | REAL | Above this → Critical status |
| `description` | TEXT | Human-readable label |

### Default targets

| Metric | Target | Warning | Critical |
|---|---|---|---|
| AQI | 50 | 100 | 150 |
| Water Usage | 1 000 L | 1 200 L | 1 500 L |
| Energy Consumption | 500 kWh | 600 kWh | 750 kWh |
| Waste Generated | 100 kg | 150 kg | 200 kg |
| CO₂ Emissions | 200 kg | 250 kg | 300 kg |

---

## 🤖 Machine Learning

### Forecasting — Linear Regression

Each metric is modelled independently using a **scikit-learn `LinearRegression`** trained on the full historical time series. The date is converted to an ordinal integer (days since a fixed epoch) and used as the single feature.

**Confidence levels** are derived from the in-sample R² score:

| R² | Confidence |
|---|---|
| > 0.70 | High |
| 0.40 – 0.70 | Medium |
| < 0.40 | Low |

The predicted value for 30 days ahead is shown alongside the confidence level and raw R² on the Dashboard.

### Anomaly Detection — Z-Score

Statistical outliers are flagged using the **Z-score method**:

```
z = (x − μ) / σ
```

Points where `|z| > 2.5` are labelled anomalies. Results are displayed in an expandable table per metric on the Dashboard.

### Spike Detection

Day-over-day ratio spikes are flagged when:

```
current_value / previous_value ≥ 1.3   (i.e., ≥ 30% increase)
```

The three most recent spikes per metric appear in the **Active Alerts** section.

---

## 🔧 Troubleshooting

| Problem | Solution |
|---|---|
| `ModuleNotFoundError` | Run `pip install -r requirements.txt` inside your virtual environment |
| Dashboard shows no data | Go to **Data Management → Reset & Reload Sample Data** |
| CSV upload rejected | Check column names match exactly (see [CSV format](#csv-data-format)) |
| Duplicate date warning | Re-uploading the same file is safe — duplicates are silently skipped |
| Weather shows "unavailable" | The wttr.in API is optional; the rest of the dashboard is unaffected |
| `FutureWarning: resample 'M'` | Update to the latest code — this was fixed in `kpi_engine.py` (`resample('ME')`) |

---

## 🔮 Future Enhancements

- [ ] ARIMA / Prophet forecasting for non-linear trends
- [ ] PDF report export
- [ ] Email / SMS threshold alerts
- [ ] Multi-site / multi-organisation support
- [ ] Real-time IoT sensor data ingestion
- [ ] User authentication

---

## 👤 Author

Built as a portfolio project demonstrating real-world Python engineering:
**ETL pipeline · SQLite DB design · ML forecasting · Streamlit dashboards · Streamlit Cloud deployment.**

---

*Happy monitoring! 🌍♻️*
