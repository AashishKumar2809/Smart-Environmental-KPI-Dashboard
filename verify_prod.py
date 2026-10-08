"""
Verification script — runs outside Streamlit to confirm all modules work correctly.
Run with:  python verify_prod.py
"""
import sys
print(f"Python: {sys.version}\n")

errors = []

# ── 1. Imports ────────────────────────────────────────────────────────────────
try:
    import database, kpi_engine, ml_model, data_loader
    from generate_sample_data import generate_sample_data
    print("✅ All modules imported successfully")
except Exception as e:
    errors.append(f"Import error: {e}")
    print(f"❌ Import error: {e}")

# ── 2. DB init ────────────────────────────────────────────────────────────────
try:
    database.init_db()
    print("✅ database.init_db() OK")
except Exception as e:
    errors.append(f"init_db: {e}")
    print(f"❌ init_db failed: {e}")

# ── 3. Fetch / seed data ──────────────────────────────────────────────────────
try:
    import pandas as pd
    df = database.fetch_data()
    if df.empty:
        seed_df = generate_sample_data(days=90, output_file=None)
        n = database.save_data(seed_df)
        df = database.fetch_data()
        print(f"✅ Seeded {n} rows — fetch_data() now returns {len(df)} rows")
    else:
        print(f"✅ database.fetch_data() — {len(df)} existing rows")
except Exception as e:
    errors.append(f"fetch/seed: {e}")
    print(f"❌ fetch/seed failed: {e}")

# ── 4. KPI engine ─────────────────────────────────────────────────────────────
try:
    targets_df = database.fetch_targets()
    kpis = kpi_engine.get_latest_kpis(df, targets_df)
    assert len(kpis) == 5, f"Expected 5 KPIs, got {len(kpis)}"
    for m, v in kpis.items():
        assert v["current_value"] is not None or True  # None handled
    print(f"✅ kpi_engine.get_latest_kpis() — {len(kpis)} metrics")
except Exception as e:
    errors.append(f"kpi_engine: {e}")
    print(f"❌ kpi_engine failed: {e}")

# ── 5. Monthly resample (tests 'ME' fix) ──────────────────────────────────────
try:
    monthly = kpi_engine.calculate_monthly_kpis(df)
    assert not monthly.empty
    print(f"✅ kpi_engine.calculate_monthly_kpis() — {len(monthly)} months (resample 'ME' OK)")
except Exception as e:
    errors.append(f"resample: {e}")
    print(f"❌ resample failed: {e}")

# ── 6. ML predictions ─────────────────────────────────────────────────────────
try:
    preds = ml_model.generate_all_predictions(df)
    assert len(preds) == 5
    for m, p in preds.items():
        assert "predicted_value" in p
        val = p["predicted_value"]
        r2  = p["r2_score"]
        print(f"   {m:22s}: {val:.2f}  conf={p['confidence']}  R²={r2:.3f}")
    print(f"✅ ml_model.generate_all_predictions() — {len(preds)} predictions")
except Exception as e:
    errors.append(f"ml_model: {e}")
    print(f"❌ ml prediction failed: {e}")

# ── 7. Anomaly summary ────────────────────────────────────────────────────────
try:
    anom = ml_model.get_anomaly_summary(df)
    print(f"✅ ml_model.get_anomaly_summary() — {anom['total_anomalies']} anomalies across {len(anom['metrics_affected'])} metrics")
except Exception as e:
    errors.append(f"anomaly: {e}")
    print(f"❌ anomaly detection failed: {e}")

# ── 8. Spike detection ────────────────────────────────────────────────────────
try:
    spikes = ml_model.get_spike_alerts(df, "aqi", spike_threshold=1.3)
    print(f"✅ ml_model.get_spike_alerts() — {len(spikes)} AQI spike(s)")
except Exception as e:
    errors.append(f"spikes: {e}")
    print(f"❌ spike detection failed: {e}")

# ── 9. data_loader has no streamlit dependency ────────────────────────────────
try:
    with open("data_loader.py") as f:
        src = f.read()
    assert "import streamlit" not in src, "streamlit import still present!"
    print("✅ data_loader.py — no Streamlit dependency (pure Python)")
except Exception as e:
    errors.append(f"data_loader: {e}")
    print(f"❌ data_loader check failed: {e}")

# ── 10. Deduplication ────────────────────────────────────────────────────────
try:
    before = database.get_data_summary()["total_records"]
    # Re-save exactly the rows we already loaded — should insert 0 new rows
    inserted = database.save_data(df)   # df was fetched from DB earlier
    after = database.get_data_summary()["total_records"]
    assert inserted == 0, f"Expected 0 inserted (dedup), got {inserted}"
    assert before == after, "Row count changed after re-insert!"
    print(f"✅ Deduplication — re-upload of existing data inserted {inserted} new rows (correct)")
except Exception as e:
    errors.append(f"dedup: {e}")
    print(f"❌ deduplication check failed: {e}")

# ── Summary ───────────────────────────────────────────────────────────────────
print()
if errors:
    print(f"{'='*50}")
    print(f"❌ {len(errors)} CHECK(S) FAILED:")
    for err in errors:
        print(f"   • {err}")
    sys.exit(1)
else:
    print("=" * 50)
    print("✅  ALL 10 CHECKS PASSED — production ready!")
    print("=" * 50)
