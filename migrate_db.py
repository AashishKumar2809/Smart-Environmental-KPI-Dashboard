"""
One-time DB migration: adds UNIQUE constraint on date by rebuilding the table.
Safe to run multiple times (checks if migration is needed first).
Run with:  python migrate_db.py
"""
import sqlite3
import os

DB_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "environmental_kpi.db")

def migrate():
    conn = sqlite3.connect(DB_PATH)
    cur = conn.cursor()

    # Check current schema
    cur.execute("SELECT sql FROM sqlite_master WHERE type='table' AND name='environmental_data'")
    row = cur.fetchone()
    if row is None:
        print("Table does not exist yet — no migration needed.")
        conn.close()
        return

    current_sql = row[0].upper()
    if "UNIQUE" in current_sql:
        print("Schema already has UNIQUE constraint — no migration needed.")
        conn.close()
        return

    print("Migrating: adding UNIQUE constraint on date column...")

    # SQLite cannot ALTER TABLE to add UNIQUE — must rebuild
    cur.executescript("""
        BEGIN;

        -- Step 1: create new table with UNIQUE on date
        CREATE TABLE IF NOT EXISTS environmental_data_new (
            id                   INTEGER PRIMARY KEY AUTOINCREMENT,
            date                 TEXT    NOT NULL UNIQUE,
            aqi                  REAL,
            water_usage          REAL,
            energy_consumption   REAL,
            waste_generated      REAL,
            co2_emissions        REAL,
            created_at           TIMESTAMP DEFAULT CURRENT_TIMESTAMP
        );

        -- Step 2: copy data (dedup: keep newest row per date via MAX(id))
        INSERT OR IGNORE INTO environmental_data_new
            (id, date, aqi, water_usage, energy_consumption, waste_generated, co2_emissions, created_at)
        SELECT id, date, aqi, water_usage, energy_consumption, waste_generated, co2_emissions, created_at
        FROM environmental_data
        WHERE id IN (
            SELECT MAX(id) FROM environmental_data GROUP BY date
        );

        -- Step 3: swap tables
        DROP TABLE environmental_data;
        ALTER TABLE environmental_data_new RENAME TO environmental_data;

        COMMIT;
    """)

    cur.execute("SELECT COUNT(*) FROM environmental_data")
    count = cur.fetchone()[0]
    conn.close()
    print(f"Migration complete — {count} rows retained.")

if __name__ == "__main__":
    migrate()
