"""
db/load_parquet.py
==================
One-time migration: loads all parquet/CSV data into Supabase Postgres.

Load order (respects foreign key constraints):
  1. markets      ← polymarket_markets_meta.csv
  2. snapshots    ← polymarket_ml_dataset_clean.parquet
  3. trends       ← category_trends_features.parquet
  4. model_runs   ← one row per model variant (created here)
  5. predictions  ← models/*/predictions/*.csv (where snapshot_timestamp exists)

Idempotent: safe to re-run; uses ON CONFLICT DO NOTHING throughout.
SVM predictions are skipped — their CSVs lack snapshot_timestamp.

Usage:
    python db/load_parquet.py
    python db/load_parquet.py --skip-snapshots   # re-run just predictions
    python db/load_parquet.py --skip-predictions
"""

import argparse
import os
import sys
import time
from pathlib import Path

import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
MODELS    = ROOT / "models"

load_dotenv(ROOT / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Copy .env.example → .env and fill it in.")
    sys.exit(1)

SEP = "─" * 60

# Columns the snapshots table accepts (excludes outcome/split/category/question
# which live on the markets table).
SNAPSHOT_COLS = [
    "market_id", "snapshot_timestamp",
    "days_before_close", "pct_lifetime_elapsed", "duration_days",
    "price_at_snapshot", "price_deviation_from_half",
    "total_volume", "log_volume",
    "price_mean_7d",  "price_volatility_7d",  "price_min_7d",  "price_max_7d",
    "price_change_7d",  "price_range_7d",  "price_trend_7d",
    "price_mean_14d", "price_volatility_14d", "price_min_14d", "price_max_14d",
    "price_change_14d", "price_range_14d", "price_trend_14d",
]


# ── helpers ───────────────────────────────────────────────────────────────────

def nan_to_none(df: pd.DataFrame) -> pd.DataFrame:
    """Replace NaN/NaT with None so psycopg2 writes SQL NULL."""
    return df.where(pd.notnull(df), other=None)


def bulk_insert(conn, table: str, df: pd.DataFrame,
                chunk_size: int = 5_000, conflict: str = "DO NOTHING") -> int:
    """Bulk-insert a DataFrame via execute_values. Returns row count."""
    if df.empty:
        return 0
    df = nan_to_none(df)
    cols    = list(df.columns)
    col_str = ", ".join(f'"{c}"' for c in cols)
    query   = f"INSERT INTO {table} ({col_str}) VALUES %s ON CONFLICT {conflict}"
    data    = [tuple(row) for row in df.itertuples(index=False, name=None)]
    total   = 0
    with conn.cursor() as cur:
        for i in range(0, len(data), chunk_size):
            chunk = data[i : i + chunk_size]
            psycopg2.extras.execute_values(cur, query, chunk, page_size=chunk_size)
            total += len(chunk)
    conn.commit()
    return total


def ts_key(ts) -> int:
    """Convert a timestamp to integer seconds-since-epoch for dict lookup."""
    if hasattr(ts, "timestamp"):
        return int(ts.timestamp())
    return int(pd.Timestamp(ts, tz="UTC").timestamp())


# ── loaders ───────────────────────────────────────────────────────────────────

def load_markets(conn) -> int:
    print(f"\n{SEP}\n  markets\n{SEP}")
    csv = DATA_DIR / "polymarket_markets_meta.csv"

    if csv.exists():
        df = pd.read_csv(csv)
        df = df[[
            "market_id", "clob_token_id", "question", "category",
            "start_date", "end_date", "duration_days", "total_volume",
            "yes_final_price", "outcome", "split",
        ]]
        df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
        df["end_date"]   = pd.to_datetime(df["end_date"],   utc=True, errors="coerce")
        print(f"  Source: {csv.name}")
    else:
        # CSV is gitignored — derive unique market rows from the snapshot parquet.
        # start_date, end_date, clob_token_id, yes_final_price will be NULL;
        # run fetch_markets.py and re-run this script to fill them in.
        print(f"  {csv.name} not found — deriving from snapshot parquet (date fields will be NULL)")
        pq = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
        snap = pd.read_parquet(pq, columns=[
            "market_id", "split", "category", "question",
            "outcome", "total_volume", "duration_days",
        ])
        df = snap.drop_duplicates(subset=["market_id"]).reset_index(drop=True)
        df = df[["market_id", "question", "category", "outcome",
                 "split", "total_volume", "duration_days"]]

    print(f"  Rows: {len(df):,}")
    n = bulk_insert(conn, "markets", df, conflict="(market_id) DO NOTHING")
    print(f"  Inserted: {n:,}")
    return n


def load_snapshots(conn) -> int:
    print(f"\n{SEP}\n  snapshots  (1.4M rows — takes a few minutes)\n{SEP}")
    pq = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
    if not pq.exists():
        print(f"  SKIP: {pq.name} not found")
        return 0

    print(f"  Reading {pq.name} ...")
    df = pd.read_parquet(pq, columns=SNAPSHOT_COLS)
    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], utc=True)
    df = nan_to_none(df)
    print(f"  Rows: {len(df):,}")

    # Check how many are already in — ON CONFLICT DO NOTHING is idempotent,
    # but we skip already-loaded markets to avoid re-inserting 1M rows.
    with conn.cursor() as cur:
        cur.execute("SELECT DISTINCT market_id FROM snapshots")
        done_markets = {r[0] for r in cur.fetchall()}

    if done_markets:
        before = len(df)
        df = df[~df["market_id"].isin(done_markets)].reset_index(drop=True)
        print(f"  Resuming — skipping {len(done_markets):,} already-loaded markets "
              f"({before - len(df):,} rows skipped, {len(df):,} remaining)")

    if df.empty:
        print(f"  All snapshots already loaded.")
        return 0

    cols    = list(df.columns)
    col_str = ", ".join(f'"{c}"' for c in cols)
    query   = f'INSERT INTO snapshots ({col_str}) VALUES %s ON CONFLICT DO NOTHING'
    data    = [tuple(r) for r in df.itertuples(index=False, name=None)]
    CHUNK        = 10_000
    COMMIT_EVERY = 5        # commit every 50k rows so progress survives a dropped connection
    total        = 0
    t0           = time.time()

    with conn.cursor() as cur:
        for i in range(0, len(data), CHUNK):
            chunk = data[i : i + CHUNK]
            psycopg2.extras.execute_values(cur, query, chunk, page_size=CHUNK)
            total += len(chunk)
            batch_num = i // CHUNK
            if batch_num % COMMIT_EVERY == 0:
                conn.commit()
            if batch_num % 100 == 0:
                elapsed = time.time() - t0
                pct     = i / len(data) * 100
                rate    = total / elapsed if elapsed > 0 else 0
                eta     = (len(data) - total) / rate if rate > 0 else 0
                print(f"  [{pct:5.1f}%]  {total:>9,} rows  "
                      f"{elapsed:>5.0f}s elapsed  ETA ~{eta:.0f}s")
    conn.commit()
    print(f"  Done: {total:,} rows in {time.time()-t0:.0f}s")
    return total


def load_trends(conn) -> int:
    print(f"\n{SEP}\n  trends\n{SEP}")
    pq = DATA_DIR / "category_trends_features.parquet"
    if not pq.exists():
        print(f"  SKIP: {pq.name} not found")
        return 0

    df = pd.read_parquet(pq)
    df = df[["category", "week_start", "trend_value", "trend_ma4",
             "trend_change_4w", "trend_spike"]]
    df["week_start"] = pd.to_datetime(df["week_start"]).dt.date
    print(f"  Rows: {len(df):,}")

    n = bulk_insert(conn, "trends", df,
                    conflict="(category, week_start) DO NOTHING")
    print(f"  Inserted: {n:,}")
    return n


def build_snapshot_index(conn) -> dict:
    """Return {(market_id, unix_ts_int): snapshot_id} for all snapshots."""
    print(f"\n  Building snapshot index ...")
    with conn.cursor() as cur:
        cur.execute("SELECT id, market_id, snapshot_timestamp FROM snapshots")
        rows = cur.fetchall()
    idx = {(r[1], ts_key(r[2])): r[0] for r in rows}
    print(f"  Index: {len(idx):,} entries")
    return idx


def upsert_model_run(conn, model_name: str, dataset_tag: str, notes: str) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_runs (model_name, dataset_tag, notes)
               VALUES (%s, %s, %s)
               ON CONFLICT (model_name, dataset_tag) DO NOTHING
               RETURNING id""",
            (model_name, dataset_tag, notes),
        )
        row = cur.fetchone()
        if row is None:
            cur.execute(
                "SELECT id FROM model_runs WHERE model_name = %s AND dataset_tag = %s",
                (model_name, dataset_tag),
            )
            row = cur.fetchone()
    conn.commit()
    return row[0]


def load_predictions_from_csv(conn, snap_idx: dict,
                               csv_path: Path, model_name: str,
                               dataset_tag: str, prob_col: str,
                               raw_col: str | None, label_col: str | None) -> int:
    if not csv_path.exists():
        print(f"  SKIP {model_name:<45} {csv_path.name} not found")
        return 0

    df = pd.read_csv(csv_path)
    if prob_col not in df.columns:
        print(f"  SKIP {model_name:<45} column '{prob_col}' not in CSV")
        return 0

    model_run_id = upsert_model_run(
        conn, model_name, dataset_tag, f"Migrated from {csv_path.name}"
    )

    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], utc=True)
    df["_ts_key"] = df["snapshot_timestamp"].apply(ts_key)
    df["snapshot_id"] = df.apply(
        lambda r: snap_idx.get((r["market_id"], r["_ts_key"])), axis=1
    )

    missing = df["snapshot_id"].isna().sum()
    if missing:
        print(f"  WARN {model_name}: {missing:,} rows had no matching snapshot — dropped")
    df = df.dropna(subset=["snapshot_id"])
    df["snapshot_id"] = df["snapshot_id"].astype(int)

    pred_df = pd.DataFrame({
        "model_run_id": model_run_id,
        "snapshot_id":  df["snapshot_id"],
        "pred_prob":    df[prob_col],
        "pred_prob_raw": df[raw_col]   if raw_col   and raw_col   in df.columns else None,
        "pred_label":   df[label_col] if label_col and label_col in df.columns else None,
    })

    n = bulk_insert(conn, "predictions", pred_df,
                    conflict="(model_run_id, snapshot_id) DO NOTHING")
    print(f"  {model_name:<45} run_id={model_run_id:<4} rows={n:,}")
    return n


def load_predictions(conn, snap_idx: dict) -> int:
    print(f"\n{SEP}\n  predictions\n{SEP}")

    LR  = MODELS / "logistic_regression/predictions/predictions.csv"
    GB  = MODELS / "gradient_boosting/predictions/predictions.csv"
    RF  = MODELS / "random_forest/predictions/test_predictions.csv"
    RFT = MODELS / "random_forest_trends/predictions/trends_test_predictions.csv"

    sources = [
        # (csv,  model_name,                          dataset_tag,       prob_col,                raw_col, label_col)
        (LR,  "logistic_regression_base",             "clean_v1",        "pred_prob_base",         None,   "pred_label_base"),
        (LR,  "logistic_regression_trends",           "clean_v1_trends", "pred_prob_trends",       None,   "pred_label_trends"),
        (GB,  "xgboost_base",                         "clean_v1",        "pred_prob_base",         None,   "pred_label_base"),
        (GB,  "xgboost_trends",                       "clean_v1_trends", "pred_prob_trends",       None,   "pred_label_trends"),
        (RF,  "random_forest_price_only",             "clean_v1",        "proba_price_only",       None,   None),
        (RF,  "random_forest_full",                   "clean_v1",        "proba_full",             None,   None),
        (RF,  "random_forest_full_calibrated",        "clean_v1",        "proba_full_calibrated",  None,   None),
        (RFT, "random_forest_trends_full",            "clean_v1_trends", "proba_full",             None,   None),
        (RFT, "random_forest_trends_calibrated",      "clean_v1_trends", "proba_trends_calibrated",None,   None),
    ]

    total = 0
    for csv, model_name, dataset_tag, prob_col, raw_col, label_col in sources:
        total += load_predictions_from_csv(
            conn, snap_idx, csv, model_name, dataset_tag, prob_col, raw_col, label_col
        )

    print(f"\n  NOTE: SVM predictions skipped — CSVs use target_percentile, not")
    print(f"        snapshot_timestamp. Reload SVM by rerunning svm.py after")
    print(f"        updating it to write directly to the database.")
    return total


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Load parquet data into Supabase")
    parser.add_argument("--skip-snapshots",   action="store_true")
    parser.add_argument("--skip-predictions", action="store_true")
    args = parser.parse_args()

    print(f"\n{'█'*60}")
    print(f"  POLYMARKET — PARQUET → SUPABASE")
    print(f"{'█'*60}")

    conn = psycopg2.connect(
        DATABASE_URL,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    print(f"\nConnected.")

    load_markets(conn)

    if not args.skip_snapshots:
        load_snapshots(conn)
    else:
        print(f"\n── snapshots ── SKIPPED")

    load_trends(conn)

    if not args.skip_predictions:
        snap_idx = build_snapshot_index(conn)
        load_predictions(conn, snap_idx)
    else:
        print(f"\n── predictions ── SKIPPED")

    # Final counts
    print(f"\n{SEP}\n  Row counts\n{SEP}")
    with conn.cursor() as cur:
        for table in ["markets", "snapshots", "trends", "model_runs", "predictions"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  {table:<15} {cur.fetchone()[0]:>12,}")

    conn.close()
    print(f"\n{'='*60}\n  DONE\n{'='*60}\n")


if __name__ == "__main__":
    main()
