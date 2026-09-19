"""
db/load_parquet.py
==================
Operational-layer migration: loads metadata and model registry into Supabase.

Loads (small — completes in seconds):
  1. markets      ← polymarket_markets_meta.csv  (or snapshot parquet fallback)
  2. trends       ← category_trends_features.parquet
  3. model_runs   ← metrics computed from models/*/predictions/*.csv

Does NOT load:
  - snapshots   (1.4M historical rows — lives in parquet, read directly by training scripts)
  - predictions (per-snapshot rows — DB will hold live market predictions only)

The DB is the operational/live layer; parquet files are the training data store.
See ADR-011 for the offline/online split decision.

Before running, apply the pending migration:
    db/migrations/004_add_model_run_metrics.sql

Usage:
    python db/load_parquet.py
    python db/load_parquet.py --skip-markets
    python db/load_parquet.py --skip-trends
    python db/load_parquet.py --skip-model-runs
"""

import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import psycopg2
import psycopg2.extras
from dotenv import load_dotenv
from sklearn.metrics import (
    average_precision_score,
    brier_score_loss,
    f1_score,
    log_loss,
    roc_auc_score,
)

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
MODELS   = ROOT / "models"

load_dotenv(ROOT / ".env")
DATABASE_URL = os.getenv("DATABASE_URL")
if not DATABASE_URL:
    print("ERROR: DATABASE_URL not set. Copy .env.example → .env and fill it in.")
    sys.exit(1)

SEP = "─" * 60


# ── helpers ───────────────────────────────────────────────────────────────────

def nan_to_none(df: pd.DataFrame) -> pd.DataFrame:
    return df.where(pd.notnull(df), other=None)


def bulk_insert(conn, table: str, df: pd.DataFrame,
                chunk_size: int = 5_000, conflict: str = "DO NOTHING") -> int:
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
        # start_date, end_date, clob_token_id, yes_final_price will be NULL.
        # Run fetch_markets.py to regenerate the CSV, then re-run to fill them in.
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


def compute_metrics(y_true, y_prob) -> dict:
    y_true = np.asarray(y_true, dtype=int)
    y_prob = np.asarray(y_prob, dtype=float)
    mask   = ~np.isnan(y_prob)
    y_true, y_prob = y_true[mask], y_prob[mask]

    # threshold that maximises F1
    thresholds  = np.linspace(0.1, 0.9, 81)
    best_thresh = max(
        thresholds,
        key=lambda t: f1_score(y_true, (y_prob >= t).astype(int), zero_division=0),
    )
    y_pred = (y_prob >= best_thresh).astype(int)

    return {
        "auc_roc":   round(float(roc_auc_score(y_true, y_prob)), 4),
        "pr_auc":    round(float(average_precision_score(y_true, y_prob)), 4),
        "log_loss":  round(float(log_loss(y_true, y_prob)), 4),
        "brier":     round(float(brier_score_loss(y_true, y_prob)), 4),
        "f1":        round(float(f1_score(y_true, y_pred, zero_division=0)), 4),
        "threshold": round(float(best_thresh), 3),
        "n_test":    int(len(y_true)),
    }


def upsert_model_run(conn, model_name: str, dataset_tag: str,
                     notes: str, metrics: dict) -> int:
    with conn.cursor() as cur:
        cur.execute(
            """INSERT INTO model_runs (model_name, dataset_tag, notes, metrics)
               VALUES (%s, %s, %s, %s::jsonb)
               ON CONFLICT (model_name, dataset_tag)
               DO UPDATE SET metrics = EXCLUDED.metrics, notes = EXCLUDED.notes
               RETURNING id""",
            (model_name, dataset_tag, notes, json.dumps(metrics)),
        )
        row = cur.fetchone()
    conn.commit()
    return row[0]


def load_model_runs(conn) -> int:
    print(f"\n{SEP}\n  model_runs  (metrics from prediction CSVs)\n{SEP}")

    LR  = MODELS / "logistic_regression/predictions/predictions.csv"
    GB  = MODELS / "gradient_boosting/predictions/predictions.csv"
    RF  = MODELS / "random_forest/predictions/test_predictions.csv"
    RFT = MODELS / "random_forest_trends/predictions/trends_test_predictions.csv"

    sources = [
        # (csv,  model_name,                          dataset_tag,        prob_col)
        (LR,  "logistic_regression_base",             "clean_v1",         "pred_prob_base"),
        (LR,  "logistic_regression_trends",           "clean_v1_trends",  "pred_prob_trends"),
        (GB,  "xgboost_base",                         "clean_v1",         "pred_prob_base"),
        (GB,  "xgboost_trends",                       "clean_v1_trends",  "pred_prob_trends"),
        (RF,  "random_forest_price_only",             "clean_v1",         "proba_price_only"),
        (RF,  "random_forest_full",                   "clean_v1",         "proba_full"),
        (RF,  "random_forest_full_calibrated",        "clean_v1",         "proba_full_calibrated"),
        (RFT, "random_forest_trends_full",            "clean_v1_trends",  "proba_full"),
        (RFT, "random_forest_trends_calibrated",      "clean_v1_trends",  "proba_trends_calibrated"),
    ]

    loaded    = 0
    csv_cache: dict[Path, pd.DataFrame] = {}

    for csv_path, model_name, dataset_tag, prob_col in sources:
        if not csv_path.exists():
            print(f"  SKIP {model_name:<45} {csv_path.name} not found")
            continue
        if csv_path not in csv_cache:
            csv_cache[csv_path] = pd.read_csv(csv_path)
        df = csv_cache[csv_path]

        if prob_col not in df.columns:
            print(f"  SKIP {model_name:<45} column '{prob_col}' not found")
            continue
        if "outcome" not in df.columns:
            print(f"  SKIP {model_name:<45} no 'outcome' column — cannot compute metrics")
            continue

        metrics = compute_metrics(df["outcome"], df[prob_col])
        run_id  = upsert_model_run(
            conn, model_name, dataset_tag,
            f"Metrics from {csv_path.name}", metrics,
        )
        print(f"  {model_name:<45} run_id={run_id:<4} "
              f"auc={metrics['auc_roc']:.4f}  pr_auc={metrics['pr_auc']:.4f}  "
              f"n={metrics['n_test']:,}")
        loaded += 1

    print(f"\n  NOTE: SVM not included — SVM CSVs use target_percentile, not probabilities.")
    return loaded


# ── main ──────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Load markets metadata + model registry into Supabase"
    )
    parser.add_argument("--skip-markets",    action="store_true")
    parser.add_argument("--skip-trends",     action="store_true")
    parser.add_argument("--skip-model-runs", action="store_true")
    args = parser.parse_args()

    print(f"\n{'█'*60}")
    print(f"  POLYMARKET — PARQUET → SUPABASE  (operational layer)")
    print(f"{'█'*60}")

    conn = psycopg2.connect(
        DATABASE_URL,
        keepalives=1,
        keepalives_idle=30,
        keepalives_interval=10,
        keepalives_count=5,
    )
    print(f"\nConnected.")

    if not args.skip_markets:
        load_markets(conn)
    else:
        print(f"\n── markets ── SKIPPED")

    if not args.skip_trends:
        load_trends(conn)
    else:
        print(f"\n── trends ── SKIPPED")

    if not args.skip_model_runs:
        load_model_runs(conn)
    else:
        print(f"\n── model_runs ── SKIPPED")

    print(f"\n{SEP}\n  Row counts\n{SEP}")
    with conn.cursor() as cur:
        for table in ["markets", "trends", "model_runs"]:
            cur.execute(f"SELECT COUNT(*) FROM {table}")
            print(f"  {table:<15} {cur.fetchone()[0]:>10,}")

    conn.close()
    print(f"\n{'='*60}\n  DONE\n{'='*60}\n")


if __name__ == "__main__":
    main()
