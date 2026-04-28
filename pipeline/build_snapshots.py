"""
Step 2: Build Snapshot Dataset
================================
Reads polymarket_markets_meta.csv produced by fetch_markets.py,
fetches CLOB price history for each market, and builds a snapshot
dataset with rolling price features.

Output: polymarket_ml_dataset.csv

Run fetch_markets.py first.
"""

import requests
import pandas as pd
import numpy as np
import time
import os
from datetime import timedelta
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

CLOB_URL            = "https://clob.polymarket.com"

MIN_PRICE_OBS       = 20        # minimum price observations to use a market
SNAPSHOT_INTERVAL_H = 12        # hours between snapshots
MIN_HISTORY_DAYS    = 14        # burn-in before first snapshot
CUTOFF_BEFORE_CLOSE = 14        # days before endDate to stop snapshots
ROLLING_WINDOWS     = [7, 14]   # rolling feature windows in days

SLEEP_BETWEEN_CALLS = 0.15

INPUT_META          = DATA_DIR / "polymarket_markets_meta.csv"
OUTPUT_DATASET      = DATA_DIR / "polymarket_ml_dataset.csv"


# ─────────────────────────────────────────────
# FETCH PRICE HISTORY
# ─────────────────────────────────────────────

def fetch_price_history(clob_token_id: str) -> Optional[pd.DataFrame]:
    """
    Fetch full price history for a YES token from CLOB API.
    Returns DataFrame(timestamp, price) sorted ascending, or None.
    """
    try:
        r = requests.get(
            f"{CLOB_URL}/prices-history",
            params={"market": clob_token_id, "interval": "max", "fidelity": 720},
            timeout=20,
        )
        r.raise_for_status()
        history = r.json().get("history", [])
        if not history:
            return None

        df = pd.DataFrame(history)
        df = df.rename(columns={"t": "timestamp", "p": "price"})
        df["timestamp"] = pd.to_datetime(df["timestamp"], unit="s", utc=True)
        df["price"]     = pd.to_numeric(df["price"], errors="coerce")
        df = df.dropna(subset=["price"]).sort_values("timestamp").reset_index(drop=True)

        return df if len(df) >= MIN_PRICE_OBS else None

    except Exception:
        return None


# ─────────────────────────────────────────────
# COMPUTE SNAPSHOTS
# ─────────────────────────────────────────────

def compute_snapshots(market: pd.Series, price_df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """
    Generate one row per 12-hour snapshot for a market.

    Window:
      snap_start = createdAt + MIN_HISTORY_DAYS   (burn-in for feature stability)
      snap_end   = endDate   - CUTOFF_BEFORE_CLOSE (no peeking near resolution)

    All features use price history strictly BEFORE each snapshot timestamp.
    """
    start    = market["start_date"]
    end      = market["end_date"]
    duration = market["duration_days"]
    outcome  = market["outcome"]

    snap_start = start + timedelta(days=MIN_HISTORY_DAYS)
    snap_end   = end   - timedelta(days=CUTOFF_BEFORE_CLOSE)

    if snap_start >= snap_end:
        return None

    snap_times = pd.date_range(
        start=snap_start, end=snap_end,
        freq=f"{SNAPSHOT_INTERVAL_H}h", tz="UTC"
    )
    if len(snap_times) == 0:
        return None

    rows = []
    for snap_ts in snap_times:

        hist = price_df[price_df["timestamp"] <= snap_ts]
        if len(hist) < MIN_PRICE_OBS:
            continue

        price_now         = float(hist["price"].values[-1])
        days_before_close = (end - snap_ts).total_seconds() / 86400
        pct_elapsed       = float(np.clip(
            (snap_ts - start).total_seconds() / (end - start).total_seconds(), 0, 1
        ))

        feat = {
            "market_id":                 market["market_id"],
            "snapshot_timestamp":        snap_ts,
            "days_before_close":         round(days_before_close, 2),
            "pct_lifetime_elapsed":      round(pct_elapsed, 4),
            "duration_days":             duration,
            "price_at_snapshot":         round(price_now, 4),
            "price_deviation_from_half": round(abs(price_now - 0.5), 4),
            "total_volume":              market["total_volume"],
            "log_volume":                round(float(np.log1p(market["total_volume"])), 4),
            "outcome":                   outcome,
        }

        for window_days in ROLLING_WINDOWS:
            label        = f"{window_days}d"
            window_start = snap_ts - timedelta(days=window_days)
            w_prices     = hist[hist["timestamp"] >= window_start]["price"].values

            if len(w_prices) >= 2:
                x     = np.arange(len(w_prices), dtype=float)
                slope = float(np.polyfit(x, w_prices, 1)[0]) if x.std() > 0 else 0.0
                feat[f"price_mean_{label}"]       = round(float(np.mean(w_prices)), 4)
                feat[f"price_volatility_{label}"] = round(float(np.std(w_prices)), 4)
                feat[f"price_min_{label}"]        = round(float(np.min(w_prices)), 4)
                feat[f"price_max_{label}"]        = round(float(np.max(w_prices)), 4)
                feat[f"price_change_{label}"]     = round(float(w_prices[-1] - w_prices[0]), 4)
                feat[f"price_range_{label}"]      = round(float(np.max(w_prices) - np.min(w_prices)), 4)
                feat[f"price_trend_{label}"]      = round(slope, 6)
            else:
                for s in ["mean", "volatility", "min", "max", "change", "range", "trend"]:
                    feat[f"price_{s}_{label}"] = np.nan

        rows.append(feat)

    return pd.DataFrame(rows) if rows else None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def load_processed_ids() -> set:
    """Resume support — read already-processed market IDs from existing CSV."""
    if OUTPUT_DATASET.exists():
        existing = pd.read_csv(OUTPUT_DATASET, usecols=["market_id"])
        ids = set(existing["market_id"].unique())
        print(f"  Resuming — found {len(ids):,} already-processed markets in {OUTPUT_DATASET}")
        return ids
    return set()

def append_to_dataset(snap_df: pd.DataFrame, first_write: bool):
    snap_df.to_csv(OUTPUT_DATASET, mode="a", header=first_write, index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  POLYMARKET — BUILD SNAPSHOT DATASET")
    print("█" * 60 + "\n")

    # Load markets meta
    if not INPUT_META.exists():
        print(f"ERROR: {INPUT_META} not found. Run fetch_markets.py first.")
        return

    markets_df = pd.read_csv(INPUT_META)
    markets_df["start_date"] = pd.to_datetime(markets_df["start_date"], format="ISO8601", utc=True)
    markets_df["end_date"]   = pd.to_datetime(markets_df["end_date"], format="ISO8601", utc=True)

    print(f"Loaded {len(markets_df):,} markets from {INPUT_META}")
    print(f"  Outcome: YES={markets_df['outcome'].mean():.1%} | "
          f"NO={(1-markets_df['outcome'].mean()):.1%}")
    print(f"  Duration: min={markets_df['duration_days'].min()}d  "
          f"median={markets_df['duration_days'].median():.0f}d  "
          f"max={markets_df['duration_days'].max()}d\n")

    print("=" * 60)
    print("Fetching price history + computing snapshots")
    print("=" * 60)

    processed_ids  = load_processed_ids()
    n_markets      = len(markets_df)
    first_write    = not OUTPUT_DATASET.exists()

    # Initialise counters from existing CSV if resuming
    if not first_write:
        existing = pd.read_csv(OUTPUT_DATASET, usecols=["market_id", "outcome"])
        n_success  = existing["market_id"].nunique()
        total_rows = len(existing)
    else:
        n_success  = 0
        total_rows = 0

    n_no_history   = 0
    n_no_snapshots = 0

    remaining_df = markets_df[~markets_df["market_id"].isin(processed_ids)].reset_index(drop=True)
    n_remaining  = len(remaining_df)
    print(f"  {len(processed_ids):,} already done, {n_remaining:,} remaining\n")

    for i, (_, market) in enumerate(remaining_df.iterrows()):
        market_id = market["market_id"]

        if i % 50 == 0:
            print(f"  [{i/n_remaining*100:5.1f}%] {i}/{n_remaining} remaining | "
                  f"ok={n_success} no_hist={n_no_history} "
                  f"no_snap={n_no_snapshots} rows={total_rows:,}")

        price_df = fetch_price_history(market["clob_token_id"])
        time.sleep(SLEEP_BETWEEN_CALLS)

        if price_df is None:
            n_no_history += 1
            processed_ids.add(market_id)
            continue

        snap_df = compute_snapshots(market, price_df)

        if snap_df is None or snap_df.empty:
            n_no_snapshots += 1
            processed_ids.add(market_id)
            continue

        snap_df["split"]    = market["split"]
        snap_df["category"] = market["category"]
        snap_df["question"] = market["question"]

        append_to_dataset(snap_df, first_write)
        first_write = False
        total_rows += len(snap_df)
        n_success  += 1
        processed_ids.add(market_id)



    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Markets processed:   {n_markets:>7,}")
    print(f"  Successful:          {n_success:>7,}")
    print(f"  No/sparse history:   {n_no_history:>7,}")
    print(f"  No valid snapshots:  {n_no_snapshots:>7,}")
    print(f"  Total rows:          {total_rows:>7,}")

    if OUTPUT_DATASET.exists():
        df = pd.read_csv(OUTPUT_DATASET)
        print(f"\nDataset validation:")
        print(f"  Rows:                    {len(df):,}")
        print(f"  Markets:                 {df['market_id'].nunique():,}")
        print(f"  Outcome: YES={df['outcome'].mean():.1%} | NO={(1-df['outcome'].mean()):.1%}")
        print(f"  Snapshots/market (mean): {len(df)/df['market_id'].nunique():.1f}")
        print(f"  Train: {(df['split']=='train').sum():,} | Test: {(df['split']=='test').sum():,}")
        nan_counts = df.isnull().sum()
        if nan_counts.any():
            print(f"\n  NaN counts:\n{nan_counts[nan_counts > 0].to_string()}")


if __name__ == "__main__":
    main()