"""
Step 2: Build Snapshot Dataset
================================
Reads polymarket_markets_meta.csv produced by fetch_markets.py,
fetches CLOB price history for each market, and builds a snapshot
dataset with rolling price features.

Output: polymarket_ml_dataset.csv

Run fetch_markets.py first.

    python data_collection_pipeline/build_snapshots.py          # new markets only
    python data_collection_pipeline/build_snapshots.py --full   # rebuild every market

--full ignores the merged parquet and rebuilds all markets into the CSV
(resumable). It refuses to resume from a CSV built with different settings.
"""

import argparse
import json
import logging
import pandas as pd
import numpy as np
import time
import sys
from datetime import timedelta
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor

from features import MIN_PRICE_OBS, ROLLING_WINDOWS, FetchError, fetch_price_history, raw_features
from meta_checks import assert_meta_ok
from typing import Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

SNAPSHOT_INTERVAL_H = 12        # hours between snapshots
MIN_HISTORY_DAYS    = 14        # burn-in before first snapshot
CUTOFF_BEFORE_CLOSE = 0         # days before endDate to stop snapshots (was 14; ADR-022)

SLEEP_BETWEEN_CALLS = 0.15
WORKERS             = 4         # processes; 2 physical cores here, fetch waits overlap

# Column order of the CSV (a resumed build appends without a header)
TIME_PRICE_COLUMNS  = ["days_before_close", "pct_lifetime_elapsed", "duration_days",
                       "price_at_snapshot", "price_deviation_from_half"]

INPUT_META          = DATA_DIR / "polymarket_markets_meta.csv"
OUTPUT_DATASET      = DATA_DIR / "polymarket_ml_dataset.csv"
BUILD_SETTINGS      = DATA_DIR / "polymarket_ml_dataset.csv.build.json"
CLOSED_TIMES_CSV    = DATA_DIR / "market_closed_times.csv"

# Settings that change which rows a market gets; a resumed build must match them
SETTINGS = {
    "snapshot_interval_h": SNAPSHOT_INTERVAL_H,
    "min_history_days":    MIN_HISTORY_DAYS,
    "cutoff_before_close": CUTOFF_BEFORE_CLOSE,
    "rolling_windows":     ROLLING_WINDOWS,
    "min_price_obs":       MIN_PRICE_OBS,
}

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# COMPUTE SNAPSHOTS
# ─────────────────────────────────────────────

def compute_snapshots(market: pd.Series, price_df: pd.DataFrame,
                      closed_time: Optional[pd.Timestamp] = None) -> Optional[pd.DataFrame]:
    """
    Generate one row per 12-hour snapshot for a market.

    Window:
      snap_start = createdAt + MIN_HISTORY_DAYS    (burn-in for feature stability)
      snap_end   = endDate   - CUTOFF_BEFORE_CLOSE (ADR-022: 0 — near-certain rows
                   are removed by fix_leakage.py's price rule instead)
                   and no later than closed_time when known: fix_leakage.py drops
                   post-close rows anyway, so don't build them

    All features use price history up to each snapshot timestamp (features.py).
    """
    start    = market["start_date"]
    end      = market["end_date"]
    duration = market["duration_days"]
    outcome  = market["outcome"]

    snap_start = start + timedelta(days=MIN_HISTORY_DAYS)
    snap_end   = end   - timedelta(days=CUTOFF_BEFORE_CLOSE)
    if closed_time is not None and not pd.isna(closed_time):
        snap_end = min(snap_end, closed_time)

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

        raw  = raw_features(hist, snap_ts, start, end, duration)
        feat = {
            "market_id":          market["market_id"],
            "snapshot_timestamp": snap_ts,
            **{k: raw[k] for k in TIME_PRICE_COLUMNS},
            "total_volume":       market["total_volume"],
            "log_volume":         round(float(np.log1p(market["total_volume"])), 4),
            "outcome":            outcome,
            **{k: v for k, v in raw.items() if k not in TIME_PRICE_COLUMNS},
        }
        rows.append(feat)

    return pd.DataFrame(rows) if rows else None


def process_market(market: pd.Series, closed_time) -> tuple[str, Optional[pd.DataFrame], Optional[str]]:
    """Fetch + compute one market (runs in a worker process).
    Returns (status, snapshots or None, error or None); status is one of
    ok / no_history / no_snapshots / failed."""
    token = market["clob_token_id"]
    if pd.isna(token) or not str(token).strip():
        return "no_history", None, None
    try:
        price_df = fetch_price_history(token)
    except FetchError as e:
        return "failed", None, str(e)
    finally:
        time.sleep(SLEEP_BETWEEN_CALLS)   # per worker: keeps the request rate modest
    if price_df is None:
        return "no_history", None, None

    snap_df = compute_snapshots(market, price_df, closed_time)
    if snap_df is None or snap_df.empty:
        return "no_snapshots", None, None
    snap_df["split"]    = market["split"]
    snap_df["category"] = market["category"]
    snap_df["question"] = market["question"]
    return "ok", snap_df, None


# ─────────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────────

def check_build_settings(full: bool) -> None:
    """Refuse to resume a CSV built with different settings or mode (e.g. an
    incremental CSV under --full, or the old 14-day cutoff)."""
    wanted = {**SETTINGS, "full": full}
    if OUTPUT_DATASET.exists():
        found = json.loads(BUILD_SETTINGS.read_text()) if BUILD_SETTINGS.exists() else None
        if found != wanted:
            raise SystemExit(
                f"ERROR: {OUTPUT_DATASET.name} was built with settings {found}, not {wanted}.\n"
                f"Move or delete it (and {BUILD_SETTINGS.name}) before building."
            )
    else:
        BUILD_SETTINGS.write_text(json.dumps(wanted, indent=2) + "\n")


def load_closed_times() -> dict:
    if not CLOSED_TIMES_CSV.exists():
        return {}
    ct = pd.read_csv(CLOSED_TIMES_CSV).dropna(subset=["closed_time"])
    return dict(zip(ct["market_id"], pd.to_datetime(ct["closed_time"], utc=True, format="mixed")))


def load_processed_ids(full: bool = False) -> set:
    """Resume support — union of markets in the in-progress CSV and the merged parquet.

    The CSV only holds markets from the current (possibly interrupted) run, so it
    must never replace the parquet as the source of already-processed IDs.
    With full=True the parquet is ignored: every market is rebuilt.
    """
    ids = set()
    if OUTPUT_DATASET.exists():
        csv_ids = set(pd.read_csv(OUTPUT_DATASET, usecols=["market_id"])["market_id"].unique())
        print(f"  Resuming — {len(csv_ids):,} already-processed markets in CSV")
        ids |= csv_ids
    parquet_path = OUTPUT_DATASET.with_suffix(".parquet")
    if parquet_path.exists() and not full:
        pq_ids = set(pd.read_parquet(parquet_path, columns=["market_id"])["market_id"].unique())
        print(f"  Resuming — {len(pq_ids):,} already-processed markets in parquet")
        ids |= pq_ids
    return ids

def append_to_dataset(snap_df: pd.DataFrame, first_write: bool):
    snap_df.to_csv(OUTPUT_DATASET, mode="a", header=first_write, index=False)


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main(full: bool = False, workers: int = WORKERS):
    check_build_settings(full)
    print("\n" + "█" * 60)
    print("  POLYMARKET — BUILD SNAPSHOT DATASET")
    print("█" * 60 + "\n")

    # Load markets meta
    if not INPUT_META.exists():
        print(f"ERROR: {INPUT_META} not found. Run fetch_markets.py first.")
        sys.exit(1)

    markets_df = pd.read_csv(INPUT_META, dtype={"clob_token_id": str})
    assert_meta_ok(markets_df)   # ADR-023: never build from estimated or placeholder values
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

    processed_ids  = load_processed_ids(full)
    closed_times   = load_closed_times()
    print(f"  closedTime known for {len(closed_times):,} markets (snapshots stop there)")
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
    n_fetch_failed = 0

    remaining_df = markets_df[~markets_df["market_id"].isin(processed_ids)].reset_index(drop=True)
    n_remaining  = len(remaining_df)
    print(f"  {len(processed_ids):,} already done, {n_remaining:,} remaining\n")

    # Each worker process fetches and computes one market at a time; only this
    # process writes the CSV, one whole market per append, so resume still works.
    markets = [row for _, row in remaining_df.iterrows()]
    cts     = [closed_times.get(m["market_id"]) for m in markets]
    print(f"  Using {workers} worker processes\n")

    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = ex.map(process_market, markets, cts, chunksize=4)
        for i, (market, (status, snap_df, err)) in enumerate(zip(markets, results)):
            market_id = market["market_id"]

            if i % 50 == 0:
                print(f"  [{i/n_remaining*100:5.1f}%] {i}/{n_remaining} remaining | "
                      f"ok={n_success} no_hist={n_no_history} "
                      f"no_snap={n_no_snapshots} failed={n_fetch_failed} rows={total_rows:,}",
                      flush=True)

            if status == "failed":
                # Not added to the CSV, so the next run retries it
                n_fetch_failed += 1
                log.warning("  fetch failed for %s: %s", market_id, err)
                continue
            if status == "no_history":
                n_no_history += 1
                continue
            if status == "no_snapshots":
                n_no_snapshots += 1
                continue

            append_to_dataset(snap_df, first_write)
            first_write = False
            total_rows += len(snap_df)
            n_success  += 1

    print(f"\n{'=' * 60}")
    print(f"  COMPLETE")
    print(f"{'=' * 60}")
    print(f"  Markets processed:   {n_markets:>7,}")
    print(f"  Successful:          {n_success:>7,}")
    print(f"  No/sparse history:   {n_no_history:>7,}")
    print(f"  No valid snapshots:  {n_no_snapshots:>7,}")
    print(f"  Fetch failed:        {n_fetch_failed:>7,}  (retried on the next run)")
    print(f"  Total rows:          {total_rows:>7,}")

    if OUTPUT_DATASET.exists():
        # Read in chunks: a full rebuild is several GB once loaded (ADR-022)
        n_rows, yes, split_rows, ids, nan_counts = 0, 0, {}, set(), None
        for df in pd.read_csv(OUTPUT_DATASET, chunksize=500_000, low_memory=False):
            n_rows += len(df)
            yes    += int(df["outcome"].sum())
            ids.update(df["market_id"].unique())
            for k, v in df["split"].value_counts().items():
                split_rows[k] = split_rows.get(k, 0) + int(v)
            c = df.isnull().sum()
            nan_counts = c if nan_counts is None else nan_counts.add(c, fill_value=0)
        print(f"\nDataset validation:")
        print(f"  Rows:                    {n_rows:,}")
        print(f"  Markets:                 {len(ids):,}")
        print(f"  Outcome: YES={yes / n_rows:.1%} | NO={1 - yes / n_rows:.1%}")
        print(f"  Snapshots/market (mean): {n_rows / len(ids):.1f}")
        print(f"  Train: {split_rows.get('train', 0):,} | Test: {split_rows.get('test', 0):,}")
        if nan_counts is not None and nan_counts.any():
            print(f"\n  NaN counts:\n{nan_counts[nan_counts > 0].astype(int).to_string()}")

    if n_fetch_failed:
        print(f"\n  {n_fetch_failed:,} markets failed to fetch — re-run to retry them.")
    return n_fetch_failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Build the snapshot dataset CSV")
    parser.add_argument("--full", action="store_true",
                        help="Rebuild every market, ignoring the merged parquet (ADR-022)")
    parser.add_argument("--workers", type=int, default=WORKERS,
                        help=f"parallel worker processes (default {WORKERS})")
    args = parser.parse_args()
    n_failed = main(full=args.full, workers=args.workers)
    # Non-zero so run_pipeline.py stops before merging an incomplete build
    sys.exit(1 if n_failed else 0)