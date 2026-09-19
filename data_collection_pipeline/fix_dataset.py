"""
Fix Dataset
===========
Applies three data quality fixes in place:
  1. Deduplicate market_ids in polymarket_markets_meta.csv (keep first)
  2. Clip price_at_snapshot to [0, 1] in polymarket_ml_dataset.csv
  3. Fill NaN rolling features using the current snapshot price:
       price_mean/min/max  → price_at_snapshot (no trading = flat price)
       volatility/change/range/trend → 0.0

Run after build_snapshots.py, before training any models.
"""

import pandas as pd
from pathlib import Path

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"

META_CSV    = DATA_DIR / "polymarket_markets_meta.csv"
DATASET_CSV = DATA_DIR / "polymarket_ml_dataset.csv"
SEP         = "=" * 60

ROLLING_FILL_PRICE = ["price_mean", "price_min", "price_max"]
ROLLING_FILL_ZERO  = ["price_volatility", "price_change", "price_range", "price_trend"]
WINDOWS            = ["7d", "14d"]


def fix_meta(meta: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}\n  FIX 1: Deduplicate meta CSV\n{SEP}")

    before = len(meta)
    dupes  = meta["market_id"].duplicated().sum()
    print(f"  Before: {before:,} rows  ({dupes} duplicate market_ids)")

    meta = meta.drop_duplicates(subset=["market_id"], keep="first").reset_index(drop=True)

    after = len(meta)
    print(f"  After:  {after:,} rows  (removed {before - after})")
    return meta


def fix_dataset(df: pd.DataFrame) -> pd.DataFrame:
    # ── Fix 2: clip prices ──────────────────────────────
    print(f"\n{SEP}\n  FIX 2: Clip price_at_snapshot to [0, 1]\n{SEP}")

    out_of_range = ((df["price_at_snapshot"] < 0) | (df["price_at_snapshot"] > 1)).sum()
    print(f"  Rows out of [0,1] before: {out_of_range}")
    df["price_at_snapshot"] = df["price_at_snapshot"].clip(0.0, 1.0)
    df["price_deviation_from_half"] = (df["price_at_snapshot"] - 0.5).abs().round(4)
    out_after = ((df["price_at_snapshot"] < 0) | (df["price_at_snapshot"] > 1)).sum()
    print(f"  Rows out of [0,1] after:  {out_after}")

    # ── Fix 3: fill NaN rolling features ───────────────
    print(f"\n{SEP}\n  FIX 3: Fill NaN rolling features\n{SEP}")

    total_nan_before = sum(
        df[f"{feat}_{w}"].isna().sum()
        for feat in ROLLING_FILL_PRICE + ROLLING_FILL_ZERO
        for w in WINDOWS
    )
    print(f"  Total NaN rolling cells before: {total_nan_before:,}")

    for w in WINDOWS:
        nan_mask = df[f"price_mean_{w}"].isna()
        n_nan    = nan_mask.sum()

        for feat in ROLLING_FILL_PRICE:
            col = f"{feat}_{w}"
            df.loc[nan_mask, col] = df.loc[nan_mask, "price_at_snapshot"].round(4)

        for feat in ROLLING_FILL_ZERO:
            col = f"{feat}_{w}"
            df.loc[nan_mask, col] = 0.0

        print(f"  {w}: filled {n_nan:,} NaN rows")

    total_nan_after = sum(
        df[f"{feat}_{w}"].isna().sum()
        for feat in ROLLING_FILL_PRICE + ROLLING_FILL_ZERO
        for w in WINDOWS
    )
    print(f"  Total NaN rolling cells after:  {total_nan_after:,}  <- should be 0")

    return df


def main():
    print(f"\n{'█' * 60}")
    print(f"  POLYMARKET — FIX DATASET")
    print(f"{'█' * 60}")

    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found.")
        return
    if not DATASET_CSV.exists():
        print(f"ERROR: {DATASET_CSV} not found.")
        return

    # ── Load ────────────────────────────────────────────
    meta = pd.read_csv(META_CSV)
    meta["start_date"] = pd.to_datetime(meta["start_date"], format="ISO8601", utc=True)
    meta["end_date"]   = pd.to_datetime(meta["end_date"],   format="ISO8601", utc=True)

    print("\nLoading dataset CSV (4M rows, ~30s)...")
    df = pd.read_csv(DATASET_CSV, parse_dates=["snapshot_timestamp"], low_memory=False)

    # ── Apply fixes ─────────────────────────────────────
    meta = fix_meta(meta)
    df   = fix_dataset(df)

    # ── Save ────────────────────────────────────────────
    print(f"\n{SEP}\n  SAVING\n{SEP}")

    meta.to_csv(META_CSV, index=False)
    print(f"  Saved {META_CSV}  ({len(meta):,} rows)")

    print(f"  Writing dataset CSV (this takes ~60s)...")
    df.to_csv(DATASET_CSV, index=False)
    print(f"  Saved {DATASET_CSV}  ({len(df):,} rows)")

    parquet_path = DATASET_CSV.with_suffix(".parquet")
    if parquet_path.exists():
        print(f"  Merging with existing parquet...")
        existing_pq = pd.read_parquet(parquet_path)
        existing_pq = existing_pq[~existing_pq["market_id"].isin(set(df["market_id"]))].copy()
        df = pd.concat([existing_pq, df], ignore_index=True)
        print(f"  Merged total: {len(df):,} rows")
    print(f"  Writing dataset parquet...")
    df.to_parquet(parquet_path, index=False)
    print(f"  Saved {parquet_path}  ({len(df):,} rows)")

    # ── Final validation ─────────────────────────────────
    print(f"\n{SEP}\n  VALIDATION\n{SEP}")
    print(f"  Meta market_ids unique:    {meta['market_id'].nunique():,}  (dupes: {meta['market_id'].duplicated().sum()})")
    print(f"  price_at_snapshot out of [0,1]: {((df['price_at_snapshot'] < 0) | (df['price_at_snapshot'] > 1)).sum()}")
    remaining_nan = df[[f"{f}_{w}" for f in ROLLING_FILL_PRICE + ROLLING_FILL_ZERO for w in WINDOWS]].isna().sum().sum()
    print(f"  Remaining NaN in rolling features: {remaining_nan}")
    print(f"\n  All fixes applied.\n")


if __name__ == "__main__":
    main()
