"""
Fix Dataset
===========
Applies three data quality fixes in place:
  1. Deduplicate market_ids in polymarket_markets_meta.csv (keep first)
  2. Clip price_at_snapshot to [0, 1] in polymarket_ml_dataset.csv
  3. Fill NaN rolling features using the current snapshot price:
       price_mean/min/max  → price_at_snapshot (no trading = flat price)
       volatility/change/range/trend → 0.0

Then merges the fixed CSV rows into polymarket_ml_dataset.parquet (replacing
any older rows for the same markets) and re-stamps every row's split from meta.

Everything streams in chunks (stream_parquet.py) so the full dataset is never
held in memory.

Run after build_snapshots.py, before training any models.
"""

import argparse
import logging
import os
import sys
import pandas as pd
from pathlib import Path

from stream_parquet import BATCH_ROWS, FrameWriter, iter_frames

log = logging.getLogger(__name__)

ROOT        = Path(__file__).resolve().parent.parent
DATA_DIR    = ROOT / "data"

META_CSV    = DATA_DIR / "polymarket_markets_meta.csv"
DATASET_CSV = DATA_DIR / "polymarket_ml_dataset.csv"
SEP         = "=" * 60

ROLLING_FILL_PRICE = ["price_mean", "price_min", "price_max"]
ROLLING_FILL_ZERO  = ["price_volatility", "price_change", "price_range", "price_trend"]
WINDOWS            = ["7d", "14d"]
TEXT_COLUMNS       = ["market_id", "split", "category", "question"]


def fix_meta(meta: pd.DataFrame) -> pd.DataFrame:
    print(f"\n{SEP}\n  FIX 1: Deduplicate meta CSV\n{SEP}")

    before = len(meta)
    dupes  = meta["market_id"].duplicated().sum()
    print(f"  Before: {before:,} rows  ({dupes} duplicate market_ids)")

    meta = meta.drop_duplicates(subset=["market_id"], keep="first").reset_index(drop=True)

    after = len(meta)
    print(f"  After:  {after:,} rows  (removed {before - after})")
    return meta


def fix_chunk(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Fixes 2 and 3 on one chunk of CSV rows. Returns (df, counts)."""
    counts = {}

    # ── Fix 2: clip prices ──────────────────────────────
    counts["clipped"] = int(((df["price_at_snapshot"] < 0) | (df["price_at_snapshot"] > 1)).sum())
    df["price_at_snapshot"] = df["price_at_snapshot"].clip(0.0, 1.0)
    df["price_deviation_from_half"] = (df["price_at_snapshot"] - 0.5).abs().round(4)

    # ── Fix 3: fill NaN rolling features ───────────────
    for w in WINDOWS:
        nan_mask = df[f"price_mean_{w}"].isna()
        counts[f"filled_{w}"] = int(nan_mask.sum())

        for feat in ROLLING_FILL_PRICE:
            df.loc[nan_mask, f"{feat}_{w}"] = df.loc[nan_mask, "price_at_snapshot"].round(4)
        for feat in ROLLING_FILL_ZERO:
            df.loc[nan_mask, f"{feat}_{w}"] = 0.0

    return df, counts


def finish_chunk(df: pd.DataFrame, split_by_market: pd.Series) -> tuple[pd.DataFrame, dict]:
    """Normalise the timestamp type and re-stamp split from meta (ADR-013)."""
    # Older parquet rows store snapshot_timestamp as strings; CSV rows parse to
    # Timestamps. A mixed object column fails the Arrow write, so normalise.
    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], format="mixed", utc=True)

    # Chunked read_csv infers dtypes per chunk: an all-empty `category` arrives
    # as float NaN, which Arrow would cast to the string "nan". Use real nulls.
    for col in TEXT_COLUMNS:
        df[col] = df[col].astype(object).where(df[col].notna(), None)

    # The meta CSV is the source of truth for the split. Snapshot rows get their
    # label when first built, so re-stamp after a split recompute.
    new_split = df["market_id"].map(split_by_market)
    n_missing = int(new_split.isna().sum())
    new_split = new_split.fillna(df["split"])
    n_changed = int((new_split != df["split"]).sum())
    df["split"] = new_split

    rolling = [f"{f}_{w}" for f in ROLLING_FILL_PRICE + ROLLING_FILL_ZERO for w in WINDOWS]
    return df, {
        "split_missing": n_missing,
        "split_changed": n_changed,
        "out_of_range":  int(((df["price_at_snapshot"] < 0) | (df["price_at_snapshot"] > 1)).sum()),
        "nan_rolling":   int(df[rolling].isna().sum().sum()),
    }


def add(total: dict, counts: dict) -> None:
    for k, v in counts.items():
        total[k] = total.get(k, 0) + v


def main(replace: bool = False):
    print(f"\n{'█' * 60}")
    print(f"  POLYMARKET — FIX DATASET")
    print(f"{'█' * 60}")

    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found.")
        sys.exit(1)
    if not DATASET_CSV.exists():
        print(f"ERROR: {DATASET_CSV} not found.")
        sys.exit(1)

    # ── Fix 1: meta (small — fits in memory) ───────────
    meta = pd.read_csv(META_CSV)
    meta = fix_meta(meta)
    meta.to_csv(META_CSV, index=False)
    print(f"  Saved {META_CSV}  ({len(meta):,} rows)")
    split_by_market = meta.set_index("market_id")["split"]

    # Markets in the new CSV replace any older rows for the same market
    csv_ids: set = set()
    for chunk in pd.read_csv(DATASET_CSV, usecols=["market_id"], chunksize=BATCH_ROWS):
        csv_ids.update(chunk["market_id"].unique())
    log.info("  %s markets in the new CSV", f"{len(csv_ids):,}")

    parquet_path = DATASET_CSV.with_suffix(".parquet")
    csv_tmp      = DATASET_CSV.with_name(DATASET_CSV.name + ".tmp")
    fixed, final = {}, {}

    print(f"\n{SEP}\n  FIXES 2–3 + MERGE (streaming, {BATCH_ROWS:,} rows per chunk)\n{SEP}")
    try:
        with FrameWriter(parquet_path) as out:
            # Existing parquet rows first, minus markets the CSV supersedes
            if parquet_path.exists() and replace:
                log.info("  --replace: the CSV is a full rebuild — existing parquet rows discarded")
            elif parquet_path.exists():
                kept = 0
                for df in iter_frames(parquet_path):
                    df = df[~df["market_id"].isin(csv_ids)]
                    kept += len(df)
                    df, counts = finish_chunk(df, split_by_market)
                    add(final, counts)
                    out.write(df)
                log.info("  Existing parquet rows kept: %s", f"{kept:,}")

            # Then the fixed CSV rows (also written back to the CSV, as before)
            first = True
            for df in pd.read_csv(DATASET_CSV, chunksize=BATCH_ROWS, low_memory=False):
                df, counts = fix_chunk(df)
                add(fixed, counts)
                df.to_csv(csv_tmp, mode="w" if first else "a", header=first, index=False)
                first = False
                df, counts = finish_chunk(df, split_by_market)
                add(final, counts)
                out.write(df)
        os.replace(csv_tmp, DATASET_CSV)
    finally:
        if csv_tmp.exists():
            csv_tmp.unlink()

    print(f"  Prices clipped to [0,1]:     {fixed.get('clipped', 0):,}")
    for w in WINDOWS:
        print(f"  {w}: filled {fixed.get(f'filled_{w}', 0):,} NaN rows")
    if final.get("split_missing"):
        log.warning("  %s snapshot rows have no market in meta — kept their old split",
                    f"{final['split_missing']:,}")
    log.info("  Split re-stamped from meta: %s rows changed", f"{final.get('split_changed', 0):,}")
    print(f"  Saved {parquet_path}  ({out.rows:,} rows)")

    # ── Final validation ─────────────────────────────────
    print(f"\n{SEP}\n  VALIDATION\n{SEP}")
    print(f"  Meta market_ids unique:    {meta['market_id'].nunique():,}  (dupes: {meta['market_id'].duplicated().sum()})")
    print(f"  price_at_snapshot out of [0,1]: {final.get('out_of_range', 0)}")
    print(f"  Remaining NaN in rolling features: {final.get('nan_rolling', 0)}")
    print(f"\n  All fixes applied.\n")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Clean the snapshot CSV and merge it into the parquet")
    parser.add_argument("--replace", action="store_true",
                        help="The CSV is a full rebuild: discard the existing parquet instead of merging")
    main(replace=parser.parse_args().replace)
