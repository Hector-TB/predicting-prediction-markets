"""
scripts/recompute_split.py
===========================
Recomputes the temporal 80/20 train/test split on the full market corpus.

Run this after a bulk data fetch (e.g. going from 21k → 40k+ markets) where
the frozen incremental split would produce an inverted ratio.  The script
reads polymarket_markets_meta.csv, sorts by start_date, applies a fresh 80/20
cutoff, and writes the updated split column back.

The parquet datasets (polymarket_ml_dataset*.parquet) must be regenerated
afterwards via run_pipeline.py or fix_dataset.py → fix_leakage.py so that
every snapshot row carries the new split assignment.

Usage:
    python scripts/recompute_split.py [--dry-run]
"""

import argparse
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_CSV = DATA_DIR / "polymarket_markets_meta.csv"


def main(dry_run: bool = False) -> None:
    if not META_CSV.exists():
        print(f"ERROR: {META_CSV} not found. Run fetch_markets.py first.")
        sys.exit(1)

    df = pd.read_csv(META_CSV)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    df = df.sort_values("start_date").reset_index(drop=True)

    n           = len(df)
    cutoff_idx  = int(n * 0.8)
    cutoff_date = df.iloc[cutoff_idx]["start_date"]

    old_counts = df["split"].value_counts().to_dict() if "split" in df.columns else {}
    new_split  = ["train"] * cutoff_idx + ["test"] * (n - cutoff_idx)

    print(f"Markets in corpus : {n:,}")
    print(f"Split cutoff index: {cutoff_idx:,}  ({cutoff_date.date()})")
    print(f"New  split: train={cutoff_idx:,}  test={n - cutoff_idx:,}  "
          f"({cutoff_idx/n:.1%} / {(n-cutoff_idx)/n:.1%})")
    if old_counts:
        print(f"Old  split: train={old_counts.get('train', 0):,}  "
              f"test={old_counts.get('test', 0):,}")

    if dry_run:
        print("\nDry run — no changes written.")
        return

    df["split"] = new_split
    df.to_csv(META_CSV, index=False)
    print(f"\nSaved updated split → {META_CSV}")
    print("Next: regenerate parquets with  python data_collection_pipeline/run_pipeline.py")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
