"""
scripts/recompute_split.py
===========================
Recomputes the train/test split by resolution time (ADR-021, replacing the
start-date rule of ADR-001).

Markets are sorted by closedTime (when they resolved); the first 80% are train,
the rest test. Every train label is therefore known by the cutoff T. To make the
test set fully leak-free, fix_leakage.py also drops test rows dated before T
(split = "test_pre_cutoff"), so the model is only scored on snapshots taken
after everything it learned from had already resolved.

closedTime comes from data/market_closed_times.csv (looked up from Gamma for any
market not yet cached). Markets without one fall back to their scheduled end_date.

The parquet datasets must be regenerated afterwards (fix_dataset.py re-stamps
the split from meta; fix_leakage.py applies the pre-cutoff rule):
    python data_collection_pipeline/run_pipeline.py --skip-markets --skip-snapshots

Usage:
    python scripts/recompute_split.py [--dry-run]
"""

import argparse
import logging
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_CSV = DATA_DIR / "polymarket_markets_meta.csv"
TRAIN_FRACTION = 0.8

sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
from fix_leakage import fetch_closed_times  # noqa: E402

log = logging.getLogger(__name__)


def main(dry_run: bool = False) -> None:
    if not META_CSV.exists():
        log.error("ERROR: %s not found. Run fetch_markets.py first.", META_CSV)
        sys.exit(1)

    meta = pd.read_csv(META_CSV)
    closed = fetch_closed_times(meta["market_id"].tolist()).set_index("market_id")["closed_time"]
    resolved_at = meta["market_id"].map(closed)
    n_fallback = int(resolved_at.isna().sum())
    resolved_at = resolved_at.fillna(pd.to_datetime(meta["end_date"], format="ISO8601", utc=True))

    order      = resolved_at.sort_values(kind="stable").index
    cutoff_idx = int(len(meta) * TRAIN_FRACTION)
    cutoff     = resolved_at.loc[order[cutoff_idx - 1]]

    new_split = pd.Series("test", index=meta.index)
    new_split.loc[order[:cutoff_idx]] = "train"

    old = meta["split"] if "split" in meta.columns else pd.Series(index=meta.index, dtype=object)
    log.info("Markets: %s  (%s without closedTime → scheduled end_date)", f"{len(meta):,}", n_fallback)
    log.info("Cutoff T (last train resolution): %s", cutoff)
    log.info("New split: train=%s  test=%s", f"{(new_split == 'train').sum():,}", f"{(new_split == 'test').sum():,}")
    log.info("Old split: train=%s  test=%s", f"{(old == 'train').sum():,}", f"{(old == 'test').sum():,}")
    log.info("Markets changing split: %s", f"{(new_split != old).sum():,}")

    if dry_run:
        log.info("\nDry run — no changes written.")
        return

    meta["split"] = new_split
    meta.to_csv(META_CSV, index=False)
    log.info("\nSaved updated split → %s", META_CSV)
    log.info("Next: python data_collection_pipeline/run_pipeline.py --skip-markets --skip-snapshots")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser()
    parser.add_argument("--dry-run", action="store_true",
                        help="Print what would change without writing")
    args = parser.parse_args()
    main(dry_run=args.dry_run)
