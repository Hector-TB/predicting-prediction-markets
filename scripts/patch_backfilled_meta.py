"""
scripts/patch_backfilled_meta.py
================================
Fills in real values for meta rows written by rebuild_meta_from_parquet.py.

Those rows have an empty clob_token_id and a placeholder yes_final_price
(0.95 / 0.05). The fresh Gamma fetch in data/fetch_cache/ has the real values,
so copy them over. Dates, duration and split are left alone: the existing
snapshots were built from them, and changing start_date could move markets
across the train/test cutoff.

Usage:
    python scripts/patch_backfilled_meta.py            # dry run
    python scripts/patch_backfilled_meta.py --apply
"""

import argparse
import logging
import shutil
from pathlib import Path

import pandas as pd

ROOT      = Path(__file__).resolve().parent.parent
DATA_DIR  = ROOT / "data"
META_CSV  = DATA_DIR / "polymarket_markets_meta.csv"
CACHE_DIR = DATA_DIR / "fetch_cache"
BACKUP    = DATA_DIR / "polymarket_markets_meta.pre_patch.bak.csv"

PATCH_COLUMNS = ["clob_token_id", "yes_final_price"]

log = logging.getLogger(__name__)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[3])
    parser.add_argument("--apply", action="store_true", help="Write the patched meta CSV")
    args = parser.parse_args()

    meta = pd.read_csv(META_CSV, dtype={"clob_token_id": str})
    backfilled = meta["clob_token_id"].fillna("").str.strip() == ""
    log.info("Meta rows: %s — back-filled (empty clob_token_id): %s",
             f"{len(meta):,}", f"{backfilled.sum():,}")

    cache = pd.concat(
        [pd.read_csv(f, dtype={"clob_token_id": str}) for f in sorted(CACHE_DIR.glob("*.csv"))],
        ignore_index=True,
    ).drop_duplicates("market_id", keep="last").set_index("market_id")

    ids = meta.loc[backfilled, "market_id"]
    found = ids[ids.isin(cache.index)]
    log.info("Found in fetch cache: %s — not found (left as is): %s",
             f"{len(found):,}", f"{len(ids) - len(found):,}")

    # Labels must agree, or the placeholder price wasn't the only thing wrong
    cached_outcome = found.map(cache["outcome"])
    mismatched = (meta.loc[found.index, "outcome"] != cached_outcome).sum()
    if mismatched:
        raise SystemExit(f"{mismatched} outcome mismatches between meta and cache — not patching")

    for col in PATCH_COLUMNS:
        meta.loc[found.index, col] = found.map(cache[col])

    still_placeholder = meta["yes_final_price"].isin([0.05, 0.95]).sum()
    log.info("Rows still at a 0.05/0.95 placeholder price: %s", f"{still_placeholder:,}")

    if not args.apply:
        log.info("Dry run — re-run with --apply to write %s", META_CSV.name)
        return
    shutil.copy2(META_CSV, BACKUP)
    meta.to_csv(META_CSV, index=False)
    log.info("Saved %s (backup: %s)", META_CSV.name, BACKUP.name)


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
