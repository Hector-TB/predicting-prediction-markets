"""
scripts/rebuild_meta_from_parquet.py
======================================
Reconstructs polymarket_markets_meta.csv for any markets that are in the
parquet but missing from the current meta CSV (i.e. the re-fetch missed them).

Markets already in the meta CSV are kept unchanged. Missing markets are
back-filled from the parquet with approximate start/end dates (derived from
snapshot timestamps + pct_lifetime_elapsed). clob_token_id is left empty for
back-filled markets since their snapshots are already in the parquet.

After running this, call scripts/recompute_split.py to reset the 80/20 split
on the full corpus, then run_pipeline.py --skip-markets --skip-trends to
regenerate the clean parquets.

Usage:
    python scripts/rebuild_meta_from_parquet.py
"""

from pathlib import Path

import numpy as np
import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
META_CSV = DATA_DIR / "polymarket_markets_meta.csv"
PARQUET  = DATA_DIR / "polymarket_ml_dataset_with_trends_clean.parquet"


def approx_start_date(group: pd.DataFrame) -> pd.Timestamp:
    """Estimate market start date from the earliest snapshot."""
    row = group.loc[group["pct_lifetime_elapsed"].idxmin()]
    first_ts = pd.to_datetime(row["snapshot_timestamp"], utc=True)
    return first_ts - pd.to_timedelta(
        float(row["pct_lifetime_elapsed"]) * float(row["duration_days"]), unit="D"
    )


def main() -> None:
    meta = pd.read_csv(META_CSV)
    meta["start_date"] = pd.to_datetime(meta["start_date"], utc=True, errors="coerce")
    meta_ids = set(meta["market_id"])
    print(f"Current meta CSV : {len(meta):,} markets")

    pq = pd.read_parquet(PARQUET, columns=[
        "market_id", "snapshot_timestamp", "pct_lifetime_elapsed",
        "duration_days", "total_volume", "outcome", "category", "question", "split",
    ])
    pq["snapshot_timestamp"] = pd.to_datetime(pq["snapshot_timestamp"], utc=True, errors="coerce")
    pq_ids = set(pq["market_id"].unique())
    missing_ids = pq_ids - meta_ids
    print(f"Parquet markets  : {len(pq_ids):,}")
    print(f"Missing from meta: {len(missing_ids):,}  — back-filling ...")

    if not missing_ids:
        print("Nothing to back-fill.")
        return

    missing_pq = pq[pq["market_id"].isin(missing_ids)].copy()

    rows = []
    for mid, grp in missing_pq.groupby("market_id"):
        start = approx_start_date(grp)
        dur   = int(grp["duration_days"].iloc[0])
        end   = start + pd.Timedelta(days=dur)
        rows.append({
            "market_id":       mid,
            "clob_token_id":   "",           # already have snapshots; not needed
            "question":        grp["question"].iloc[0],
            "category":        grp["category"].iloc[0],
            "start_date":      start,
            "end_date":        end,
            "duration_days":   dur,
            "total_volume":    float(grp["total_volume"].iloc[0]),
            "yes_final_price": 0.95 if grp["outcome"].iloc[0] == 1 else 0.05,
            "outcome":         int(grp["outcome"].iloc[0]),
            "split":           grp["split"].iloc[0],
        })

    backfill_df = pd.DataFrame(rows)
    print(f"  Back-filled {len(backfill_df):,} markets from parquet")

    combined = pd.concat([meta, backfill_df], ignore_index=True)
    combined  = combined.sort_values("start_date").reset_index(drop=True)
    combined.to_csv(META_CSV, index=False)

    print(f"\nSaved {len(combined):,} markets → {META_CSV}")
    print(f"  train={combined['split'].eq('train').sum():,}  "
          f"test={combined['split'].eq('test').sum():,}")
    print("\nNext: python scripts/recompute_split.py")


if __name__ == "__main__":
    main()
