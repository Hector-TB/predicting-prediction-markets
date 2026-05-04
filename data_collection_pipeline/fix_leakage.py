"""
fix_leakage.py
==============
Fetches closedTime for every market in polymarket_markets_meta.csv,
then removes post-resolution snapshots from all parquet files.

SAFETY: originals are NEVER modified. All output goes to *_clean.parquet files.

Uses paginated batch fetching (~50 calls) instead of one call per market,
so the API step takes ~30 seconds rather than ~1 hour.

Run from the project root:
    python data_collection_pipeline/fix_leakage.py
"""

import pathlib
import time
import requests
import pandas as pd

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT     = pathlib.Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# Read-only inputs — never written to
META_CSV      = DATA_DIR / "polymarket_markets_meta.csv"
PARQUET_MAIN  = DATA_DIR / "polymarket_ml_dataset.parquet"
PARQUET_T1    = DATA_DIR / "polymarket_ml_dataset_with_trends_part1.parquet"
PARQUET_T2    = DATA_DIR / "polymarket_ml_dataset_with_trends_part2.parquet"

# New output files — originals are untouched
CLOSED_TIMES_CSV        = DATA_DIR / "market_closed_times.csv"
PARQUET_MAIN_CLEAN      = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
PARQUET_TRENDS_CLEAN    = DATA_DIR / "polymarket_ml_dataset_with_trends_clean.parquet"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GAMMA_URL           = "https://gamma-api.polymarket.com"
BATCH_SIZE          = 500
SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES         = 3


# ─────────────────────────────────────────────
# STEP 1 — FETCH closedTime VIA PAGINATION
# ─────────────────────────────────────────────

def fetch_closed_times(market_ids: list[str]) -> pd.DataFrame:
    """
    Sweep the Gamma API using the same paginated batch endpoint used by
    fetch_markets.py (500 markets per call).  Extracts conditionId →
    closedTime for every market that appears in market_ids.

    Returns DataFrame with columns [market_id, closed_time].
    """
    target_ids = set(market_ids)
    found: dict[str, str | None] = {}

    offset               = 0
    total_calls          = 0
    offset_attempts      = 0   # retry counter for the current offset
    consecutive_skips    = 0
    MAX_OFFSET_ATTEMPTS  = 5
    MAX_CONSECUTIVE_SKIP = 3

    print(f"  Sweeping Gamma API in batches of {BATCH_SIZE}...")

    while len(found) < len(target_ids):
        batch = None
        try:
            r = requests.get(
                f"{GAMMA_URL}/markets",
                params={
                    "closed":    "true",
                    "limit":     BATCH_SIZE,
                    "offset":    offset,
                    "order":     "startDate",
                    "ascending": "true",
                },
                timeout=20,
            )
            r.raise_for_status()
            batch = r.json()
        except Exception as e:
            offset_attempts += 1
            wait = 2 ** offset_attempts
            if offset_attempts < MAX_OFFSET_ATTEMPTS:
                print(f"  Attempt {offset_attempts}/{MAX_OFFSET_ATTEMPTS} failed "
                      f"at offset {offset}: {e} — retrying in {wait}s")
                time.sleep(wait)
                continue  # retry same offset
            else:
                # Exhausted retries for this offset — skip it
                print(f"  Giving up on offset {offset} after {MAX_OFFSET_ATTEMPTS} attempts — skipping.")
                offset          += BATCH_SIZE
                offset_attempts  = 0
                consecutive_skips += 1
                if consecutive_skips >= MAX_CONSECUTIVE_SKIP:
                    print(f"  {MAX_CONSECUTIVE_SKIP} consecutive skips — stopping sweep.")
                    break
                continue

        # Success
        offset_attempts   = 0
        consecutive_skips = 0

        for m in batch:
            cid = m.get("conditionId")
            if cid and cid in target_ids:
                found[cid] = m.get("closedTime")  # may be None

        total_calls += 1
        print(f"  offset {offset:>6}  |  batch {len(batch):>4}  "
              f"|  matched {len(found):>5} / {len(target_ids):,}")

        if len(batch) < BATCH_SIZE:
            break  # last page

        offset += BATCH_SIZE
        time.sleep(SLEEP_BETWEEN_CALLS)

    # Any market not seen in the sweep gets None
    for mid in target_ids:
        if mid not in found:
            found[mid] = None

    rows = [
        {"market_id": mid, "closed_time": pd.to_datetime(ct, utc=True) if ct else None}
        for mid, ct in found.items()
    ]
    result = pd.DataFrame(rows)
    result.to_csv(CLOSED_TIMES_CSV, index=False)
    print(f"\n  API calls made: {total_calls}  |  Saved to {CLOSED_TIMES_CSV.name}")
    return result


# ─────────────────────────────────────────────
# STEP 2 — FILTER PARQUET
# ─────────────────────────────────────────────

PRICE_THRESHOLD  = 0.95                                    # settled if price >= 0.95 or <= 0.05
INACTIVITY_DAYS  = 14                                      # keep N days after settlement/last change
HARD_CUTOFF      = pd.Timestamp("2026-05-01", tz="UTC")   # absolute ceiling


def price_based_cutoffs(df: pd.DataFrame) -> pd.Series:
    """
    Derive a per-market cutoff from price history for ALL markets.

    Pass A — threshold: find the first snapshot where price >= 0.95 or <= 0.05.
              cutoff = that timestamp + INACTIVITY_DAYS.

    Pass B — inactivity: for markets where price never crossed the threshold,
              find the last snapshot where the price changed by > 0.001.
              cutoff = that timestamp + INACTIVITY_DAYS.

    Returns a Series indexed by market_id. Markets with no detectable change
    fall back to HARD_CUTOFF.
    """
    sub = df.sort_values(["market_id", "snapshot_timestamp"])

    # Pass A: first extreme price
    extreme = (sub["price_at_snapshot"] >= PRICE_THRESHOLD) | \
              (sub["price_at_snapshot"] <= 1 - PRICE_THRESHOLD)
    first_extreme = (
        sub[extreme]
        .groupby("market_id")["snapshot_timestamp"]
        .min()
        .rename("first_extreme_ts")
    )

    # Pass B: last price change
    sub["prev_price"] = sub.groupby("market_id")["price_at_snapshot"].shift(1)
    sub["changed"] = (
        (sub["price_at_snapshot"] - sub["prev_price"].fillna(sub["price_at_snapshot"]))
        .abs() > 0.001
    )
    last_change = (
        sub[sub["changed"]]
        .groupby("market_id")["snapshot_timestamp"]
        .max()
        .rename("last_change_ts")
    )

    cutoffs = (
        pd.DataFrame({"market_id": sub["market_id"].unique()})
        .set_index("market_id")
        .join(first_extreme)
        .join(last_change)
    )

    # Prefer first_extreme; fall back to last_change; fall back to HARD_CUTOFF
    cutoffs["cutoff"] = (
        cutoffs["first_extreme_ts"]
        .fillna(cutoffs["last_change_ts"])
        .fillna(HARD_CUTOFF)
    ) + pd.Timedelta(days=INACTIVITY_DAYS)

    cutoffs["cutoff"] = cutoffs["cutoff"].clip(upper=HARD_CUTOFF)

    return cutoffs["cutoff"]


def filter_parquet(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    closed_times: pd.DataFrame,
) -> None:
    """
    Three-pass filter (writes to output_path — input_path is never modified):

      Pass 1 — closedTime:  drop rows after API-reported close for markets that have one.
      Pass 2 — price/inact: drop rows more than 14 days after the price first crossed
                            0.95/0.05 (or after the last price change). Applied to ALL
                            markets so that even closedTime markets don't keep months of
                            stale prices.
      Pass 3 — hard cap:    drop any remaining rows after 2026-05-01.
    """
    print(f"\n  Reading {input_path.name} ...")
    df = pd.read_parquet(input_path)
    original_rows = len(df)
    print(f"  Rows before: {original_rows:,}")

    df["snapshot_timestamp"] = pd.to_datetime(
        df["snapshot_timestamp"], format="mixed", utc=True
    )

    # ── Pass 1: closedTime ────────────────────────────────
    ct = closed_times.copy()
    ct["closed_time"] = pd.to_datetime(ct["closed_time"], utc=True, errors="coerce")
    populated = ct.dropna(subset=["closed_time"])
    print(f"  Markets with closedTime: {len(populated):,} / {len(ct):,}")

    df = df.merge(populated[["market_id", "closed_time"]], on="market_id", how="left")
    mask1    = df["closed_time"].isna() | (df["snapshot_timestamp"] <= df["closed_time"])
    removed1 = (~mask1).sum()
    df = df.loc[mask1].drop(columns=["closed_time"]).copy()

    # ── Pass 2: price-based cutoff for ALL markets ────────
    print(f"  Computing price-based cutoffs for {df['market_id'].nunique():,} markets...")
    cutoffs  = price_based_cutoffs(df)
    df       = df.merge(cutoffs.rename("price_cutoff").reset_index(), on="market_id", how="left")
    mask2    = df["snapshot_timestamp"] <= df["price_cutoff"]
    removed2 = (~mask2).sum()
    df       = df.loc[mask2].drop(columns=["price_cutoff"]).copy()

    # ── Pass 3: hard cap ─────────────────────────────────────
    mask3    = df["snapshot_timestamp"] <= HARD_CUTOFF
    removed3 = (~mask3).sum()
    df_clean = df.loc[mask3].copy()

    total_removed = original_rows - len(df_clean)
    print(f"  Removed — closedTime:  {removed1:,}")
    print(f"  Removed — price/inact: {removed2:,}")
    print(f"  Removed — hard cap:    {removed3:,}")
    print(f"  Total removed:         {total_removed:,} ({total_removed/original_rows:.2%})")
    print(f"  Rows after:            {len(df_clean):,}")
    print(f"  Max snapshot:          {df_clean['snapshot_timestamp'].max()}")

    print(f"  Writing {output_path.name} ...")
    df_clean.to_parquet(output_path, index=False)
    print(f"  Saved → {output_path}")


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  FIX LEAKAGE — REMOVE POST-RESOLUTION SNAPSHOTS")
    print("█" * 60)

    # Confirm originals exist
    for p in [META_CSV, PARQUET_MAIN, PARQUET_T1, PARQUET_T2]:
        if not p.exists():
            print(f"\nERROR: {p} not found.")
            return

    # Confirm we are NOT about to overwrite originals
    assert PARQUET_MAIN != PARQUET_MAIN_CLEAN, "SAFETY: output path equals input path"
    assert PARQUET_TRENDS_CLEAN != PARQUET_T1,  "SAFETY: output path equals input path"
    assert PARQUET_TRENDS_CLEAN != PARQUET_T2,  "SAFETY: output path equals input path"

    print("\n" + "=" * 60)
    print("STEP 1 — Fetch closedTime from Gamma API")
    print("=" * 60)

    if CLOSED_TIMES_CSV.exists():
        print(f"  Cache found — loading {CLOSED_TIMES_CSV.name}  (delete file to re-fetch)")
        closed_times = pd.read_csv(CLOSED_TIMES_CSV)
    else:
        meta = pd.read_csv(META_CSV)
        market_ids = meta["market_id"].tolist()
        print(f"  Markets in meta: {len(market_ids):,}")
        closed_times = fetch_closed_times(market_ids)

    populated = closed_times["closed_time"].notna().sum()
    print(f"\n  closedTime summary:")
    print(f"    Populated : {populated:,} ({populated/len(closed_times):.1%})")
    print(f"    Null      : {closed_times['closed_time'].isna().sum():,}")

    print("\n" + "=" * 60)
    print("STEP 2 — Filter parquet files")
    print("=" * 60)

    # Base dataset — filter directly
    filter_parquet(PARQUET_MAIN, PARQUET_MAIN_CLEAN, closed_times)

    # Trends dataset — part1 and part2 were split by row position so 449 markets
    # have their rows divided across both files.  Filtering them separately would
    # compute price_based_cutoffs on incomplete market histories, producing slightly
    # different cutoffs for split markets vs filtering the full history.
    # Fix: combine into one dataset, filter as a unit, save as a single clean file.
    print(f"\n  Combining {PARQUET_T1.name} + {PARQUET_T2.name} ...")
    t1 = pd.read_parquet(PARQUET_T1)
    t2 = pd.read_parquet(PARQUET_T2)
    trends_combined_path = DATA_DIR / "_trends_combined_tmp.parquet"
    pd.concat([t1, t2], ignore_index=True).to_parquet(trends_combined_path, index=False)
    del t1, t2
    print(f"  Combined rows: {pd.read_parquet(trends_combined_path, columns=['market_id']).shape[0]:,}")

    filter_parquet(trends_combined_path, PARQUET_TRENDS_CLEAN, closed_times)

    trends_combined_path.unlink()  # remove temp file

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Original files unchanged:")
    print(f"    {PARQUET_MAIN.name}")
    print(f"    {PARQUET_T1.name}")
    print(f"    {PARQUET_T2.name}")
    print(f"\n  Clean files written:")
    print(f"    {PARQUET_MAIN_CLEAN.name}")
    print(f"    {PARQUET_TRENDS_CLEAN.name}")
    print(f"\n  closedTime cache: {CLOSED_TIMES_CSV.name}")


if __name__ == "__main__":
    main()
