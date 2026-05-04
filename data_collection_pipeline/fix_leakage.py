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
CLOSED_TIMES_CSV   = DATA_DIR / "market_closed_times.csv"
PARQUET_MAIN_CLEAN = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
PARQUET_T1_CLEAN   = DATA_DIR / "polymarket_ml_dataset_with_trends_part1_clean.parquet"
PARQUET_T2_CLEAN   = DATA_DIR / "polymarket_ml_dataset_with_trends_part2_clean.parquet"

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

def filter_parquet(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    closed_times: pd.DataFrame,
) -> None:
    """
    Remove rows where snapshot_timestamp > closedTime for that market.
    Markets with null closedTime are kept entirely.
    Writes to output_path — input_path is never modified.
    """
    print(f"\n  Reading {input_path.name} ...")
    df = pd.read_parquet(input_path)
    original_rows = len(df)
    print(f"  Rows before: {original_rows:,}")

    df["snapshot_timestamp"] = pd.to_datetime(
        df["snapshot_timestamp"], format="mixed", utc=True
    )

    # Build market_id → closed_time lookup
    ct = closed_times.copy()
    ct["closed_time"] = pd.to_datetime(ct["closed_time"], utc=True, errors="coerce")
    populated = ct.dropna(subset=["closed_time"])
    print(f"  Markets with closedTime populated: {len(populated):,} "
          f"/ {len(ct):,}")

    # Vectorised filter: merge closed_time onto df, then drop rows past it
    df = df.merge(
        populated[["market_id", "closed_time"]],
        on="market_id",
        how="left",
    )
    # Keep row if: no closedTime for this market  OR  snapshot is before closedTime
    mask_keep = df["closed_time"].isna() | (df["snapshot_timestamp"] <= df["closed_time"])
    removed   = (~mask_keep).sum()

    df_clean = df.loc[mask_keep].drop(columns=["closed_time"]).copy()

    print(f"  Rows removed (post-closedTime): {removed:,} ({removed/original_rows:.2%})")
    print(f"  Rows after:  {len(df_clean):,}")
    print(f"  Markets affected: {df.loc[~mask_keep, 'market_id'].nunique():,}")

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
    for src, dst in [
        (PARQUET_MAIN, PARQUET_MAIN_CLEAN),
        (PARQUET_T1,   PARQUET_T1_CLEAN),
        (PARQUET_T2,   PARQUET_T2_CLEAN),
    ]:
        assert src != dst, f"SAFETY: output path equals input path for {src}"

    print("\n" + "=" * 60)
    print("STEP 1 — Fetch closedTime from Gamma API")
    print("=" * 60)

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

    for src, dst in [
        (PARQUET_MAIN, PARQUET_MAIN_CLEAN),
        (PARQUET_T1,   PARQUET_T1_CLEAN),
        (PARQUET_T2,   PARQUET_T2_CLEAN),
    ]:
        filter_parquet(src, dst, closed_times)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Original files unchanged:")
    print(f"    {PARQUET_MAIN.name}")
    print(f"    {PARQUET_T1.name}")
    print(f"    {PARQUET_T2.name}")
    print(f"\n  Clean files written:")
    print(f"    {PARQUET_MAIN_CLEAN.name}")
    print(f"    {PARQUET_T1_CLEAN.name}")
    print(f"    {PARQUET_T2_CLEAN.name}")
    print(f"\n  closedTime cache: {CLOSED_TIMES_CSV.name}")


if __name__ == "__main__":
    main()
