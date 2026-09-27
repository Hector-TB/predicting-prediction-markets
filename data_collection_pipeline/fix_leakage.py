"""
fix_leakage.py
==============
Fetches closedTime for every market in the snapshot parquet, then removes
post-resolution and already-settled snapshots from all parquet files (ADR-005, ADR-014).

SAFETY: originals are NEVER modified. All output goes to *_clean.parquet files.

closedTime is looked up by conditionId in batches of 100 (~450 calls for 45k
markets) and cached in market_closed_times.csv; later runs only fetch markets
missing from the cache.

Run from the project root:
    python data_collection_pipeline/fix_leakage.py
"""

import logging
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
META_CSV       = DATA_DIR / "polymarket_markets_meta.csv"
PARQUET_MAIN   = DATA_DIR / "polymarket_ml_dataset.parquet"
PARQUET_TRENDS = DATA_DIR / "polymarket_ml_dataset_with_trends.parquet"

# New output files — originals are untouched
CLOSED_TIMES_CSV        = DATA_DIR / "market_closed_times.csv"
PARQUET_MAIN_CLEAN      = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
PARQUET_TRENDS_CLEAN    = DATA_DIR / "polymarket_ml_dataset_with_trends_clean.parquet"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GAMMA_URL           = "https://gamma-api.polymarket.com"
BATCH_SIZE          = 100       # condition_ids per request (Gamma page cap)
SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES         = 8         # Gamma has multi-minute 500 outages (ADR-013)
RETRY_BACKOFF       = 5         # seconds before first retry (doubles each attempt)
RETRY_BACKOFF_MAX   = 60

log = logging.getLogger(__name__)


# ─────────────────────────────────────────────
# STEP 1 — FETCH closedTime VIA PAGINATION
# ─────────────────────────────────────────────

def _fetch_closed_batch(ids: list[str]) -> dict[str, str | None]:
    """Look up closedTime for up to BATCH_SIZE conditionIds.

    `closed=true` is required: without it Gamma returns nothing for resolved markets.
    Raises RuntimeError after MAX_RETRIES so a gap is never silently cached (ADR-013).
    """
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            r = requests.get(
                f"{GAMMA_URL}/markets",
                params={"condition_ids": ids, "closed": "true", "limit": BATCH_SIZE},
                timeout=30,
            )
            r.raise_for_status()
            return {m["conditionId"]: m.get("closedTime") for m in r.json() if m.get("conditionId")}
        except Exception as e:
            if attempt == MAX_RETRIES:
                raise RuntimeError(f"Gamma closedTime lookup failed {MAX_RETRIES}x: {e}") from e
            wait = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
            log.warning("  closedTime batch attempt %d/%d: %s — retry in %ds",
                        attempt, MAX_RETRIES, e, wait)
            time.sleep(wait)


def fetch_closed_times(market_ids: list[str]) -> pd.DataFrame:
    """
    Return [market_id, closed_time] for every id in market_ids.

    Markets already in CLOSED_TIMES_CSV with a populated closed_time are reused;
    the rest are looked up by conditionId in batches. Markets Gamma does not
    return keep closed_time = None and are retried on the next run.
    """
    cached: dict[str, str] = {}
    if CLOSED_TIMES_CSV.exists():
        prev = pd.read_csv(CLOSED_TIMES_CSV).dropna(subset=["closed_time"])
        cached = dict(zip(prev["market_id"], prev["closed_time"]))

    todo = [mid for mid in market_ids if mid not in cached]
    log.info("  closedTime cache: %s cached, %s to fetch", f"{len(cached):,}", f"{len(todo):,}")

    fetched: dict[str, str | None] = {}
    for i in range(0, len(todo), BATCH_SIZE):
        fetched.update(_fetch_closed_batch(todo[i:i + BATCH_SIZE]))
        done = min(i + BATCH_SIZE, len(todo))
        if (i // BATCH_SIZE) % 50 == 0 or done == len(todo):
            log.info("    %s / %s looked up", f"{done:,}", f"{len(todo):,}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    rows = [
        {"market_id": mid, "closed_time": cached.get(mid) or fetched.get(mid)}
        for mid in market_ids
    ]
    result = pd.DataFrame(rows)
    result["closed_time"] = pd.to_datetime(result["closed_time"], utc=True, format="mixed", errors="coerce")
    result.to_csv(CLOSED_TIMES_CSV, index=False)
    log.info("  Saved %s", CLOSED_TIMES_CSV.name)
    return result


# ─────────────────────────────────────────────
# STEP 2 — FILTER PARQUET
# ─────────────────────────────────────────────

PRICE_THRESHOLD  = 0.95                                                          # settled if price >= 0.95 or <= 0.05
INACTIVITY_DAYS  = 14                                                            # keep N days after last price change
HARD_CUTOFF      = (pd.Timestamp.now(tz="UTC") + pd.Timedelta(days=1)).normalize()  # tomorrow midnight — includes today


def inactivity_cutoffs(df: pd.DataFrame) -> pd.Series:
    """
    Per-market cutoff = last snapshot where the price changed by > 0.001,
    plus INACTIVITY_DAYS. Markets with no detectable change fall back to
    HARD_CUTOFF. Returns a Series indexed by market_id.
    """
    sub = df.sort_values(["market_id", "snapshot_timestamp"])
    prev = sub.groupby("market_id")["price_at_snapshot"].shift(1)
    changed = (sub["price_at_snapshot"] - prev.fillna(sub["price_at_snapshot"])).abs() > 0.001
    last_change = sub[changed].groupby("market_id")["snapshot_timestamp"].max()

    cutoff = (
        last_change.reindex(pd.Index(sub["market_id"].unique(), name="market_id"))
        .fillna(HARD_CUTOFF)
        + pd.Timedelta(days=INACTIVITY_DAYS)
    )
    return cutoff.clip(upper=HARD_CUTOFF)


def filter_snapshots(df: pd.DataFrame, closed_times: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """
    Four-pass leakage filter (ADR-005, revised by ADR-014). Returns (clean_df, removed_counts).

      Pass 1 — closedTime:  drop rows after the API-reported close.
      Pass 2 — settled:     drop rows whose own price is >= 0.95 or <= 0.05. Decided per
                            row from the snapshot price alone, so no future information
                            selects which rows survive and live scoring can apply the
                            same rule.
      Pass 3 — inactivity:  markets WITHOUT a closedTime only — drop rows more than
                            INACTIVITY_DAYS after the last price change.
      Pass 4 — hard cap:    drop rows after HARD_CUTOFF (tomorrow midnight UTC).
    """
    df = df.copy()
    df["snapshot_timestamp"] = pd.to_datetime(df["snapshot_timestamp"], format="mixed", utc=True)

    ct = closed_times[["market_id", "closed_time"]].copy()
    ct["closed_time"] = pd.to_datetime(ct["closed_time"], utc=True, format="mixed", errors="coerce")
    df = df.merge(ct.dropna(subset=["closed_time"]), on="market_id", how="left")
    removed = {}

    # ── Pass 1: closedTime ────────────────────────────────
    keep = df["closed_time"].isna() | (df["snapshot_timestamp"] <= df["closed_time"])
    removed["closedTime"] = int((~keep).sum())
    df = df.loc[keep]

    # ── Pass 2: settled price at the snapshot itself ──────
    settled = (df["price_at_snapshot"] >= PRICE_THRESHOLD) | \
              (df["price_at_snapshot"] <= 1 - PRICE_THRESHOLD)
    removed["settled price"] = int(settled.sum())
    df = df.loc[~settled]

    # ── Pass 3: inactivity, only where closedTime is unknown ─
    no_ct = df["closed_time"].isna()
    if no_ct.any():
        cutoffs = inactivity_cutoffs(df.loc[no_ct]).rename("inactive_cutoff")
        df = df.merge(cutoffs.reset_index(), on="market_id", how="left")
        keep = df["inactive_cutoff"].isna() | (df["snapshot_timestamp"] <= df["inactive_cutoff"])
        df = df.loc[keep].drop(columns=["inactive_cutoff"])
        removed["inactivity"] = int((~keep).sum())
    else:
        removed["inactivity"] = 0

    # ── Pass 4: hard cap ──────────────────────────────────
    keep = df["snapshot_timestamp"] <= HARD_CUTOFF
    removed["hard cap"] = int((~keep).sum())
    df = df.loc[keep].drop(columns=["closed_time"]).reset_index(drop=True)

    return df, removed


def filter_parquet(
    input_path: pathlib.Path,
    output_path: pathlib.Path,
    closed_times: pd.DataFrame,
) -> None:
    """Apply filter_snapshots to input_path and write output_path (input is never modified)."""
    print(f"\n  Reading {input_path.name} ...")
    df = pd.read_parquet(input_path)
    original_rows = len(df)
    print(f"  Rows before: {original_rows:,}")
    print(f"  Markets with closedTime: {closed_times['closed_time'].notna().sum():,} / {len(closed_times):,}")

    df_clean, removed = filter_snapshots(df, closed_times)

    total_removed = original_rows - len(df_clean)
    for name, n in removed.items():
        print(f"  Removed — {name + ':':<15} {n:,}")
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
    for p in [META_CSV, PARQUET_MAIN, PARQUET_TRENDS]:
        if not p.exists():
            print(f"\nERROR: {p} not found.")
            return

    # Confirm we are NOT about to overwrite originals
    assert PARQUET_MAIN   != PARQUET_MAIN_CLEAN,   "SAFETY: output path equals input path"
    assert PARQUET_TRENDS != PARQUET_TRENDS_CLEAN, "SAFETY: output path equals input path"

    print("\n" + "=" * 60)
    print("STEP 1 — Fetch closedTime from Gamma API")
    print("=" * 60)

    # Read market IDs from the parquet so we cover all markets, not just the meta CSV
    market_ids = (
        pd.read_parquet(PARQUET_MAIN, columns=["market_id"])["market_id"]
        .unique().tolist()
    )
    print(f"  Markets in parquet: {len(market_ids):,}")
    closed_times = fetch_closed_times(market_ids)

    populated = closed_times["closed_time"].notna().sum()
    print(f"\n  closedTime summary:")
    print(f"    Populated : {populated:,} ({populated/len(closed_times):.1%})")
    print(f"    Null      : {closed_times['closed_time'].isna().sum():,}")

    print("\n" + "=" * 60)
    print("STEP 2 — Filter parquet files")
    print("=" * 60)

    # Base dataset
    filter_parquet(PARQUET_MAIN, PARQUET_MAIN_CLEAN, closed_times)

    # Trends dataset (single merged file produced by merge_trends.py)
    filter_parquet(PARQUET_TRENDS, PARQUET_TRENDS_CLEAN, closed_times)

    print("\n" + "=" * 60)
    print("  DONE")
    print("=" * 60)
    print(f"  Original files unchanged:")
    print(f"    {PARQUET_MAIN.name}")
    print(f"    {PARQUET_TRENDS.name}")
    print(f"\n  Clean files written:")
    print(f"    {PARQUET_MAIN_CLEAN.name}")
    print(f"    {PARQUET_TRENDS_CLEAN.name}")
    print(f"\n  closedTime cache: {CLOSED_TIMES_CSV.name}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
