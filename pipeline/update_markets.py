"""
Step 1b: Incremental Market Update
=====================================
Reads the most recent start_date from polymarket_markets_meta.csv
and fetches only markets created after that date.
Appends new markets to the existing CSV without touching old ones.

Run after fetch_markets.py has been run at least once.
Then run build_snapshots.py to fetch price history for the new markets.
"""

import requests
import pandas as pd
import numpy as np
import json
import time
import os
from pathlib import Path
from typing import Optional

# ─────────────────────────────────────────────
# PATHS
# ─────────────────────────────────────────────

ROOT     = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

# ─────────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────────

GAMMA_URL = "https://gamma-api.polymarket.com"

MARKET_FETCH_LIMIT  = 500
MAX_MARKETS         = None
VOLUME_NUM_MIN      = 1_000
MIN_DURATION_DAYS   = 30
OUTCOME_THRESHOLD   = 0.95
SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES         = 5
RETRY_BACKOFF       = 5

OUTPUT_META         = DATA_DIR / "polymarket_markets_meta.csv"


# ─────────────────────────────────────────────
# HELPERS (shared with fetch_markets.py)
# ─────────────────────────────────────────────

def parse_json_field(value, default=None):
    if value is None:
        return default
    if isinstance(value, (list, dict)):
        return value
    try:
        return json.loads(value)
    except:
        return default

def parse_dt(value) -> Optional[pd.Timestamp]:
    if not value:
        return None
    try:
        ts = pd.to_datetime(value, utc=True)
        return ts if not pd.isna(ts) else None
    except:
        return None


# ─────────────────────────────────────────────
# GET CUTOFF DATE FROM EXISTING CSV
# ─────────────────────────────────────────────

def get_cutoff_date() -> str:
    """
    Read the most recent start_date from the existing meta CSV.
    Returns it as a date string (YYYY-MM-DD) to use as start_date_min.
    """
    if not OUTPUT_META.exists():
        print(f"ERROR: {OUTPUT_META} not found. Run fetch_markets.py first.")
        return None

    existing = pd.read_csv(OUTPUT_META, usecols=["market_id", "start_date"])
    existing["start_date"] = pd.to_datetime(existing["start_date"], format="ISO8601", utc=True)
    cutoff = existing["start_date"].max()
    cutoff_str = cutoff.strftime("%Y-%m-%d")

    print(f"  Existing markets: {len(existing):,}")
    print(f"  Most recent start_date: {cutoff_str}")
    print(f"  Fetching markets created after: {cutoff_str}")

    return cutoff_str, set(existing["market_id"].unique())


# ─────────────────────────────────────────────
# FETCH NEW MARKETS
# ─────────────────────────────────────────────

def fetch_new_markets(start_date_min: str) -> list[dict]:
    print("\n" + "=" * 60)
    print("Fetching new markets from Gamma API")
    print("=" * 60)
    print(f"  start_date_min : {start_date_min}")
    print(f"  volume_num_min : {VOLUME_NUM_MIN:,}\n")

    all_markets = []
    offset = 0
    consecutive_failures = 0
    MAX_CONSECUTIVE_FAILURES = 3

    while True:
        params = {
            "closed":         "true",
            "limit":          MARKET_FETCH_LIMIT,
            "offset":         offset,
            "order":          "startDate",
            "ascending":      "true",
            "volume_num_min": VOLUME_NUM_MIN,
            "start_date_min": start_date_min,
        }

        batch = None
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(f"{GAMMA_URL}/markets", params=params, timeout=20)
                r.raise_for_status()
                batch = r.json()
                break
            except Exception as e:
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                if attempt < MAX_RETRIES:
                    print(f"  Error at offset {offset} (attempt {attempt}/{MAX_RETRIES}): {e} "
                          f"— retrying in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"  Error at offset {offset} (attempt {attempt}/{MAX_RETRIES}): {e} "
                          f"— skipping offset.")

        if batch is None:
            consecutive_failures += 1
            if consecutive_failures >= MAX_CONSECUTIVE_FAILURES:
                print(f"  {consecutive_failures} consecutive failures — stopping.")
                break
            offset += MARKET_FETCH_LIMIT
            continue

        consecutive_failures = 0

        if not batch:
            print(f"  Empty batch at offset {offset} — done.")
            break

        all_markets.extend(batch)

        oldest = batch[0].get("startDate", "?")[:10]
        newest = batch[-1].get("startDate", "?")[:10]
        print(f"  offset {offset:>6} | batch {len(batch):>4} | "
              f"dates {oldest} → {newest} | total {len(all_markets):>6}")

        if MAX_MARKETS and len(all_markets) >= MAX_MARKETS:
            all_markets = all_markets[:MAX_MARKETS]
            break

        offset += MARKET_FETCH_LIMIT
        time.sleep(SLEEP_BETWEEN_CALLS)

    print(f"\nTotal fetched: {len(all_markets):,}")
    return all_markets


# ─────────────────────────────────────────────
# PARSE AND FILTER
# ─────────────────────────────────────────────

def parse_and_filter_markets(raw_markets: list[dict],
                              existing_ids: set) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("Parsing and filtering new markets")
    print("=" * 60)

    records = []
    skipped = {
        "already_exists":    0,
        "not_binary":        0,
        "bad_dates":         0,
        "negative_duration": 0,
        "too_short":         0,
        "no_outcome_field":  0,
        "ambiguous_outcome": 0,
        "no_clob_token":     0,
    }

    for m in raw_markets:

        # Skip if already in existing CSV
        market_id = m.get("conditionId") or str(m.get("id"))
        if market_id in existing_ids:
            skipped["already_exists"] += 1
            continue

        # Binary check
        outcomes = parse_json_field(m.get("outcomes"), [])
        if len(outcomes) != 2:
            skipped["not_binary"] += 1
            continue

        # Dates
        start = parse_dt(m.get("createdAt"))
        end   = parse_dt(m.get("endDate"))
        if start is None or end is None:
            skipped["bad_dates"] += 1
            continue

        duration_days = (end - start).days
        if duration_days <= 0:
            skipped["negative_duration"] += 1
            continue
        if duration_days < MIN_DURATION_DAYS:
            skipped["too_short"] += 1
            continue

        # Outcome
        op = parse_json_field(m.get("outcomePrices"), [])
        if not op or len(op) < 2:
            skipped["no_outcome_field"] += 1
            continue
        try:
            yes_price = float(op[0])
        except:
            skipped["no_outcome_field"] += 1
            continue

        if yes_price >= OUTCOME_THRESHOLD:
            outcome = 1
        elif yes_price <= (1 - OUTCOME_THRESHOLD):
            outcome = 0
        else:
            skipped["ambiguous_outcome"] += 1
            continue

        # CLOB token
        clob_tokens = parse_json_field(m.get("clobTokenIds"), [])
        if not clob_tokens:
            skipped["no_clob_token"] += 1
            continue

        records.append({
            "market_id":       market_id,
            "clob_token_id":   clob_tokens[0],
            "question":        m.get("question", ""),
            "category":        m.get("category", ""),
            "start_date":      start,
            "end_date":        end,
            "duration_days":   duration_days,
            "total_volume":    float(m.get("volumeNum") or 0),
            "yes_final_price": round(yes_price, 6),
            "outcome":         outcome,
        })

    df = pd.DataFrame(records)

    print(f"\nFilter results:")
    print(f"  Input:                        {len(raw_markets):>7,}")
    for reason, count in skipped.items():
        print(f"  Skipped ({reason:<24}): {count:>6,}")
    print(f"  {'New passing markets':<32}: {len(df):>6,}")

    if len(df) > 0:
        print(f"\n  Outcome:  YES={df['outcome'].mean():.1%}  NO={(1-df['outcome'].mean()):.1%}")
        print(f"  Duration: min={df['duration_days'].min()}d  "
              f"median={df['duration_days'].median():.0f}d  "
              f"max={df['duration_days'].max()}d")
        print(f"  Dates:    {df['start_date'].min().date()} → {df['start_date'].max().date()}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  POLYMARKET — INCREMENTAL MARKET UPDATE")
    print("█" * 60 + "\n")

    # Get cutoff date and existing IDs from CSV
    result = get_cutoff_date()
    if result is None:
        return
    cutoff_str, existing_ids = result

    # Fetch new markets
    raw_markets = fetch_new_markets(cutoff_str)
    if not raw_markets:
        print("No new markets fetched.")
        return

    # Filter — skip already existing market IDs
    new_df = parse_and_filter_markets(raw_markets, existing_ids)
    if new_df.empty:
        print("\nNo new markets to add.")
        return

    # Assign split — new markets get random assignment consistent with 80/20
    np.random.seed(None)  # no fixed seed for incremental updates
    new_df["split"] = np.where(
        np.random.rand(len(new_df)) < 0.8, "train", "test"
    )

    # Append to existing CSV
    new_df.to_csv(OUTPUT_META, mode="a", header=False, index=False)

    print(f"\nAppended {len(new_df):,} new markets to {OUTPUT_META}")
    print(f"  Total markets now: {len(existing_ids) + len(new_df):,}")
    print(f"\nNext: run build_snapshots.py to fetch price history for new markets")


if __name__ == "__main__":
    main()