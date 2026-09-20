"""
Step 1: Fetch and Filter Markets
==================================
Fetches resolved binary markets from Polymarket Gamma API,
applies filters, and saves a clean market metadata file.

Output: polymarket_markets_meta.csv

Run this first, then run build_snapshots.py
"""

import requests
import pandas as pd
import numpy as np
import json
import time
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

# Server-side API filters
MARKET_FETCH_LIMIT  = 500
MAX_MARKETS         = None          # None = fetch all
VOLUME_NUM_MIN      = 1_000         # eliminates intraday/thin markets
START_DATE_MIN      = "2023-01-01"  # CLOB era only

# Client-side filters
MIN_DURATION_DAYS   = 30            # no upper bound
OUTCOME_THRESHOLD   = 0.95          # outcomePrices[0] >= this → YES, <= 0.05 → NO

SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES         = 5         # retries per offset on server error
RETRY_BACKOFF       = 5         # seconds to wait before first retry (doubles each attempt)

OUTPUT_META         = DATA_DIR / "polymarket_markets_meta.csv"


# ─────────────────────────────────────────────
# INCREMENTAL SUPPORT
# ─────────────────────────────────────────────

def load_existing_meta() -> tuple:
    """
    Returns (existing_df | None, fetch_from_date, split_cutoff | None).
    fetch_from_date: start_date_min to pass to the API.
    split_cutoff:    start_date of the first test market (freeze existing split).
    """
    if not OUTPUT_META.exists():
        return None, START_DATE_MIN, None

    df = pd.read_csv(OUTPUT_META)
    df["start_date"] = pd.to_datetime(df["start_date"], utc=True, errors="coerce")
    fetch_from   = (df["start_date"].max() - pd.Timedelta(days=7)).strftime("%Y-%m-%d")
    test_cutoff  = df[df["split"] == "test"]["start_date"].min() if "split" in df.columns else None
    split_cutoff = test_cutoff.strftime("%Y-%m-%d") if test_cutoff is not pd.NaT and test_cutoff is not None else None
    print(f"  Found {len(df):,} existing markets — fetching from {fetch_from}")
    return df, fetch_from, split_cutoff


# ─────────────────────────────────────────────
# HELPERS
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
# FETCH
# ─────────────────────────────────────────────

def _date_windows(start: str, end: str, months: int = 1):
    """Yield (window_start, window_end) string pairs in N-month chunks."""
    current = pd.Timestamp(start)
    target  = pd.Timestamp(end)
    while current < target:
        wend = min(current + pd.DateOffset(months=months), target)
        yield current.strftime("%Y-%m-%d"), wend.strftime("%Y-%m-%d")
        current = wend


def _fetch_window(start_date: str, end_date: str, depth: int = 0) -> list[dict]:
    """Offset-paginate one date window.

    If the API returns 422 (offset cap hit), automatically splits the window
    in half and recurses — handles high-volume months like election periods.
    Max recursion depth of 6 gives windows as small as ~half a day.
    """
    MAX_DEPTH = 6
    markets   = []
    offset    = 0
    failures  = 0

    while True:
        params = {
            "closed":         "true",
            "limit":          MARKET_FETCH_LIMIT,
            "offset":         offset,
            "order":          "startDate",
            "ascending":      "true",
            "volume_num_min": VOLUME_NUM_MIN,
            "start_date_min": start_date,
            "start_date_max": end_date,
        }

        cap_hit = False
        batch   = None

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(f"{GAMMA_URL}/markets", params=params, timeout=20)
                if r.status_code == 422:
                    cap_hit = True
                    break
                r.raise_for_status()
                batch   = r.json()
                failures = 0
                break
            except Exception as e:
                if hasattr(e, "response") and getattr(e.response, "status_code", None) == 422:
                    cap_hit = True
                    break
                wait = RETRY_BACKOFF * (2 ** (attempt - 1))
                if attempt < MAX_RETRIES:
                    print(f"    offset {offset} attempt {attempt}/{MAX_RETRIES}: {e} — retry in {wait}s")
                    time.sleep(wait)
                else:
                    print(f"    offset {offset}: all retries failed — skipping")

        if cap_hit:
            if depth < MAX_DEPTH:
                # Split window in half and recurse
                mid = (pd.Timestamp(start_date) +
                       (pd.Timestamp(end_date) - pd.Timestamp(start_date)) / 2
                       ).strftime("%Y-%m-%d")
                if mid <= start_date:
                    print(f"    WARNING: window too small to split: {start_date} → {end_date}")
                    break
                indent = "  " * depth
                print(f"    {indent}↳ cap hit — splitting {start_date}→{end_date} at {mid}")
                left  = _fetch_window(start_date, mid, depth + 1)
                right = _fetch_window(mid, end_date, depth + 1)
                return markets + left + right
            else:
                print(f"    WARNING: hit cap at max depth ({MAX_DEPTH}) for {start_date}→{end_date}")
            break

        if batch is None:
            failures += 1
            if failures >= 3:
                break
            offset += MARKET_FETCH_LIMIT
            continue

        if not batch:
            break

        markets.extend(batch)
        offset += MARKET_FETCH_LIMIT
        time.sleep(SLEEP_BETWEEN_CALLS)

    return markets


def fetch_all_markets(start_date_min: str = START_DATE_MIN) -> list[dict]:
    """Fetch all markets using monthly date windows to avoid the API 2500-record offset cap."""
    from datetime import date as _date
    today   = _date.today().strftime("%Y-%m-%d")
    windows = list(_date_windows(start_date_min, today, months=1))

    print("=" * 60)
    print("STEP 1: Fetching markets from Gamma API")
    print("=" * 60)
    print(f"  volume_num_min : {VOLUME_NUM_MIN:,}")
    print(f"  windows        : {len(windows)} monthly ({start_date_min} → {today})\n")

    all_markets = []
    seen_ids    = set()

    for i, (wstart, wend) in enumerate(windows, 1):
        batch = _fetch_window(wstart, wend)
        new   = []
        for m in batch:
            mid = m.get("conditionId") or str(m.get("id"))
            if mid not in seen_ids:
                seen_ids.add(mid)
                new.append(m)
        all_markets.extend(new)
        if new or batch:
            print(f"  [{i:>3}/{len(windows)}] {wstart} → {wend}  "
                  f"+{len(new):>4} markets  total {len(all_markets):>6}")

        if MAX_MARKETS and len(all_markets) >= MAX_MARKETS:
            all_markets = all_markets[:MAX_MARKETS]
            print(f"  Hit MAX_MARKETS cap ({MAX_MARKETS}).")
            break

    print(f"\nTotal markets fetched: {len(all_markets):,}")
    return all_markets


# ─────────────────────────────────────────────
# FILTER AND PARSE
# ─────────────────────────────────────────────


def parse_and_filter_markets(raw_markets: list[dict]) -> pd.DataFrame:
    print("\n" + "=" * 60)
    print("STEP 2: Parsing and filtering markets")
    print("=" * 60)

    records = []
    skipped = {
        "not_binary":        0,
        "bad_dates":         0,
        "negative_duration": 0,
        "too_short":         0,
        "no_outcome_field":  0,
        "ambiguous_outcome": 0,
        "no_clob_token":     0,
    }

    for m in raw_markets:

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
            "market_id":       m.get("conditionId") or str(m.get("id")),
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
    print(f"  {'Passing markets':<32}: {len(df):>6,}")

    if len(df) > 0:
        print(f"\n  Outcome:  YES={df['outcome'].mean():.1%}  NO={(1-df['outcome'].mean()):.1%}")
        print(f"  Duration: min={df['duration_days'].min()}d  "
              f"median={df['duration_days'].median():.0f}d  "
              f"max={df['duration_days'].max()}d")
        print(f"  Dates:    {df['start_date'].min().date()} → {df['start_date'].max().date()}")
        print(f"  Volume:   min=${df['total_volume'].min():,.0f}  "
              f"median=${df['total_volume'].median():,.0f}  "
              f"max=${df['total_volume'].max():,.0f}")

    return df


# ─────────────────────────────────────────────
# MAIN
# ─────────────────────────────────────────────

def main():
    print("\n" + "█" * 60)
    print("  POLYMARKET — FETCH & FILTER MARKETS")
    print("█" * 60 + "\n")

    existing_df, fetch_from, split_cutoff = load_existing_meta()

    raw_markets = fetch_all_markets(start_date_min=fetch_from)

    new_df = parse_and_filter_markets(raw_markets)
    if new_df.empty and existing_df is None:
        print("No markets passed filters. Exiting.")
        return

    if existing_df is not None:
        known_ids = set(existing_df["market_id"])
        new_df = new_df[~new_df["market_id"].isin(known_ids)].reset_index(drop=True)
        print(f"\n  {len(new_df):,} genuinely new markets after dedup")

        if new_df.empty:
            print("  Nothing new — output unchanged.")
            return

        # New markets are temporally after the existing split cutoff → assign test
        if split_cutoff:
            cutoff_ts = pd.Timestamp(split_cutoff, tz="UTC")
            new_df["split"] = new_df["start_date"].apply(
                lambda d: "test" if d >= cutoff_ts else "train"
            )
        else:
            new_df["split"] = "test"

        markets_df = pd.concat([existing_df, new_df], ignore_index=True)
        markets_df = markets_df.sort_values("start_date").reset_index(drop=True)
    else:
        # First run — compute 80/20 split on full corpus
        markets_df = new_df.sort_values("start_date").reset_index(drop=True)
        n           = len(markets_df)
        cutoff_idx  = int(n * 0.8)
        cutoff_date = markets_df.iloc[cutoff_idx]["start_date"]
        markets_df["split"] = ["train"] * cutoff_idx + ["test"] * (n - cutoff_idx)
        print(f"  Split cutoff: {cutoff_date.date()} — train={cutoff_idx:,}  test={n-cutoff_idx:,}")

    markets_df.to_csv(OUTPUT_META, index=False)
    print(f"\nSaved: {OUTPUT_META}")
    print(f"  Markets: {len(markets_df):,}")
    print(f"  Columns: {list(markets_df.columns)}")
    print(f"\nNext: run build_snapshots.py")


if __name__ == "__main__":
    main()