"""
Step 1: Fetch and Filter Markets
==================================
Fetches resolved binary markets from Polymarket Gamma API,
applies filters, and saves a clean market metadata file.

Output: polymarket_markets_meta.csv

Run this first, then run build_snapshots.py
"""

import argparse
import logging
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
MARKET_FETCH_LIMIT  = 100         # Gamma returns at most 100 rows per page
MAX_MARKETS         = None          # None = fetch all
VOLUME_NUM_MIN      = 1_000         # eliminates intraday/thin markets
START_DATE_MIN      = "2023-01-01"  # CLOB era only

# Client-side filters
MIN_DURATION_DAYS   = 30            # no upper bound
OUTCOME_THRESHOLD   = 0.95          # outcomePrices[0] >= this → YES, <= 0.05 → NO

SLEEP_BETWEEN_CALLS = 0.15
MAX_RETRIES         = 8         # attempts per page; Gamma has multi-minute 500 outages
RETRY_BACKOFF       = 5         # seconds before first retry (doubles each attempt)
RETRY_BACKOFF_MAX   = 60        # cap per wait → ~4 min total before failing loudly

OUTPUT_META         = DATA_DIR / "polymarket_markets_meta.csv"
FETCH_CACHE_DIR     = DATA_DIR / "fetch_cache"   # per-window checkpoints for --full

log = logging.getLogger(__name__)

META_COLUMNS = [
    "market_id", "clob_token_id", "question", "category", "start_date", "end_date",
    "duration_days", "total_volume", "yes_final_price", "outcome",
]


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


def _fetch_window(start_date: str, end_date: str) -> list[dict]:
    """Cursor-paginate one date window via /markets/keyset (ADR-013).

    Raises RuntimeError if a page still fails after MAX_RETRIES — a skipped
    page would leave a gap the incremental watermark never revisits.
    """
    markets = []
    cursor  = None

    while True:
        params = {
            "closed":         "true",
            "limit":          MARKET_FETCH_LIMIT,
            "volume_num_min": VOLUME_NUM_MIN,
            "start_date_min": start_date,
            "start_date_max": end_date,
        }
        if cursor:
            params["after_cursor"] = cursor

        for attempt in range(1, MAX_RETRIES + 1):
            try:
                r = requests.get(f"{GAMMA_URL}/markets/keyset", params=params, timeout=30)
                r.raise_for_status()
                page = r.json()
                break
            except Exception as e:
                if attempt == MAX_RETRIES:
                    raise RuntimeError(
                        f"Gamma keyset page failed {MAX_RETRIES}x for "
                        f"{start_date} → {end_date} (cursor={cursor}): {e}"
                    ) from e
                wait = min(RETRY_BACKOFF * (2 ** (attempt - 1)), RETRY_BACKOFF_MAX)
                log.warning("    %s → %s attempt %d/%d: %s — retry in %ds",
                            start_date, end_date, attempt, MAX_RETRIES, e, wait)
                time.sleep(wait)

        batch = page.get("markets") or []
        markets.extend(batch)
        cursor = page.get("next_cursor")
        if not batch or not cursor:
            return markets
        time.sleep(SLEEP_BETWEEN_CALLS)


def fetch_filtered_markets(start_date_min: str, cache_dir: Optional[Path] = None) -> pd.DataFrame:
    """Fetch monthly windows from start_date_min to today and filter each as it arrives.

    Filtering per window keeps memory flat: most raw markets are short-duration
    and discarded immediately. With cache_dir set, each fully elapsed window is
    written to a CSV and reused on the next run, so an interrupted full fetch resumes.
    """
    today   = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    windows = list(_date_windows(start_date_min, today, months=1))

    log.info("=" * 60)
    log.info("STEP 1: Fetching markets from Gamma API (keyset pagination)")
    log.info("=" * 60)
    log.info("  volume_num_min : %s", f"{VOLUME_NUM_MIN:,}")
    log.info("  windows        : %d monthly (%s → %s)", len(windows), start_date_min, today)
    if cache_dir:
        cache_dir.mkdir(parents=True, exist_ok=True)
        log.info("  checkpoints    : %s", cache_dir)

    frames = []
    for i, (wstart, wend) in enumerate(windows, 1):
        cache_file = cache_dir / f"{wstart}_{wend}.csv" if cache_dir else None
        if cache_file and cache_file.exists():
            df = pd.read_csv(cache_file)
            df["start_date"] = pd.to_datetime(df["start_date"], utc=True, format="ISO8601")
            df["end_date"]   = pd.to_datetime(df["end_date"], utc=True, format="ISO8601")
            log.info("  [%3d/%d] %s → %s  %5d passing (cached)", i, len(windows), wstart, wend, len(df))
        else:
            raw = _fetch_window(wstart, wend)
            df  = parse_and_filter_markets(raw, verbose=False)
            # Only checkpoint windows that have fully elapsed; the current one can still grow.
            if cache_file and wend < today:
                df.to_csv(cache_file, index=False)
            log.info("  [%3d/%d] %s → %s  %7d fetched  %5d passing",
                     i, len(windows), wstart, wend, len(raw), len(df))
        frames.append(df)

    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame()
    markets_df = pd.concat(frames, ignore_index=True)
    # Window bounds are inclusive on both ends, so boundary markets can appear twice.
    markets_df = markets_df.drop_duplicates(subset=["market_id"], keep="first").reset_index(drop=True)
    if MAX_MARKETS:
        markets_df = markets_df.head(MAX_MARKETS)
    log.info("\nTotal markets passing filters: %s", f"{len(markets_df):,}")
    return markets_df


# ─────────────────────────────────────────────
# FILTER AND PARSE
# ─────────────────────────────────────────────


def parse_and_filter_markets(raw_markets: list[dict], verbose: bool = True) -> pd.DataFrame:
    if verbose:
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

    df = pd.DataFrame(records, columns=META_COLUMNS)
    if not verbose:
        return df

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

def main(full: bool = False):
    print("\n" + "█" * 60)
    print("  POLYMARKET — FETCH & FILTER MARKETS")
    print("█" * 60 + "\n")

    existing_df, fetch_from, split_cutoff = load_existing_meta()
    if full:
        # Re-fetch everything (ADR-013). Existing rows still win on merge below,
        # so LLM categories and previously stored markets are preserved.
        fetch_from = START_DATE_MIN
        log.info("  --full: re-fetching from %s with checkpoints", fetch_from)

    new_df = fetch_filtered_markets(fetch_from, cache_dir=FETCH_CACHE_DIR if full else None)
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
    if full and existing_df is not None:
        log.warning("\nFull re-fetch merged under the OLD frozen split. "
                    "Run scripts/recompute_split.py before build_snapshots.py (ADR-013).")
    else:
        print(f"\nNext: run build_snapshots.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Fetch and filter resolved Polymarket markets")
    parser.add_argument("--full", action="store_true",
                        help="Re-fetch all markets since START_DATE_MIN (ADR-013), resumable via data/fetch_cache/")
    main(full=parser.parse_args().full)