"""
Step 1: Fetch and Filter Markets
==================================
Fetches resolved binary markets from Polymarket Gamma API,
applies filters, and saves a clean market metadata file.

Output: polymarket_markets_meta.csv

Run this first, then run build_snapshots.py

Modes:
  (default)   incremental (ADR-020): closed markets whose *scheduled end date* is
              on or after (last fetch − 30 days), no upper bound — catches markets
              that closed on time, early (future end date) or up to 30 days late.
              The last fetch time comes from data/fetch_state.json, else the
              local dataset manifest's fetched_on.
  --full      re-fetch everything since START_DATE_MIN by start date (ADR-013).
              Monthly checkpoints are only reused with --resume (an interrupted run).
  --dry-run   fetch and report what would change; write nothing.

Every run saves the Gamma records it used to data/fetch_cache/ (meta_checks.py
verifies the meta against them, ADR-023) and, when it writes the meta, records
its start time in data/fetch_state.json.
"""

import argparse
import logging
import requests
import pandas as pd
import numpy as np
import json
import time
from datetime import datetime, timezone
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

LOOKBACK_DAYS       = 30            # markets closing up to this late after their scheduled end (ADR-020)

OUTPUT_META         = DATA_DIR / "polymarket_markets_meta.csv"
FETCH_CACHE_DIR     = DATA_DIR / "fetch_cache"   # Gamma records (evidence, ADR-023) + --full checkpoints
FETCH_STATE         = DATA_DIR / "fetch_state.json"  # when the meta's markets were last fetched (ADR-020)
LOCAL_MANIFEST      = DATA_DIR / "manifest.json"

log = logging.getLogger(__name__)

META_COLUMNS = [
    "market_id", "clob_token_id", "question", "category", "start_date", "end_date",
    "duration_days", "total_volume", "yes_final_price", "outcome",
]


# ─────────────────────────────────────────────
# INCREMENTAL SUPPORT
# ─────────────────────────────────────────────

# Our own columns, kept from the existing meta on merge; everything else is
# Gamma's and is always replaced by the fresh fetch (ADR-023)
OWN_COLUMNS = ["category", "split"]


def merge_fresh(existing: pd.DataFrame, fresh: pd.DataFrame) -> tuple[pd.DataFrame, int, int]:
    """
    Merge freshly fetched markets into the existing meta.

    - Markets in both: every Gamma field comes from `fresh`; only `category`
      (LLM) and `split` are kept from `existing`. Keeping the old row instead
      once preserved estimated dates for 14,886 markets (ADR-023).
    - Markets only in `fresh`: added with split = "test" (ADR-021).
    - Markets only in `existing`: kept unchanged (not returned by this fetch).

    Returns (merged, n_updated, n_added).
    """
    fresh = fresh.drop_duplicates("market_id", keep="last")
    gamma_cols = [c for c in fresh.columns if c not in OWN_COLUMNS]
    old = existing.set_index("market_id")
    new = fresh.set_index("market_id")

    both = old.index.intersection(new.index)
    updated = new.loc[both, [c for c in gamma_cols if c != "market_id"]].join(old.loc[both, OWN_COLUMNS])
    added = new.loc[new.index.difference(old.index)].copy()
    added["split"] = "test"
    kept = old.loc[old.index.difference(new.index)]

    merged = pd.concat([f for f in (kept, updated, added) if len(f)]).reset_index()
    merged = merged[existing.columns.tolist()]
    merged["_start"] = pd.to_datetime(merged["start_date"], format="ISO8601", utc=True)
    merged = merged.sort_values("_start").drop(columns="_start").reset_index(drop=True)
    return merged, len(both), len(added)


def load_existing_meta() -> Optional[pd.DataFrame]:
    if not OUTPUT_META.exists():
        return None
    df = pd.read_csv(OUTPUT_META, dtype={"clob_token_id": str})
    print(f"  Found {len(df):,} existing markets")
    return df


def last_fetch_time() -> Optional[pd.Timestamp]:
    """
    When the markets in the meta were last fetched: data/fetch_state.json, else
    the checked-out dataset version's `fetched_on` (ADR-016). Stored explicitly,
    never inferred from the data (ADR-020).
    """
    if FETCH_STATE.exists():
        return pd.Timestamp(json.loads(FETCH_STATE.read_text())["last_fetch"])
    if LOCAL_MANIFEST.exists():
        fetched_on = json.loads(LOCAL_MANIFEST.read_text()).get("fetched_on")
        if fetched_on:
            return pd.Timestamp(fetched_on, tz="UTC")
    return None


def save_fetch_state(started: pd.Timestamp, mode: str, n_markets: int) -> None:
    FETCH_STATE.write_text(json.dumps({
        "last_fetch": started.isoformat(), "mode": mode, "markets": n_markets,
    }, indent=2) + "\n")


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


def _fetch_window(start_date: str, end_date: Optional[str], by: str = "start") -> list[dict]:
    """Cursor-paginate one date window via /markets/keyset (ADR-013).

    by="start" windows on Gamma's startDate (the full fetch); by="end" on the
    scheduled endDate (the incremental fetch and scripts/coverage_check.py,
    ADR-020). end_date=None leaves the window open-ended.

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
            f"{by}_date_min": start_date,
        }
        if end_date:
            params[f"{by}_date_max"] = end_date
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


def in_scope(raw_markets: list[dict]) -> list[dict]:
    """Markets whose Gamma startDate is on or after START_DATE_MIN — the scope the
    full fetch gets from its start_date_min filter. Needed when fetching by end date."""
    floor = pd.Timestamp(START_DATE_MIN, tz="UTC")
    out = []
    for m in raw_markets:
        start = parse_dt(m.get("startDate"))
        if start is not None and start >= floor:
            out.append(m)
    return out


def fetch_incremental(since: pd.Timestamp) -> pd.DataFrame:
    """
    Closed markets whose scheduled end date is on or after `since`, with no upper
    bound (ADR-020), filtered like the full fetch. Walked in monthly windows up
    to today, then one open-ended window for markets that closed early.
    """
    today = pd.Timestamp.now(tz="UTC").strftime("%Y-%m-%d")
    windows = list(_date_windows(since.strftime("%Y-%m-%d"), today, months=1)) + [(today, None)]
    log.info("=" * 60)
    log.info("STEP 1: Incremental fetch by scheduled end date (ADR-020)")
    log.info("=" * 60)
    log.info("  end_date >= %s (last fetch − %d days), no upper bound; volume >= %s",
             since.date(), LOOKBACK_DAYS, f"{VOLUME_NUM_MIN:,}")

    frames = []
    for i, (wstart, wend) in enumerate(windows, 1):
        raw = _fetch_window(wstart, wend, by="end")
        df = parse_and_filter_markets(in_scope(raw), verbose=False)
        log.info("  [%d/%d] end %s → %s  %7d fetched  %5d passing",
                 i, len(windows), wstart, wend or "open", len(raw), len(df))
        frames.append(df)
    frames = [f for f in frames if not f.empty]
    if not frames:
        return pd.DataFrame(columns=META_COLUMNS)
    return pd.concat(frames, ignore_index=True).drop_duplicates("market_id", keep="first").reset_index(drop=True)


def fetch_filtered_markets(start_date_min: str, cache_dir: Optional[Path] = None,
                           resume: bool = False) -> pd.DataFrame:
    """Fetch monthly windows from start_date_min to today and filter each as it arrives.

    Filtering per window keeps memory flat: most raw markets are short-duration
    and discarded immediately. With cache_dir set, each fully elapsed window is
    written to a CSV. Those files are reused only with resume=True (continuing an
    interrupted run): markets that started in a month keep closing afterwards, so
    a checkpoint from an earlier run is stale (ADR-020).
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
        if resume and cache_file and cache_file.exists():
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

def main(full: bool = False, resume: bool = False, dry_run: bool = False):
    print("\n" + "█" * 60)
    print("  POLYMARKET — FETCH & FILTER MARKETS")
    print("█" * 60 + "\n")

    # Recorded before fetching: a market that closes mid-run is picked up next time
    started = pd.Timestamp.now(tz="UTC").floor("s")
    existing_df = load_existing_meta()

    if full or existing_df is None:
        # Re-fetch everything (ADR-013). Gamma fields are refreshed on merge;
        # LLM categories and the split are kept (ADR-023).
        mode = "full"
        log.info("  Full fetch from %s%s", START_DATE_MIN, " (resuming from checkpoints)" if resume else "")
        new_df = fetch_filtered_markets(START_DATE_MIN, cache_dir=FETCH_CACHE_DIR, resume=resume)
    else:
        mode = "incremental"
        last = last_fetch_time()
        if last is None:
            raise SystemExit("ERROR: no record of the last fetch (data/fetch_state.json or data/manifest.json "
                             "fetched_on). Pull a dataset version or run with --full (ADR-020).")
        log.info("  Last fetch: %s", last)
        new_df = fetch_incremental(last - pd.Timedelta(days=LOOKBACK_DAYS))

    if new_df.empty and existing_df is None:
        print("No markets passed filters. Exiting.")
        return

    if existing_df is not None:
        # Fresh Gamma values replace the stored ones for every re-fetched market;
        # only category and split are ours (ADR-023). New markets always go to
        # test (ADR-021) — safe even for one that resolved before the cutoff:
        # fix_leakage.py drops every test row dated before the cutoff.
        markets_df, n_updated, n_added = merge_fresh(existing_df, new_df)
        print(f"\n  {n_updated:,} existing markets refreshed from Gamma, {n_added:,} new")
    else:
        # First run — the split needs resolution times, which recompute_split.py
        # looks up (ADR-021). Until then everything is test.
        markets_df = new_df.sort_values("start_date").reset_index(drop=True)
        markets_df["split"] = "test"
        log.warning("  First run: all markets marked test — run scripts/recompute_split.py next (ADR-021).")

    from meta_checks import check_meta
    problems = check_meta(markets_df)
    if problems:
        raise SystemExit(f"ERROR: merged meta fails integrity checks, not saved: "
                         f"{ {k: len(v) for k, v in problems.items()} } (ADR-023)")

    if dry_run:
        print(f"\nDry run — nothing written. Would save {len(markets_df):,} markets "
              f"({n_added if existing_df is not None else len(markets_df):,} new).")
        return new_df

    # Keep the exact Gamma records behind this meta, so meta_checks.py can
    # verify every row (ADR-023). One file per run, never overwritten; the
    # newest file wins in load_fetch_cache().
    FETCH_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(FETCH_CACHE_DIR / f"fetch_{started:%Y%m%d_%H%M%S}_{mode}.csv", index=False)

    markets_df.to_csv(OUTPUT_META, index=False)
    save_fetch_state(started, mode, len(markets_df))
    print(f"\nSaved: {OUTPUT_META}")
    print(f"  Markets: {len(markets_df):,}")
    print(f"  Last fetch recorded: {started} ({FETCH_STATE.name})")
    if full and existing_df is not None:
        log.warning("\nFull re-fetch merged under the OLD frozen split. "
                    "Run scripts/recompute_split.py before build_snapshots.py (ADR-013, ADR-021).")
    else:
        print(f"\nNext: run build_snapshots.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Fetch and filter resolved Polymarket markets")
    parser.add_argument("--full", action="store_true",
                        help="Re-fetch all markets since START_DATE_MIN (ADR-013)")
    parser.add_argument("--resume", action="store_true",
                        help="with --full: reuse checkpoints in data/fetch_cache/ from an interrupted run")
    parser.add_argument("--dry-run", action="store_true",
                        help="fetch and report, but write nothing")
    args = parser.parse_args()
    if args.resume and not args.full:
        parser.error("--resume only applies to --full")
    main(full=args.full, resume=args.resume, dry_run=args.dry_run)
