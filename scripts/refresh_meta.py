"""
scripts/refresh_meta.py
=======================
Makes every Gamma field in polymarket_markets_meta.csv match Gamma (ADR-023).

Source of truth, in order:
  1. data/fetch_cache/*.csv — the latest full keyset fetch (fetch_markets.py --full)
  2. a direct Gamma lookup by conditionId for meta markets missing from the cache,
     parsed and filtered with fetch_markets.py's own rules; saved to
     data/fetch_cache/lookups.csv so later checks can verify them

Every Gamma field is replaced; only `category` and `split` are kept. Markets
Gamma no longer returns, or that now fail the market filters, are dropped:
their values can't be verified. Outcome changes are reported separately.

Afterwards, re-run scripts/recompute_split.py (the market set may change).

Usage:
    python scripts/refresh_meta.py            # dry run
    python scripts/refresh_meta.py --apply
"""

import argparse
import logging
import shutil
import sys
import time
from pathlib import Path

import pandas as pd
import requests

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
from fetch_markets import (  # noqa: E402
    FETCH_CACHE_DIR, GAMMA_URL, OUTPUT_META, VOLUME_NUM_MIN, merge_fresh, parse_and_filter_markets,
)
from meta_checks import check_against_gamma, check_meta, load_fetch_cache  # noqa: E402

LOOKUPS_CSV = FETCH_CACHE_DIR / "lookups.csv"
BATCH_SIZE  = 100

log = logging.getLogger(__name__)


def lookup(ids: list[str]) -> tuple[pd.DataFrame, set]:
    """Fetch markets by conditionId; return (parsed + filtered, ids Gamma returned)."""
    raw = []
    for i in range(0, len(ids), BATCH_SIZE):
        r = requests.get(f"{GAMMA_URL}/markets",
                         params={"condition_ids": ids[i:i + BATCH_SIZE], "closed": "true",
                                 "limit": BATCH_SIZE},
                         timeout=30)
        r.raise_for_status()
        raw.extend(r.json())
        time.sleep(0.2)
    returned = {m.get("conditionId") for m in raw}
    # Same volume floor the keyset fetch applies server-side
    raw = [m for m in raw if float(m.get("volumeNum") or 0) >= VOLUME_NUM_MIN]
    return parse_and_filter_markets(raw, verbose=False), returned


def main(apply: bool) -> None:
    meta  = pd.read_csv(OUTPUT_META, dtype={"clob_token_id": str})
    cache = load_fetch_cache()
    log.info("Meta: %s markets — fetch cache: %s markets", f"{len(meta):,}", f"{cache['market_id'].nunique():,}")

    missing = sorted(set(meta["market_id"]) - set(cache["market_id"]))
    log.info("Not in the cache: %s — looking them up on Gamma ...", f"{len(missing):,}")
    looked_up, returned = lookup(missing) if missing else (pd.DataFrame(columns=cache.columns), set())
    log.info("  Gamma returned %s; %s pass the market filters", f"{len(returned):,}", f"{len(looked_up):,}")

    fresh = pd.concat([cache, looked_up], ignore_index=True).drop_duplicates("market_id", keep="last")
    fresh = fresh[fresh["market_id"].isin(meta["market_id"])]
    for col in ("start_date", "end_date"):
        fresh[col] = pd.to_datetime(fresh[col], format="ISO8601", utc=True)

    # Unverifiable markets are dropped rather than kept with stale values
    unverifiable = meta[~meta["market_id"].isin(fresh["market_id"])]
    for _, row in unverifiable.iterrows():
        why = "fails market filters" if row["market_id"] in returned else "not returned by Gamma"
        log.info("  drop %s  (%s)  %s", row["market_id"][:12], why, str(row["question"])[:60])
    kept = meta[meta["market_id"].isin(fresh["market_id"])]

    merged, n_updated, n_added = merge_fresh(kept, fresh)
    assert n_added == 0, "refresh must never add markets"

    before = kept.set_index("market_id")
    after  = merged.set_index("market_id").loc[before.index]
    for col in ["clob_token_id", "question", "start_date", "end_date", "duration_days",
                "total_volume", "yes_final_price", "outcome"]:
        a, b = before[col].astype(str), after[col].astype(str)
        if col in ("start_date", "end_date"):
            a = pd.to_datetime(before[col], format="ISO8601", utc=True).astype(str)
            b = pd.to_datetime(after[col], format="ISO8601", utc=True).astype(str)
        log.info("  %-16s changed for %s markets", col, f"{(a != b).sum():,}")
    flipped = before.index[before["outcome"] != after["outcome"]]
    if len(flipped):
        log.warning("  OUTCOME CHANGED for %d markets (label changes!): %s", len(flipped), list(flipped[:5]))

    problems = check_meta(merged)
    problems.update(check_against_gamma(merged, fresh.reset_index(drop=True)))
    if problems:
        for name, ids in problems.items():
            log.error("  still failing — %s: %s markets", name, f"{len(ids):,}")
        raise SystemExit("Refreshed meta still fails checks — not writing.")
    log.info("\nRefreshed meta: %s markets (%s dropped), all checks pass",
             f"{len(merged):,}", f"{len(unverifiable):,}")

    if not apply:
        log.info("Dry run — re-run with --apply to write.")
        return
    backup = OUTPUT_META.with_name(OUTPUT_META.stem + ".pre_refresh.bak.csv")
    shutil.copy2(OUTPUT_META, backup)
    merged.to_csv(OUTPUT_META, index=False)
    if len(looked_up):
        looked_up.to_csv(LOOKUPS_CSV, index=False)
    log.info("Saved %s (backup: %s)%s", OUTPUT_META.name, backup.name,
             f"; {len(looked_up)} lookups → {LOOKUPS_CSV.name}" if len(looked_up) else "")
    log.info("Next: python scripts/recompute_split.py")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Make meta match Gamma (ADR-023)")
    parser.add_argument("--apply", action="store_true")
    main(parser.parse_args().apply)
