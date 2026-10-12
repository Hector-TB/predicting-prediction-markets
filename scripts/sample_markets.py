"""
Daily random sample of open markets for the live track record (ADR-026)
======================================================================
Draws markets uniformly at random from the open markets the models were trained
on (ADR-008 filters, 14-day burn-in, unsettled price, end date in the future),
skipping any market sampled before, predicts each with the production release
and stores it like a lookup with source 'sample'. The draw order is seeded by
the UTC date. Run daily, after resolve_lookups.py.

    python scripts/sample_markets.py               # 20 markets
    python scripts/sample_markets.py --n 5 --dry-run   # predict, store nothing
"""

import argparse
import logging
import random
import sys
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
sys.path.insert(0, str(ROOT))

from features import FetchError  # noqa: E402
from fetch_markets import (  # noqa: E402
    MIN_DURATION_DAYS,
    START_DATE_MIN,
    VOLUME_NUM_MIN,
    parse_dt,
    parse_json_field,
)

import api  # noqa: E402,F401  (sets up the import paths)
from api import gamma, releases, store  # noqa: E402
from api.predict import BURN_IN_DAYS, SETTLED, PredictError, predict_raw  # noqa: E402

log = logging.getLogger(__name__)

N_PER_DAY = 20
# ADR-025 warnings that put a market outside the training data (ADR-026); near_close,
# long_market and no_category stay in, as in the backtest test set
REJECT = frozenset({"few_prices", "young_market", "settled_price", "outside_filters",
                    "no_end_date", "past_end_date"})


def eligible(m: dict, now: pd.Timestamp) -> bool:
    """Whether a Gamma market passes the training filters as far as its listing shows."""
    created, end, start = parse_dt(m.get("createdAt")), parse_dt(m.get("endDate")), parse_dt(m.get("startDate"))
    price = gamma.yes_price(m)
    return (gamma.is_binary(m) and not m.get("closed") and m.get("conditionId") is not None
            and bool(parse_json_field(m.get("clobTokenIds"), []))
            and created is not None and end is not None and start is not None
            and start >= pd.Timestamp(START_DATE_MIN, tz="UTC")
            and float(m.get("volumeNum") or 0) >= VOLUME_NUM_MIN
            and end > now and (end - created).days >= MIN_DURATION_DAYS
            and created <= now - pd.Timedelta(days=BURN_IN_DAYS)
            and price is not None and 1 - SETTLED < price < SETTLED)


def draw_order(markets: list[dict], seed: str) -> list[dict]:
    """The markets in a random order that depends only on `seed` and the set of markets."""
    ordered = sorted(markets, key=lambda m: m["conditionId"])
    random.Random(seed).shuffle(ordered)
    return ordered


def main(n: int, dry_run: bool) -> None:
    release = releases.load()
    now = pd.Timestamp(datetime.now(timezone.utc))
    listing = gamma.open_markets(volume_num_min=VOLUME_NUM_MIN, start_date_min=START_DATE_MIN,
                                 end_date_min=now.strftime("%Y-%m-%d"))
    with store.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT DISTINCT market_id FROM snapshots WHERE source = 'sample'")
        sampled = {r[0] for r in cur.fetchall()}
    conn.close()
    # Shuffled before dropping sampled markets, so a re-run on the same day keeps the order
    order = draw_order([m for m in listing if eligible(m, now)], now.strftime("%Y-%m-%d"))
    pool = [m for m in order if m["conditionId"] not in sampled]
    log.info("Open markets listed: %d; eligible: %d; not sampled before: %d",
             len(listing), len(order), len(pool))

    stored, skipped = 0, {}
    for m in pool:
        if stored == n:
            break
        try:
            result, market = predict_raw(m, release, reject=REJECT)
        except (PredictError, FetchError, gamma.GammaError) as e:
            code = getattr(e, "code", type(e).__name__)
            skipped[code] = skipped.get(code, 0) + 1
            log.info("  skip %s (%s: %s)", m["conditionId"][:12], code, e)
            continue
        snapshot_id = None if dry_run else store.save_lookup(result, market, release, source="sample")
        stored += 1
        head = next(p for p in result["predictions"] if p["model"] == release.headline)
        log.info("  %2d. %s price %.3f  %s %.3f  %s  %s", stored, market.market_id[:12], market.price,
                 release.headline, head["probability"], [w["code"] for w in result["warnings"]],
                 market.question[:60])
    log.info("%s %d markets with %s (skipped: %s)%s", "Predicted" if dry_run else "Stored", stored,
             release.name, skipped or "none", " (dry run: nothing written)" if dry_run else "")
    if stored < n:
        raise SystemExit(f"Only {stored} of {n} markets could be sampled.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--n", type=int, default=N_PER_DAY)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    main(args.n, args.dry_run)
