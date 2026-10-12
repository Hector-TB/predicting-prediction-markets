"""
Record the outcomes of looked-up and sampled markets (ADR-026)
=============================================================
Every market with resolution_status 'open' is looked up on Gamma; closed ones
get the training outcome rule (ADR-004): final YES price >= 0.95 -> YES,
<= 0.05 -> NO, otherwise 'ambiguous'. Run daily, before sample_markets.py.

    python scripts/resolve_lookups.py
    python scripts/resolve_lookups.py --dry-run     # report, write nothing
"""

import argparse
import logging
import sys
from datetime import datetime, timezone
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
sys.path.insert(0, str(ROOT))

from fetch_markets import OUTCOME_THRESHOLD, parse_dt  # noqa: E402

import api  # noqa: E402,F401  (sets up the import paths)
from api import gamma, store  # noqa: E402

log = logging.getLogger(__name__)


def resolution(m: dict) -> tuple[str, int | None]:
    """(resolution_status, outcome) of a closed Gamma market, by the ADR-004 rule."""
    price = gamma.yes_price(m)
    if price is not None and price >= OUTCOME_THRESHOLD:
        return "resolved", 1
    if price is not None and price <= 1 - OUTCOME_THRESHOLD:
        return "resolved", 0
    return "ambiguous", None


def main(dry_run: bool) -> None:
    with store.connect() as conn, conn.cursor() as cur:
        cur.execute("SELECT market_id FROM markets WHERE resolution_status = 'open'")
        open_ids = [r[0] for r in cur.fetchall()]
        log.info("%d open markets", len(open_ids))
        closed = gamma.closed_markets(open_ids) if open_ids else []
        now = datetime.now(timezone.utc)
        counts = {"resolved": 0, "ambiguous": 0}
        for m in closed:
            status, outcome = resolution(m)
            counts[status] += 1
            log.info("  %s %s outcome=%s  %s", status, m["conditionId"][:12], outcome,
                     (m.get("question") or "")[:70])
            if not dry_run:
                cur.execute(
                    """UPDATE markets SET resolution_status = %s, outcome = %s, closed_time = %s,
                                          yes_final_price = %s, resolved_at = %s
                       WHERE market_id = %s AND resolution_status = 'open'""",
                    (status, outcome, parse_dt(m.get("closedTime")), gamma.yes_price(m), now,
                     m["conditionId"]))
        if dry_run:
            conn.rollback()
    conn.close()
    log.info("Closed: %d resolved, %d ambiguous; %d still open%s", counts["resolved"],
             counts["ambiguous"], len(open_ids) - len(closed), " (dry run: nothing written)" if dry_run else "")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dry-run", action="store_true")
    from dotenv import load_dotenv
    load_dotenv(ROOT / ".env")
    main(parser.parse_args().dry_run)
