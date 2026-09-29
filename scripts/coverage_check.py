"""
scripts/coverage_check.py
=========================
Independent check that the meta CSV isn't missing markets (ADR-020).

fetch_markets.py finds markets by Gamma startDate. This script walks the same
Gamma catalogue by *endDate* instead, applies exactly the same filters
(volume floor server-side, parse_and_filter_markets), and compares:

  - qualifying markets not in meta, split into
      out of scope   Gamma startDate before START_DATE_MIN (by design)
      new            closed after the meta was fetched (the next fetch adds them)
      MISSED         everything else: a real gap
  - meta markets this route doesn't return (worth a look, not necessarily wrong)

Exits non-zero if any market was MISSED. Writes the missing markets to
logs/coverage_<date>.csv.

Usage:
    python scripts/coverage_check.py [--fetched-on 2026-09-25]   # default: the recorded last fetch
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
from fetch_markets import (  # noqa: E402
    OUTPUT_META, SLEEP_BETWEEN_CALLS, START_DATE_MIN, _date_windows, _fetch_window,
    last_fetch_time, parse_and_filter_markets,
)

END_DATE_MAX = "2031-01-01"   # closed markets can carry far-future scheduled end dates
log = logging.getLogger(__name__)


def main(fetched_on: str) -> int:
    meta = pd.read_csv(OUTPUT_META, usecols=["market_id"])
    meta_ids = set(meta["market_id"])
    fetched = pd.Timestamp(fetched_on, tz="UTC")
    log.info("Meta: %s markets (fetched %s)", f"{len(meta_ids):,}", fetched.date())

    raw_info: dict[str, dict] = {}
    passing = []
    windows = list(_date_windows(START_DATE_MIN, END_DATE_MAX, months=1))
    for i, (ws, we) in enumerate(windows, 1):
        raw = _fetch_window(ws, we, by="end")
        for m in raw:
            if m.get("conditionId"):
                raw_info[m["conditionId"]] = {"gamma_start": m.get("startDate"),
                                              "closed_time": m.get("closedTime")}
        df = parse_and_filter_markets(raw, verbose=False)
        passing.append(df)
        if i % 6 == 0 or i == len(windows):
            log.info("  [%3d/%d] end %s → %s: %s raw, %s passing so far",
                     i, len(windows), ws, we, f"{len(raw_info):,}",
                     f"{sum(len(d) for d in passing):,}")
        time.sleep(SLEEP_BETWEEN_CALLS)

    found = pd.concat([d for d in passing if len(d)], ignore_index=True).drop_duplicates("market_id")
    info = pd.DataFrame.from_dict(raw_info, orient="index")
    found = found.join(info, on="market_id")
    found["gamma_start"] = pd.to_datetime(found["gamma_start"], utc=True, format="ISO8601", errors="coerce")
    found["closed_time"] = pd.to_datetime(found["closed_time"], utc=True, format="mixed", errors="coerce")

    missing = found[~found["market_id"].isin(meta_ids)].copy()
    out_of_scope = missing["gamma_start"].isna() | (missing["gamma_start"] < pd.Timestamp(START_DATE_MIN, tz="UTC"))
    new = ~out_of_scope & (missing["closed_time"] > fetched)
    missing["reason"] = "MISSED"
    missing.loc[out_of_scope, "reason"] = "out of scope (startDate < 2023 or missing)"
    missing.loc[new, "reason"] = "new (closed after fetch)"
    not_found = meta_ids - set(found["market_id"])

    log.info("\nBy-end-date route: %s qualifying markets", f"{len(found):,}")
    log.info("  in meta:                 %s", f"{len(found) - len(missing):,}")
    for reason, n in missing["reason"].value_counts().items():
        log.info("  not in meta — %-40s %s", reason + ":", f"{n:,}")
    log.info("Meta markets this route didn't return: %s", f"{len(not_found):,}")

    out = ROOT / "logs" / f"coverage_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
    missing.to_csv(out, index=False)
    log.info("Missing markets written to %s", out.relative_to(ROOT))

    n_missed = int((missing["reason"] == "MISSED").sum())
    if n_missed:
        log.error("\n%d qualifying markets were MISSED by the fetch.", n_missed)
    return n_missed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Check meta covers every qualifying Gamma market")
    parser.add_argument("--fetched-on",
                        help="when the meta's markets were fetched (markets closed later count as new); "
                             "default: data/fetch_state.json, else the dataset manifest (ADR-020)")
    fetched_on = parser.parse_args().fetched_on or last_fetch_time()
    if fetched_on is None:
        sys.exit("ERROR: no recorded fetch time — pass --fetched-on")
    sys.exit(1 if main(str(fetched_on)) else 0)
