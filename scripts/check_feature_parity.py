"""
Check that the serving feature path reproduces the training snapshots
=====================================================================
Samples markets from the local clean parquet, re-fetches their CLOB price
history and recomputes every stored snapshot with `features.features_at` (the
function the API uses). Every model feature must match the stored row.

    python scripts/check_feature_parity.py              # 50 markets
    python scripts/check_feature_parity.py --n 200 --seed 1

Needs network access (CLOB). Exits 1 on any mismatch.
"""

import argparse
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
sys.path.insert(0, str(ROOT))

from features import FetchError, fetch_price_history, features_at  # noqa: E402
from models.logistic_regression.train import NUMERIC_FEATURES  # noqa: E402

DATA_DIR = ROOT / "data"
CLEAN    = DATA_DIR / "polymarket_ml_dataset_clean.parquet"
META     = DATA_DIR / "polymarket_markets_meta.csv"
ATOL     = 1e-9

log = logging.getLogger(__name__)


def main(n: int, seed: int) -> int:
    meta = pd.read_csv(META, dtype={"clob_token_id": str}).set_index("market_id")
    meta["start_date"] = pd.to_datetime(meta["start_date"], format="ISO8601", utc=True)
    meta["end_date"]   = pd.to_datetime(meta["end_date"], format="ISO8601", utc=True)

    ids = pd.read_parquet(CLEAN, columns=["market_id"])["market_id"].unique()
    sample = pd.Series(ids).sample(min(n, len(ids)), random_state=seed).tolist()
    rows = pd.read_parquet(CLEAN, columns=["market_id", "snapshot_timestamp"] + NUMERIC_FEATURES,
                           filters=[("market_id", "in", sample)])
    rows["snapshot_timestamp"] = pd.to_datetime(rows["snapshot_timestamp"], utc=True)
    log.info("Checking %d markets, %s snapshot rows, %d features",
             len(sample), f"{len(rows):,}", len(NUMERIC_FEATURES))

    n_rows = n_bad_rows = n_skipped = 0
    bad_features: dict[str, float] = {}
    for i, market_id in enumerate(sample, 1):
        m = meta.loc[market_id]
        try:
            price_df = fetch_price_history(m["clob_token_id"])
        except FetchError as e:
            log.warning("  %s: fetch failed (%s) — skipped", market_id, e)
            n_skipped += 1
            continue
        if price_df is None:
            log.warning("  %s: no price history now — skipped", market_id)
            n_skipped += 1
            continue

        stored = rows[rows["market_id"] == market_id]
        market_bad = 0
        for _, r in stored.iterrows():
            got = features_at(price_df, r["snapshot_timestamp"], m["start_date"],
                              m["end_date"], int(m["duration_days"]))
            n_rows += 1
            if got is None:
                market_bad += 1
                bad_features["<too few prices>"] = np.inf
                continue
            diffs = {f: abs(float(got[f]) - float(r[f])) for f in NUMERIC_FEATURES}
            wrong = {f: d for f, d in diffs.items() if not d <= ATOL}
            if wrong:
                market_bad += 1
                for f, d in wrong.items():
                    bad_features[f] = max(bad_features.get(f, 0.0), d)
        n_bad_rows += market_bad
        if market_bad:
            log.warning("  %s: %d of %d rows differ", market_id, market_bad, len(stored))
        if i % 10 == 0:
            log.info("  %d/%d markets, %s rows, %d differ", i, len(sample), f"{n_rows:,}", n_bad_rows)
        time.sleep(0.15)

    log.info("\nRows compared: %s  (markets skipped: %d)", f"{n_rows:,}", n_skipped)
    if n_bad_rows:
        log.error("FAIL: %s rows differ. Max abs difference per feature:", f"{n_bad_rows:,}")
        for f, d in sorted(bad_features.items(), key=lambda x: -x[1]):
            log.error("  %-28s %g", f, d)
        return 1
    log.info("OK: every feature matches the training rows")
    return 0


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    parser.add_argument("--n", type=int, default=50, help="markets to sample (default 50)")
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    sys.exit(main(args.n, args.seed))
