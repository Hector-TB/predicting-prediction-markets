"""
meta_checks.py
==============
Integrity checks for polymarket_markets_meta.csv (ADR-023).

Every field except `category` (LLM, ADR-008) and `split` (ADR-021) must be
exactly what Gamma returned. Nothing may be estimated or filled with a
placeholder. These checks catch the patterns that let estimated rows slip
into the dataset before:

  - empty clob_token_id
  - yes_final_price at the 0.05 / 0.95 placeholders, or inconsistent with outcome
  - duration_days that doesn't match end_date − start_date

build_snapshots.py and sync.py publish refuse to run while any check fails.

    python data_collection_pipeline/meta_checks.py      # report and exit non-zero on failure
"""

import logging
import sys
from pathlib import Path

import pandas as pd

ROOT     = Path(__file__).resolve().parent.parent
META_CSV = ROOT / "data" / "polymarket_markets_meta.csv"

OUTCOME_THRESHOLD = 0.95          # same rule as fetch_markets.py (ADR-004)
PLACEHOLDER_PRICES = {0.05, 0.95}  # written by the retired rebuild_meta_from_parquet.py
VALID_SPLITS = {"train", "test"}
REQUIRED = ["market_id", "clob_token_id", "question", "start_date", "end_date",
            "duration_days", "total_volume", "yes_final_price", "outcome", "split"]

log = logging.getLogger(__name__)


def check_meta(meta: pd.DataFrame) -> dict[str, list]:
    """Return {problem: [market_id, ...]} for every failed check (empty = all good)."""
    problems: dict[str, list] = {}

    def flag(name: str, mask: pd.Series) -> None:
        ids = meta.loc[mask, "market_id"].tolist() if "market_id" in meta else []
        if mask.any():
            problems[name] = ids

    missing_cols = [c for c in REQUIRED if c not in meta.columns]
    if missing_cols:
        return {f"missing columns: {missing_cols}": []}

    flag("duplicate market_id", meta["market_id"].duplicated(keep=False))
    for col in REQUIRED:
        empty = meta[col].isna() | (meta[col].astype(str).str.strip() == "")
        flag(f"empty {col}", empty)

    price = pd.to_numeric(meta["yes_final_price"], errors="coerce")
    flag("placeholder yes_final_price (0.05 / 0.95)", price.isin(PLACEHOLDER_PRICES))
    implied = pd.Series(pd.NA, index=meta.index, dtype="Int64")
    implied[price >= OUTCOME_THRESHOLD] = 1
    implied[price <= 1 - OUTCOME_THRESHOLD] = 0
    flag("outcome inconsistent with yes_final_price", implied.isna() | (implied != meta["outcome"]))

    start = pd.to_datetime(meta["start_date"], format="ISO8601", utc=True, errors="coerce")
    end   = pd.to_datetime(meta["end_date"], format="ISO8601", utc=True, errors="coerce")
    flag("unparseable start/end date", start.isna() | end.isna())
    flag("end_date not after start_date", end <= start)
    flag("duration_days != (end − start).days", (end - start).dt.days != meta["duration_days"])

    flag("split not train/test", ~meta["split"].isin(VALID_SPLITS))
    return problems


# Fields that come from Gamma unchanged; category and split are ours (ADR-023)
GAMMA_FIELDS = ["clob_token_id", "question", "start_date", "end_date", "duration_days",
                "total_volume", "yes_final_price", "outcome"]


def check_against_gamma(meta: pd.DataFrame, fresh: pd.DataFrame) -> dict[str, list]:
    """Compare meta with the latest Gamma records (e.g. data/fetch_cache/).

    This is the only check that catches *plausible but invented* values, such as
    dates estimated from snapshots. Markets missing from `fresh` are reported
    too: their values can't be verified.
    """
    problems: dict[str, list] = {}
    fresh = fresh.drop_duplicates("market_id", keep="last").set_index("market_id")
    missing = ~meta["market_id"].isin(fresh.index)
    if missing.any():
        problems["not in the latest Gamma fetch (unverifiable)"] = meta.loc[missing, "market_id"].tolist()
    m = meta.loc[~missing].set_index("market_id")
    f = fresh.loc[m.index]
    for col in GAMMA_FIELDS:
        a, b = m[col], f[col]
        if col in ("start_date", "end_date"):
            a = pd.to_datetime(a, format="ISO8601", utc=True, errors="coerce")
            b = pd.to_datetime(b, format="ISO8601", utc=True, errors="coerce")
            differs = (a - b).abs() > pd.Timedelta(seconds=1)
        elif col in ("total_volume", "yes_final_price"):
            differs = (pd.to_numeric(a) - pd.to_numeric(b)).abs() > 1e-6
        else:
            differs = a.astype(str) != b.astype(str)
        if differs.any():
            problems[f"{col} differs from Gamma"] = m.index[differs.values].tolist()
    return problems


def load_fetch_cache(cache_dir: Path = ROOT / "data" / "fetch_cache") -> pd.DataFrame:
    """Every Gamma record fetch_markets.py / refresh_meta.py have saved; for a
    market in several files, the most recently written file wins."""
    files = sorted(Path(cache_dir).glob("*.csv"), key=lambda f: f.stat().st_mtime)
    if not files:
        return pd.DataFrame()
    return pd.concat([pd.read_csv(f, dtype={"clob_token_id": str}) for f in files], ignore_index=True)


def assert_meta_ok(meta: pd.DataFrame, against_gamma: bool = True) -> None:
    """Raise SystemExit with a readable report if any check fails. With
    against_gamma, also compare with the saved Gamma records (fetch cache)."""
    problems = check_meta(meta)
    if against_gamma:
        cache = load_fetch_cache()
        if not len(cache):
            raise SystemExit("ERROR: no Gamma records in data/fetch_cache/ to verify meta against (ADR-023).")
        problems.update(check_against_gamma(meta, cache))
    if not problems:
        return
    lines = [f"  {name}: {len(ids):,} markets  (e.g. {ids[:3]})" for name, ids in problems.items()]
    raise SystemExit("ERROR: polymarket_markets_meta.csv failed integrity checks (ADR-023):\n"
                     + "\n".join(lines)
                     + "\nFix the meta CSV (scripts/refresh_meta.py) before building or publishing.")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    meta = pd.read_csv(META_CSV, dtype={"clob_token_id": str})
    problems = check_meta(meta)
    cache = load_fetch_cache()
    if len(cache):
        problems.update(check_against_gamma(meta, cache))
    else:
        log.warning("no fetch cache — values not compared with Gamma")
    if not problems:
        log.info("meta OK — %s markets pass every check", f"{len(meta):,}")
        sys.exit(0)
    for name, ids in problems.items():
        log.error("  %s: %s markets  (e.g. %s)", name, f"{len(ids):,}", ids[:3])
    sys.exit(1)
