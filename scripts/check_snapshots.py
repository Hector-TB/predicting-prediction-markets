"""
scripts/check_snapshots.py
==========================
Full quality check of a snapshot build before it is published (ADR-023).
Streams the data in chunks, so it runs in well under 1 GB of RAM.

1. Raw build (data/polymarket_ml_dataset.csv), every row:
   structure (duplicates, one block per market, exact 12h spacing), window
   (burn-in, scheduled end, closedTime), agreement with meta (outcome, split,
   category), value invariants (price range, rolling-feature consistency).
2. Comparison with a published version's raw snapshots (default v2) on shared
   markets. Differences must be explained:
     - v3 raw rows are cleaned the same way first (fix_dataset.fix_chunk)
     - time features may differ only by a constant per market (dates changed)
     - rows only in the old version must be after closedTime
     - rows only in the new one must be in the last ~14 days (ADR-022)
3. Training files (*_clean.parquet): no market in two splits, cutoff respected
   (ADR-021), no settled prices, no NaN features, outcome = meta, and both
   clean files hold exactly the same rows.

Exits non-zero if anything is unexplained.

Usage:
    python scripts/check_snapshots.py [--compare-with v2]
"""

import argparse
import logging
import os
import sys
import tempfile
from collections import Counter
from pathlib import Path

import boto3
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
DATA = ROOT / "data"
sys.path.insert(0, str(ROOT / "data_collection_pipeline"))
from fix_dataset import fix_chunk  # noqa: E402
from fix_leakage import PRICE_THRESHOLD, split_cutoff  # noqa: E402
from stream_parquet import iter_frames  # noqa: E402

RAW_CSV    = DATA / "polymarket_ml_dataset.csv"
CLEAN      = DATA / "polymarket_ml_dataset_clean.parquet"
CLEAN_TR   = DATA / "polymarket_ml_dataset_with_trends_clean.parquet"
CHUNK_ROWS = 400_000
TEXT_COLS  = {"market_id", "snapshot_timestamp", "split", "category", "question", "ts"}
NEW_WINDOW_DAYS = 14.5     # ADR-022 rows, plus 12h for rounding at the old boundary

log = logging.getLogger(__name__)
EXAMPLES: dict[str, set] = {}     # issue -> market ids, written to logs/ for follow-up


def note_ids(name: str, ids) -> None:
    EXAMPLES.setdefault(name, set()).update(ids)


def market_chunks(path: Path, rows: int = CHUNK_ROWS):
    """Yield DataFrames of whole markets (a market is never split across chunks)."""
    carry = None
    for df in pd.read_csv(path, chunksize=rows, low_memory=False):
        if carry is not None:
            df = pd.concat([carry, df], ignore_index=True)
        last = df["market_id"].iloc[-1]
        carry = df[df["market_id"] == last]
        out = df[df["market_id"] != last]
        if len(out):
            yield out.copy()
    if carry is not None and len(carry):
        yield carry.copy()


def load_reference(meta: pd.DataFrame) -> pd.DataFrame:
    closed = pd.read_csv(DATA / "market_closed_times.csv")
    closed["closed_time"] = pd.to_datetime(closed["closed_time"], utc=True, format="mixed")
    m = meta.merge(closed, on="market_id", how="left").set_index("market_id")
    m["start"] = pd.to_datetime(m["start_date"], format="ISO8601", utc=True)
    m["end"]   = pd.to_datetime(m["end_date"], format="ISO8601", utc=True)
    return m


def check_raw(m: pd.DataFrame, old_path: Path | None) -> tuple[Counter, Counter, int]:
    issues, notes = Counter(), Counter()
    seen: set = set()
    n_rows = 0
    for i, df in enumerate(market_chunks(RAW_CSV), 1):
        n_rows += len(df)
        df["ts"] = pd.to_datetime(df["snapshot_timestamp"], format="mixed", utc=True)
        ids = df["market_id"]

        # structure
        issues["duplicate (market, timestamp)"] += int(df.duplicated(["market_id", "ts"]).sum())
        chunk_ids = set(ids.unique())
        issues["market in more than one block"] += len(chunk_ids & seen)
        seen |= chunk_ids
        blocks = (ids != ids.shift()).cumsum()
        issues["market in more than one block"] += int((blocks.groupby(ids).nunique() > 1).sum())
        issues["gap != 12h within a market"] += int((df.groupby("market_id")["ts"].diff().dropna()
                                                     != pd.Timedelta(hours=12)).sum())
        # window + meta
        start, end, closed = ids.map(m["start"]), ids.map(m["end"]), ids.map(m["closed_time"])
        second = pd.Timedelta(seconds=1)
        issues["market not in meta"] += int(start.isna().sum())
        issues["before 14-day burn-in"] += int((df["ts"] < start + pd.Timedelta(days=14) - second).sum())
        issues["after scheduled end"] += int((df["ts"] > end + second).sum())
        issues["after closedTime"] += int((df["ts"] > closed + second).sum())
        for col in ["outcome", "split", "category"]:
            issues[f"{col} differs from meta"] += int((df[col].astype(str) != ids.map(m[col]).astype(str)).sum())
        # values
        notes["price outside [0,1] (clipped by fix_dataset)"] += int(
            ((df["price_at_snapshot"] < 0) | (df["price_at_snapshot"] > 1)).sum())
        issues["days_before_close < 0"] += int((df["days_before_close"] < -0.01).sum())
        issues["pct_lifetime_elapsed outside [0,1]"] += int(
            ((df["pct_lifetime_elapsed"] < 0) | (df["pct_lifetime_elapsed"] > 1)).sum())
        for w in ["7d", "14d"]:
            x = df[df[f"price_mean_{w}"].notna()]
            issues[f"{w}: not min <= mean <= max"] += int(
                ((x[f"price_min_{w}"] > x[f"price_mean_{w}"] + 1e-4) |
                 (x[f"price_mean_{w}"] > x[f"price_max_{w}"] + 1e-4)).sum())
            issues[f"{w}: range != max - min"] += int(
                ((x[f"price_range_{w}"] - (x[f"price_max_{w}"] - x[f"price_min_{w}"])).abs() > 2e-4).sum())
            notes[f"{w}: NaN rolling features (filled by fix_dataset)"] += int(df[f"price_mean_{w}"].isna().sum())

        if old_path is not None:
            compare_chunk(df, m, old_path, issues, notes)
        if i % 3 == 0:
            log.info("  ... %s rows checked", f"{n_rows:,}")
    return issues, notes, n_rows


def compare_chunk(new: pd.DataFrame, m: pd.DataFrame, old_path: Path,
                  issues: Counter, notes: Counter) -> None:
    ids = list(new["market_id"].unique())
    old = pq.read_table(old_path, filters=[("market_id", "in", ids)]).to_pandas()
    if old.empty:
        notes["markets not in the old version"] += len(ids)
        return
    old["ts"] = pd.to_datetime(old["snapshot_timestamp"], format="mixed", utc=True)
    notes["markets not in the old version"] += len(set(ids) - set(old["market_id"]))
    new = new[new["market_id"].isin(set(old["market_id"]))].copy()
    new, _ = fix_chunk(new)                      # same cleaning the old parquet had

    j = old.merge(new, on=["market_id", "ts"], how="outer", suffixes=("_old", "_new"), indicator=True)
    both = j[j["_merge"] == "both"]
    notes["rows in both versions"] += len(both)

    # Markets whose end date changed since the old build: days_before_close is
    # shifted by the same amount on every shared row. Their window and
    # pct_lifetime_elapsed legitimately differ too (ADR-023 date corrections).
    dbc = both["days_before_close_new"] - both["days_before_close_old"]
    shifted = dbc.abs() > 1e-6
    spread = dbc[shifted].groupby(both.loc[shifted, "market_id"]).agg(["min", "max"])
    end_changed = set(spread.index[(spread["max"] - spread["min"]).abs() <= 0.02])
    issues["days_before_close: non-constant differences"] += int(len(spread) - len(end_changed))
    note_ids("days_before_close: non-constant differences", set(spread.index) - end_changed)
    notes["markets whose end date changed since the old build"] += len(end_changed)

    def classify(rows: pd.DataFrame, ok: pd.Series, ok_note: str, bad_issue: str) -> None:
        dated = ~ok & rows["market_id"].isin(end_changed)
        notes[ok_note] += int(ok.sum())
        notes[f"{bad_issue} — explained by a changed end date"] += int(dated.sum())
        issues[bad_issue] += int((~ok & ~dated).sum())
        note_ids(bad_issue, rows.loc[~ok & ~dated, "market_id"])

    old_only = j[j["_merge"] == "left_only"]
    classify(old_only, old_only["ts"] > old_only["market_id"].map(m["closed_time"]),
             "old-only rows after closedTime (no longer built)", "old-only rows NOT after closedTime")
    new_only = j[j["_merge"] == "right_only"]
    classify(new_only, new_only["days_before_close_new"] <= NEW_WINDOW_DAYS,
             "new-only rows in the last 14 days (ADR-022)", "new-only rows outside the last 14 days")

    feats = [c[:-4] for c in both.columns if c.endswith("_old") and c[:-4] not in TEXT_COLS]
    for c in feats:
        if c == "days_before_close":
            continue                                    # handled above
        a, b = both[c + "_old"].astype(float), both[c + "_new"].astype(float)
        # pct_lifetime_elapsed is stored at 4 decimals: a sub-second start-time
        # difference can move it by one unit of rounding
        tol = 1.01e-4 if c == "pct_lifetime_elapsed" else 1e-6
        differs = pd.Series(~np.isclose(a, b, atol=tol, equal_nan=True), index=both.index)
        if c in ("pct_lifetime_elapsed", "duration_days"):
            differs &= ~both["market_id"].isin(end_changed)   # follows from the changed dates
        issues[f"{c}: values differ"] += int(differs.sum())
        note_ids(f"{c}: values differ", both.loc[differs, "market_id"])


def check_clean(meta: pd.DataFrame, closed_times: pd.DataFrame) -> tuple[Counter, dict]:
    issues = Counter()
    cutoff = split_cutoff(closed_times)
    outcome = meta.set_index("market_id")["outcome"]
    markets: dict[str, set] = {}
    rng: dict[str, list] = {}
    rows: Counter = Counter()
    digest = {}
    for path in (CLEAN, CLEAN_TR):
        h, n = 0, 0
        for df in iter_frames(path):
            n += len(df)
            h = (h + int(pd.util.hash_pandas_object(df[["market_id", "snapshot_timestamp"]],
                                                     index=False).sum())) % 2**64
            if path != CLEAN:
                continue
            ts = pd.to_datetime(df["snapshot_timestamp"], format="mixed", utc=True)
            for split, g in df.groupby("split"):
                markets.setdefault(split, set()).update(g["market_id"].unique())
                rows[split] += len(g)
                t = ts[g.index]
                lo, hi = t.min(), t.max()
                rng[split] = [lo, hi] if split not in rng else [min(rng[split][0], lo), max(rng[split][1], hi)]
            p = df["price_at_snapshot"]
            issues["settled price in training data"] += int(((p >= PRICE_THRESHOLD) | (p <= 1 - PRICE_THRESHOLD)).sum())
            feats = [c for c in df.columns if c not in TEXT_COLS]
            issues["NaN in features"] += int(df[feats].isna().sum().sum())
            issues["outcome differs from meta"] += int((df["outcome"] != df["market_id"].map(outcome)).sum())
        digest[path.name] = (n, h)

    issues["market in train and test"] = len(markets.get("train", set()) &
                                             (markets.get("test", set()) | markets.get("test_pre_cutoff", set())))
    issues["train row at/after cutoff"] = int(rng["train"][1] >= cutoff)
    issues["test row before cutoff"] = int(rng["test"][0] < cutoff)
    issues["test_pre_cutoff row at/after cutoff"] = int(rng.get("test_pre_cutoff", [cutoff, cutoff - pd.Timedelta(1)])[1] >= cutoff)
    issues["clean files differ in rows"] = int(len(set(digest.values())) != 1)
    summary = {"cutoff": cutoff, "rows": dict(rows), "markets": {k: len(v) for k, v in markets.items()},
               "range": {k: [str(a), str(b)] for k, (a, b) in rng.items()}, "files": digest}
    return issues, summary


def report(title: str, issues: Counter, notes: Counter | None = None) -> int:
    log.info("\n%s", title)
    for k, v in issues.items():
        log.info("  %s %s: %s", "OK " if v == 0 else "!! ", k, f"{v:,}")
    for k, v in (notes or {}).items():
        log.info("  ·   %s: %s", k, f"{v:,}")
    return sum(1 for v in issues.values() if v)


def main(compare_with: str | None) -> int:
    meta = pd.read_csv(DATA / "polymarket_markets_meta.csv", dtype={"clob_token_id": str})
    m = load_reference(meta)
    failed = 0
    with tempfile.TemporaryDirectory() as tmp:
        old_path = None
        if compare_with:
            load_dotenv(ROOT / ".env")
            old_path = Path(tmp) / f"{compare_with}_raw.parquet"
            log.info("Downloading %s raw snapshots for comparison ...", compare_with)
            boto3.client("s3", region_name=os.getenv("AWS_REGION", "us-east-1")).download_file(
                os.environ["S3_BUCKET"], f"datasets/{compare_with}/polymarket_ml_dataset.parquet", str(old_path))
        log.info("Checking %s ...", RAW_CSV.name)
        issues, notes, n = check_raw(m, old_path)
        failed += report(f"RAW BUILD — {n:,} rows" + (f" (vs {compare_with})" if compare_with else ""), issues, notes)

    closed = pd.read_csv(DATA / "market_closed_times.csv")
    closed["closed_time"] = pd.to_datetime(closed["closed_time"], utc=True, format="mixed")
    issues, s = check_clean(meta, closed)
    failed += report("TRAINING FILES", issues)
    log.info("  cutoff T: %s", s["cutoff"])
    for split in s["rows"]:
        log.info("  %-16s %10s rows  %7s markets  %s → %s", split, f"{s['rows'][split]:,}",
                 f"{s['markets'][split]:,}", s["range"][split][0][:16], s["range"][split][1][:16])
    if EXAMPLES:
        out = ROOT / "logs" / f"check_snapshots_markets_{pd.Timestamp.now():%Y%m%d_%H%M}.csv"
        pd.DataFrame([(k, i) for k, ids in EXAMPLES.items() for i in sorted(ids)],
                     columns=["issue", "market_id"]).to_csv(out, index=False)
        log.info("\nMarkets behind each flagged difference: %s", out.relative_to(ROOT))
    log.info("\n%s", "ALL CHECKS PASSED" if not failed else f"{failed} CHECKS FAILED")
    return failed


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    parser = argparse.ArgumentParser(description="Full quality check of a snapshot build")
    parser.add_argument("--compare-with", default="v2", help="published version to compare with ('' to skip)")
    sys.exit(1 if main(parser.parse_args().compare_with or None) else 0)
