import logging
from pathlib import Path

import pandas as pd

from stream_parquet import FrameWriter, iter_frames

log = logging.getLogger(__name__)

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SNAPSHOTS_PATH = DATA_DIR / "polymarket_ml_dataset.parquet"
TRENDS_PATH    = DATA_DIR / "category_trends_features.parquet"
OUTPUT_PATH    = DATA_DIR / "polymarket_ml_dataset_with_trends.parquet"

TREND_COLS = ["trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data"]


def merge_batch(snapshots: pd.DataFrame, trends: pd.DataFrame) -> pd.DataFrame:
    """Attach each snapshot's most recent weekly trend row for its category."""
    snapshots["snapshot_timestamp"] = pd.to_datetime(snapshots["snapshot_timestamp"], format="ISO8601", utc=True)
    merged = pd.merge_asof(
        snapshots.sort_values("snapshot_timestamp"),
        trends,
        left_on="snapshot_timestamp",
        right_on="week_start",
        by="category",
        direction="backward",
    ).drop(columns=["week_start"])

    # Fill snapshots that predate the trends data or have no category mapping
    for col in ["trend_value", "trend_ma4", "trend_change_4w"]:
        merged[col] = merged[col].fillna(0.0)
    for col in ["trend_spike", "has_trend_data"]:
        merged[col] = merged[col].fillna(0).astype(int)
    return merged


def main():
    log.info("Loading trend features...")
    trends = pd.read_parquet(TRENDS_PATH)
    trends["week_start"] = pd.to_datetime(trends["week_start"], utc=True)
    trends = trends[["category", "week_start"] + TREND_COLS].sort_values("week_start")
    log.info("  %s rows across %d categories", f"{len(trends):,}", trends["category"].nunique())

    # Streamed in batches: the full dataset doesn't fit in memory on 8 GB machines.
    # merge_asof is per-row (each snapshot looks up its own category's latest
    # week), so batching doesn't change the result.
    log.info("Merging %s in batches...", SNAPSHOTS_PATH.name)
    rows_in = 0
    has_trend = 0
    with FrameWriter(OUTPUT_PATH) as out:
        for snapshots in iter_frames(SNAPSHOTS_PATH):
            rows_in += len(snapshots)
            merged = merge_batch(snapshots, trends)
            has_trend += int(merged["has_trend_data"].sum())
            out.write(merged)

    assert out.rows == rows_in, f"Row count mismatch: {out.rows} != {rows_in}"

    log.info("\nSaved %s rows to %s", f"{out.rows:,}", OUTPUT_PATH)
    log.info("New columns: %s", TREND_COLS)
    log.info("has_trend_data: %s of %s rows (%.1f%%)",
             f"{has_trend:,}", f"{rows_in:,}", 100 * has_trend / max(rows_in, 1))


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    main()
