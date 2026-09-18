from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

SNAPSHOTS_PATH = DATA_DIR / "polymarket_ml_dataset.parquet"
TRENDS_PATH    = DATA_DIR / "category_trends_features.parquet"
OUTPUT_PATH    = DATA_DIR / "polymarket_ml_dataset_with_trends.parquet"

TREND_COLS = ["trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data"]


def main():
    print("Loading snapshots...")
    snapshots = pd.read_parquet(SNAPSHOTS_PATH)
    original_rows = len(snapshots)
    print(f"  {original_rows:,} rows, columns: {list(snapshots.columns)}")

    print("Loading trend features...")
    trends = pd.read_parquet(TRENDS_PATH)
    trends["week_start"] = pd.to_datetime(trends["week_start"], utc=True)
    print(f"  {len(trends):,} rows across {trends['category'].nunique()} categories")

    snapshots["snapshot_timestamp"] = pd.to_datetime(snapshots["snapshot_timestamp"], format="ISO8601", utc=True)

    print("Merging...")
    merged_parts = []
    for category, snap_group in snapshots.groupby("category"):
        trend_group = trends[trends["category"] == category][["week_start"] + TREND_COLS].sort_values("week_start")
        snap_group  = snap_group.sort_values("snapshot_timestamp")

        merged = pd.merge_asof(
            snap_group,
            trend_group,
            left_on="snapshot_timestamp",
            right_on="week_start",
            direction="backward",
        ).drop(columns=["week_start"])

        merged_parts.append(merged)

    result = pd.concat(merged_parts, ignore_index=True)

    # Fill any snapshots that predate the trends data (before 2023-01-01)
    for col in ["trend_value", "trend_ma4", "trend_change_4w"]:
        result[col] = result[col].fillna(0.0)
    for col in ["trend_spike", "has_trend_data"]:
        result[col] = result[col].fillna(0).astype(int)

    assert len(result) == original_rows, f"Row count mismatch: {len(result)} != {original_rows}"

    result.to_parquet(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(result):,} rows to {OUTPUT_PATH}")
    print(f"New columns: {TREND_COLS}")
    print(f"\nhas_trend_data distribution:")
    print(result["has_trend_data"].value_counts())
    print(f"\ntrend_value stats:")
    print(result["trend_value"].describe().round(2))


if __name__ == "__main__":
    main()
