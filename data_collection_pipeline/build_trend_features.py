import pandas as pd

INPUT_PATH  = "data/category_trends_raw.csv"
OUTPUT_PATH = "data/category_trends_features.parquet"

ALL_CATEGORIES = [
    "politics_us", "politics_global", "crypto", "sports",
    "finance", "geopolitics", "science_tech", "entertainment", "other",
]


def build_features(df):
    df = df.copy()
    df["week_start"] = pd.to_datetime(df["week_start"])
    df = df.sort_values(["category", "week_start"]).reset_index(drop=True)

    rows = []
    for category, group in df.groupby("category"):
        group = group.set_index("week_start").sort_index()
        group["trend_ma4"]      = group["trend_value"].rolling(4, min_periods=1).mean().round(2)
        group["trend_change_4w"] = (group["trend_value"] - group["trend_value"].shift(4)).round(2)
        group["trend_spike"]    = (group["trend_value"] > 1.5 * group["trend_ma4"]).astype(int)
        group["has_trend_data"] = 1
        group["category"]       = category
        group[["trend_change_4w"]] = group[["trend_change_4w"]].fillna(0)
        rows.append(group.reset_index())

    result = pd.concat(rows, ignore_index=True)

    # Add other category with all zeros
    other_weeks = result[result["category"] == result["category"].iloc[0]][["week_start"]].copy()
    other_weeks["category"]       = "other"
    other_weeks["trend_value"]    = 0.0
    other_weeks["trend_ma4"]      = 0.0
    other_weeks["trend_change_4w"] = 0.0
    other_weeks["trend_spike"]    = 0
    other_weeks["has_trend_data"] = 0
    result = pd.concat([result, other_weeks], ignore_index=True)

    return result[["category", "week_start", "trend_value", "trend_ma4", "trend_change_4w", "trend_spike", "has_trend_data"]]


def main():
    df = pd.read_csv(INPUT_PATH)
    print(f"Loaded {len(df)} rows from {INPUT_PATH}")

    features = build_features(df)
    features.to_parquet(OUTPUT_PATH, index=False)

    print(f"Saved {len(features)} rows to {OUTPUT_PATH}")
    print(f"\nCategories: {sorted(features['category'].unique())}")
    print(f"Date range: {features['week_start'].min().date()} to {features['week_start'].max().date()}")
    print(f"\nSpike counts per category:")
    print(features.groupby("category")["trend_spike"].sum().sort_values(ascending=False))


if __name__ == "__main__":
    main()
