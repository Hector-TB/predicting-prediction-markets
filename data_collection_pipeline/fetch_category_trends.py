import time
from pathlib import Path

import pandas as pd
from pytrends.request import TrendReq

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"

CATEGORY_KEYWORDS = {
    "politics_us":     ["Trump", "Biden", "Congress", "White House", "election"],
    "politics_global": ["world news", "international news", "United Nations", "foreign policy", "global politics"],
    "crypto":          ["Bitcoin", "Ethereum", "cryptocurrency", "crypto", "blockchain"],
    "sports":          ["NFL", "NBA", "soccer", "MLB", "tennis"],
    "finance":         ["stock market", "Federal Reserve", "interest rates", "inflation", "S&P 500"],
    "geopolitics":     ["Ukraine war", "Middle East", "Gaza", "Russia", "NATO"],
    "science_tech":    ["artificial intelligence", "ChatGPT", "SpaceX", "climate change", "technology"],
    "entertainment":   ["Taylor Swift", "Netflix", "movies", "music", "celebrity"],
}

TIMEFRAME = "2023-01-01 2026-02-06"
OUTPUT_PATH = DATA_DIR / "category_trends_raw.csv"


def fetch_category(pytrends, category, keywords):
    print(f"Fetching: {category} {keywords}")
    pytrends.build_payload(keywords, timeframe=TIMEFRAME, geo="")
    df = pytrends.interest_over_time()

    if df.empty:
        print(f"  WARNING: empty response for {category}")
        return None

    if "isPartial" in df.columns:
        df = df[~df["isPartial"]].drop(columns=["isPartial"])

    df["trend_value"] = df[keywords].mean(axis=1).round(2)
    df["category"] = category

    result = df[["category", "trend_value"]].reset_index()
    result.rename(columns={"date": "week_start"}, inplace=True)
    print(f"  {len(result)} weeks fetched. Peak: {df['trend_value'].max():.1f} on {df['trend_value'].idxmax().date()}")
    return result


def main():
    pytrends = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    all_results = []

    for i, (category, keywords) in enumerate(CATEGORY_KEYWORDS.items()):
        result = fetch_category(pytrends, category, keywords)
        if result is not None:
            all_results.append(result)
        if i < len(CATEGORY_KEYWORDS) - 1:
            time.sleep(3)

    if not all_results:
        print("No data fetched.")
        return

    combined = pd.concat(all_results, ignore_index=True)
    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH}")
    print(combined.groupby("category")["trend_value"].describe().round(2))


if __name__ == "__main__":
    main()
