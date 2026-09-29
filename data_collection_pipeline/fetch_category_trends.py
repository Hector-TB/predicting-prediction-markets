import sys
import time
from datetime import date
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

OUTPUT_PATH = DATA_DIR / "category_trends_raw.csv"


START_DATE = "2023-01-01"


def get_timeframe() -> tuple[str, str]:
    """Return (start_date, end_date): always the full range, START_DATE to today.

    Google scales each request to 0–100 over its own timeframe, and returns daily
    (not weekly) points for ranges under ~9 months. Appending a short incremental
    fetch would therefore mix scales and granularities, so every run re-fetches
    the whole range and replaces the file.
    """
    return START_DATE, date.today().strftime("%Y-%m-%d")


def fetch_category(pytrends, category, keywords, timeframe: str):
    print(f"Fetching: {category} {keywords}")
    pytrends.build_payload(keywords, timeframe=timeframe, geo="")
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
    start_date, end_date = get_timeframe()
    timeframe = f"{start_date} {end_date}"

    print(f"Fetching trends for timeframe: {timeframe}")
    pytrends    = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    all_results = []
    failed      = []

    for i, (category, keywords) in enumerate(CATEGORY_KEYWORDS.items()):
        try:
            result = fetch_category(pytrends, category, keywords, timeframe)
        except Exception as e:
            print(f"  ERROR: {category}: {e}")
            result = None
        if result is None:
            failed.append(category)
        else:
            all_results.append(result)
        if i < len(CATEGORY_KEYWORDS) - 1:
            time.sleep(3)

    # Keep the previous file unless every category came back: a partial file
    # would silently zero out the missing categories' trend features.
    if failed:
        print(f"\nFAILED for {failed} — {OUTPUT_PATH.name} left unchanged.")
        sys.exit(1)

    combined = pd.concat(all_results, ignore_index=True)
    combined["week_start"] = pd.to_datetime(combined["week_start"])
    gaps = combined.sort_values("week_start").groupby("category")["week_start"].diff().dropna()
    if not (gaps == pd.Timedelta(days=7)).all():
        print(f"\nERROR: expected weekly points, got spacings {sorted(gaps.unique())} — "
              f"{OUTPUT_PATH.name} left unchanged.")
        sys.exit(1)

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH} "
          f"({combined['week_start'].min().date()} to {combined['week_start'].max().date()})")
    print(combined.groupby("category")["trend_value"].describe().round(2))


if __name__ == "__main__":
    main()
