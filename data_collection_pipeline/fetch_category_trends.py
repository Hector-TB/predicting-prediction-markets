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


def get_timeframe() -> tuple[str, str]:
    """Return (start_date, end_date) for the API call.

    On first run: full history from 2023-01-01.
    On subsequent runs: picks up from the week after the last fetched week.
    """
    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        max_week  = pd.to_datetime(existing["week_start"]).max()
        start_str = (max_week + pd.Timedelta(weeks=1)).strftime("%Y-%m-%d")
    else:
        start_str = "2023-01-01"
    end_str = date.today().strftime("%Y-%m-%d")
    return start_str, end_str


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

    if start_date >= end_date:
        print(f"Trends already up to date (coverage through {start_date}). Nothing to fetch.")
        return

    print(f"Fetching trends for timeframe: {timeframe}")
    pytrends    = TrendReq(hl="en-US", tz=0, timeout=(10, 25))
    all_results = []

    for i, (category, keywords) in enumerate(CATEGORY_KEYWORDS.items()):
        result = fetch_category(pytrends, category, keywords, timeframe)
        if result is not None:
            all_results.append(result)
        if i < len(CATEGORY_KEYWORDS) - 1:
            time.sleep(3)

    if not all_results:
        print("No data fetched.")
        return

    new_data = pd.concat(all_results, ignore_index=True)

    if OUTPUT_PATH.exists():
        existing = pd.read_csv(OUTPUT_PATH)
        combined = pd.concat([existing, new_data], ignore_index=True)
        combined = combined.drop_duplicates(subset=["category", "week_start"]).reset_index(drop=True)
    else:
        combined = new_data

    combined.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved {len(combined)} rows to {OUTPUT_PATH} ({len(new_data)} new rows appended)")
    print(combined.groupby("category")["trend_value"].describe().round(2))


if __name__ == "__main__":
    main()
