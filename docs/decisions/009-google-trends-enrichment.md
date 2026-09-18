# ADR-009: Google Trends Enrichment at Category Level

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/fetch_category_trends.py`, `build_trend_features.py`, `merge_trends.py`

---

## Context

The core hypothesis of RQ3 is that external signals (beyond the market price) add predictive value, especially early in a market's lifecycle. Google Trends is a free, publicly available signal that measures search interest — a proxy for public attention, which may lead market price movements.

The question is: at what granularity should we fetch Trends data? Market-level (one query per market question) or category-level (one query per category)?

## Decision

Fetch Trends at the **category level**: one set of keywords per category, one weekly time series per category, covering 2023-01-01 to 2026-02-06.

Each snapshot is enriched by joining to the category's Trends time series on the most recent week before the snapshot timestamp. Four features are added:

| Feature | Description |
|---|---|
| `trend_value` | Weekly search interest (0–100) |
| `trend_ma4` | 4-week moving average |
| `trend_change_4w` | Change from 4 weeks prior |
| `trend_spike` | Binary: interest > 1.5× MA4 |

## Rationale

**Category-level vs market-level:** Market-level queries would require one pytrends API call per market (~24k calls), which hits rate limits and takes days. Category-level requires only 8 calls. More importantly, most individual market questions are too specific for Google Trends to return meaningful data (Trends requires a certain search volume threshold).

**Weekly granularity:** Trends API returns weekly data; daily data requires a shorter time range. Given our 12-hour snapshot interval, weekly Trends granularity means many snapshots in the same week share the same trend value — but this is a known limitation.

**Coverage: 99.6%** of snapshots have `has_trend_data = 1`. The 0.4% without it are mostly the `other` category (no defined keyword set) and a small number of early `politics_us` snapshots predating the 2023 trends window.

## Alternatives considered

- **Market-level Trends queries** — more precise but rate-limited and practically infeasible at scale; rejected
- **Daily Trends data** — would require fetching multiple shorter time ranges and stitching; adds complexity with marginal benefit at 12h snapshot frequency; rejected
- **No external enrichment** — baseline approach; included for comparison (non-trends models)
- **News API or LLM sentiment** — richer but costly and complex; out of scope for course project

## Assumptions

1. Category-level search interest is a reasonable proxy for market-level attention
2. The 8-keyword sets per category are representative (e.g., sports = ["NFL", "NBA", "soccer", "MLB", "tennis"])
3. Weekly granularity is sufficient for the prediction task
4. The `TIMEFRAME = "2023-01-01 2026-02-06"` covers the full market date range

## Consequences

- Adds 5 columns; 99.6% coverage
- The `other` category (0.6% of markets) has zero trend coverage — those markets are either excluded from trends models or zero-filled
- Trends model and non-trends model cannot be fairly compared on identical test sets unless the `other` category is excluded
- The trends enrichment requires pytrends library and has no API key requirement (Google public data), but is subject to rate limiting

## Related ADRs

- ADR-008: Market filters (the 9 categories that Trends is built on top of)
- ADR-006: SVM subsampling (SVM trains two versions: with and without trends)
