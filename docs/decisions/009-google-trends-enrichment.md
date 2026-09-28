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

## Amendment (2026-09-28): join on completed weeks, full-range re-fetch

Found while reviewing the pipeline before the first v2 run.

1. **Join on completed weeks.** The Decision above says each snapshot joins to the most recent week *before* the snapshot. The code joined on `week_start`, which picks the week that *contains* the snapshot. That week isn't over yet, so the features could see up to 6 days after the snapshot. `merge_trends.py` now joins on `available_at = week_start + 7 days`. Snapshots therefore see trend values one week later than before.
2. **Full-range re-fetch every run.** `fetch_category_trends.py` used to fetch only the weeks after the last saved one and append them. Google scales each request to 0–100 over that request's own timeframe, and returns daily points for ranges shorter than about 9 months. A 2-week incremental fetch would have appended daily rows on a different scale. The script now always fetches from 2023-01-01 to today, checks that the points are weekly, and replaces the file only when every category succeeds (it exits non-zero otherwise).

**Consequences:** Trend values in v2 are not comparable with v1: the scale changes with the timeframe, and the join is shifted by a week. A limitation remains: because each series is scaled over the whole timeframe, its 0–100 scale depends on the later peak. That is mild look-ahead, and it is unavoidable with Google Trends. Live prediction (ADR-017) must use the same completed-week rule.

## Related ADRs

- ADR-008: Market filters (the 9 categories that Trends is built on top of)
- ADR-006: SVM subsampling (SVM trains two versions: with and without trends)
