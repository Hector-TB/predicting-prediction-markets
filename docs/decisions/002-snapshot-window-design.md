# ADR-002: Snapshot Window Design (12h interval, 14d burn-in, 14d cutoff)

**Date:** 2026-09-18  
**Status:** Accepted  
**Deciders:** Dhairya Dhamani, Hector Thompson Baroni, Sachin Sastri  
**Code location:** `data_collection_pipeline/build_snapshots.py:37–39`

---

## Context

For each market we need to decide: how frequently to sample, when to start sampling, and when to stop. These choices affect the size of the dataset, the quality of rolling features, and whether we inadvertently include periods where the outcome is already obvious.

## Decision

- **Interval:** one snapshot every **12 hours**
- **Burn-in:** first snapshot no earlier than `createdAt + 14 days`
- **Cutoff:** last snapshot no later than `endDate − 14 days`
- Markets where `burn-in start >= cutoff end` are dropped entirely

## Rationale

**12-hour interval:** Balances dataset size vs information density. 24h is too coarse — significant price moves happen overnight. 6h triples the dataset with minimal new signal (prices are autocorrelated at sub-day scale).

**14-day burn-in:** The 7d and 14d rolling features require 7–14 days of price history to be non-trivially computed. Starting earlier means those features would be NaN or filled with the first price, adding noise. 14 days ensures both windows are populated for every snapshot we keep.

**14-day cutoff before close:** Prediction markets often converge sharply to 0 or 1 in the final days as the outcome becomes obvious. Including those snapshots would inflate AUC (predicting from near-certain prices is trivially easy) and mislead evaluation. 14 days is a conservative buffer.

## Alternatives considered

- **6h interval** — doubles dataset size for marginal additional signal; chose 12h
- **7-day burn-in** — shorter burn-in means more early snapshots, but 7d rolling features would be partially NaN; rejected
- **7-day cutoff before close** — tighter, would include more late-market "easy" predictions; 14d is more conservative
- **Variable cutoff based on price** — eventually replaced the fixed 14-day rule via `fix_leakage.py` (see ADR-005); this decision represents the original design intent

## Assumptions

1. 12-hour granularity is sufficient to capture meaningful price dynamics
2. Rolling windows up to 14 days are the most informative timescales (supported by feature correlation analysis in README)
3. The last 14 days before close are not useful for the prediction task

## Consequences

- 3,144 of 24,092 markets (13%) are dropped for having insufficient snapshot windows (too short to satisfy both burn-in and cutoff simultaneously)
- Dropped markets are not systematically biased by category or volume
- Dataset size: ~1.4M rows (manageable in memory with parquet)

## Related ADRs

- ADR-003: Rolling window selection (why 7d and 14d)
- ADR-005: Leakage fix (replaces the fixed 14d cutoff with a price-based one)
